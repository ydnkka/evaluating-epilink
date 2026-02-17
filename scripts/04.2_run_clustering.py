#!/usr/bin/env python3
"""
scripts/04.2_run_clustering.py

Build weighted networks from saved synthetic pairwise datasets and run Leiden
community detection across a grid of resolution parameters.

Outputs
-------
data/processed/synthetic/scenario=<name>/
  - leiden_partitions.parquet  (rows: case_id, gamma, cluster_id)
tables/supplementary/
  - leiden_summary.csv (optional quick summary)

Config
------
config/paths.yaml
config/clustering.yaml:
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import igraph as ig

from utils import *


def build_igraph_from_pairwise(df: pd.DataFrame, weight_col: str, min_w: float) -> ig.Graph:
    """
    Build an undirected weighted igraph from a pairwise table with columns:
      - NodeA, NodeB
      - weight_col
    """
    if min_w > 0:
        df = df[df[weight_col] >= min_w].copy()

    edges = df[["NodeA", "NodeB", weight_col]].to_records(index=False).tolist()

    return ig.Graph.TupleList(
        edges=edges,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=weight_col
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/simulate_datasets.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    ensure_dirs(processed_dir)

    # Resolution grid
    gmin = float(deep_get(clus_cfg, ["community_detection", "resolution", "min"], 0.1))
    gmax = float(deep_get(clus_cfg, ["community_detection", "resolution", "max"], 1.0))
    gstep = float(deep_get(clus_cfg, ["community_detection", "resolution", "step"], 0.05))
    gammas = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["MechProbLinearDist"]))

    # Network sparsification
    min_w = float(deep_get(clus_cfg, ["network", "sparsify", "min_edge_weight"], 0.01))
    sparsify = bool(deep_get(clus_cfg, ["network", "sparsify", "enabled"], True))
    if not sparsify:
        min_w = 0.0

    rng_seed = int(deep_get(clus_cfg, ["community_detection", "rng_seed"], 12345))
    import random
    random.seed(rng_seed)
    n_restarts = int(deep_get(clus_cfg, ["community_detection", "n_restarts"], 10))

    scenarios = deep_get(scenarios_cfg, ["scenarios"], {})
    for scen in scenarios.keys():
        print(f">>> Clustering: {scen}")
        sc_dir = processed_dir / f"scenario={scen}"
        pw_path = sc_dir / "pairwise_eval.parquet"
        if not pw_path.exists():
            continue

        df = pd.read_parquet(pw_path)
        df = df[df["Sampled"]].copy()

        rows = []
        for weight_col in weight_columns:
            graph = build_igraph_from_pairwise(df[["NodeA", "NodeB", weight_col]].dropna(), weight_col, min_w=min_w)
            for gamma in gammas:
                best = None
                best_q = -np.inf

                # Restarts help avoid local optima
                for _ in range(n_restarts):
                    part = graph.community_leiden(
                        weights=weight_col,
                        resolution=float(gamma),
                        n_iterations=-1
                    )

                    # Use modularity as a simple tie-breaker
                    q = graph.modularity(
                        membership=part,
                        weights=weight_col,
                        resolution=gamma,
                        directed=False
                    )

                    if q > best_q:
                        best_q = q
                        best = part

                memb = best.membership
                rows.append(pd.DataFrame({
                    "case_id": graph.vs["case_id"],
                    "gamma": float(gamma),
                    "cluster_id": np.array(memb, dtype=int),
                    "weight_col": weight_col,
                }))

        out = pd.concat(rows, ignore_index=True)
        out.to_parquet(sc_dir / "leiden_partitions.parquet", index=False)
        print(f"Saved Leiden partitions for {scen} to: {sc_dir / 'leiden_partitions.parquet'}")

    print("Done.")


if __name__ == "__main__":
    main()
