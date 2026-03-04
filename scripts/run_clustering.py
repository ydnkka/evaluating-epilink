#!/usr/bin/env python3
"""
scripts/run_clustering.py

Build weighted networks from saved synthetic pairwise datasets and run Leiden
community detection across a grid of resolution parameters.

Config
------
config/paths.yaml
config/generate_datasets.yaml
config/clustering.yaml

Outputs
-------
data/processed/synthetic/scenario=<name>/
  - leiden_partitions.parquet (rows: case_id, gamma, cluster_id)
"""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import igraph as ig

from utils import *

_WORKER_CONFIG: dict[str, Any] | None = None


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


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    rng_seed = _WORKER_CONFIG.get("rng_seed", 12345)
    if rng_seed is not None:
        random.seed(rng_seed)


def cluster_scenario(
    scen: str,
    processed_dir: Path = None,
    weight_columns: list[str] = None,
    min_w: float = None,
    resolutions: np.ndarray = None,
    n_restarts: int = None,
) -> bool:
    if processed_dir is None:
        if _WORKER_CONFIG is None:
            raise RuntimeError("Worker config not initialized")
        processed_dir = _WORKER_CONFIG["processed_dir"]
        weight_columns = _WORKER_CONFIG["weight_columns"]
        min_w = _WORKER_CONFIG["min_w"]
        resolutions = _WORKER_CONFIG["resolutions"]
        n_restarts = _WORKER_CONFIG["n_restarts"]

    sc_dir = processed_dir / f"scenario={scen}"
    pw_path = sc_dir / "pairwise.parquet"
    if not pw_path.exists():
        return False

    df = pd.read_parquet(pw_path)
    df = df[df["Sampled"]].copy()

    rows = []
    for weight_col in weight_columns:
        graph = build_igraph_from_pairwise(df[["NodeA", "NodeB", weight_col]].dropna(), weight_col, min_w=min_w)
        for res in resolutions:
            best = None
            best_q = -np.inf

            # Restarts help avoid local optima
            for _ in range(n_restarts):
                part = graph.community_leiden(
                    weights=weight_col,
                    resolution=res,
                    n_iterations=-1
                )

                # Use modularity as a simple tie-breaker
                q = graph.modularity(
                    membership=part,
                    weights=weight_col,
                    resolution=res,
                    directed=False
                )

                if q > best_q:
                    best_q = q
                    best = part

            memb = best.membership
            rows.append(pd.DataFrame({
                "case_id": graph.vs["case_id"],
                "resolution": res,
                "cluster_id": np.array(memb, dtype=int),
                "weight_col": weight_col,
            }))

    if not rows:
        return False

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(sc_dir / "leiden_partitions.parquet", index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--datasets", default="../config/generate_datasets.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (<=1 for sequential)",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    datasets_cfg = load_yaml(Path(args.datasets))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))

    # Resolution grid
    gmin = float(deep_get(clus_cfg, ["community_detection", "resolution", "min"], 0.1))
    gmax = float(deep_get(clus_cfg, ["community_detection", "resolution", "max"], 1.0))
    gstep = float(deep_get(clus_cfg, ["community_detection", "resolution", "step"], 0.05))
    resolutions = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)
    n_restarts = int(deep_get(clus_cfg, ["community_detection", "n_restarts"], 10))
    rng_seed = int(deep_get(clus_cfg, ["rng_seed"], 12345))
    random.seed(rng_seed)
    weight_columns = list(deep_get(clus_cfg, ["network", "weight_columns"], []))
    min_w = float(deep_get(clus_cfg, ["network", "sparsify", "min_edge_weight"], 0.0001))
    sparsify = bool(deep_get(clus_cfg, ["network", "sparsify", "enabled"], True))
    if not sparsify:
        min_w = 0.0

    scenarios = deep_get(datasets_cfg, ["scenarios"], {})
    scenario_list = list(scenarios.keys())
    jobs = args.jobs if args.jobs is not None else (round(os.cpu_count() * 0.75) or 1)

    if jobs <= 1 or len(scenario_list) <= 1:
        for scen in scenario_list:
            print(f">>> Clustering: {scen}")
            cluster_scenario(
                scen=scen,
                processed_dir=processed_dir,
                weight_columns=weight_columns,
                min_w=min_w,
                resolutions=resolutions,
                n_restarts=n_restarts,
            )
    else:
        worker_config = {
            "processed_dir": processed_dir,
            "weight_columns": weight_columns,
            "min_w": min_w,
            "resolutions": resolutions,
            "n_restarts": n_restarts,
            "rng_seed": rng_seed,
        }
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(worker_config,),
        ) as executor:
            futures = {executor.submit(cluster_scenario, scen): scen for scen in scenario_list}
            for future in as_completed(futures):
                scen = futures[future]
                wrote = future.result()
                if wrote:
                    print(f">>> Clustered: {scen}")

    print(f"Saved tables to: {processed_dir}")

if __name__ == "__main__":
    main()
