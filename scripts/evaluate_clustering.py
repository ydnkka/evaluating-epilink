#!/usr/bin/env python3
"""
scripts/evaluate_clustering.py

Evaluate clustering partitions against simulation ground truth using BCubed
and quantify resolution stability (ARI between consecutive gammas).

This is a backbone: you’ll plug in your exact ground-truth construction and
BCubed implementation (especially for overlapping truth).

Outputs
-------
tables/supplementary/
  - clustering_metrics.parquet
  - clustering_stability.parquet

Config
------
config/paths.yaml
config/generate_datasets.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import igraph as ig
import bcubed

from utils import *


# -----------------------------
# Placeholders you should replace with your final definitions
# -----------------------------

def build_ground_truth_memberships(tree_path: Path) -> Dict[int, set[int]]:
    """
    Placeholder: construct overlapping ground-truth clusters from a populated tree.
    Return a mapping:
      case_id -> list of truth cluster IDs it belongs to

    Replace with your actual definition:
      - index case + direct descendants
      - or any other “true outbreak cluster” notion
    """
    # Construct overlapping ground truth clusters:
    # Each node is clustered with its direct successors (out-neighbours)
    g = ig.Graph.Read_GML(str(tree_path))
    clusters = []
    for node_id in range(g.vcount()):
        neighbours = set(g.successors(node_id))
        c = {node_id} | neighbours
        c = [g.vs[i]["label"] for i in c]
        clusters.append(c)

    # Build multi-membership maps
    membership = defaultdict(set)
    for clus_id, clus in enumerate(clusters):
        for node in clus:
            membership[int(node)].add(int(clus_id))
    return membership


def bcubed_scores(pred, truth):
    cases = sorted(set(pred) & set(truth))

    pred = {c: pred[c] for c in cases if pred[c]}
    truth = {c: truth[c] for c in cases if truth[c]}

    if not pred or not truth:
        raise ValueError("No valid cases with non-empty memberships")

    precision = bcubed.precision(pred, truth)
    recall = bcubed.recall(pred, truth)
    f_score = bcubed.fscore(precision, recall)
    return precision, recall, f_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/generate_datasets.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    ensure_dirs(processed_dir, tabs_dir)

    tree_path = Path(deep_get(scenarios_cfg, ["backbone", "tree_gml"],
                              "../data/processed/synthetic/scovmod_tree.gml"))
    scenarios = deep_get(scenarios_cfg, ["scenarios"], {})

    metrics_rows = []
    stability_rows = []
    truth = build_ground_truth_memberships(tree_path)

    for scen in scenarios.keys():
        print(f">>> Evaluating: {scen}")
        sc_dir = processed_dir / f"scenario={scen}"
        part_path = sc_dir / "leiden_partitions.parquet"

        if not part_path.exists() or not tree_path.exists():
            continue

        parts = pd.read_parquet(part_path)

        # --- Clustering evaluation against ground truth
        for (weight_col, gamma), sub in parts.groupby(["weight_col", "gamma"], observed=True):
            pred = dict(zip(sub["case_id"].tolist(), sub["cluster_id"].tolist()))
            pred = {int(k): {int(v)} for k, v in pred.items()}  # make non-overlapping into overlapping
            prec, rec, f1 = bcubed_scores(pred=pred, truth=truth)
            metrics_rows.append({
                "Scenario": scen,
                "gamma": float(gamma),
                "Weight_Column": weight_col,
                "BCubed_Precision": prec,
                "BCubed_Recall": rec,
                "BCubed_F1_Score": f1,
                "N_cases": len(pred),
            })

        # --- Stability between consecutive gammas
        gammas = np.sort(parts["gamma"].unique())
        for weight_col, sub in parts.groupby("weight_col", observed=True):
            for g1, g2 in zip(gammas[:-1], gammas[1:]):
                p1 = sub[sub["gamma"] == g1]
                p2 = sub[sub["gamma"] == g2]
                p1_mem = dict(zip(p1["case_id"].tolist(), p1["cluster_id"].tolist()))
                p1_mem = {int(k): {int(v)} for k, v in p1_mem.items()}
                p2_mem = dict(zip(p2["case_id"].tolist(), p2["cluster_id"].tolist()))
                p2_mem = {int(k): {int(v)} for k, v in p2_mem.items()}
                prec, rec, f1 = bcubed_scores(pred=p1_mem, truth=p2_mem)
                stability_rows.append({
                    "Scenario": scen,
                    "Weight_Column": weight_col,
                    "gamma1": float(g1),
                    "gamma2": float(g2),
                    "BCubed_Precision": prec,
                    "BCubed_Recall": rec,
                    "BCubed_F1_Score": f1,
                })

    pd.DataFrame(metrics_rows).to_parquet(tabs_dir / "clustering_metrics.parquet", index=False)
    pd.DataFrame(stability_rows).to_parquet(tabs_dir / "clustering_stability.parquet", index=False)

    print(f"Saved clustering metrics to: {tabs_dir / 'clustering_metrics.parquet'}")
    print(f"Saved clustering stability to: {tabs_dir / 'clustering_stability.parquet'}")

if __name__ == "__main__":
    main()
