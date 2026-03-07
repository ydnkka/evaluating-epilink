#!/usr/bin/env python3
"""
scripts/evaluate_clustering.py

Evaluate clustering partitions against simulation ground truth using BCubed
and quantify resolution stability (BCubed F1-score between consecutive gammas).

Config
------
config/paths.yaml
config/generate_datasets.yaml

Outputs
-------
tables/clustering/
  - clustering_metrics.parquet
  - clustering_stability.parquet
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

import igraph as ig
import bcubed

from utils import *

_WORKER_TRUTH: Optional[dict[int, set[int]]] = None


# -----------------------------
# Placeholders you should replace with your final definitions
# -----------------------------

def build_ground_truth_memberships(tree_path: Path) -> dict[int, set[int]]:
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


def _init_worker(truth: dict[int, set[int]]) -> None:
    global _WORKER_TRUTH
    _WORKER_TRUTH = truth


def evaluate_scenario(
    scen: str,
    processed_dir: Path,
    tree_path: Path,
    truth: Optional[dict[int, set[int]]] = None,
) -> tuple[list[dict], list[dict]]:
    if truth is None:
        truth = _WORKER_TRUTH
        if truth is None:
            truth = build_ground_truth_memberships(tree_path)

    sc_dir = processed_dir / f"scenario={scen}"
    part_path = sc_dir / "leiden_partitions.parquet"

    if not part_path.exists() or not tree_path.exists():
        return [], []

    parts = pd.read_parquet(part_path)
    metrics_rows = []
    stability_rows = []

    # --- Clustering evaluation against ground truth
    for (weight_col, res), sub in parts.groupby(["weight_col", "resolution"], observed=True):
        pred = {int(k): {int(v)} for k, v in zip(sub["case_id"].tolist(), sub["cluster_id"].tolist())}
        prec, rec, f1 = bcubed_scores(pred=pred, truth=truth)
        metrics_rows.append({
            "Scenario": scen,
            "Resolution": res,
            "Weight_Column": weight_col,
            "BCubed_Precision": prec,
            "BCubed_Recall": rec,
            "BCubed_F1_Score": f1,
            "N_cases": len(pred),
        })

    # --- Stability between consecutive resolutions
    resolutions = np.sort(parts["resolution"].unique())
    for weight_col, sub in parts.groupby("weight_col", observed=True):
        for res1, res2 in zip(resolutions[:-1], resolutions[1:]):
            p1 = sub[sub["resolution"] == res1]
            p2 = sub[sub["resolution"] == res2]
            p1_mem = dict(zip(p1["case_id"].tolist(), p1["cluster_id"].tolist()))
            p1_mem = {int(k): {int(v)} for k, v in p1_mem.items()}
            p2_mem = dict(zip(p2["case_id"].tolist(), p2["cluster_id"].tolist()))
            p2_mem = {int(k): {int(v)} for k, v in p2_mem.items()}
            prec, rec, f1 = bcubed_scores(pred=p1_mem, truth=p2_mem)
            stability_rows.append({
                "Scenario": scen,
                "Weight_Column": weight_col,
                "Res1": res1,
                "Res2": res2,
                "BCubed_Precision": prec,
                "BCubed_Recall": rec,
                "BCubed_F1_Score": f1,
            })

    return metrics_rows, stability_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--datasets", default="../config/generate_datasets.yaml")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (<=1 for sequential)",
    )
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    datasets_cfg = load_yaml(Path(args.datasets))

    tree_path = Path(
        deep_get(datasets_cfg, ["backbone", "tree_gml"], "../data/processed/synthetic/scovmod_tree.gml")
    )
    processed_dir = Path(
        deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic")
    )
    tabs_dir = Path(
        deep_get(paths_cfg, ["outputs", "tables"], "../tables")
    )
    tabs_dir = tabs_dir / "clustering"
    ensure_dirs(tabs_dir)

    scenarios = deep_get(datasets_cfg, ["scenarios"], {})

    truth = build_ground_truth_memberships(tree_path)

    metrics_rows = []
    stability_rows = []
    scenario_list = list(scenarios.keys())
    jobs = args.jobs if args.jobs is not None else (round(os.cpu_count() * 0.75)  or 1)

    if jobs <= 1 or len(scenario_list) <= 1:
        for scen in scenario_list:
            print(f">>> Evaluating: {scen}")
            scen_metrics, scen_stability = evaluate_scenario(
                scen=scen,
                processed_dir=processed_dir,
                tree_path=tree_path,
                truth=truth,
            )
            metrics_rows.extend(scen_metrics)
            stability_rows.extend(scen_stability)
    else:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(truth,),
        ) as executor:
            futures = {
                executor.submit(
                    evaluate_scenario,
                    scen,
                    processed_dir,
                    tree_path,
                    None,
                ): scen
                for scen in scenario_list
            }
            for future in as_completed(futures):
                scen = futures[future]
                scen_metrics, scen_stability = future.result()
                if scen_metrics or scen_stability:
                    print(f">>> Evaluated: {scen}")
                metrics_rows.extend(scen_metrics)
                stability_rows.extend(scen_stability)

    pd.DataFrame(metrics_rows).to_parquet(tabs_dir / "clustering_metrics.parquet", index=False)
    pd.DataFrame(stability_rows).to_parquet(tabs_dir / "clustering_stability.parquet", index=False)

    print(f"Saved clustering metrics to: {tabs_dir / 'clustering_metrics.parquet'}")
    print(f"Saved clustering stability to: {tabs_dir / 'clustering_stability.parquet'}")

if __name__ == "__main__":
    main()
