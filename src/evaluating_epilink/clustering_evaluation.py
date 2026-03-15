#!/usr/bin/env python3
"""Evaluate synthetic clustering results against overlapping reference clusters."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import config_value, ensure_directories, load_merged_config, load_yaml, resolve_path
from .execution import finish_stage_run, start_stage_run
from .metrics import bcubed_scores, build_star_memberships

_WORKER_TRUTH: Optional[dict[int, set[int]]] = None


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
            truth = build_star_memberships(tree_path)

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
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/synthetic.yaml")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (<=1 for sequential)",
    )
    args = parser.parse_args()
    stage_run = start_stage_run("clustering_evaluation", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    scenario_config_path = config_value(study_config, ["scenario_config"], "configs/scenarios/synthetic.yaml")
    scenario_config = load_yaml(scenario_config_path)

    tree_path = resolve_path(config_value(study_config, ["backbone", "tree_gml"]))
    processed_dir = resolve_path(config_value(study_config, ["paths", "data", "processed", "synthetic"]))
    results_dir = resolve_path(config_value(study_config, ["paths", "results", "clustering"]))
    ensure_directories(results_dir)

    scenarios = config_value(scenario_config, ["scenarios"], {})

    truth = build_star_memberships(tree_path)

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

    clustering_metrics_path = results_dir / "clustering_metrics.parquet"
    clustering_stability_path = results_dir / "clustering_stability.parquet"
    pd.DataFrame(metrics_rows).to_parquet(clustering_metrics_path, index=False)
    pd.DataFrame(stability_rows).to_parquet(clustering_stability_path, index=False)

    manifest_config = study_config | {"scenario_definitions": scenario_config}
    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "clustering_evaluation.json",
        config=manifest_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "scenario_config": resolve_path(scenario_config_path),
            "tree_path": str(tree_path),
            "processed_dir": str(processed_dir),
        },
        outputs={
            "results_dir": str(results_dir),
            "metrics_path": str(clustering_metrics_path),
            "stability_path": str(clustering_stability_path),
        },
        summary={
            "num_configured_scenarios": len(scenario_list),
            "num_metric_rows": len(metrics_rows),
            "num_stability_rows": len(stability_rows),
            "jobs": jobs,
        },
    )

    print(f"Saved clustering metrics to: {clustering_metrics_path}")
    print(f"Saved clustering stability to: {clustering_stability_path}")

if __name__ == "__main__":
    main()
