#!/usr/bin/env python3
"""Run Leiden community detection on scored synthetic pairwise datasets."""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_value, load_merged_config, load_yaml, resolve_path
from .execution import finish_stage_run, start_stage_run
from .graphs import build_weighted_graph, partition_to_frame, run_leiden_partition

_WORKER_CONFIG: dict[str, Any] | None = None


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    rng_seed = _WORKER_CONFIG.get("rng_seed", 12345)
    if rng_seed is not None:
        random.seed(rng_seed)


def cluster_scenario(
    scenario_name: str,
    processed_dir: Path | None = None,
    weight_columns: list[str] | None = None,
    min_w: float | None = None,
    resolutions: np.ndarray | None = None,
    n_restarts: int | None = None,
) -> bool:
    if processed_dir is None:
        if _WORKER_CONFIG is None:
            raise RuntimeError("Worker config not initialized")
        processed_dir = _WORKER_CONFIG["processed_dir"]
        weight_columns = _WORKER_CONFIG["weight_columns"]
        min_w = _WORKER_CONFIG["min_w"]
        resolutions = _WORKER_CONFIG["resolutions"]
        n_restarts = _WORKER_CONFIG["n_restarts"]

    scenario_dir = processed_dir / f"scenario={scenario_name}"
    pairwise_path = scenario_dir / "pairwise_scored.parquet"
    if not pairwise_path.exists():
        return False

    pairwise_frame = pd.read_parquet(pairwise_path)

    rows = []
    for weight_column in weight_columns:
        graph = build_weighted_graph(
            pairwise_frame,
            weight_column=weight_column,
            minimum_weight=float(min_w),
        )
        for resolution in resolutions:
            partition, _ = run_leiden_partition(
                graph,
                weight_column=weight_column,
                resolution=float(resolution),
                num_restarts=int(n_restarts),
                rng_seed=_WORKER_CONFIG.get("rng_seed") if _WORKER_CONFIG else None,
            )
            rows.append(
                partition_to_frame(
                    graph,
                    partition,
                    weight_column=weight_column,
                    resolution=float(resolution),
                )
            )

    if not rows:
        return False

    partition_frame = pd.concat(rows, ignore_index=True)
    partition_frame.to_parquet(scenario_dir / "leiden_partitions.parquet", index=False)
    return True


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
    stage_run = start_stage_run("synthetic_clustering", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    scenario_config_path = config_value(study_config, ["scenario_config"], "configs/scenarios/synthetic.yaml")
    scenario_config = load_yaml(scenario_config_path)

    processed_dir = resolve_path(config_value(study_config, ["paths", "data", "processed", "synthetic"]))

    gmin = float(config_value(study_config, ["clustering", "resolution", "min"], 0.1))
    gmax = float(config_value(study_config, ["clustering", "resolution", "max"], 1.0))
    gstep = float(config_value(study_config, ["clustering", "resolution", "step"], 0.05))
    resolutions = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)
    n_restarts = int(config_value(study_config, ["clustering", "n_restarts"], 10))
    rng_seed = int(config_value(study_config, ["project", "rng_seed"], 12345))
    random.seed(rng_seed)
    weight_columns = list(config_value(study_config, ["network", "weight_columns"], []))
    min_w = float(config_value(study_config, ["network", "sparsify", "min_edge_weight"], 0.0001))
    sparsify = bool(config_value(study_config, ["network", "sparsify", "enabled"], True))
    if not sparsify:
        min_w = 0.0

    scenarios = config_value(scenario_config, ["scenarios"], {})
    scenario_list = list(scenarios.keys())
    jobs = args.jobs if args.jobs is not None else (round(os.cpu_count() * 0.75) or 1)
    clustered_scenarios: list[str] = []

    if jobs <= 1 or len(scenario_list) <= 1:
        for scenario_name in scenario_list:
            print(f">>> Clustering: {scenario_name}")
            wrote = cluster_scenario(
                scenario_name=scenario_name,
                processed_dir=processed_dir,
                weight_columns=weight_columns,
                min_w=min_w,
                resolutions=resolutions,
                n_restarts=n_restarts,
            )
            if wrote:
                clustered_scenarios.append(scenario_name)
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
            futures = {
                executor.submit(cluster_scenario, scenario_name): scenario_name
                for scenario_name in scenario_list
            }
            for future in as_completed(futures):
                scenario_name = futures[future]
                wrote = future.result()
                if wrote:
                    print(f">>> Clustered: {scenario_name}")
                    clustered_scenarios.append(scenario_name)

    manifest_config = study_config | {"scenario_definitions": scenario_config}
    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "synthetic_clustering.json",
        config=manifest_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "scenario_config": resolve_path(scenario_config_path),
            "processed_dir": str(processed_dir),
        },
        outputs={"processed_dir": str(processed_dir)},
        summary={
            "num_configured_scenarios": len(scenario_list),
            "num_clustered_scenarios": len(clustered_scenarios),
            "clustered_scenarios": sorted(clustered_scenarios),
            "num_weight_columns": len(weight_columns),
            "num_resolutions": len(resolutions),
            "jobs": jobs,
        },
        extra_metadata={"weight_columns": weight_columns},
    )

    print(f"Saved tables to: {processed_dir}")

if __name__ == "__main__":
    main()
