"""Quantify retention and runtime effects of edge sparsification."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from .config import config_value, ensure_directories, load_merged_config, resolve_path
from .execution import finish_stage_run, start_stage_run
from .graphs import build_weighted_graph, total_edge_weight


def timed(function, *args, **kwargs):
    """Measure wall-clock time for a function call."""

    start = time.perf_counter()
    output = function(*args, **kwargs)
    duration_seconds = time.perf_counter() - start
    return output, duration_seconds


def sparsify_edges(pairwise_frame: pd.DataFrame, min_edge_weight: float, weight_column: str) -> pd.DataFrame:
    """Filter a pairwise table to the retained edges for a threshold."""

    threshold = float(min_edge_weight)
    if threshold <= 0:
        return pairwise_frame
    return pairwise_frame.loc[pairwise_frame[weight_column] >= threshold]


def timed_igraph_and_leiden(
    pairwise_frame: pd.DataFrame,
    *,
    weight_column: str,
    vertex_ids: pd.Index,
    resolution: float,
) -> tuple[float, float]:
    """Measure graph construction and Leiden runtime for a sparsified edge set."""

    graph, build_seconds = timed(
        build_weighted_graph,
        pairwise_frame,
        weight_column=weight_column,
        minimum_weight=0.0,
        vertex_ids=vertex_ids,
    )

    def _run_leiden():
        return graph.community_leiden(
            weights=weight_column,
            resolution=float(resolution),
            n_iterations=-1,
        )

    _, leiden_seconds = timed(_run_leiden)
    return float(build_seconds), float(leiden_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/synthetic.yaml")
    parser.add_argument("--scenario", default="baseline", help="Scenario subdir name, e.g. baseline")
    parser.add_argument("--gamma", type=float, default=0.5, help="Leiden resolution for timing diagnostics")
    args = parser.parse_args()
    stage_run = start_stage_run("sparsify_analysis", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)

    processed_dir = resolve_path(config_value(study_config, ["paths", "data", "processed", "synthetic"]))
    results_dir = resolve_path(config_value(study_config, ["paths", "results", "sparsify"]))
    ensure_directories(results_dir)

    min_edge_weights = list(
        config_value(study_config, ["network", "min_edge_weights"], [0.0, 0.0001, 0.001, 0.01, 0.1])
    )
    weight_columns = list(config_value(study_config, ["network", "weight_columns"], ["ProbLinearDist"]))
    scenario_dir = processed_dir / f"scenario={args.scenario}"
    scored_pairs = pd.read_parquet(scenario_dir / "pairwise_scored.parquet")

    weight_column = weight_columns[0]

    reference_threshold = float(min(min_edge_weights))
    reference_frame = sparsify_edges(scored_pairs, reference_threshold, weight_column)
    reference_nodes = pd.Index(pd.unique(reference_frame[["NodeA", "NodeB"]].values.ravel())).astype(str)

    reference_weight = total_edge_weight(reference_frame, weight_column=weight_column)
    reference_edge_count = int(len(reference_frame)) if len(reference_frame) else 0

    retention_rows: list[dict[str, float]] = []
    for threshold in min_edge_weights:
        filtered_pairs, sparsify_seconds = timed(sparsify_edges, scored_pairs, threshold, weight_column)
        retained_weight = total_edge_weight(filtered_pairs, weight_column=weight_column)
        retained_edges = int(len(filtered_pairs))

        build_seconds, leiden_seconds = timed_igraph_and_leiden(
            filtered_pairs,
            weight_column=weight_column,
            vertex_ids=reference_nodes,
            resolution=args.gamma,
        )
        retention_rows.append(
            {
                "min_edge_weight": float(threshold),
                "edge_retention_frac": float(retained_edges / reference_edge_count) if reference_edge_count > 0 else float("nan"),
                "weight_retention_frac": float(retained_weight / reference_weight) if reference_weight > 0 else float("nan"),
                "t_pipeline_s": float(sparsify_seconds + build_seconds + leiden_seconds),
            }
        )

    retention_frame = pd.DataFrame(retention_rows).sort_values("min_edge_weight").reset_index(drop=True)
    retention_path = results_dir / "sparsify_edge_retention.parquet"
    retention_frame.to_parquet(retention_path, index=False)

    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "sparsify_analysis.json",
        config=study_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "scenario_dir": str(scenario_dir),
        },
        outputs={
            "results_dir": str(results_dir),
            "retention_path": str(retention_path),
        },
        summary={
            "scenario": args.scenario,
            "weight_column": weight_column,
            "num_thresholds": len(min_edge_weights),
            "gamma": args.gamma,
        },
    )

    print(f"Saved tables to: {results_dir}")


if __name__ == "__main__":
    main()
