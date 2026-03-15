"""Evaluate partition stability as cases accrue over time."""

from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import pandas as pd
from epilink import build_pairwise_case_table
from sklearn.linear_model import LogisticRegression

from .config import config_value, ensure_directories, load_merged_config, resolve_path
from .epilink_adapter import (
    build_model_components,
    estimate_linkage_scores,
    simulate_epidemic_tree,
    simulate_genomic_outputs,
)
from .execution import finish_stage_run, start_stage_run
from .graphs import build_weighted_graph, partition_to_frame, run_leiden_partition, subset_pairs_for_nodes
from .metrics import bcubed_scores, build_star_memberships, overlap_metrics_between


def sampling_times(tree: nx.Graph, step_days: int) -> pd.DataFrame:
    """Assign cases to cumulative availability bins based on rounded sample dates."""

    sampling = {node_id: int(round(sample_date)) for node_id, sample_date in tree.nodes(data="sample_date")}
    case_meta = pd.DataFrame({"node": list(sampling.keys()), "sampling_time": list(sampling.values())})

    time_min = case_meta["sampling_time"].min()
    time_max = case_meta["sampling_time"].max()
    cuts = np.arange(time_min, time_max + step_days, step_days)
    bin_index = cuts.searchsorted(case_meta["sampling_time"].values, side="right") - 1
    bin_index = bin_index.clip(0, len(cuts) - 1)

    case_meta["available_bin_start"] = cuts[bin_index]
    case_meta = case_meta.sort_values(["available_bin_start", "sampling_time"]).reset_index(drop=True)
    codes, _ = pd.factorize(case_meta["available_bin_start"], sort=True)
    case_meta["available_time"] = codes
    return case_meta.sort_values("sampling_time").reset_index(drop=True)


def run_partition_for_nodes(
    pairwise_frame: pd.DataFrame,
    *,
    nodes_present: set,
    weight_column: str,
    minimum_weight: float,
    resolution: float,
    num_restarts: int,
) -> dict:
    """Infer a single Leiden partition for the available cases at one time point."""

    subgraph_pairs = subset_pairs_for_nodes(pairwise_frame, nodes_present)
    graph = build_weighted_graph(
        subgraph_pairs,
        weight_column=weight_column,
        minimum_weight=minimum_weight,
        vertex_ids=sorted(nodes_present),
    )
    partition, _ = run_leiden_partition(
        graph,
        weight_column=weight_column,
        resolution=resolution,
        num_restarts=num_restarts,
    )
    return dict(zip(graph.vs["case_id"], partition.membership))


def cumulative_stability(
    pairwise_frame: pd.DataFrame,
    case_meta: pd.DataFrame,
    *,
    weight_column: str,
    resolution: float,
    minimum_weight: float,
    num_restarts: int,
) -> pd.DataFrame:
    """Run cumulative clustering and compare consecutive weekly partitions."""

    transitions = []
    previous_time = None
    previous_labels = None

    for current_time in sorted(case_meta["available_time"].unique()):
        nodes_present = set(case_meta.loc[case_meta["available_time"] <= current_time, "node"])
        labels = run_partition_for_nodes(
            pairwise_frame,
            nodes_present=nodes_present,
            weight_column=weight_column,
            minimum_weight=minimum_weight,
            resolution=resolution,
            num_restarts=num_restarts,
        )
        if previous_labels is not None:
            overlap_frame = overlap_metrics_between(previous_labels, labels)
            overlap_frame.insert(0, "t", previous_time)
            overlap_frame.insert(1, "t1", current_time)
            transitions.append(overlap_frame)

        previous_time = current_time
        previous_labels = labels

    return pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/temporal_stability.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("temporal_stability", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    results_dir = resolve_path(config_value(study_config, ["paths", "results", "temporal_stability"]))
    ensure_directories(results_dir)

    step_days = int(config_value(study_config, ["case_sampling", "step_days"], 7))
    train_max_time_index = int(config_value(study_config, ["case_sampling", "train_max_time_index"], 2))
    num_restarts = int(config_value(study_config, ["clustering", "n_restarts"], 10))
    min_edge_weight = float(config_value(study_config, ["network", "sparsify", "min_edge_weight"], 1e-4))
    gmin = float(config_value(study_config, ["clustering", "resolution", "min"], 0.1))
    gmax = float(config_value(study_config, ["clustering", "resolution", "max"], 1.0))
    gstep = float(config_value(study_config, ["clustering", "resolution", "step"], 0.05))
    resolutions = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)

    tree_path = resolve_path(config_value(study_config, ["backbone", "tree_gml"]))
    transmission_tree = nx.read_gml(tree_path)
    model_components = build_model_components(study_config)

    populated_tree = simulate_epidemic_tree(
        model_components.transmission_profile,
        transmission_tree,
        study_config,
    )
    genomic_outputs = simulate_genomic_outputs(model_components.molecular_clock, populated_tree)
    pairwise_frame = build_pairwise_case_table(genomic_outputs["packed"], populated_tree)

    case_meta = sampling_times(populated_tree, step_days=step_days)
    case_counts = (
        case_meta.groupby("available_time", as_index=False)
        .size()
        .rename(columns={"size": "n_cases"})
    )
    case_counts.to_parquet(results_dir / "case_counts_over_time.parquet", index=False)

    pairwise_frame["ProbLinearDist"] = estimate_linkage_scores(
        model_components.transmission_profile,
        model_components.molecular_clock,
        genetic_distance=pairwise_frame["DeterministicDistance"].values,
        temporal_distance=pairwise_frame["SamplingDateDistanceDays"].values,
        config=study_config,
    )
    pairwise_frame["ProbPoissonDist"] = estimate_linkage_scores(
        model_components.transmission_profile,
        model_components.molecular_clock,
        genetic_distance=pairwise_frame["StochasticDistance"].values,
        temporal_distance=pairwise_frame["SamplingDateDistanceDays"].values,
        config=study_config,
    )

    initial_nodes = set(case_meta.loc[case_meta["available_time"] <= train_max_time_index, "node"])
    initial_pairs = subset_pairs_for_nodes(pairwise_frame, initial_nodes)
    y = initial_pairs["IsRelated"].astype(int).values
    logistic_distance_columns = {
        "LogitLinearDist": "DeterministicDistance",
        "LogitPoissonDist": "StochasticDistance",
    }
    for output_column, distance_column in logistic_distance_columns.items():
        feature_matrix = initial_pairs[["SamplingDateDistanceDays", distance_column]].values
        classifier = LogisticRegression(solver="lbfgs", max_iter=200)
        classifier.fit(feature_matrix, y)
        pairwise_frame[output_column] = classifier.predict_proba(
            pairwise_frame[["SamplingDateDistanceDays", distance_column]].values
        )[:, 1]
    initial_pairs = subset_pairs_for_nodes(pairwise_frame, initial_nodes)

    truth = build_star_memberships(tree_path)
    weight_columns = [
        "ProbLinearDist",
        "ProbPoissonDist",
        "LogitLinearDist",
        "LogitPoissonDist",
    ]

    metric_rows = []
    partition_rows = []
    for weight_column in weight_columns:
        graph = build_weighted_graph(
            initial_pairs,
            weight_column=weight_column,
            minimum_weight=min_edge_weight,
            vertex_ids=sorted(initial_nodes),
        )
        for resolution in resolutions:
            partition, _ = run_leiden_partition(
                graph,
                weight_column=weight_column,
                resolution=float(resolution),
                num_restarts=num_restarts,
            )
            partition_rows.append(
                partition_to_frame(
                    graph,
                    partition,
                    weight_column=weight_column,
                    resolution=float(resolution),
                )
            )

    partitions = pd.concat(partition_rows, ignore_index=True) if partition_rows else pd.DataFrame()
    for (weight_column, resolution), subframe in partitions.groupby(["weight_col", "resolution"], observed=True):
        predicted = {
            int(case_id): {int(cluster_id)}
            for case_id, cluster_id in zip(subframe["case_id"].tolist(), subframe["cluster_id"].tolist())
        }
        precision, recall, f1_score = bcubed_scores(predicted, truth)
        metric_rows.append(
            {
                "Resolution": resolution,
                "Weight": weight_column,
                "BCubed_Precision": precision,
                "BCubed_Recall": recall,
                "BCubed_F1_Score": f1_score,
                "N_cases": len(predicted),
            }
        )

    evaluation_metrics = pd.DataFrame(metric_rows)
    best_index = evaluation_metrics.groupby("Weight")["BCubed_F1_Score"].idxmax()
    best_models = evaluation_metrics.loc[best_index, ["Weight", "Resolution"]]
    model_resolution_map = best_models.set_index("Weight")["Resolution"].to_dict()

    methods = {
        "prob_linear": {"weight_column": "ProbLinearDist"},
        "prob_poisson": {"weight_column": "ProbPoissonDist"},
        "logit_linear": {"weight_column": "LogitLinearDist"},
        "logit_poisson": {"weight_column": "LogitPoissonDist"},
    }
    for method_name, method_config in methods.items():
        print(f"Analysing: {method_name}")
        stability_frame = cumulative_stability(
            pairwise_frame,
            case_meta,
            weight_column=method_config["weight_column"],
            minimum_weight=min_edge_weight,
            resolution=float(model_resolution_map[method_config["weight_column"]]),
            num_restarts=num_restarts,
        )
        stability_summary = stability_frame.groupby("t1")[["forward", "backward", "jaccard"]].mean()
        stability_summary.to_parquet(results_dir / f"temporal_stability_{method_name}.parquet", index=True)

    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "temporal_stability.json",
        config=study_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "tree_path": str(tree_path),
        },
        outputs={"results_dir": str(results_dir)},
        summary={
            "step_days": step_days,
            "train_max_time_index": train_max_time_index,
            "num_cases": len(case_meta),
            "num_initial_cases": len(initial_nodes),
            "num_methods": len(methods),
            "num_resolutions": len(resolutions),
        },
    )


if __name__ == "__main__":
    main()
