"""Evaluation helpers shared across manuscript workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import bcubed
import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import chisquare


def build_star_memberships(tree_path) -> dict[int, set[int]]:
    """Build overlapping reference clusters from each case and its direct infectees."""

    graph = ig.Graph.Read_GML(str(tree_path))
    clusters = []
    for node_index in range(graph.vcount()):
        direct_infectees = set(graph.successors(node_index))
        cluster_members = {node_index} | direct_infectees
        cluster_labels = [graph.vs[index]["label"] for index in cluster_members]
        clusters.append(cluster_labels)

    memberships: dict[int, set[int]] = defaultdict(set)
    for cluster_id, cluster_members in enumerate(clusters):
        for node_label in cluster_members:
            memberships[int(node_label)].add(int(cluster_id))
    return memberships


def bcubed_scores(
    predicted_memberships: dict[int, set[int]],
    reference_memberships: dict[int, set[int]],
) -> tuple[float, float, float]:
    """Compute BCubed precision, recall, and F1 for overlapping memberships."""

    shared_cases = sorted(set(predicted_memberships) & set(reference_memberships))
    filtered_predicted = {case_id: predicted_memberships[case_id] for case_id in shared_cases if predicted_memberships[case_id]}
    filtered_reference = {case_id: reference_memberships[case_id] for case_id in shared_cases if reference_memberships[case_id]}

    if not filtered_predicted or not filtered_reference:
        raise ValueError("No valid cases with non-empty memberships.")

    precision = bcubed.precision(filtered_predicted, filtered_reference)
    recall = bcubed.recall(filtered_predicted, filtered_reference)
    f1_score = bcubed.fscore(precision, recall)
    return float(precision), float(recall), float(f1_score)


def make_cluster_sets(labels_by_case: dict[Any, Any]) -> dict[Any, set[Any]]:
    """Invert case-to-cluster labels into cluster membership sets."""

    cluster_sets: dict[Any, set[Any]] = defaultdict(set)
    for case_id, cluster_id in labels_by_case.items():
        cluster_sets[cluster_id].add(case_id)
    return cluster_sets


def overlap_metrics_between(
    earlier_labels: dict[Any, Any],
    later_labels: dict[Any, Any],
) -> pd.DataFrame:
    """Compute forward, backward, and Jaccard overlap between two partitions."""

    earlier_clusters = make_cluster_sets(earlier_labels)
    later_clusters = make_cluster_sets(later_labels)
    shared_cases = sorted(set(earlier_labels) & set(later_labels))

    rows = []
    for case_id in shared_cases:
        earlier_cluster = earlier_clusters[earlier_labels[case_id]]
        later_cluster = later_clusters[later_labels[case_id]]
        overlap = earlier_cluster & later_cluster
        union = earlier_cluster | later_cluster

        rows.append(
            {
                "node": case_id,
                "forward": len(overlap) / len(later_cluster) if later_cluster else np.nan,
                "backward": len(overlap) / len(earlier_cluster) if earlier_cluster else np.nan,
                "jaccard": len(overlap) / len(union) if union else np.nan,
                "earlier_cluster_size": len(earlier_cluster),
                "later_cluster_size": len(later_cluster),
                "overlap_size": len(overlap),
            }
        )

    return pd.DataFrame(rows)


def analyse_partition_composition(
    partition: ig.VertexClustering,
    *,
    node_attribute: str,
    edge_attributes: list[str] | None = None,
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """Summarise within-cluster composition and edge properties."""

    edge_attributes = [] if edge_attributes is None else edge_attributes
    graph = partition.graph
    results = []

    clusters = [
        (cluster_id, members)
        for cluster_id, members in enumerate(partition)
        if len(members) >= min_cluster_size
    ]

    overall_values = list(graph.vs[node_attribute])
    overall_counts = Counter(overall_values)
    total_nodes = len(overall_values)
    global_categories = list(overall_counts.keys())

    category_order = Counter()
    for _, members in clusters:
        category_order.update(graph.vs[members][node_attribute])
    ordered_categories = [category for category, _ in category_order.most_common()]

    for cluster_id, members in clusters:
        cluster_subgraph = graph.subgraph(members)
        cluster_values = list(cluster_subgraph.vs[node_attribute])
        cluster_counts = Counter(cluster_values)

        observed = np.asarray(
            [cluster_counts.get(category, 0) for category in global_categories],
            dtype=float,
        )
        chi_square_p_value = None
        if observed.sum() > 0 and total_nodes > 0:
            expected_proportions = np.asarray(
                [overall_counts.get(category, 0) / total_nodes for category in global_categories],
                dtype=float,
            )
            expected = expected_proportions * observed.sum()
            valid = expected > 0
            if valid.sum() >= 1:
                _, chi_square_p_value = chisquare(f_obs=observed[valid], f_exp=expected[valid])

        within_edges = graph.es.select(_within=members)
        outside_members = list(set(range(graph.vcount())) - set(members))
        between_edges = graph.es.select(_between=(members, outside_members))

        edge_summary: dict[str, Any] = {}
        for edge_attribute in edge_attributes:
            within_values = within_edges[edge_attribute]
            between_values = between_edges[edge_attribute]
            has_within = len(within_values) > 0
            has_between = len(between_values) > 0

            edge_summary[f"intra_mean_{edge_attribute}"] = np.mean(within_values) if has_within else None
            edge_summary[f"intra_max_{edge_attribute}"] = np.max(within_values) if has_within else None
            edge_summary[f"intra_min_{edge_attribute}"] = np.min(within_values) if has_within else None
            edge_summary[f"inter_mean_{edge_attribute}"] = np.mean(between_values) if has_between else None
            edge_summary[f"inter_min_{edge_attribute}"] = np.min(between_values) if has_between else None

        row = {
            "cluster_id": cluster_id,
            "size": len(members),
            f"{node_attribute}_dist": dict(cluster_counts),
            "chi2_p_value": chi_square_p_value,
            **edge_summary,
        }

        cluster_total = max(1, sum(cluster_counts.values()))
        for category in ordered_categories:
            count = cluster_counts.get(category, 0)
            row[f"count::{category}"] = count
            row[f"prop::{category}"] = count / cluster_total

        results.append(row)

    result_frame = pd.DataFrame(results)
    if not result_frame.empty:
        result_frame = result_frame.sort_values("size", ascending=False).reset_index(drop=True)
    return result_frame
