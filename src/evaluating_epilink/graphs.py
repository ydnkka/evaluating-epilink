"""Graph construction and Leiden community detection helpers."""

from __future__ import annotations

import random
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd


def build_weighted_graph(
    pairwise_frame: pd.DataFrame,
    *,
    weight_column: str,
    minimum_weight: float,
    source_column: str = "CaseA",
    target_column: str = "CaseB",
    vertex_ids: list[str] | pd.Index | None = None,
) -> ig.Graph:
    """Build an undirected weighted graph from a pairwise edge table."""

    edge_frame = pairwise_frame[[source_column, target_column, weight_column]].dropna().copy()
    if minimum_weight > 0:
        edge_frame = edge_frame.loc[edge_frame[weight_column] >= minimum_weight]

    graph = ig.Graph.TupleList(
        edges=edge_frame.to_records(index=False).tolist(),
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=weight_column,
    )

    if vertex_ids is not None:
        vertex_id_list = [str(vertex_id) for vertex_id in vertex_ids]
        present = set(graph.vs["case_id"]) if graph.vcount() else set()
        missing = [vertex_id for vertex_id in vertex_id_list if vertex_id not in present]
        if missing:
            graph.add_vertices(missing)
            if "case_id" not in graph.vs.attributes():
                graph.vs["case_id"] = graph.vs["name"]
            graph.vs.select(case_id_in=missing)["case_id"] = missing

    return graph


def run_leiden_partition(
    graph: ig.Graph,
    *,
    weight_column: str,
    resolution: float,
    num_restarts: int,
    rng_seed: int | None = None,
) -> tuple[ig.VertexClustering, float]:
    """Run Leiden repeatedly and keep the highest-modularity partition."""

    if rng_seed is not None:
        random.seed(rng_seed)

    best_partition: ig.VertexClustering | None = None
    best_modularity = -np.inf

    for _ in range(num_restarts):
        partition = graph.community_leiden(
            weights=weight_column,
            resolution=resolution,
            n_iterations=-1,
        )
        modularity = graph.modularity(
            membership=partition,
            weights=weight_column,
            resolution=resolution,
            directed=False,
        )
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    if best_partition is None:
        raise RuntimeError("Leiden did not produce a partition.")
    return best_partition, float(best_modularity)


def partition_to_frame(
    graph: ig.Graph,
    partition: ig.VertexClustering,
    *,
    weight_column: str,
    resolution: float,
) -> pd.DataFrame:
    """Convert an igraph partition into a tidy output table."""

    return pd.DataFrame(
        {
            "case_id": graph.vs["case_id"],
            "cluster_id": np.asarray(partition.membership, dtype=int),
            "resolution": float(resolution),
            "weight_col": weight_column,
        }
    )


def total_edge_weight(pairwise_frame: pd.DataFrame, *, weight_column: str) -> float:
    """Compute total retained edge weight for a given score column."""

    if pairwise_frame.empty:
        return 0.0
    return float(pairwise_frame[weight_column].sum())


def subset_pairs_for_nodes(pairwise_frame: pd.DataFrame, node_ids: set[Any]) -> pd.DataFrame:
    """Return the induced pairwise subgraph for a set of nodes."""

    mask = pairwise_frame["CaseA"].isin(node_ids) & pairwise_frame["CaseB"].isin(node_ids)
    return pairwise_frame.loc[mask].copy()
