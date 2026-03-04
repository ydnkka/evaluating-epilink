#!/usr/bin/env python3
"""
scripts/boston_clustering.py

Boston empirical analysis pipeline:
  - load metadata + precomputed pairwise distances
  - compute linkage probabilities (epilink)
  - run Leiden community detection at a fixed resolution
  - summarize cluster composition and export manuscript tables

Config
------
config/paths.yaml
config/boston.yaml

Outputs
-------
tables/boston/
  - boston_cluster_summary.parquet
  - boston_cluster_composition.parquet
"""
from __future__ import annotations

import argparse
from collections import Counter

import igraph as ig
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.stats import chisquare

from epilink import (
    TOIT,
    MolecularClock,
    InfectiousnessParams,
    linkage_probability
)

from utils import *


def add_temporal_distance(
    pairwise_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_col_1: str,
    id_col_2: str,
    date_col: str,
    out_col: str,
) -> pd.DataFrame:
    date_map = metadata_df.set_index("SeqID")[date_col]
    d1 = pd.to_datetime(pairwise_df[id_col_1].map(date_map))
    d2 = pd.to_datetime(pairwise_df[id_col_2].map(date_map))
    pairwise_df[out_col] = (d1 - d2).abs().dt.days.astype(int)
    return pairwise_df


def build_graph(
    pairwise_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_col_1: str,
    id_col_2: str,
    exposure_col: str,
    minimum_weight: float = 0.0001,
    probability_col: str = "Probability",
) -> ig.Graph:
    all_ids = pd.unique(pairwise_df[[id_col_1, id_col_2]].values.ravel())
    id_to_index = {seq_id: i for i, seq_id in enumerate(all_ids)}
    metadata_dict = metadata_df.set_index("SeqID").to_dict(orient="index")

    g = ig.Graph(n=len(all_ids))
    g.vs["SeqID"] = all_ids.tolist()
    g.vs["Date"] = [metadata_dict.get(sid, {}).get("Date") for sid in all_ids]
    g.vs["Clade"] = [metadata_dict.get(sid, {}).get("Clade") for sid in all_ids]
    g.vs[exposure_col] = [metadata_dict.get(sid, {}).get(exposure_col) for sid in all_ids]

    filtered = pairwise_df[pairwise_df[probability_col] >= minimum_weight]
    edges = list(zip(filtered[id_col_1].map(id_to_index), filtered[id_col_2].map(id_to_index)))
    g.add_edges(edges)

    for col in ["TN93_Distance", "SNP_Distance", "Temporal_Distance", probability_col]:
        if col in filtered.columns:
            g.es[col] = filtered[col].tolist()

    return g


def analyse_partition(
    part: ig.VertexClustering,
    node_attribute: str,
    edge_attributes: list = None,
    min_size: int = 1,
) -> pd.DataFrame:
    if edge_attributes is None:
        edge_attributes = []

    g = part.graph
    results = []

    clusters = [
        (cid, members)
        for cid, members in enumerate(part)
        if len(members) >= min_size
    ]

    all_node_attrs = list(g.vs[node_attribute])
    overall_attr_counts = Counter(all_node_attrs)
    total_nodes = len(all_node_attrs)
    unique_global_attrs = list(overall_attr_counts.keys())

    all_composition_vals = Counter()
    for _, members in clusters:
        vals = g.vs[members][node_attribute]
        all_composition_vals.update(vals)
    all_composition_attrs = [attr for attr, _ in all_composition_vals.most_common()]

    for cid, members in clusters:
        comm_size = len(members)
        subgraph = g.subgraph(members)

        comm_attrs = list(subgraph.vs[node_attribute])
        comm_attr_counts = Counter(comm_attrs)

        observed = np.array([comm_attr_counts.get(attr, 0) for attr in unique_global_attrs], dtype=float)
        p_value = None
        if observed.sum() > 0 and total_nodes > 0:
            expected_props = np.array(
                [overall_attr_counts.get(attr, 0) / total_nodes for attr in unique_global_attrs],
                dtype=float,
            )
            expected = expected_props * observed.sum()
            keep = expected > 0
            if keep.sum() >= 1:
                _, p_value = chisquare(f_obs=observed[keep], f_exp=expected[keep])

        intra_es = g.es.select(_within=members)
        others = list(set(range(g.vcount())) - set(members))
        inter_es = g.es.select(_between=(members, others))

        edge_stats: dict[str, Any] = {}
        for attr in edge_attributes:
            intra_vals = intra_es[attr]
            inter_vals = inter_es[attr]

            has_intra = len(intra_vals) > 0
            has_inter = len(inter_vals) > 0

            intra_max = np.max(intra_vals) if has_intra else 0

            edge_stats[f"intra_mean_{attr}"] = np.mean(intra_vals) if has_intra else None
            edge_stats[f"intra_max_{attr}"] = intra_max if has_intra else None
            edge_stats[f"intra_min_{attr}"] = np.min(intra_vals) if has_intra else None

            edge_stats[f"inter_min_{attr}"] = np.min(inter_vals) if has_inter else None
            edge_stats[f"inter_mean_{attr}"] = np.mean(inter_vals) if has_inter else None

        row = {
            "cluster_id": cid,
            "size": comm_size,
            f"{node_attribute}_dist": dict(comm_attr_counts),
            "chi2_p_value": p_value,
            **edge_stats,
        }

        total_in_cluster = max(1, sum(comm_attr_counts.values()))
        for attr in all_composition_attrs:
            count = comm_attr_counts.get(attr, 0)
            row[f"count::{attr}"] = count
            row[f"prop::{attr}"] = count / total_in_cluster

        results.append(row)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("size", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--boston", default="../config/boston.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bos_cfg = load_yaml(Path(args.boston))

    data_boston = Path(deep_get(paths_cfg, ["data", "processed", "boston"], "../data/processed/boston"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables"], "../tables"))
    tabs_dir = tabs_dir / "boston"
    ensure_dirs( tabs_dir)

    metadata_path = data_boston / str(deep_get(bos_cfg, ["inputs", "metadata"]))
    pairwise_path = data_boston / str(deep_get(bos_cfg, ["inputs", "pairwise_distances"]))

    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(f"Boston metadata file not found: {metadata_path}")
    if metadata_path is None or not pairwise_path.exists():
        raise FileNotFoundError(f"Boston pairwise file not found: {pairwise_path}")

    metadata = pd.read_parquet(metadata_path)
    pair_data = pd.read_parquet(pairwise_path)

    id_col_1 = str(deep_get(bos_cfg, ["inputs", "id_col_1"], "SeqID1"))
    id_col_2 = str(deep_get(bos_cfg, ["inputs", "id_col_2"], "SeqID2"))
    date_col = str(deep_get(bos_cfg, ["inputs", "date_col"], "Date"))
    exposure_col = str(deep_get(bos_cfg, ["inputs", "exposure_col"], "Exposure"))
    temporal_col = str(deep_get(bos_cfg, ["inputs", "temporal_col"], "Temporal_Distance"))
    genetic_col = str(deep_get(bos_cfg, ["inputs", "genetic_col"], "SNP_Distance"))

    required = {id_col_1, id_col_2, genetic_col}
    missing = required - set(pair_data.columns)
    if missing:
        raise ValueError(f"Missing required columns in Boston pairwise file: {missing}")

    pair_data = add_temporal_distance(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=id_col_1,
        id_col_2=id_col_2,
        date_col=date_col,
        out_col=temporal_col,
    )

    rng_seed = int(deep_get(bos_cfg, ["rng_seed"], 12345))
    toit_cfg = deep_get(bos_cfg, ["infectiousness_params"], {})
    clock_cfg = deep_get(bos_cfg, ["clock"], {})
    inference_cfg = deep_get(bos_cfg, ["inference"], {})
    inference_cfg["intermediate_generations"] = tuple(inference_cfg["intermediate_generations"])

    rng = default_rng(rng_seed)

    params = InfectiousnessParams(**toit_cfg)
    toit = TOIT(params=params, rng=rng)
    clock = MolecularClock(**clock_cfg)

    pair_data["Probability"] = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=pair_data[genetic_col].values,
        temporal_distance=pair_data[temporal_col].values,
        **inference_cfg
    )

    resolution = float(deep_get(bos_cfg, ["analysis", "resolution"], 0.3))
    minimum_weight = float(deep_get(bos_cfg, ["network", "sparsify", "min_edge_weight"], 0.0001))
    sparsify = bool(deep_get(bos_cfg, ["network", "sparsify", "enabled"], True))
    if not sparsify:
        minimum_weight = 0.0
    min_cluster_size = int(deep_get(bos_cfg, ["analysis", "min_cluster_size"], 2))
    focus_exposures = list(deep_get(bos_cfg, ["analysis", "focus_exposures"], ["Conference", "SNF"]))

    g = build_graph(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=id_col_1,
        id_col_2=id_col_2,
        exposure_col=exposure_col,
        minimum_weight=minimum_weight,
        probability_col="Probability",
    )

    part = g.community_leiden(
        weights="Probability",
        resolution=resolution,
        n_iterations=-1,
    )

    prob_results = analyse_partition(
        part,
        node_attribute=exposure_col,
        edge_attributes=[genetic_col, temporal_col],
        min_size=min_cluster_size,
    )

    if focus_exposures:
        focus_cols = [f"count::{label}" for label in focus_exposures if f"count::{label}" in prob_results.columns]
        if not focus_cols:
            raise ValueError(f"Focus exposures not found in cluster summary: {focus_exposures}")
        prob_focus = prob_results[prob_results[focus_cols].sum(axis=1) > 0]
    else:
        prob_focus = prob_results

    prob_summary = prob_focus[[
        "cluster_id",
        "size",
        f"intra_mean_{genetic_col}",
        f"intra_max_{genetic_col}",
        f"intra_mean_{temporal_col}",
        f"intra_max_{temporal_col}",
        f"inter_mean_{genetic_col}",
    ]].copy()

    prob_summary.rename(
        columns={
            "cluster_id": "Cluster ID",
            "size": "Size",
            f"intra_mean_{genetic_col}": "Intra-SNP (Mean)",
            f"intra_max_{genetic_col}": "Intra-SNP(Max)",
            f"intra_mean_{temporal_col}": "Intra-Time (Mean)",
            f"intra_max_{temporal_col}": "Intra-Time (Max)",
            f"inter_mean_{genetic_col}": "Inter-SNP (Mean)",
        },
        inplace=True,
    )

    prob_summary.to_parquet(tabs_dir / "boston_cluster_summary.parquet", index=False)
    prob_focus.to_parquet(tabs_dir / "boston_cluster_composition.parquet", index=False)

    print(f"Saved Boston outputs to: {tabs_dir}")


if __name__ == "__main__":
    main()
