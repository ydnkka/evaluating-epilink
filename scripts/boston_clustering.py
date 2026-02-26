#!/usr/bin/env python3
"""
scripts/boston_clustering.py

Boston empirical analysis pipeline:
  - load metadata + precomputed pairwise distances
  - compute mechanistic linkage probabilities (epilink)
  - run Leiden community detection at a fixed resolution
  - summarize cluster composition and export manuscript tables

Config
------
config/paths.yaml
config/boston.yaml

Outputs
-------
tables/main/
  - boston_cluster_summary.parquet
  - boston_cluster_composition.parquet
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from epilink import TOIT, InfectiousnessParams, estimate_linkage_probabilities

from utils import deep_get, ensure_dirs, load_yaml


def hart_default_params() -> InfectiousnessParams:
    return InfectiousnessParams(
        k_inc=5.807,
        scale_inc=0.948,
        k_E=3.38,
        mu=0.37,
        k_I=1,
        alpha=2.29,
    )


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
    edge_attributes: Optional[Iterable[str]] = None,
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

        edge_stats: Dict[str, Any] = {}
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

            dunn_index = (np.mean(inter_vals) / intra_max) if has_inter and has_intra and intra_max > 0 else None
            edge_stats[f"dunn_index_{attr}"] = dunn_index

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
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bos_cfg = load_yaml(Path(args.boston))

    out_root = Path(args.out_root) if args.out_root else None

    data_boston = Path(deep_get(paths_cfg, ["data", "processed", "boston"], "../data/processed/boston"))
    if out_root and not data_boston.is_absolute():
        data_boston = out_root / data_boston

    tables_main = Path(deep_get(paths_cfg, ["outputs", "tables", "main"], "../tables/main"))
    if out_root:
        if not tables_main.is_absolute():
            tables_main = out_root / tables_main

    ensure_dirs(data_boston, tables_main)

    metadata_path = data_boston / str(deep_get(bos_cfg, ["inputs", "metadata"], "boston_metadata.parquet"))
    pairwise_path = data_boston / str(deep_get(bos_cfg, ["inputs", "pairwise_distances"], "boston_pairwise_distances.parquet"))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Boston metadata file not found: {metadata_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Boston pairwise file not found: {pairwise_path}")

    metadata = pd.read_parquet(metadata_path)
    pair_data = pd.read_parquet(pairwise_path)

    id_col_1 = str(deep_get(bos_cfg, ["inputs", "id_col_1"], "SeqID1"))
    id_col_2 = str(deep_get(bos_cfg, ["inputs", "id_col_2"], "SeqID2"))
    date_col = str(deep_get(bos_cfg, ["inputs", "date_col"], "Date"))
    exposure_col = str(deep_get(bos_cfg, ["inputs", "exposure_col"], "Exposure"))
    temporal_col = str(deep_get(bos_cfg, ["inputs", "temporal_col"], "Temporal_Distance"))

    required = {id_col_1, id_col_2, "SNP_Distance"}
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

    params = hart_default_params()
    rng_seed = int(deep_get(bos_cfg, ["inference", "rng_seed"], 42))
    subs_rate = float(deep_get(bos_cfg, ["inference", "subs_rate"], 1e-3))
    relax_rate = bool(deep_get(bos_cfg, ["inference", "relax_rate"], False))
    sigma = float(deep_get(bos_cfg, ["inference", "subs_rate_sigma"], 0.0))
    mutation_model = str(deep_get(bos_cfg, ["inference", "mutation_model"], "deterministic"))

    toit = TOIT(params=params, rng_seed=rng_seed, subs_rate=subs_rate, relax_rate=relax_rate, subs_rate_sigma=sigma)
    pair_data["Probability"] = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=pair_data["SNP_Distance"].values,
        temporal_distance=pair_data[temporal_col].values,
        mutation_model=mutation_model,
    )
    pair_data.to_parquet(data_boston / "boston_pairwise_with_probs.parquet", index=False)

    resolution = float(deep_get(bos_cfg, ["analysis", "resolution"], 0.3))
    probability_threshold = float(deep_get(bos_cfg, ["analysis", "probability_threshold"], 0.0001))
    min_cluster_size = int(deep_get(bos_cfg, ["analysis", "min_cluster_size"], 2))
    focus_exposures = list(deep_get(bos_cfg, ["analysis", "focus_exposures"], ["Conference", "SNF"]))

    g = build_graph(
        pairwise_df=pair_data,
        metadata_df=metadata,
        id_col_1=id_col_1,
        id_col_2=id_col_2,
        exposure_col=exposure_col,
        minimum_weight=probability_threshold,
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
        edge_attributes=["SNP_Distance", temporal_col],
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
        "intra_mean_SNP_Distance",
        "intra_max_SNP_Distance",
        f"intra_mean_{temporal_col}",
        f"intra_max_{temporal_col}",
        "inter_mean_SNP_Distance",
        "dunn_index_SNP_Distance",
    ]].copy()

    prob_summary.rename(
        columns={
            "cluster_id": "Cluster ID",
            "size": "Size",
            "intra_mean_SNP_Distance": "Intra-SNP (Mean)",
            "intra_max_SNP_Distance": "Intra-SNP(Max)",
            f"intra_mean_{temporal_col}": "Intra-Time (Mean)",
            f"intra_max_{temporal_col}": "Intra-Time (Max)",
            "inter_mean_SNP_Distance": "Inter-SNP (Mean)",
            "dunn_index_SNP_Distance": "Dunn Index (Genetic)",
        },
        inplace=True,
    )

    prob_summary.to_parquet(tables_main / "boston_cluster_summary.parquet", index=False)
    prob_focus.to_parquet(tables_main / "boston_cluster_composition.parquet", index=False)

    print(f"Saved Boston outputs to: {tables_main}")


if __name__ == "__main__":
    main()
