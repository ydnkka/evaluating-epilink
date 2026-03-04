#!/usr/bin/env python3
"""
scripts/temporal_stability.py

Evaluate cluster stability over time for different edge-weight models.

Config
------
config/paths.yaml
config/default_parameters.yaml
config/generate_datasets.yaml
config/temporal_stability.yaml

Outputs
-------
tables/supplementary/temporal_stability/
  - case_counts_over_time.parquet
  - stability_summary_<method>.parquet
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from numpy.random import default_rng
import pandas as pd
import networkx as nx
import igraph as ig
import bcubed

from sklearn.linear_model import LogisticRegression

from epilink import (
    TOIT,
    MolecularClock,
    InfectiousnessParams,
    populate_epidemic_data,
    simulate_genomic_data,
    generate_pairwise_data,
    linkage_probability,
)

from utils import *


def sampling_times(tree: nx.Graph, step: int) -> pd.DataFrame:
    """Create per-case availability bins."""
    sampling = {n: int(round(d)) for n, d in tree.nodes(data="sample_date")}
    case_meta = pd.DataFrame({
        "node": list(sampling.keys()),
        "sampling_time": list(sampling.values()),
    })

    tmin = case_meta["sampling_time"].min()
    tmax = case_meta["sampling_time"].max()

    cuts = np.arange(tmin, tmax + step, step)
    idx = cuts.searchsorted(case_meta["sampling_time"].values, side="right") - 1
    idx = idx.clip(0, len(cuts) - 1)

    case_meta["available_bin_start"] = cuts[idx]

    case_meta = case_meta.sort_values(["available_bin_start", "sampling_time"]).reset_index(drop=True)
    codes, _ = pd.factorize(case_meta["available_bin_start"], sort=True)
    case_meta["available_time"] = codes

    case_meta = case_meta.sort_values("sampling_time").reset_index(drop=True)
    return case_meta


def build_igraph(
    df: pd.DataFrame,
    weight_col: str,
    minimum_weight: float,
    all_nodes: list,
) -> ig.Graph:
    """Build an undirected igraph graph from an edge list."""
    sub_df = df[df[weight_col] >= minimum_weight]
    edges = sub_df[["NodeA", "NodeB", weight_col]].to_records(index=False).tolist()

    g = ig.Graph.TupleList(
        edges=edges,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=weight_col,
    )

    present = set(g.vs["case_id"]) if g.vcount() else set()
    missing = [n for n in all_nodes if n not in present]
    if missing:
        g.add_vertices(missing, attributes={"case_id": missing})
    return g


def build_ground_truth_memberships(tree_path: Path) -> dict[int, set[int]]:
    g = ig.Graph.Read_GML(str(tree_path))

    clusters = []
    for node_id in range(g.vcount()):
        neighbours = set(g.successors(node_id))
        c = {node_id} | neighbours
        c = [g.vs[i]["label"] for i in c]
        clusters.append(c)

    membership = defaultdict(set)
    for clus_id, clus in enumerate(clusters):
        for node in clus:
            membership[int(node)].add(int(clus_id))
    return membership


def bcubed_scores(
    pred: dict[int, set[int]],
    truth: dict[int, set[int]],
) -> tuple[float, float, float]:
    cases = sorted(set(pred) & set(truth))
    pred = {c: pred[c] for c in cases if pred[c]}
    truth = {c: truth[c] for c in cases if truth[c]}

    if not pred or not truth:
        raise ValueError("No valid cases with non-empty memberships")

    precision = bcubed.precision(pred, truth)
    recall = bcubed.recall(pred, truth)
    f_score = bcubed.fscore(precision, recall)
    return precision, recall, f_score


def run_clustering(
    df: pd.DataFrame,
    nodes_present: set,
    weight_col: str,
    minimum_weight: float,
    resolution: float,
    n_restarts: int,
) -> dict:
    g = build_igraph(
        df[["NodeA", "NodeB", weight_col]],
        weight_col,
        minimum_weight=minimum_weight,
        all_nodes=sorted(nodes_present),
    )

    best = None
    best_q = -np.inf

    for _ in range(n_restarts):
        part = g.community_leiden(
            weights=weight_col,
            resolution=resolution,
            n_iterations=-1,
        )

        q = g.modularity(
            membership=part,
            weights=weight_col,
            resolution=resolution,
        )

        if q > best_q:
            best_q = q
            best = part

    memb = best.membership if best is not None else None

    if memb is None:
        memb = list(range(g.vcount()))

    return dict(zip(g.vs["case_id"], memb))


def subset_pairs(df: pd.DataFrame, nodes_present: set) -> pd.DataFrame:
    m = df["NodeA"].isin(nodes_present) & df["NodeB"].isin(nodes_present)
    return df.loc[m].copy()


def make_cluster_sets(labels: dict) -> dict:
    clusters = defaultdict(set)
    for n, c in labels.items():
        clusters[c].add(n)
    return clusters


def overlap_metrics_between(labels_t: dict, labels_t1: dict) -> pd.DataFrame:
    """Per-node overlap between two consecutive snapshots."""
    clusters_t = make_cluster_sets(labels_t)
    clusters_t1 = make_cluster_sets(labels_t1)

    common_nodes = sorted(set(labels_t).intersection(labels_t1))

    rows = []
    for n in common_nodes:
        ct = clusters_t[labels_t[n]]
        ct1 = clusters_t1[labels_t1[n]]

        inter = ct & ct1
        union = ct | ct1

        o = len(inter)
        rows.append({
            "node": n,
            "forward": o / len(ct1) if len(ct1) else float("nan"),
            "backward": o / len(ct) if len(ct) else float("nan"),
            "jaccard": o / len(union) if len(union) else float("nan"),
            "Ct_size": len(ct),
            "Ct1_size": len(ct1),
            "overlap_size": o,
        })

    return pd.DataFrame(rows)


def cumulative_stability(
    pairwise_df: pd.DataFrame,
    case_meta: pd.DataFrame,
    weight_col: str,
    resolution: float,
    minimum_weight: float,
    n_restarts: int,
) -> pd.DataFrame:
    """Run cumulative clustering and compute overlap metrics for each transition."""

    times = sorted(case_meta["available_time"].unique())

    transitions = []
    prev_t = None
    prev_labels = None

    for t in times:
        nodes_present = set(case_meta.loc[case_meta["available_time"] <= t, "node"])
        sub_df = subset_pairs(pairwise_df, nodes_present)

        labels = run_clustering(
            sub_df,
            nodes_present=nodes_present,
            weight_col=weight_col,
            resolution=resolution,
            minimum_weight=minimum_weight,
            n_restarts=n_restarts,
        )

        if prev_labels is not None:
            df = overlap_metrics_between(prev_labels, labels)
            df.insert(0, "t", prev_t)
            df.insert(1, "t1", t)
            transitions.append(df)

        prev_t = t
        prev_labels = labels

    return pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    parser.add_argument("--datasets", default="../config/generate_datasets.yaml")
    parser.add_argument("--stability", default="../config/temporal_stability.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    defaults_cfg = load_yaml(Path(args.defaults))
    datasets_cfg = load_yaml(Path(args.datasets))
    stability_cfg = load_yaml(Path(args.stability))

    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables"], "../tables"))
    tabs_dir =tabs_dir / "temporal_stability"
    ensure_dirs(tabs_dir)

    step_days = int(deep_get(stability_cfg, ["case_sampling", "step_days"], 7))
    train_max_time_index = int(deep_get(stability_cfg, ["case_sampling", "train_max_time_index"], 2))
    rng_seed = int(deep_get(defaults_cfg, ["rng_seed"], 12345))
    toit_cfg = deep_get(defaults_cfg, ["infectiousness_params"], {})
    clock_cfg = deep_get(defaults_cfg, ["clock"], {})
    inference_cfg = deep_get(defaults_cfg, ["inference"], {})
    inference_cfg["intermediate_generations"] = tuple(inference_cfg["intermediate_generations"])
    n_restarts = int(deep_get(stability_cfg, ["community_detection", "n_restarts"], 10))
    min_edge_weight = float(deep_get(stability_cfg, ["community_detection", "min_edge_weight"], 1e-4))
    gmin = float(deep_get(stability_cfg, ["community_detection", "resolution", "min"], 0.1))
    gmax = float(deep_get(stability_cfg, ["community_detection", "resolution", "max"], 1.0))
    gstep = float(deep_get(stability_cfg, ["community_detection", "resolution", "step"], 0.05))
    resolutions = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)

    tree_path = Path(
        deep_get(datasets_cfg, ["backbone", "tree_gml"],"../data/processed/synthetic/scovmod/scovmod_tree.gml")
    )

    trans_tree = nx.read_gml(tree_path)
    rng = default_rng(rng_seed)

    params = InfectiousnessParams(**toit_cfg)
    toit = TOIT(params=params, rng=rng)
    clock = MolecularClock(**clock_cfg)

    populated_tree = populate_epidemic_data(toit=toit, tree=trans_tree)
    gen_results = simulate_genomic_data(clock=clock, tree=populated_tree)
    pairwise = generate_pairwise_data(
        packed_genomic_data=gen_results["packed"],
        tree=populated_tree,
    )

    sample_dates = sampling_times(populated_tree, step=step_days)

    case_counts = (
        sample_dates.groupby("available_time", as_index=False)
        .size()
        .rename(columns={"size": "n_cases"})
    )
    case_counts.to_parquet(tabs_dir / "case_counts_over_time.parquet", index=False)

    pairwise["ProbLinearDist"] = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=pairwise["LinearDist"].values,
        temporal_distance=pairwise["TemporalDist"].values,
        **inference_cfg
    )

    pairwise["ProbPoissonDist"] = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=pairwise["PoissonDist"].values,
        temporal_distance=pairwise["TemporalDist"].values,
        **inference_cfg
    )

    initial_nodes = set(sample_dates.loc[sample_dates["available_time"] <= train_max_time_index, "node"])
    initial_data = pairwise[
        pairwise["NodeA"].isin(initial_nodes) &
        pairwise["NodeB"].isin(initial_nodes)
    ]

    y = initial_data["Related"].astype(int).values
    for dist_col in ("LinearDist", "PoissonDist"):
        X = initial_data[["TemporalDist", dist_col]].values
        col = f"Logit{dist_col}"
        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(X, y)
        pairwise[col] = clf.predict_proba(pairwise[["TemporalDist", dist_col]].values)[:, 1]

    truth = build_ground_truth_memberships(tree_path)

    # Infer best resolution
    initial_data = pairwise[
        pairwise["NodeA"].isin(initial_nodes) &
        pairwise["NodeB"].isin(initial_nodes)
        ]

    prob_weight_columns = [
        "ProbLinearDist",
        "ProbPoissonDist",
        "LogitLinearDist",
        "LogitPoissonDist",
    ]

    metrics_rows = []
    rows = []
    for weight in prob_weight_columns:
        graph = build_igraph(
            initial_data[["NodeA", "NodeB", weight]],
            weight,
            minimum_weight=min_edge_weight,
            all_nodes=sorted(initial_nodes),
        )

        for res in resolutions:
            best = None
            best_q = -np.inf

            for _ in range(n_restarts):
                part = graph.community_leiden(
                    weights=weight,
                    resolution=res,
                    n_iterations=-1,
                )

                q = graph.modularity(
                    membership=part,
                    weights=weight,
                    resolution=res,
                    directed=False,
                )

                if q > best_q:
                    best_q = q
                    best = part

            memb = best.membership if best is not None else None
            if memb is None:
                memb = list(range(graph.vcount()))
            rows.append(pd.DataFrame({
                "case_id": graph.vs["case_id"],
                "resolution": res,
                "cluster_id": np.array(memb, dtype=int),
                "weight": weight,
            }))

    partitions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    for (weight, res), sub in partitions.groupby(["weight", "resolution"], observed=True):
        pred = {int(k): {int(v)} for k, v in zip(sub["case_id"].tolist(), sub["cluster_id"].tolist())}
        prec, rec, f1 = bcubed_scores(pred=pred, truth=truth)
        metrics_rows.append({
            "Resolution": res,
            "Weight": weight,
            "BCubed_Precision": prec,
            "BCubed_Recall": rec,
            "BCubed_F1_Score": f1,
            "N_cases": len(pred),
        })

    eval_metrics = pd.DataFrame(metrics_rows)
    idx = eval_metrics.groupby("Weight")["BCubed_F1_Score"].idxmax()
    best_f1 = eval_metrics.loc[idx, ["Weight", "BCubed_F1_Score", "Resolution"]]
    model_res = best_f1.set_index("Weight")["Resolution"].to_dict()

    methods = {
        "prob_linear": {
            "weight_col": "ProbLinearDist",
            "minimum_weight": min_edge_weight,
            "resolution": model_res.get("ProbLinearDist"),
        },
        "prob_poisson": {
            "weight_col": "ProbPoissonDist",
            "minimum_weight": min_edge_weight,
            "resolution": model_res.get("ProbPoissonDist"),
        },
        "logit_linear": {
            "weight_col": "LogitLinearDist",
            "minimum_weight": min_edge_weight,
            "resolution": model_res.get("LogitLinearDist"),
        },
        "logit_poisson": {
            "weight_col": "LogitPoissonDist",
            "minimum_weight": min_edge_weight,
            "resolution": model_res.get("LogitPoissonDist"),
        }
    }

    for i, name in enumerate(methods.keys()):
        cfg = methods[name]
        print(f"Analysing: {name}")
        stability_df = cumulative_stability(
            pairwise,
            sample_dates,
            n_restarts=n_restarts,
            **cfg,
        )
        summary = stability_df.groupby("t1")[["forward", "backward", "jaccard"]].mean()
        summary.to_parquet(tabs_dir / f"temporal_stability_{name}.parquet", index=True)

if __name__ == "__main__":
    main()
