#!/usr/bin/env python3
"""
scripts/cluster_stability.py

Evaluate cluster stability over time for different edge-weight models.

Outputs
-------
figures/main/
  - cluster_stability.(png|pdf)
figures/supplementary/stability/
  - case_counts_over_time.(png|pdf)
tables/main/
  - stability_summary_<method>.csv

Config
------
config/paths.yaml
config/default_parameters.yaml
config/generate_datasets.yaml
config/cluster_stability.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import bcubed

from sklearn.linear_model import LogisticRegression

from epilink import (
    TOIT,
    InfectiousnessParams,
    populate_epidemic_data,
    simulate_genomic_data,
    generate_pairwise_data,
    estimate_linkage_probabilities,
)

from utils import *


set_seaborn_paper_context()


def _gamma_grid(cfg: dict) -> tuple[float, ...]:
    gammas = deep_get(cfg, ["leiden", "gammas"], None)
    if isinstance(gammas, dict):
        gmin = float(gammas.get("min", 0.0))
        gmax = float(gammas.get("max", 1.0))
        gstep = float(gammas.get("step", 0.05))
        return tuple(np.round(np.arange(gmin, gmax + 1e-9, gstep), 10))
    if isinstance(gammas, Iterable):
        return tuple(float(g) for g in gammas)
    return tuple(np.round(np.arange(0, 1 + 1e-9, 0.05), 10))


def _build_toit(defaults_cfg: dict) -> tuple[TOIT, dict]:
    rng_seed = int(deep_get(defaults_cfg, ["toit", "rng_seed"], 12345))
    params_cfg = deep_get(defaults_cfg, ["toit", "infectiousness_params"], {})
    evol_cfg = deep_get(defaults_cfg, ["toit", "evolution"], {})

    params = InfectiousnessParams(**params_cfg)
    toit = TOIT(
        params=params,
        rng_seed=rng_seed,
        subs_rate=float(evol_cfg["subs_rate"]),
        relax_rate=bool(evol_cfg["relax_rate"]),
        subs_rate_sigma=float(evol_cfg["subs_rate_sigma"]),
        gen_len=int(evol_cfg["gen_length"]),
    )

    inference_cfg = deep_get(defaults_cfg, ["inference"], {})
    return toit, inference_cfg


def sampling_times(tree: nx.Graph, step: int = 7) -> pd.DataFrame:
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
    minimum_weight: float | None = None,
    probability: bool = True,
    all_nodes: list | None = None,
) -> ig.Graph:
    """Build an undirected igraph graph from an edge list."""
    work = df[["NodeA", "NodeB", weight_col]].copy()

    if minimum_weight is not None:
        if probability:
            work = work[work[weight_col] >= float(minimum_weight)]
        else:
            thr = 1.0 / (float(minimum_weight) + 1.0)
            work = work[work[weight_col] >= thr]

    edges = work[["NodeA", "NodeB", weight_col]].to_records(index=False).tolist()

    g = ig.Graph.TupleList(
        edges=edges,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=weight_col,
    )

    if all_nodes is not None:
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
    resolution: float | None = None,
    minimum_weight: float | None = None,
    probability: bool = True,
    n_restarts: int = 10,
) -> dict:
    g = build_igraph(
        df[["NodeA", "NodeB", weight_col]],
        weight_col,
        minimum_weight=minimum_weight,
        probability=probability,
        all_nodes=sorted(nodes_present),
    )

    if probability:
        best = None
        best_q = -np.inf

        for _ in range(n_restarts):
            part = g.community_leiden(
                weights=weight_col,
                resolution=resolution,
                n_iterations=-1,
            )

            q = g.modularity(membership=part, weights=weight_col)

            if q > best_q:
                best_q = q
                best = part

        memb = best.membership if best is not None else None
    else:
        part = g.connected_components(mode="weak")
        memb = part.membership if part is not None else None

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
    resolution: float | None = None,
    minimum_weight: float | None = None,
    probability: bool = True,
    n_restarts: int = 10,
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
            probability=probability,
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


def plot_stability_over_time(summary: pd.DataFrame, title: str, ax) -> None:
    cols = ["forward", "backward", "jaccard"]
    missing = [c for c in cols if c not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")

    for c, style in zip(cols, ["--", "-.", "-"]):
        ax.plot(summary.index, summary[c], linestyle=style, linewidth=2, label=c.capitalize())

    ax.set_xlabel("Week")
    ax.set_ylabel("Mean stability")
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", pad=20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    parser.add_argument("--datasets", default="../config/generate_datasets.yaml")
    parser.add_argument("--stability", default="../config/cluster_stability.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    defaults_cfg = load_yaml(Path(args.defaults))
    datasets_cfg = load_yaml(Path(args.datasets))
    stability_cfg = load_yaml(Path(args.stability))

    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "main"], "../figures/main"))
    sup_figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "main"], "../tables/main"))

    stability_subdir = deep_get(stability_cfg, ["plots", "stability_subdir"], "stability")
    sup_figs_dir = sup_figs_dir / stability_subdir
    ensure_dirs(figs_dir, sup_figs_dir, tabs_dir)

    step_days = int(deep_get(stability_cfg, ["sampling", "step_days"], 7))
    train_max_time_index = int(deep_get(stability_cfg, ["sampling", "train_max_time_index"], 2))

    n_restarts = int(deep_get(stability_cfg, ["leiden", "n_restarts"], 10))
    min_prob_weight = float(deep_get(stability_cfg, ["leiden", "min_prob_weight"], 1e-4))
    gammas = _gamma_grid(stability_cfg)

    snp_thresholds = tuple(deep_get(stability_cfg, ["scores", "snp_thresholds"], [0, 1, 2, 3, 4, 5]))
    logit_max_iter = int(deep_get(stability_cfg, ["logistic_regression", "max_iter"], 200))
    save_formats = list(deep_get(stability_cfg, ["plots", "save_formats"], ["png", "pdf"]))

    tree_path = Path(deep_get(datasets_cfg, ["backbone", "tree_gml"],
                              "../data/processed/synthetic/scovmod_tree.gml"))

    toit, inference_cfg = _build_toit(defaults_cfg)

    trans_tree = nx.read_gml(tree_path)

    populated_tree = populate_epidemic_data(toit=toit, tree=trans_tree)
    gen_results = simulate_genomic_data(toit=toit, tree=populated_tree)
    pairwise = generate_pairwise_data(
        packed_genomic_data=gen_results["packed"],
        tree=populated_tree,
    )

    sample_dates = sampling_times(populated_tree, step=step_days)

    fig = plt.figure(figsize=(10, 5))
    sns.countplot(data=sample_dates, x="available_time")
    plt.xlabel("Week")
    plt.ylabel("Number of cases")
    save_figure(fig, sup_figs_dir / "case_counts_over_time", save_formats)
    plt.close(fig)

    inter_gens = tuple(deep_get(inference_cfg, ["inter_generations"], [0, 1]))
    no_intermediates = int(deep_get(inference_cfg, ["num_intermediates"], 10))
    num_simulations = int(deep_get(inference_cfg, ["num_simulations"], 10000))

    pairwise["MechProbLinearDist"] = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=pairwise["LinearDist"].values,
        temporal_distance=pairwise["TemporalDist"].values,
        intermediate_generations=inter_gens,
        no_intermediates=no_intermediates,
        num_simulations=num_simulations,
    )

    pairwise["MechProbPoissonDist"] = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=pairwise["PoissonDist"].values,
        temporal_distance=pairwise["TemporalDist"].values,
        intermediate_generations=inter_gens,
        no_intermediates=no_intermediates,
        num_simulations=num_simulations,
    )

    initial_nodes = set(sample_dates.loc[sample_dates["available_time"] <= train_max_time_index, "node"])
    initial_data = pairwise[
        pairwise["NodeA"].isin(initial_nodes) &
        pairwise["NodeB"].isin(initial_nodes)
    ]

    y = initial_data["Related"].astype(int).values
    for dist_col in ("LinearDist", "PoissonDist"):
        X = initial_data[["TemporalDist", dist_col]].values
        col = f"LogitProb{dist_col}"
        clf = LogisticRegression(solver="lbfgs", max_iter=logit_max_iter)
        clf.fit(X, y)
        pairwise[col] = clf.predict_proba(pairwise[["TemporalDist", dist_col]].values)[:, 1]

    pairwise["LinearDistScore"] = 1.0 / (pairwise["LinearDist"] + 1.0)
    pairwise["PoissonDistScore"] = 1.0 / (pairwise["PoissonDist"] + 1.0)

    truth = build_ground_truth_memberships(tree_path)

    prob_weight_columns = [
        "MechProbLinearDist",
        "MechProbPoissonDist",
        "LogitProbLinearDist",
        "LogitProbPoissonDist",
    ]

    metrics_rows = []
    rows = []
    for weight in prob_weight_columns:
        graph = build_igraph(
            initial_data[["NodeA", "NodeB", weight]],
            weight,
            minimum_weight=min_prob_weight,
            probability=True,
            all_nodes=sorted(initial_nodes),
        )

        for gamma in gammas:
            best = None
            best_q = -np.inf

            for _ in range(n_restarts):
                part = graph.community_leiden(
                    weights=weight,
                    resolution=float(gamma),
                    n_iterations=-1,
                )

                q = graph.modularity(
                    membership=part,
                    weights=weight,
                    resolution=gamma,
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
                "gamma": gamma,
                "cluster_id": np.array(memb, dtype=int),
                "weight": weight,
            }))

    partitions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    for (weight, gamma), sub in partitions.groupby(["weight", "gamma"], observed=True):
        pred = dict(zip(sub["case_id"].tolist(), sub["cluster_id"].tolist()))
        pred = {int(k): {int(v)} for k, v in pred.items()}
        prec, rec, f1 = bcubed_scores(pred=pred, truth=truth)
        metrics_rows.append({
            "gamma": float(gamma),
            "Weight": weight,
            "BCubed_Precision": prec,
            "BCubed_Recall": rec,
            "BCubed_F1_Score": f1,
            "N_cases": len(pred),
        })

    eval_metrics = pd.DataFrame(metrics_rows)
    idx = eval_metrics.groupby("Weight")["BCubed_F1_Score"].idxmax()
    best_f1 = eval_metrics.loc[idx, ["Weight", "BCubed_F1_Score", "gamma"]]
    model_res = best_f1.set_index("Weight")["gamma"].to_dict()

    score_weight_columns = [
        "LinearDistScore",
        "PoissonDistScore",
    ]

    metrics_rows = []
    rows = []
    for weight in score_weight_columns:
        for snp in snp_thresholds:
            graph = build_igraph(
                initial_data[["NodeA", "NodeB", weight]],
                weight,
                minimum_weight=snp,
                probability=False,
                all_nodes=sorted(initial_nodes),
            )

            cluster = graph.connected_components(mode="weak")
            memb = cluster.membership if cluster is not None else None

            if memb is None:
                memb = list(range(graph.vcount()))
            rows.append(pd.DataFrame({
                "case_id": graph.vs["case_id"],
                "snp": snp,
                "cluster_id": np.array(memb, dtype=int),
                "weight": weight,
            }))

    partitions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    for (weight, snp), sub in partitions.groupby(["weight", "snp"], observed=True):
        pred = dict(zip(sub["case_id"].tolist(), sub["cluster_id"].tolist()))
        pred = {int(k): {int(v)} for k, v in pred.items()}
        prec, rec, f1 = bcubed_scores(pred=pred, truth=truth)
        metrics_rows.append({
            "snp": snp,
            "Weight": weight,
            "BCubed_Precision": prec,
            "BCubed_Recall": rec,
            "BCubed_F1_Score": f1,
            "N_cases": len(pred),
        })

    eval_metrics = pd.DataFrame(metrics_rows)
    idx = eval_metrics.groupby("Weight")["BCubed_F1_Score"].idxmax()
    best_f1 = eval_metrics.loc[idx, ["Weight", "BCubed_F1_Score", "snp"]]
    model_snp = best_f1.set_index("Weight")["snp"].to_dict()

    method_titles = {
        "mech_linear": "(a) Mechanistic (Deterministic)",
        "mech_poisson": "(b) Mechanistic (Stochastic)",
        "logit_linear": "(c) Logistic (Deterministic)",
        "logit_poisson": "(d) Logistic (Stochastic)",
        "snp_linear": "(e) SNP (Deterministic)",
        "snp_poisson": "(f) SNP (Stochastic)",
    }

    custom_titles = deep_get(stability_cfg, ["methods", "titles"], {})
    if isinstance(custom_titles, dict):
        method_titles.update(custom_titles)

    methods = {
        "mech_linear": {
            "weight_col": "MechProbLinearDist",
            "probability": True,
            "minimum_weight": min_prob_weight,
            "resolution": model_res.get("MechProbLinearDist"),
        },
        "mech_poisson": {
            "weight_col": "MechProbPoissonDist",
            "probability": True,
            "minimum_weight": min_prob_weight,
            "resolution": model_res.get("MechProbPoissonDist"),
        },
        "logit_linear": {
            "weight_col": "LogitProbLinearDist",
            "probability": True,
            "minimum_weight": min_prob_weight,
            "resolution": model_res.get("LogitProbLinearDist"),
        },
        "logit_poisson": {
            "weight_col": "LogitProbPoissonDist",
            "probability": True,
            "minimum_weight": min_prob_weight,
            "resolution": model_res.get("LogitProbPoissonDist"),
        },
        "snp_linear": {
            "weight_col": "LinearDistScore",
            "probability": False,
            "minimum_weight": model_snp.get("LinearDistScore"),
            "resolution": None,
        },
        "snp_poisson": {
            "weight_col": "PoissonDistScore",
            "probability": False,
            "minimum_weight": model_snp.get("PoissonDistScore"),
            "resolution": None,
        },
    }

    enabled_methods = deep_get(
        stability_cfg,
        ["methods", "enabled"],
        ["mech_linear", "mech_poisson", "logit_linear", "logit_poisson"],
    )
    enabled_methods = [m for m in enabled_methods if m in methods]
    enabled_methods = [m for m in enabled_methods if methods[m].get("resolution") is not None
                       or methods[m].get("minimum_weight") is not None]

    n_methods = len(enabled_methods)
    if n_methods == 0:
        raise ValueError("No valid methods enabled for stability plotting.")

    ncols = 2 if n_methods > 1 else 1
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(enabled_methods):
        cfg = methods[name]
        print(f"Analysing: {name}")
        stability_df = cumulative_stability(
            pairwise,
            sample_dates,
            n_restarts=n_restarts,
            **cfg,
        )
        summary = stability_df.groupby("t1")[["forward", "backward", "jaccard"]].mean()
        summary.to_csv(tabs_dir / f"stability_summary_{name}.csv", index=False)
        plot_stability_over_time(summary, title=method_titles.get(name, name), ax=axes[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save_figure(fig, figs_dir / "cluster_stability", save_formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
