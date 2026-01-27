#!/usr/bin/env python3
"""
scripts/04.1_sparsify_effects.py

Edge- and neighbourhood-level diagnostics for graph sparsification.

Rationale
---------
Sparsification acts on edges, so we evaluate its effect *before* community detection:
  1) How much total edge weight ("probability mass") is retained?
  2) How many edges survive?
  3) Does sparsification preserve each node's high-probability neighbourhood?

This script operates on a *pairwise* DataFrame with mechanistic probabilities:
  - Case1, Case2, Prob_Mech
Optionally (synthetic):
  - Related (binary label indicating recent linkage in ground truth)

Outputs
-------
tables/supplementary/
  - sparsify_edge_retention.csv
  - sparsify_neighbourhood_stability.csv
  - sparsify_node_strength_distortion.csv
  - sparsify_components_summary.csv

figures/supplementary/sparsify/
  - weight_retention_curve.(png|pdf)
  - edge_retention_curve.(png|pdf)
  - neighbourhood_jaccard_boxplot_by_minw.(png|pdf)
  - node_strength_distortion_boxplot.(png|pdf)
  - components_vs_minw.(png|pdf)

Usage
-----
python scripts/07a_sparsify_edge_stability.py \
  --paths config/paths.yaml \
  --input data/processed/synthetic/scenario=Baseline/pairwise.parquet \
  --minw 0,0.001,0.005,0.01,0.02,0.05,0.1 \
  --topk 5,10,20 \
  --max-nodes 5000 \
  --plots

Notes
-----
- Neighbourhood stability is computed relative to a reference graph at minw=0 (or the smallest threshold given).
- Graph building is undirected; if your analysis is directed later, keep this as a diagnostic stage.
- For very large datasets, use --max-nodes or --max-edges and/or increase thresholds.

Dependencies
------------
pandas, numpy, matplotlib, networkx, pyyaml
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import igraph as ig
import networkx as nx

from utils import *


# -----------------------------
# Sparsification + graph stats
# -----------------------------

def sparsify_edges(df: pd.DataFrame, min_w: float, weight_col: str) -> pd.DataFrame:
    """Filter edges by min edge weight."""
    if min_w <= 0:
        return df
    return df.loc[df[weight_col] >= float(min_w)].copy()

def build_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Undirected weighted graph with edge attribute 'weight'.
    """
    edges = df[["NodeA", "NodeB", "MechProbLinearDist"]].to_records(index=False).tolist()
    g = nx.Graph()
    g.add_weighted_edges_from(
        edges,
        weight="weight",
    )
    return g

def graph_summary(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else np.nan

    # Components
    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": np.nan,
            "n_components": 0,
            "giant_component_size": 0,
            "giant_component_frac": np.nan,
        }

    comps = list(nx.connected_components(g))
    sizes = np.array([len(c) for c in comps], dtype=int)
    giant = int(sizes.max()) if sizes.size else 0
    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(density),
        "n_components": int(len(comps)),
        "giant_component_size": giant,
        "giant_component_frac": float(giant / n) if n > 0 else np.nan,
    }

def total_edge_weight(df: pd.DataFrame, weight_col: str) -> float:
    return float(df[weight_col].sum())


# -----------------------------
# Neighbourhood stability
# -----------------------------

def topk_neighbours_from_edges(
        df: pd.DataFrame,
        k: int,
        weight_col: str,
        nodes: pd.Index = None,
) -> Dict[Any, List[Any]]:
    """
    Compute top-k neighbours per node from an edge list.

    Implementation is designed to avoid building a full adjacency matrix:
      - group edges by endpoint and sort by weight
      - return neighbour IDs

    If nodes is provided, return at least those nodes (possibly empty lists).
    """
    # Ensure undirected representation by duplicating edges
    a = df[["NodeA", "NodeB", weight_col]].rename(columns={"NodeA": "u", "NodeB": "v"})
    b = df[["NodeB", "NodeA", weight_col]].rename(columns={"NodeB": "u", "NodeA": "v"})
    long = pd.concat([a, b], ignore_index=True)

    # Sort so groupby head(k) yields top-k efficiently
    long = long.sort_values(["u", weight_col], ascending=[True, False])

    top = long.groupby("u", sort=False).head(k)

    neigh = top.groupby("u")["v"].apply(list).to_dict()

    if nodes is not None:
        for n in nodes:
            neigh.setdefault(n, [])

    return neigh


def jaccard(a, b) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def neighbourhood_jaccard(
    ref_neigh: Dict[Any, List[Any]],
    cur_neigh: Dict[Any, List[Any]],
    nodes: Iterable[Any],
) -> np.ndarray:
    out = np.empty(len(list(nodes)), dtype=float)
    for i, n in enumerate(nodes):
        out[i] = jaccard(ref_neigh.get(n, []), cur_neigh.get(n, []))
    return out


# -----------------------------
# Node strength distortion
# -----------------------------

def node_strengths_from_edges(df: pd.DataFrame, weight_col: str, nodes: Optional[pd.Index] = None) -> pd.Series:
    """
    Weighted degree ("strength") per node from edge list.
    """
    a = df[["NodeA", weight_col]].rename(columns={"NodeA": "node", weight_col: "w"})
    b = df[["NodeB", weight_col]].rename(columns={"NodeB": "node", weight_col: "w"})
    s = pd.concat([a, b], ignore_index=True).groupby("node")["w"].sum()

    if nodes is not None:
        s = s.reindex(nodes, fill_value=0.0)
    return s


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_curve(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df[x].values, df[y].values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def boxplot_by_threshold(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Convert thresholds to strings for stable categorical ordering
    cats = df[x].astype(str)
    ax.boxplot(
        [df.loc[cats == c, y].values for c in sorted(cats.unique(), key=lambda z: float(z))],
        tick_labels=[c for c in sorted(cats.unique(), key=lambda z: float(z))],
        showfliers=False,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    return fig



# -----------------------------
# Main analysis
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/simulate_datasets.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    figs_dir = figs_dir / "sparsify"

    sparsify_dir = processed_dir / "sparsify"
    ensure_dirs(tabs_dir, processed_dir, sparsify_dir, figs_dir)

    min_ws = list(deep_get(clus_cfg, ["network", "min_edge_weights"], [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))
    topks = list(deep_get(clus_cfg, ["network", "topks"], [5, 10, 20]))
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["MechProbLinearDist"]))
    formats = list(deep_get(clus_cfg, ["save_formats"], ["png", "pdf"]))


    sc_dir = processed_dir / f"scenario=baseline"
    df = pd.read_parquet(sc_dir / "pairwise_eval.parquet")

    # Establish reference threshold: smallest minw supplied
    ref_minw = min(min_ws)
    df_ref = sparsify_edges(df, ref_minw, weight_columns[0])

    # Reference node set: nodes present in reference edges
    ref_nodes = pd.Index(pd.unique(df_ref[["NodeA", "NodeB"]].values.ravel()))

    # Precompute reference neighbourhoods and strengths for each k
    ref_neigh_by_k = {k: topk_neighbours_from_edges(df_ref, k, weight_columns[0], ref_nodes) for k in topks}
    ref_strength = node_strengths_from_edges(df_ref, weight_columns[0], nodes=ref_nodes)

    # Reference total weight and edge count
    w_ref = total_edge_weight(df_ref,  weight_columns[0])
    m_ref = len(df_ref)

    # Collect outputs
    retention_rows = []
    stability_rows = []
    strength_rows = []
    components_rows = []

    for minw in min_ws:
        df_w = sparsify_edges(df, minw, weight_columns[0])

        # Restrict to reference node set for comparability
        df_w = df_w[df_w["NodeA"].isin(ref_nodes) & df_w["NodeB"].isin(ref_nodes)].copy()

        # Edge retention
        w_w = total_edge_weight(df_w, weight_columns[0]) if len(df_w) else 0.0
        m_w = int(len(df_w))
        retention_rows.append({
            "min_edge_weight": float(minw),
            "n_edges": m_w,
            "edge_retention_frac": float(m_w / m_ref) if m_ref > 0 else np.nan,
            "total_weight": float(w_w),
            "weight_retention_frac": float(w_w / w_ref) if w_ref > 0 else np.nan,
        })

        # Node strength distortion (on reference node set)
        s_w = node_strengths_from_edges(df_w,  weight_columns[0], nodes=ref_nodes)
        # Use log-ratio with small epsilon to handle zeros gracefully
        eps = 1e-12
        log_ratio = np.log((s_w + eps) / (ref_strength + eps))
        abs_log_ratio = np.abs(log_ratio)

        strength_rows.append(pd.DataFrame({
            "min_edge_weight": float(minw),
            "case_id": ref_nodes.to_numpy(dtype=object),
            "strength_ref": ref_strength.values,
            "strength_cur": s_w.values,
            "log_ratio": log_ratio,
            "abs_log_ratio": abs_log_ratio,
        }))

        # Neighbourhood stability for each k
        for k in topks:
            cur_neigh = topk_neighbours_from_edges(df_w, k, weight_columns[0], ref_nodes)
            j = neighbourhood_jaccard(ref_neigh_by_k[k], cur_neigh, ref_nodes)

            stab = pd.DataFrame({
                "min_edge_weight": float(minw),
                "k": int(k),
                "case_id": ref_nodes.to_numpy(dtype=object),
                "jaccard": j,
            })

            # Optional: stratify by Related if present (synthetic)
            if "Related" in df.columns:
                # Node-level “relatedness exposure” is not uniquely defined; we provide an edge-derived proxy:
                # fraction of incident edges that are 'Related' in the *reference* graph.
                # This helps see whether sparsification disproportionately affects outbreak-connected nodes.
                pass

            stability_rows.append(stab)

        # Graph-level component stats (on current threshold graph restricted to ref nodes)
        g = build_graph(df_w)
        # Ensure isolated reference nodes are counted (important for fragmentation diagnosis)
        g.add_nodes_from(ref_nodes.tolist())

        comp = graph_summary(g)
        comp["min_edge_weight"] = float(minw)
        components_rows.append(comp)

    retention_df = pd.DataFrame(retention_rows).sort_values("min_edge_weight")
    stability_df = pd.concat(stability_rows, ignore_index=True)
    strength_df = pd.concat(strength_rows, ignore_index=True)
    components_df = pd.DataFrame(components_rows).sort_values("min_edge_weight")

    # Save tables
    retention_df.to_csv(tabs_dir / "sparsify_edge_retention.csv", index=False)
    stability_df.to_csv(tabs_dir / "sparsify_neighbourhood_stability.csv", index=False)
    strength_df.to_csv(tabs_dir / "sparsify_node_strength_distortion.csv", index=False)
    components_df.to_csv(tabs_dir / "sparsify_components_summary.csv", index=False)

    # Plots
    fig = plot_curve(
        retention_df, x="min_edge_weight", y="weight_retention_frac",
        title="Total edge-weight retained under sparsification",
        xlabel="min_edge_weight", ylabel="Weight retention fraction"
    )
    save_figure(fig, figs_dir / "weight_retention_curve", formats)
    plt.close(fig)

    fig = plot_curve(
        retention_df, x="min_edge_weight", y="edge_retention_frac",
        title="Edge count retained under sparsification",
        xlabel="min_edge_weight", ylabel="Edge retention fraction"
    )
    save_figure(fig, figs_dir / "edge_retention_curve", formats)
    plt.close(fig)

    # Neighbourhood stability boxplots per k
    for k in topks:
        sub = stability_df[stability_df["k"] == k].copy()
        fig = boxplot_by_threshold(
            sub, x="min_edge_weight", y="jaccard",
            title=f"Top-{k} neighbourhood Jaccard vs min_edge_weight (ref={ref_minw})",
            xlabel="min_edge_weight", ylabel="Jaccard similarity"
        )
        save_figure(fig, figs_dir / f"neighbourhood_jaccard_k{k}", formats)
        plt.close(fig)

    # Strength distortion: |log ratio| boxplot
    fig = boxplot_by_threshold(
        strength_df, x="min_edge_weight", y="abs_log_ratio",
        title="Node strength distortion under sparsification (|log ratio|)",
        xlabel="min_edge_weight", ylabel="|log(strength / strength_ref)|"
    )
    save_figure(fig, figs_dir / "node_strength_distortion_boxplot", formats)
    plt.close(fig)

    # Components vs threshold
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(components_df["min_edge_weight"], components_df["n_components"], marker="o", label="components")
    ax2 = ax.twinx()
    ax2.plot(components_df["min_edge_weight"], components_df["giant_component_frac"], marker="o", linestyle="--",
             label="giant component fraction")
    ax.set_title("Graph fragmentation under sparsification")
    ax.set_xlabel("min_edge_weight")
    ax.set_ylabel("Number of components")
    ax2.set_ylabel("Giant component fraction")
    ax.grid(True, alpha=0.3)

    # Build a combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    save_figure(fig, figs_dir / "components_vs_minw", formats)
    plt.close(fig)

    print(f"Saved tables to: {tabs_dir}")
    print(f"Saved figures to: {figs_dir}")
    print("Done.")


if __name__ == "__main__":
    main()