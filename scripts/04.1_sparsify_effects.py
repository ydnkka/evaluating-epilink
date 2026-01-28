#!/usr/bin/env python3
"""
scripts/04.1_sparsify_effects.py

Edge- and neighbourhood-level diagnostics for graph sparsification.

How much does sparsification change the total transmission probability mass attached to each case?

“We selected sparsification thresholds that preserve node strength distributions, ensuring that downstream clustering does not artefactually suppress highly connected cases.”

Rationale
---------
Sparsification acts on edges, so we evaluate its effect *before* community detection:
  1) How much total edge weight ("probability mass") is retained?
  2) How many edges survive?

This script operates on a *pairwise* DataFrame with mechanistic probabilities:
  - NodeA, NodeB, "MechProbLinearDist"
  - Related (binary label indicating recent linkage in ground truth)

Outputs
-------
tables/supplementary/
  - sparsify_edge_retention.csv
  - sparsify_node_strength_distortion.csv
  - sparsify_components_summary.csv

figures/supplementary/sparsify/
  - weight_retention_curve.(png|pdf)
  - edge_retention_curve.(png|pdf)
  - node_strength_distortion_boxplot.(png|pdf)
  - components_vs_minw.(png|pdf)

Notes
-----
- Graph building is undirected; if your analysis is directed later, keep this as a diagnostic stage.
- For very large datasets, use --max-nodes or --max-edges and/or increase thresholds.

Dependencies
------------
pandas, numpy, matplotlib, networkx, pyyaml
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from utils import *


# -----------------------------
# Sparsification + graph stats
# -----------------------------

def sparsify_edges(df: pd.DataFrame, min_w: float, weight_col: str) -> pd.DataFrame:
    """Filter edges by min edge weight."""
    if float(min_w) <= 0:
        return df
    return df.loc[df[weight_col] >= float(min_w)].copy()


def build_graph(df: pd.DataFrame, weight_col: str) -> nx.Graph:
    """Undirected weighted graph with edge attribute 'weight'."""
    edges = df[["NodeA", "NodeB", weight_col]].to_records(index=False).tolist()
    g = nx.Graph()
    g.add_weighted_edges_from(edges, weight="weight")
    return g


def graph_summary(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else np.nan

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
# Node strength distortion
# -----------------------------

def node_strengths_from_edges(df: pd.DataFrame, weight_col: str, nodes: Optional[pd.Index] = None) -> pd.Series:
    """Weighted degree ("strength") per node from edge list."""
    if len(df) == 0:
        s = pd.Series(dtype=float)
    else:
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
    cats = df[x].astype(str)
    order = sorted(cats.unique(), key=lambda z: float(z))
    ax.boxplot(
        [df.loc[cats == c, y].values for c in order],
        tick_labels=order,
        showfliers=False,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def grouped_boxplot_by_threshold(
    df: pd.DataFrame,
    threshold_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """
    Grouped boxplot: for each threshold, show a pair (or multiple) of boxes by group_col.
    Pure matplotlib (no seaborn).
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))

    thr = df[threshold_col].astype(float)
    thr_levels = sorted(thr.unique())
    groups = [g for g in df[group_col].dropna().unique().tolist()]

    # Consistent ordering for booleans / categoricals
    try:
        groups = sorted(groups)
    except Exception:
        pass

    width = 0.8 / max(1, len(groups))
    positions = []
    data = []
    xticks = []

    for i, t in enumerate(thr_levels):
        xticks.append(i + 1)
        for j, g in enumerate(groups):
            subset = df[(thr == t) & (df[group_col] == g)][y_col].values
            data.append(subset)
            positions.append((i + 1) - 0.4 + width/2 + j * width)

    bp = ax.boxplot(data, positions=positions, widths=width * 0.95, showfliers=False, patch_artist=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in thr_levels], rotation=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend using dummy handles
    handles = []
    labels = []
    for g in groups:
        h = plt.Line2D([0], [0], color="black", marker="s", linestyle="None")
        handles.append(h)
        labels.append(str(g))
    ax.legend(handles, labels, title=group_col, fontsize=8, loc="best")
    return fig


# -----------------------------
# Main analysis
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/simulate_datasets.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    parser.add_argument("--scenario", default="baseline", help="Scenario subdir name, e.g. baseline")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary")) / "sparsify"

    ensure_dirs(tabs_dir, processed_dir, figs_dir)

    min_ws = list(deep_get(clus_cfg, ["network", "min_edge_weights"], [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))
    topks = list(deep_get(clus_cfg, ["network", "topks"], [5, 10, 20]))
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["MechProbLinearDist"]))
    formats = list(deep_get(clus_cfg, ["save_formats"], ["png", "pdf"]))

    sc_dir = processed_dir / f"scenario={args.scenario}"
    df = pd.read_parquet(sc_dir / "pairwise_eval.parquet")

    weight_col = weight_columns[0]

    # Establish reference threshold: smallest minw supplied
    ref_minw = float(min(min_ws))
    df_ref = sparsify_edges(df, ref_minw, weight_col)

    # Reference node set: nodes present in reference edges
    ref_nodes = pd.Index(pd.unique(df_ref[["NodeA", "NodeB"]].values.ravel()))

    # Precompute reference strengths
    ref_strength = node_strengths_from_edges(df_ref, weight_col, nodes=ref_nodes)

    # Reference total weight and edge count
    w_ref = total_edge_weight(df_ref, weight_col)
    m_ref = int(len(df_ref))

    # Collect outputs
    retention_rows: List[Dict[str, Any]] = []
    strength_rows: List[pd.DataFrame] = []
    components_rows: List[Dict[str, Any]] = []

    for minw in min_ws:
        minw = float(minw)
        df_w = sparsify_edges(df, minw, weight_col)

        # Edge retention
        w_w = total_edge_weight(df_w, weight_col) if len(df_w) else 0.0
        m_w = int(len(df_w))
        retention_rows.append({
            "min_edge_weight": float(minw),
            "n_edges": m_w,
            "edge_retention_frac": float(m_w / m_ref) if m_ref > 0 else np.nan,
            "total_weight": float(w_w),
            "weight_retention_frac": float(w_w / w_ref) if w_ref > 0 else np.nan,
        })

        # Node strength distortion (on reference node set)
        s_w = node_strengths_from_edges(df_w, weight_col, nodes=ref_nodes)
        eps = 1e-12
        log_ratio = np.log((s_w + eps) / (ref_strength + eps))
        abs_log_ratio = np.abs(log_ratio)

        strength_rows.append(pd.DataFrame({
            "min_edge_weight": float(minw),
            "case_id": ref_nodes.to_numpy(dtype=np.int64),
            "strength_ref": ref_strength.values,
            "strength_cur": s_w.values,
            "log_ratio": log_ratio.values if isinstance(log_ratio, pd.Series) else log_ratio,
            "abs_log_ratio": abs_log_ratio.values if isinstance(abs_log_ratio, pd.Series) else abs_log_ratio,
        }))

        # Graph-level component stats (restricted to ref nodes, including isolates)
        g = build_graph(df_w, weight_col)
        g.add_nodes_from(ref_nodes.tolist())
        comp = graph_summary(g)
        comp["min_edge_weight"] = float(minw)
        components_rows.append(comp)

    retention_df = pd.DataFrame(retention_rows).sort_values("min_edge_weight")
    strength_df = pd.concat(strength_rows, ignore_index=True)
    components_df = pd.DataFrame(components_rows).sort_values("min_edge_weight")

    # Save tables
    retention_df.to_csv(tabs_dir / "sparsify_edge_retention.csv", index=False)
    strength_df.to_csv(tabs_dir / "sparsify_node_strength_distortion.csv", index=False)
    components_df.to_csv(tabs_dir / "sparsify_components_summary.csv", index=False)

    # Plots: retention curves
    fig = plot_curve(
        retention_df, x="min_edge_weight", y="weight_retention_frac",
        title="Total edge-weight retained under sparsification",
        xlabel="min_edge_weight", ylabel="Weight retention fraction",
    )
    save_figure(fig, figs_dir / "weight_retention_curve", formats)
    plt.close(fig)

    fig = plot_curve(
        retention_df, x="min_edge_weight", y="edge_retention_frac",
        title="Edge count retained under sparsification",
        xlabel="min_edge_weight", ylabel="Edge retention fraction",
    )
    save_figure(fig, figs_dir / "edge_retention_curve", formats)
    plt.close(fig)

    # Strength distortion: |log ratio| boxplot
    fig = boxplot_by_threshold(
        strength_df, x="min_edge_weight", y="abs_log_ratio",
        title="Node strength distortion under sparsification (|log ratio|)",
        xlabel="min_edge_weight", ylabel="|log(strength / strength_ref)|",
    )
    save_figure(fig, figs_dir / "node_strength_distortion_boxplot", formats)
    plt.close(fig)

    # Components vs threshold
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(components_df["min_edge_weight"], components_df["n_components"], marker="o", label="components")
    ax2 = ax.twinx()
    ax2.plot(
        components_df["min_edge_weight"],
        components_df["giant_component_frac"],
        marker="o",
        linestyle="--",
        label="giant component fraction",
    )
    ax.set_title("Graph fragmentation under sparsification")
    ax.set_xlabel("min_edge_weight")
    ax.set_ylabel("Number of components")
    ax2.set_ylabel("Giant component fraction")
    ax.grid(True, alpha=0.3)

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
