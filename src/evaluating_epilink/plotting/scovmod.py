"""SCoVMod tree and topology figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from scipy import stats

from .common import COLORS, TREE_STEP_COLORS, add_panel_labels, ccdf, read_json, read_table, save_figure, tree_depths

TREE_STEP_LABELS = {
    "raw": "Raw",
    "cleaned": "Filtered",
    "selected_component": "Largest component",
    "final_tree": "Final tree",
}


def render_tree_figure(results_root: Path, processed_synthetic_dir: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render the SCoVMod transmission-tree overview figure."""

    tree = nx.read_gml(processed_synthetic_dir / "scovmod" / "scovmod_tree.gml")
    heterogeneity = read_json(processed_synthetic_dir / "scovmod" / "scovmod_tree_tree_heterogeneity.json")
    tree_out_degree = np.array([degree for _, degree in tree.out_degree(tree.nodes)], dtype=int)
    depth_values = tree_depths(tree)
    tree_summary_df = read_table(results_root, "scovmod/scovmod_tree_summary")
    tree_degree_df = read_table(results_root, "scovmod/scovmod_tree_degree_distributions")
    tree_component_df = read_table(results_root, "scovmod/scovmod_tree_component_sizes")
    tree_summary_final = tree_summary_df.loc[tree_summary_df["label"] == "final_tree"]
    if tree_summary_final.empty:
        tree_summary_final = tree_summary_df.iloc[[-1]]
    tree_summary_final = tree_summary_final.iloc[0]

    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    component_sizes = tree_component_df["component_size"].to_numpy(dtype=int)
    x_component, y_component = ccdf(component_sizes)
    ax1.plot(x_component, y_component, color=COLORS["blue"], linewidth=2.2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set(xlabel="Component size (nodes)", ylabel="CCDF")

    line_style_map = {
        "raw": ("--", "o"),
        "cleaned": ("-.", "s"),
        "selected_component": (":", "^"),
        "final_tree": ("-", "D"),
    }
    for label in ["raw", "cleaned", "selected_component"]:
        subset = tree_degree_df.loc[
            (tree_degree_df["graph"] == label) & (tree_degree_df["degree_type"] == "out"),
            "value",
        ]
        values = subset.to_numpy(dtype=int)
        values = values[values > 0]
        if values.size == 0:
            continue
        x_vals, y_vals = ccdf(values)
        linestyle, marker = line_style_map[label]
        ax2.plot(
            x_vals,
            y_vals,
            color=TREE_STEP_COLORS[label],
            linewidth=1.9,
            linestyle=linestyle,
            marker=marker,
            markersize=4.2,
            markevery=max(1, len(x_vals) // 12),
            label=TREE_STEP_LABELS[label],
        )

    final_nonzero = tree_out_degree[tree_out_degree > 0]
    if final_nonzero.size > 0:
        x_tree, y_tree = ccdf(final_nonzero)
        linestyle, marker = line_style_map["final_tree"]
        ax2.plot(
            x_tree,
            y_tree,
            color=TREE_STEP_COLORS["final_tree"],
            linewidth=2.2,
            linestyle=linestyle,
            marker=marker,
            markersize=4.2,
            markevery=max(1, len(x_tree) // 12),
            label=TREE_STEP_LABELS["final_tree"],
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set(xlabel="Out-degree (offspring)", ylabel="CCDF")
    ax2.legend(loc="best", title=None, handlelength=3)

    counts = np.unique(depth_values, return_counts=True)
    bars = ax3.bar(counts[0], counts[1], color="#DDEAF7", edgecolor=COLORS["blue"], linewidth=0.8, width=0.9)
    for patch in bars:
        patch.set_hatch("//")
    ax3.set_yscale("log")
    ax3.set(xlabel="Depth from root", ylabel="Nodes")
    summary_text = (
        f"n={int(tree_summary_final['n_nodes']):,}\n"
        f"edges={int(tree_summary_final['n_edges']):,}\n"
        f"max out-degree={int(tree_summary_final['max_out_degree'])}\n"
        f"mean out-degree={tree_summary_final['mean_out_degree']:.2f}"
    )
    ax3.text(
        0.98,
        0.98,
        summary_text,
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#444444", "alpha": 0.95},
    )

    x_max = int(np.quantile(tree_out_degree, 0.99))
    x_max = max(20, min(x_max, int(tree_out_degree.max())))
    bins = np.arange(-0.5, x_max + 1.5, 1.0)
    sns.histplot(
        tree_out_degree,
        bins=bins,
        stat="probability",
        color="#F8E4CC",
        edgecolor=COLORS["orange"],
        linewidth=0.8,
        ax=ax4,
    )
    mean_reproduction_number = float(heterogeneity.get("meanRt", np.nan))
    dispersion_k = float(heterogeneity.get("disp_k", np.nan))
    if np.isfinite(mean_reproduction_number) and np.isfinite(dispersion_k) and dispersion_k > 0:
        x_vals = np.arange(0, x_max + 1, dtype=int)
        probability = dispersion_k / (dispersion_k + mean_reproduction_number)
        pmf = stats.nbinom.pmf(x_vals, dispersion_k, probability)
        ax4.plot(x_vals, np.clip(pmf, 1e-12, None), color=COLORS["vermillion"], linewidth=2.2, label="NB fit")
        ax4.legend(loc="best")
    ax4.set_yscale("log")
    ax4.set(xlabel="Offspring count (out-degree)", ylabel="Probability")
    heterogeneity_text = (
        f"R={mean_reproduction_number:.2f}\n"
        f"k={dispersion_k:.2f}\n"
        f"Zero={heterogeneity.get('pct_zero_transmitters', np.nan):.1f}%\n"
        f"80% by {heterogeneity.get('prop_80_percent_transmitters', np.nan) * 100:.1f}%"
    )
    ax4.text(
        0.98,
        0.98,
        heterogeneity_text,
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#444444", "alpha": 0.95},
    )
    add_panel_labels([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"], x=-0.12, y=1.08)
    save_figure(fig, figures_dir / "sm_scovmod_tree_overview", formats)
    plt.close(fig)
