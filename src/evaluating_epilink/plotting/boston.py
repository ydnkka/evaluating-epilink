"""Boston empirical clustering figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .common import COLORS, add_panel_labels, read_table, save_figure, style_colorbar

EXPOSURE_LABELS = {
    "count::BHCHP": "BHCHP",
    "count::Other": "Other",
    "count::City": "City",
    "count::Conference": "Conference",
    "count::SNF": "SNF",
}


def render_boston(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render the Boston cluster figure."""

    boston_clusters_comp = read_table(results_root, "boston/boston_cluster_composition")
    cluster_sizes = read_table(results_root, "boston/boston_cluster_sizes")

    available_columns = [column for column in EXPOSURE_LABELS if column in boston_clusters_comp.columns]
    exposure_counts = boston_clusters_comp[available_columns].copy().rename(columns=EXPOSURE_LABELS)
    exposure_counts.index = boston_clusters_comp["cluster_id"].astype(int)

    size_counts = cluster_sizes["size"].value_counts().sort_index()

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])

    bars = ax1.bar(
        size_counts.index,
        size_counts.values,
        color="#DDEAF7",
        edgecolor=COLORS["blue"],
        linewidth=0.8,
        width=0.85,
    )
    for patch in bars:
        patch.set_hatch("//")
    ax1.set(xlabel="Cluster size", ylabel="Clusters")
    ax1.set_title("Distribution of reported cluster sizes", loc="left", fontsize=11, pad=6)
    ax1.text(
        0.98,
        0.98,
        f"All clusters: {len(cluster_sizes)}\nFocus clusters: {int(cluster_sizes['is_focus_cluster'].sum())}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#444444", "alpha": 0.95},
    )

    vmax = float(np.nanmax(exposure_counts.to_numpy(dtype=float))) if not exposure_counts.empty else 1.0
    heatmap = sns.heatmap(
        exposure_counts.T,
        vmin=0,
        vmax=max(1.0, vmax),
        cmap="YlOrBr",
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "Cases", "shrink": 0.7},
        ax=ax2,
    )
    ax2.tick_params(axis="y", labelsize="small", rotation=0)
    ax2.tick_params(axis="x", labelsize="small")
    ax2.set_xlabel("Focus cluster ID")
    ax2.set_ylabel("")
    ax2.set_title("Exposure composition of highlighted clusters", loc="left", fontsize=11, pad=6)
    style_colorbar(heatmap, 12)

    add_panel_labels([ax1, ax2], ["A", "B"])
    save_figure(fig, figures_dir / "boston_cluster", formats)
    plt.close(fig)
