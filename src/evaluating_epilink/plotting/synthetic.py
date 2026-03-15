"""Synthetic discrimination and clustering figures."""

from __future__ import annotations

import math
import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from .common import (
    MODEL_COLORS,
    MODEL_ORDER,
    MODELS,
    SCENARIO_ORDER,
    SCENARIOS,
    add_panel_labels,
    metric_label,
    read_table,
    save_figure,
    style_colorbar,
)


def plot_cluster_metrics(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    metric: str = "BCubed_F1_Score",
    y_axis: str = "Resolution",
    ncols: int = 3,
    figsize: tuple[float, float] = (12, 12),
) -> None:
    """Plot metric profiles across the clustering resolution sweep."""

    model_subset = ["DDP", "DDL2", "SDP", "SDL2"]
    line_styles = {"DDP": "-", "DDL2": "--", "SDP": ":", "SDL2": "-."}
    markers = {"DDP": "o", "DDL2": "s", "SDP": "^", "SDL2": "D"}

    available_scenarios = set(frame["ScenarioLabel"].dropna())
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in available_scenarios]
    panel_count = len(scenarios)
    nrows = math.ceil(panel_count / ncols)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(nrows, ncols)
    axes: list[Axes] = []
    for index in range(nrows * ncols):
        row, col = divmod(index, ncols)
        ax = fig.add_subplot(grid[row, col], sharey=axes[0] if axes else None)
        axes.append(ax)

    legend_handles = None
    legend_labels = None
    for index, scenario in enumerate(scenarios):
        ax = axes[index]
        subset = frame[frame["ScenarioLabel"] == scenario]
        for model in model_subset:
            rows = subset[subset["ModelLabel"] == model]
            if rows.empty:
                continue
            ax.plot(
                rows[y_axis],
                rows[metric],
                color=MODEL_COLORS[model],
                linestyle=line_styles[model],
                marker=markers[model],
                linewidth=1.9,
                markersize=4.3,
                label=model,
            )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(scenario, fontsize=11, pad=6, loc="left")
        ax.set_ylim((0, 1.05))
        if (index % ncols) != 0:
            ax.set_ylabel("")
        if index < (nrows - 1) * ncols:
            ax.set_xlabel("")
        panel_letter = string.ascii_uppercase[index]
        ax.text(-0.18, 1.12, f"{panel_letter})", transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")

    for index in range(panel_count, nrows * ncols):
        axes[index].axis("off")

    fig.supxlabel("Resolution", fontsize=12)
    fig.supylabel(metric_label(metric), fontsize=12)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Model",
            fontsize=10,
            title_fontsize=10.5,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=4,
            frameon=False,
            handlelength=3,
        )

    save_figure(fig, output_stem, formats)
    plt.close(fig)


def render_discrimination_and_clustering(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render discrimination and cluster-recovery figures."""

    discrimination = read_table(results_root, "discrimination/discrimination_metrics")
    clustering_metrics = read_table(results_root, "clustering/clustering_metrics")
    clustering_stability = read_table(results_root, "clustering/clustering_stability")

    discrimination["ModelLabel"] = discrimination["Model"].map(MODELS)
    discrimination["ScenarioLabel"] = discrimination["Scenario"].map(SCENARIOS)
    clustering_metrics["ModelLabel"] = clustering_metrics["Weight_Column"].map(MODELS)
    clustering_metrics["ScenarioLabel"] = clustering_metrics["Scenario"].map(SCENARIOS)
    clustering_stability["ModelLabel"] = clustering_stability["Weight_Column"].map(MODELS)
    clustering_stability["ScenarioLabel"] = clustering_stability["Scenario"].map(SCENARIOS)

    pr_heat = (
        discrimination.pivot(index="ScenarioLabel", columns="ModelLabel", values="PR_AUC")
        .reindex(index=SCENARIO_ORDER, columns=MODEL_ORDER)
    )
    best_f1 = clustering_metrics.groupby(["ScenarioLabel", "ModelLabel"], as_index=False)["BCubed_F1_Score"].max()
    best_f1_heat = (
        best_f1.pivot(index="ScenarioLabel", columns="ModelLabel", values="BCubed_F1_Score")
        .reindex(index=SCENARIO_ORDER, columns=MODEL_ORDER)
    )
    best_f1_heat.dropna(axis=1, how="all", inplace=True)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    heat1 = sns.heatmap(
        pr_heat,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5},
        cbar_kws={"label": "Average Precision (AP)"},
        ax=ax1,
    )
    ax1.set(xlabel="", ylabel="")
    style_colorbar(heat1)
    heat2 = sns.heatmap(
        best_f1_heat,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.5},
        cbar_kws={"label": "Best F1 score"},
        ax=ax2,
    )
    ax2.set(xlabel="", ylabel="")
    ax2.set_yticklabels([])
    style_colorbar(heat2)
    add_panel_labels([ax1, ax2], ["A", "B"])
    save_figure(fig, figures_dir / "discrimination_recovery", formats)
    plt.close(fig)

    plot_cluster_metrics(clustering_metrics, figures_dir / "sm_cluster_metrics_f1", formats=formats, metric="BCubed_F1_Score")
    plot_cluster_metrics(
        clustering_metrics,
        figures_dir / "sm_cluster_metrics_precision",
        formats=formats,
        metric="BCubed_Precision",
    )
    plot_cluster_metrics(
        clustering_metrics,
        figures_dir / "sm_cluster_metrics_recall",
        formats=formats,
        metric="BCubed_Recall",
    )
    plot_cluster_metrics(
        clustering_stability,
        figures_dir / "sm_clustering_stability_f1",
        formats=formats,
        metric="BCubed_F1_Score",
        y_axis="Res1",
    )
