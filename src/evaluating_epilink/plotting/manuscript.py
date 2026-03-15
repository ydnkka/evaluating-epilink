"""Render manuscript and supplementary figures from parquet outputs."""

from __future__ import annotations

import argparse
import json
import math
import string
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from ..config import config_value, ensure_directories, load_merged_config, resolve_path
from ..execution import finish_stage_run, start_stage_run

MODELS = {
    "LinearDistScore": "DDS",
    "PoissonDistScore": "SDS",
    "ProbLinearDist": "DDP",
    "ProbPoissonDist": "SDP",
    "LogitLinearDist10": "DDL1",
    "LogitPoissonDist10": "SDL1",
    "LogitLinearDist100": "DDL2",
    "LogitPoissonDist100": "SDL2",
}

SCENARIOS = {
    "baseline": "Baseline",
    "surveillance_moderate": "Surveillance (moderate)",
    "surveillance_severe": "Surveillance (severe)",
    "low_clock_signal": "Low clock signal",
    "low_incubation_shape": "Low incubation shape",
    "low_incubation_scale": "Low incubation scale",
    "high_clock_signal": "High clock signal",
    "high_incubation_shape": "High incubation shape",
    "high_incubation_scale": "High incubation scale",
    "relaxed_clock": "Relaxed clock",
    "adversarial": "Adversarial",
}

MODEL_ORDER = list(MODELS.values())
SCENARIO_ORDER = list(SCENARIOS.values())

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "gray": "#7F7F7F",
}

GENETIC_COLORS = {
    "Normalised": COLORS["green"],
}

TREE_STEP_COLORS = {
    "raw": COLORS["gray"],
    "cleaned": COLORS["sky"],
    "selected_component": COLORS["orange"],
    "final_tree": COLORS["vermillion"],
}

MODEL_COLORS = {
    "DDP": COLORS["blue"],
    "DDL2": COLORS["orange"],
    "SDP": COLORS["green"],
    "SDL2": COLORS["purple"],
}

STABILITY_COLORS = {
    "forward": COLORS["blue"],
    "backward": COLORS["orange"],
    "jaccard": COLORS["green"],
}


def set_plot_theme(font_scale: float = 1.5) -> None:
    """Apply a consistent matplotlib/seaborn theme."""

    sns.set_theme(
        context="paper",
        style="white",
        font_scale=font_scale,
        rc={
            "figure.dpi": 500,
            "savefig.dpi": 500,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "legend.frameon": False,
        },
    )


def save_figure(fig: Figure, output_stem: Path, formats: tuple[str, ...]) -> None:
    """Save a figure in one or more formats."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=500)


def read_table(results_root: Path, stem: str) -> pd.DataFrame:
    """Load a parquet result by stem relative to the results root."""

    return pd.read_parquet(results_root / f"{stem}.parquet")


def read_json(path: Path) -> dict:
    """Load a JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def add_panel_labels(
    axes: list[Axes],
    labels: list[str],
    *,
    x: float = -0.16,
    y: float = 1.08,
    size: float = 12,
) -> None:
    """Add panel labels such as A), B), C) to axes."""

    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            f"{label})",
            transform=ax.transAxes,
            fontsize=size,
            fontweight="bold",
            va="top",
        )


def style_colorbar(heatmap_artist, label_size: int = 10) -> None:
    """Apply light styling to a seaborn heatmap colorbar."""

    colorbar = heatmap_artist.collections[0].colorbar
    colorbar.outline.set_edgecolor("#333333")
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.yaxis.label.set_size(label_size)


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the complementary cumulative distribution function."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.array([]), np.array([])
    array = np.sort(array.astype(int))
    unique, first_index = np.unique(array, return_index=True)
    return unique, (array.size - first_index) / array.size


def tree_depths(graph: nx.DiGraph) -> np.ndarray:
    """Return shortest-path depths from any root in a directed tree."""

    roots = [node for node, indegree in graph.in_degree(graph.nodes) if indegree == 0]
    if not roots:
        raise ValueError("Tree has no roots.")

    depth_map: dict[object, int] = {}
    for root in roots:
        for node, distance in nx.single_source_shortest_path_length(graph, root).items():
            if node not in depth_map or distance < depth_map[node]:
                depth_map[node] = distance
    return np.asarray(list(depth_map.values()), dtype=int)


def metric_label(metric_name: str) -> str:
    """Map internal metric names to figure labels."""

    labels = {
        "BCubed_F1_Score": "BCubed F1 score",
        "BCubed_Precision": "BCubed precision",
        "BCubed_Recall": "BCubed recall",
    }
    return labels.get(metric_name, metric_name.replace("_", " "))


def render_dataframe_table(ax: Axes, frame: pd.DataFrame) -> None:
    """Render a small dataframe as a matplotlib table."""

    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        rowLabels=[str(label) for label in frame.index],
        colLabels=[str(label) for label in frame.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.6)


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
                linewidth=1.7,
                markersize=4.5,
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
        ax.text(-0.18, 1.12, f"{panel_letter})", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    for index in range(panel_count, nrows * ncols):
        axes[index].axis("off")

    fig.supxlabel("Resolution", fontsize=13)
    fig.supylabel(metric_label(metric), fontsize=13)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Model",
            fontsize=12,
            title_fontsize=13,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=4,
            frameon=False,
            handlelength=3,
        )

    save_figure(fig, output_stem, formats)
    plt.close(fig)


def plot_stability_over_time(stability_frame: pd.DataFrame, ax: Axes) -> Axes:
    """Plot forward, backward, and Jaccard stability over time."""

    line_styles = {"forward": "--", "backward": ":", "jaccard": "-"}
    markers = {"forward": "o", "backward": "s", "jaccard": "^"}
    for metric in ["forward", "backward", "jaccard"]:
        ax.plot(
            stability_frame.index,
            stability_frame[metric],
            linestyle=line_styles[metric],
            linewidth=1.8,
            color=STABILITY_COLORS[metric],
            markersize=4,
            marker=markers[metric],
            label=metric.capitalize(),
        )
    ax.set(xlabel="Epidemic week", ylabel="Mean stability")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", title=None, handlelength=3)
    return ax


def plot_cumulative_cases(case_counts: pd.DataFrame, ax: Axes) -> Axes:
    """Plot cumulative sampled cases over time."""

    cases = case_counts.copy()
    cases["cumulative"] = cases["n_cases"].cumsum()
    bars = ax.bar(
        cases["available_time"],
        cases["cumulative"],
        color="#DDEAF7",
        edgecolor=COLORS["blue"],
        linewidth=0.9,
        width=0.9,
    )
    for bar in bars:
        bar.set_hatch("..")
    ax.set(xlabel="Epidemic week", ylabel="Cumulative cases")
    ax.set_yscale("log")
    return ax


def render_model_characterisation(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render the epilink feature summary figure."""

    time_of_infection_to_transmission = read_table(results_root, "characterise_epilink/characteristic_toit_grid")
    time_of_infection_to_transmission = time_of_infection_to_transmission[
        time_of_infection_to_transmission["days"] <= 15
    ]
    time_of_symptom_onset_to_transmission = read_table(
        results_root,
        "characterise_epilink/characteristic_tost_grid",
    )
    time_of_symptom_onset_to_transmission = time_of_symptom_onset_to_transmission[
        time_of_symptom_onset_to_transmission["days"].between(-10, 10)
    ]
    temporal_linkage = read_table(results_root, "characterise_epilink/characteristic_temporal_linkage")
    genetic_linkage = read_table(results_root, "characterise_epilink/characteristic_genetic_linkage")
    probability_surface = read_table(results_root, "characterise_epilink/characteristic_probability_surface")
    genetic_scenarios = read_table(results_root, "characterise_epilink/characteristic_genetic_scenarios")

    surface_pivot = probability_surface.pivot(index="days", columns="snp", values="probability")
    scenario_pivot = genetic_scenarios.pivot(index="m", columns="snp", values="normalized")

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[1, 0])
    ax5 = fig.add_subplot(grid[1, 1])
    ax6 = fig.add_subplot(grid[1, 2])

    sns.lineplot(
        data=time_of_infection_to_transmission,
        x="days",
        y="pdf",
        ax=ax1,
        color=COLORS["blue"],
        linewidth=2.2,
    )
    ax1.axvline(0, color="#333333", linestyle=":", linewidth=1.2)
    ax1.set(xlabel="TOIT (days)", ylabel="Density")

    sns.lineplot(
        data=time_of_symptom_onset_to_transmission,
        x="days",
        y="pdf",
        ax=ax2,
        color=COLORS["vermillion"],
        linewidth=2.2,
    )
    ax2.axvline(0, color="#333333", linestyle=":", linewidth=1.2)
    ax2.set(xlabel="TOST (days)", ylabel="Density")

    sns.lineplot(
        data=temporal_linkage,
        x="days",
        y="probability",
        ax=ax3,
        color=COLORS["blue"],
        linewidth=2.2,
    )
    ax3.set(xlabel="Temporal distance (days)", ylabel="Probability")

    sns.lineplot(
        data=genetic_linkage,
        x="snp",
        y="normalized",
        ax=ax4,
        color=GENETIC_COLORS["Normalised"],
        linewidth=2,
    )
    ax4.set(xlabel="Genetic distance (SNPs)", ylabel="Probability")

    heatmap_joint = sns.heatmap(
        surface_pivot,
        cmap="mako",
        ax=ax5,
        cbar_kws={"label": "Probability"},
        linewidths=0,
    )
    ax5.set(xlabel="Genetic distance (SNPs)", ylabel="Temporal distance (days)")
    style_colorbar(heatmap_joint)

    heatmap_scenarios = sns.heatmap(
        scenario_pivot,
        cmap="cividis",
        ax=ax6,
        cbar_kws={"label": "Probability"},
        linewidths=0,
    )
    ax6.set(xlabel="Genetic distance (SNPs)", ylabel="Intermediate hosts")
    style_colorbar(heatmap_scenarios)

    add_panel_labels([ax1, ax2, ax3, ax4, ax5, ax6], ["A", "B", "C", "D", "E", "F"], x=-0.18, y=1.13, size=13)
    save_figure(fig, figures_dir / "epilink_feature_summary", formats)
    plt.close(fig)


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
            linewidth=1.8,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(1, len(x_vals) // 12),
            label=label,
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
            markersize=4,
            markevery=max(1, len(x_tree) // 12),
            label="final_tree",
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set(xlabel="Out-degree (offspring)", ylabel="CCDF")
    ax2.legend(loc="best", title=None, handlelength=3)

    counts = pd.Series(depth_values).value_counts().sort_index()
    bars = ax3.bar(counts.index, counts.values, color="#DDEAF7", edgecolor=COLORS["blue"], linewidth=0.9, width=0.9)
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
        linewidth=0.9,
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

    pr_heat = discrimination.pivot(index="ScenarioLabel", columns="ModelLabel", values="PR_AUC").reindex(index=SCENARIO_ORDER, columns=MODEL_ORDER)
    best_f1 = clustering_metrics.groupby(["ScenarioLabel", "ModelLabel"], as_index=False)["BCubed_F1_Score"].max()
    best_f1_heat = best_f1.pivot(index="ScenarioLabel", columns="ModelLabel", values="BCubed_F1_Score").reindex(index=SCENARIO_ORDER, columns=MODEL_ORDER)
    best_f1_heat.dropna(axis=1, how="all", inplace=True)

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    heat1 = sns.heatmap(pr_heat, vmin=0, vmax=1, cmap="YlGnBu", linewidths=0.1, annot=True, fmt=".2f", cbar_kws={"label": "Average Precision (AP)"}, ax=ax1)
    ax1.set(xlabel="", ylabel="")
    style_colorbar(heat1)
    heat2 = sns.heatmap(best_f1_heat, vmin=0, vmax=1, cmap="YlGnBu", linewidths=0.1, annot=True, fmt=".2f", cbar_kws={"label": "Best F1 score"}, ax=ax2)
    ax2.set(xlabel="", ylabel="")
    ax2.set_yticklabels([])
    style_colorbar(heat2)
    add_panel_labels([ax1, ax2], ["A", "B"])
    fig.tight_layout()
    save_figure(fig, figures_dir / "discrimination_recovery", formats)
    plt.close(fig)

    plot_cluster_metrics(clustering_metrics, figures_dir / "sm_cluster_metrics_f1", formats=formats, metric="BCubed_F1_Score")
    plot_cluster_metrics(clustering_metrics, figures_dir / "sm_cluster_metrics_precision", formats=formats, metric="BCubed_Precision")
    plot_cluster_metrics(clustering_metrics, figures_dir / "sm_cluster_metrics_recall", formats=formats, metric="BCubed_Recall")
    plot_cluster_metrics(clustering_stability, figures_dir / "sm_clustering_stability_f1", formats=formats, metric="BCubed_F1_Score", y_axis="Res1")


def render_temporal_stability(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render temporal-stability figures."""

    case_counts_over_time = read_table(results_root, "temporal_stability/case_counts_over_time")
    logit_linear_stability = read_table(results_root, "temporal_stability/temporal_stability_logit_linear")
    logit_poisson_stability = read_table(results_root, "temporal_stability/temporal_stability_logit_poisson")
    probability_linear_stability = read_table(results_root, "temporal_stability/temporal_stability_prob_linear")
    probability_poisson_stability = read_table(results_root, "temporal_stability/temporal_stability_prob_poisson")

    fig = plt.figure(figsize=(10, 5))
    grid = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    plot_cumulative_cases(case_counts_over_time, ax1)
    plot_stability_over_time(probability_poisson_stability, ax2)
    add_panel_labels([ax1, ax2], ["A", "B"], x=-0.14, y=1.08, size=11)
    fig.tight_layout()
    save_figure(fig, figures_dir / "temporal_stability", formats)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True)
    ax1 = plot_stability_over_time(probability_linear_stability, axes[0, 0])
    ax2 = plot_stability_over_time(probability_poisson_stability, axes[0, 1])
    ax3 = plot_stability_over_time(logit_linear_stability, axes[1, 0])
    ax4 = plot_stability_over_time(logit_poisson_stability, axes[1, 1])
    for axis in [ax1, ax2]:
        axis.set_xlabel("")
    add_panel_labels([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"], x=-0.14, y=1.08, size=11)
    fig.tight_layout()
    save_figure(fig, figures_dir / "sm_temporal_stability", formats)
    plt.close(fig)


def render_boston(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render the Boston cluster figure."""

    boston_clusters_comp = read_table(results_root, "boston/boston_cluster_composition")
    boston_clusters_sum = read_table(results_root, "boston/boston_cluster_summary")

    exposure_columns = ["count::BHCHP", "count::Other", "count::City", "count::Conference", "count::SNF"]
    exposure_counts = boston_clusters_comp[exposure_columns].copy().rename(
        columns={
            "count::BHCHP": "BHCHP",
            "count::Other": "Other",
            "count::City": "City",
            "count::Conference": "Conference",
            "count::SNF": "SNF",
        }
    )
    exposure_counts.index = boston_clusters_comp["cluster_id"]

    summary_table = boston_clusters_sum.drop(columns=["Size"]).copy()
    summary_table = summary_table.set_index("Cluster ID").round(2).T
    summary_table.index.name = "Cluster ID"

    fig = plt.figure(figsize=(10, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    heatmap = sns.heatmap(
        exposure_counts.T,
        vmin=0,
        vmax=80,
        cmap="YlOrBr",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 11},
        cbar_kws={"label": "Cases", "shrink": 0.6},
        ax=ax1,
    )
    ax1.tick_params(axis="y", labelsize="small", rotation=0)
    ax1.tick_params(axis="x", labelsize="small")
    ax1.set_xlabel("Cluster ID")
    style_colorbar(heatmap, 12)

    render_dataframe_table(ax2, summary_table)
    add_panel_labels([ax1, ax2], ["A", "B"])
    fig.tight_layout()
    save_figure(fig, figures_dir / "boston_cluster", formats)
    plt.close(fig)


def main() -> None:
    """Render all manuscript figures from workflow outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/figures.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("figures", cli_args=vars(args))

    config = load_merged_config(args.base_config, args.study_config)
    figure_formats = tuple(str(value) for value in config_value(config, ["render", "formats"], ["pdf"]))
    font_scale = float(config_value(config, ["render", "font_scale"], 1.5))

    figures_dir = resolve_path(config_value(config, ["paths", "results", "figures"]))
    processed_synthetic_dir = resolve_path(config_value(config, ["paths", "data", "processed", "synthetic"]))
    results_root = figures_dir.parent
    ensure_directories(figures_dir)
    set_plot_theme(font_scale=font_scale)

    render_model_characterisation(results_root, figures_dir, figure_formats)
    render_tree_figure(results_root, processed_synthetic_dir, figures_dir, figure_formats)
    render_discrimination_and_clustering(results_root, figures_dir, figure_formats)
    render_temporal_stability(results_root, figures_dir, figure_formats)
    render_boston(results_root, figures_dir, figure_formats)

    finish_stage_run(
        stage_run,
        resolve_path(config_value(config, ["paths", "results", "manifests"])) / "figures.json",
        config=config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "results_root": str(results_root),
            "processed_synthetic_dir": str(processed_synthetic_dir),
        },
        outputs={"figures_dir": str(figures_dir)},
        summary={
            "formats": list(figure_formats),
            "num_figure_groups": 5,
        },
    )


if __name__ == "__main__":
    main()
