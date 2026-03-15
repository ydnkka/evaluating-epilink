"""Temporal stability figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from .common import COLORS, STABILITY_COLORS, add_panel_labels, read_table, save_figure


def plot_stability_over_time(stability_frame: pd.DataFrame, ax: Axes) -> Axes:
    """Plot forward, backward, and Jaccard stability over time."""

    line_styles = {"forward": "--", "backward": ":", "jaccard": "-"}
    markers = {"forward": "o", "backward": "s", "jaccard": "^"}
    for metric in ["forward", "backward", "jaccard"]:
        ax.plot(
            stability_frame.index,
            stability_frame[metric],
            linestyle=line_styles[metric],
            linewidth=2.0,
            color=STABILITY_COLORS[metric],
            markersize=4.2,
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
        linewidth=0.8,
        width=0.9,
    )
    for bar in bars:
        bar.set_hatch("..")
    ax.set(xlabel="Epidemic week", ylabel="Cumulative cases")
    ax.set_yscale("log")
    return ax


def render_temporal_stability(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render temporal-stability figures."""

    case_counts_over_time = read_table(results_root, "temporal_stability/case_counts_over_time")
    logit_linear_stability = read_table(results_root, "temporal_stability/temporal_stability_logit_linear")
    logit_poisson_stability = read_table(results_root, "temporal_stability/temporal_stability_logit_poisson")
    probability_linear_stability = read_table(results_root, "temporal_stability/temporal_stability_prob_linear")
    probability_poisson_stability = read_table(results_root, "temporal_stability/temporal_stability_prob_poisson")

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    plot_cumulative_cases(case_counts_over_time, ax1)
    plot_stability_over_time(probability_poisson_stability, ax2)
    add_panel_labels([ax1, ax2], ["A", "B"], x=-0.14, y=1.08)
    save_figure(fig, figures_dir / "temporal_stability", formats)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, constrained_layout=True)
    ax1 = plot_stability_over_time(probability_linear_stability, axes[0, 0])
    ax2 = plot_stability_over_time(probability_poisson_stability, axes[0, 1])
    ax3 = plot_stability_over_time(logit_linear_stability, axes[1, 0])
    ax4 = plot_stability_over_time(logit_poisson_stability, axes[1, 1])
    for axis in [ax1, ax2]:
        axis.set_xlabel("")
    add_panel_labels([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"], x=-0.14, y=1.08)
    save_figure(fig, figures_dir / "sm_temporal_stability", formats)
    plt.close(fig)
