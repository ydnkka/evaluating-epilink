"""Shared helpers and styling for manuscript figures."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
            "font.family": "DejaVu Sans",
            "figure.dpi": 500,
            "savefig.dpi": 500,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10.5,
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
    colorbar.ax.tick_params(labelsize=max(label_size - 1, 8), width=0.7, length=3)


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
