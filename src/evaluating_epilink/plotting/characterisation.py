"""Manuscript figure for epilink score characterisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import add_panel_labels, read_table, save_figure

DEFAULT_SLICE_SNPS = (0, 1, 2, 5)
DEFAULT_SLICE_DAYS = (0, 2, 5, 10)


def available_slice_snps(surface_frame: pd.DataFrame) -> list[int]:
    """Return the manuscript SNP slices that are present in the results table."""

    available = set(surface_frame["snp"].astype(int))
    return [snp for snp in DEFAULT_SLICE_SNPS if snp in available]


def surface_matrix(surface_frame: pd.DataFrame, value_column: str = "probability") -> pd.DataFrame:
    """Pivot a long surface table into a days-by-SNP matrix."""

    return (
        surface_frame.pivot(index="days", columns="snp", values=value_column)
        .sort_index()
        .sort_index(axis=1)
    )


def add_surface_panel(ax, fig, surface_frame: pd.DataFrame) -> None:
    """Draw the default score surface with a few contour levels."""

    pivot = surface_matrix(surface_frame)
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    z_values = pivot.to_numpy(dtype=float)
    max_probability = max(float(np.nanmax(z_values)), 1e-6)
    filled = ax.contourf(
        x_values,
        y_values,
        z_values,
        levels=np.linspace(0.0, max_probability, 16),
        cmap="mako",
        antialiased=True,
    )
    contour_levels = [level for level in (0.1, 0.25, 0.5) if level < max_probability]
    if contour_levels:
        contours = ax.contour(
            x_values,
            y_values,
            z_values,
            levels=contour_levels,
            colors="white",
            linewidths=1.0,
        )
        ax.clabel(contours, fmt="%.2f", inline=True, fontsize=8, colors="white")
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label("P(link)")
    ax.set(xlabel="Genetic distance (SNPs)", ylabel="Temporal gap (days)")
    ax.set_title("Default score surface", loc="left", fontsize=11, pad=6)


def add_slice_panel(ax, surface_frame: pd.DataFrame) -> None:
    """Plot temporal slices through the default score surface."""

    slice_snps = available_slice_snps(surface_frame)
    if not slice_snps:
        ax.axis("off")
        ax.text(0.0, 1.0, "Slices", fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.84, "No manuscript slice SNP values were present in the surface table.", fontsize=10.5, va="top")
        return
    palette = sns.color_palette("crest", len(slice_snps))
    for color, snp in zip(palette, slice_snps):
        subset = surface_frame.loc[surface_frame["snp"] == snp].sort_values("days")
        ax.plot(
            subset["days"],
            subset["probability"],
            color=color,
            linewidth=2.1,
            label=f"{snp} SNP" if snp == 1 else f"{snp} SNPs",
        )
    ax.set(xlabel="Temporal gap (days)", ylabel="P(link)", ylim=(0, 1.0))
    ax.set_title("Temporal slices at fixed SNP distance", loc="left", fontsize=11, pad=6)
    ax.legend(title=None, loc="upper right")


def available_slice_days(surface_frame: pd.DataFrame) -> list[int]:
    """Return the manuscript day slices that are present in the results table."""

    available = set(surface_frame["days"].astype(int))
    return [day for day in DEFAULT_SLICE_DAYS if day in available]


def add_snp_slice_panel(ax, surface_frame: pd.DataFrame) -> None:
    """Plot SNP slices through the default score surface at fixed time gaps."""

    slice_days = available_slice_days(surface_frame)
    if not slice_days:
        ax.axis("off")
        ax.text(0.0, 1.0, "Slices", fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.84, "No manuscript time-gap values were present in the surface table.", fontsize=10.5, va="top")
        return
    palette = sns.color_palette("flare", len(slice_days))
    for color, day in zip(palette, slice_days):
        subset = surface_frame.loc[surface_frame["days"] == day].sort_values("snp")
        ax.plot(
            subset["snp"],
            subset["probability"],
            color=color,
            linewidth=2.1,
            label=f"{day} day" if day == 1 else f"{day} days",
        )
    ax.set(xlabel="Genetic distance (SNPs)", ylabel="P(link)", ylim=(0, 1.0))
    ax.set_title("Genetic slices at fixed temporal gaps", loc="left", fontsize=11, pad=6)
    ax.legend(title=None, loc="upper right")


def add_robustness_panel(ax, fig, default_surface: pd.DataFrame, comparison_surface: pd.DataFrame) -> pd.DataFrame:
    """Plot the change in score when intermediates are allowed."""

    delta_frame = comparison_surface.merge(
        default_surface[["snp", "days", "probability"]],
        on=["snp", "days"],
        suffixes=("_comparison", "_default"),
    )
    delta_frame["delta"] = delta_frame["probability_comparison"] - delta_frame["probability_default"]
    pivot = surface_matrix(delta_frame, value_column="delta")
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    z_values = pivot.to_numpy(dtype=float)
    max_delta = max(float(np.nanmax(np.abs(z_values))), 1e-6)
    filled = ax.contourf(
        x_values,
        y_values,
        z_values,
        levels=np.linspace(-max_delta, max_delta, 17),
        cmap="RdBu_r",
        antialiased=True,
    )
    ax.contour(x_values, y_values, z_values, levels=[0.0], colors="#222222", linewidths=1.0)
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label("Delta P(link)")
    ax.set(xlabel="Genetic distance (SNPs)", ylabel="Temporal gap (days)")
    ax.set_title(
        f"{comparison_surface['label'].iat[0]} vs default",
        loc="left",
        fontsize=11,
        pad=6,
    )
    return delta_frame


def render_model_characterisation(results_root: Path, figures_dir: Path, formats: tuple[str, ...]) -> None:
    """Render the compact epilink score characterisation figure."""

    probability_surface = read_table(results_root, "characterise_epilink/characteristic_probability_surface")
    if "scenario" not in probability_surface.columns:
        probability_surface["scenario"] = "default"
        probability_surface["label"] = "Default"

    default_surface = probability_surface.loc[probability_surface["scenario"] == "default"].copy()
    if default_surface.empty:
        raise ValueError("Characterisation table is missing the default scenario.")

    comparison_surface = probability_surface.loc[probability_surface["scenario"] != "default"].copy()
    if not comparison_surface.empty:
        comparison_name = comparison_surface["scenario"].iat[0]
        comparison_surface = comparison_surface.loc[comparison_surface["scenario"] == comparison_name].copy()
    else:
        comparison_surface = None

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    add_surface_panel(ax1, fig, default_surface)
    add_snp_slice_panel(ax2, default_surface)
    add_slice_panel(ax3, default_surface)
    if comparison_surface is not None:
        add_robustness_panel(ax4, fig, default_surface, comparison_surface)
    else:
        ax4.axis("off")
        ax4.text(0.0, 1.0, "Robustness", fontsize=12, fontweight="bold", va="top")
        ax4.text(0.0, 0.84, "No non-default characterisation surface was available.", fontsize=10.5, va="top")

    add_panel_labels([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"], x=-0.16, y=1.11, size=12.5)
    save_figure(fig, figures_dir / "epilink_feature_summary", formats)
    plt.close(fig)
