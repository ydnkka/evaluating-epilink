#!/usr/bin/env python3
"""
scripts/plot_characterise_epilink.py

Generate supplementary figures from characteristic tables.

Outputs
---------------------
figures/supplementary/
  - supplementary_characteristic_toit.(png|pdf)
  - supplementary_characteristic_tost.(png|pdf)
  - supplementary_characteristic_stages.(png|pdf)
  - supplementary_characteristic_clock.(png|pdf)
  - supplementary_characteristic_linkage_components.(png|pdf)
  - supplementary_characteristic_linkage_surface.(png|pdf)
  - supplementary_characteristic_scenario_weights.(png|pdf)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import deep_get, ensure_dirs, load_yaml, save_figure, set_seaborn_paper_context


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_parquet(path)


def parse_formats(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    tabs_dir = Path(
        deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary")
    )
    tabs_dir = tabs_dir / "characterise_epilink"

    figs_dir = Path(
        deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary")
    )
    figs_dir = figs_dir / "characterise_epilink"
    ensure_dirs(figs_dir)

    formats = parse_formats(args.formats)
    set_seaborn_paper_context(font_scale=1.4)

    # TOIT PDF/CDF
    toit_df = read_table(tabs_dir / "characteristic_toit_grid.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(toit_df["days"], toit_df["pdf"], color="#1f77b4", lw=2)
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Density")
    axes[0].set_title("TOIT PDF")
    axes[0].set_xlim(0, 20)
    axes[1].plot(toit_df["days"], toit_df["cdf"], color="#1f77b4", lw=2)
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("TOIT CDF")
    axes[1].set_xlim(0, 20)
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_toit", formats)
    plt.close(fig)

    # TOST PDF/CDF
    tost_df = read_table(tabs_dir / "characteristic_tost_grid.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(tost_df["days"], tost_df["pdf"], color="#ff7f0e", lw=2)
    axes[0].axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    axes[0].set_xlabel("Days from symptom onset")
    axes[0].set_ylabel("Density")
    axes[0].set_title("TOST PDF")
    axes[0].set_xlim(-10, 10)
    axes[1].plot(tost_df["days"], tost_df["cdf"], color="#ff7f0e", lw=2)
    axes[1].axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    axes[1].set_xlabel("Days from symptom onset")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("TOST CDF")
    axes[1].set_xlim(-10, 10)
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_tost", formats)
    plt.close(fig)

    # Stage duration distributions
    stage_df = read_table(tabs_dir / "characteristic_stage_samples.parquet")
    stages = ["latent", "presymptomatic", "symptomatic", "incubation"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for ax, stage in zip(axes.ravel(), stages):
        data = stage_df.loc[stage_df["stage"] == stage, "value"]
        sns.histplot(data, bins=50, stat="density", color="#2ca02c", ax=ax)
        ax.set_title(stage.replace("_", " ").title())
        ax.set_xlabel("Days")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_stages", formats)
    plt.close(fig)

    # Molecular clock diagnostics
    clock_rates_df = read_table(tabs_dir / "characteristic_clock_rate_samples.parquet")
    expected_df = read_table(tabs_dir / "characteristic_expected_mutations.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(
        clock_rates_df["rate_per_site_year"],
        bins=60,
        stat="density",
        color="#9467bd",
        ax=axes[0],
    )
    axes[0].set_xlabel("Substitution rate (per site per year)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Clock rate distribution")
    days = expected_df["days"].to_numpy(dtype=float)
    rates = clock_rates_df["rate_per_day"].to_numpy(dtype=float)
    rng = np.random.default_rng(123)
    n_per_day = min(300, rates.size)
    scatter_rows = []
    for day in days:
        sampled_rates = rng.choice(rates, size=n_per_day, replace=True)
        scatter_rows.append(
            pd.DataFrame(
                {
                    "days": np.full(n_per_day, day, dtype=float),
                    "expected_mutations": sampled_rates * day,
                }
            )
        )
    scatter_df = pd.concat(scatter_rows, ignore_index=True)
    sns.regplot(
        data=scatter_df,
        x="days",
        y="expected_mutations",
        scatter_kws={"alpha": 0.2, "s": 12, "color": "#9467bd"},
        line_kws={"color": "#1f1f1f", "lw": 2},
        ax=axes[1],
    )
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Expected mutations")
    axes[1].set_title("Expected mutations over time")
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_clock", formats)
    plt.close(fig)

    # Temporal-only and genetic-only components
    temporal_df = read_table(tabs_dir / "characteristic_temporal_linkage.parquet")
    temporal_df = temporal_df.loc[temporal_df["days"] >= 0]
    genetic_df = read_table(tabs_dir / "characteristic_genetic_linkage.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        temporal_df["days"],
        temporal_df["probability"],
        color="#17becf",
        lw=2,
    )
    axes[0].axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    axes[0].set_xlabel("Temporal distance (days)")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Temporal evidence")
    axes[1].plot(
        genetic_df["snp"],
        genetic_df["normalized"],
        color="#d62728",
        lw=2,
        label="normalized",
    )
    axes[1].plot(
        genetic_df["snp"],
        genetic_df["relative"],
        color="#ff9896",
        lw=1.5,
        ls="--",
        label="relative",
    )
    axes[1].set_xlabel("Genetic distance (SNPs)")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Genetic evidence")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_linkage_components", formats)
    plt.close(fig)

    # Joint linkage surface
    surface_df = read_table(tabs_dir / "characteristic_probability_surface.parquet")
    surface_pivot = surface_df.pivot(index="days", columns="snp", values="probability")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        surface_pivot,
        cmap="viridis",
        cbar_kws={"label": "Probability"},
        ax=ax,
    )
    ax.set_xlabel("Genetic distance (SNPs)")
    ax.set_ylabel("Temporal distance (days)")
    ax.set_title("Joint linkage probability")
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_linkage_surface", formats)
    plt.close(fig)

    # Scenario weights by intermediate hosts
    scenarios_df = read_table(tabs_dir / "characteristic_genetic_scenarios.parquet")
    scenario_pivot = scenarios_df.pivot(index="m", columns="snp", values="normalized")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        scenario_pivot,
        cmap="magma",
        cbar_kws={"label": "Normalized probability"},
        ax=ax,
    )
    ax.set_xlabel("Genetic distance (SNPs)")
    ax.set_ylabel("Intermediate hosts (m)")
    ax.set_title("Genetic scenario weights")
    fig.tight_layout()
    save_figure(fig, figs_dir / "supplementary_characteristic_scenario_weights", formats)
    plt.close(fig)

    print(f"Saved supplementary figures to: {figs_dir}")


if __name__ == "__main__":
    main()
