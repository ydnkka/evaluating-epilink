#!/usr/bin/env python3
"""
scripts/characterise_epilink.py

Characterise the mechanistic, threshold-free linkage model.

Outputs
---------------------
figures/supplementary/mechanism/
  - toit_density.(png|pdf)
  - generation_time_density.(png|pdf)
  - prob_vs_snp.(png|pdf)
  - prob_vs_days.(png|pdf)
  - joint_probability_heatmap.(png|pdf)
tables/supplementary/
  - mechanism_sanity_table.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

# epilink provides the mechanistic timing model + probability estimator
from epilink import (
    TOIT,
    InfectiousnessParams,
    estimate_linkage_probabilities
)

from utils import *


set_seaborn_paper_context()

@dataclass
class CharCfg:
    rng_seed: int
    k_inc: float
    scale_inc: float
    k_E: float
    mu: float
    k_I: float
    alpha: float
    subs_rate: float
    relax_rate: bool
    subs_rate_sigma: float
    gen_length: int
    n_sim: int
    inter_gen: tuple[int, int]
    n_inter: int
    max_snp: int
    snp_step: int
    max_days: int
    day_step: int


# -----------------------------
# Plots
# -----------------------------

def plot_density(samples: np.ndarray, xlabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=100, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    return fig


def heatmap(z: np.ndarray, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=(x.min(), x.max(), y.min(), y.max()),
    )
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    param_cfg = load_yaml(Path(args.defaults))

    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "figures/supplementary"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "tables/supplementary"))

    figs_dir = figs_dir / "mechanism"
    ensure_dirs(figs_dir, tabs_dir)

    formats = list(deep_get(param_cfg, ["save_formats"], ["png", "pdf"]))
    cc = CharCfg(
        rng_seed=int(deep_get(param_cfg, ["toit", "rng_seed"], 42)),
        k_inc=float(deep_get(param_cfg, ["toit", "infectiousness_params", "k_inc"], 5.807)),
        scale_inc=float(deep_get(param_cfg, ["toit", "infectiousness_params", "scale_inc"], 0.948)),
        k_E=float(deep_get(param_cfg, ["toit", "infectiousness_params", "k_E"], 3.38)),
        mu=float(deep_get(param_cfg, ["toit", "infectiousness_params", "mu"], 0.37)),
        k_I=float(deep_get(param_cfg, ["toit", "infectiousness_params", "k_I"], 1.0)),
        alpha=float(deep_get(param_cfg, ["toit", "infectiousness_params", "alpha"], 2.29)),

        subs_rate=float(deep_get(param_cfg, ["toit", "evolution", "subs_rate"], 0.001)),
        relax_rate=deep_get(param_cfg, ["toit", "evolution", "relax_rate"], True),
        subs_rate_sigma=float(deep_get(param_cfg, ["toit", "evolution", "subs_rate_sigma"], 0.33)),
        gen_length=int(deep_get(param_cfg, ["toit", "evolution", "gen_length"], 29903)),

        n_sim=int(deep_get(param_cfg, ["inference", "num_simulations"], 10_000)),
        inter_gen=(deep_get(param_cfg, ["inference", "inter_generations"], (0,1))),
        n_inter=int(deep_get(param_cfg, ["inference", "num_intermediates"], 10)),

        max_snp=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "max_snp"], 10)),
        snp_step=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "step"], 1)),
        max_days=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "max_days"], 21)),
        day_step=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "step"], 1)),
    )

    params = InfectiousnessParams(
        k_inc=cc.k_inc,
        scale_inc=cc.scale_inc,
        k_E=cc.k_E,
        mu=cc.mu,
        k_I=cc.k_I,
        alpha=cc.alpha,
    )

    toit = TOIT(
        params=params,
        rng_seed=cc.rng_seed,
        subs_rate=cc.subs_rate,
        relax_rate=cc.relax_rate,
        subs_rate_sigma=cc.subs_rate_sigma,
        gen_len=cc.gen_length,
    )

    # --- A) Timing priors: TOIT and generation time
    toit_samples = toit.rvs(cc.n_sim)
    gen_time_samples = toit.generation_time(cc.n_sim)

    fig = plot_density(toit_samples, xlabel="Days", title="")
    save_figure(fig, figs_dir / "sm1_toit_density", formats)
    plt.close(fig)

    # --- B) Plausibility surfaces: genetic-only, temporal-only, joint
    snps = np.arange(0, cc.max_snp + 1, cc.snp_step)
    days = np.arange(0, cc.max_days + 1, cc.day_step)
    Dg, Dt = np.meshgrid(snps.astype(float), days.astype(float))

    # Genetic plausibility at Dt=0: treat temporal_distance as fixed
    # Temporal synchrony at Dg=0: treat genetic_distance as fixed
    # Joint: both varying
    P_joint = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=Dg.ravel(),
        temporal_distance=Dt.ravel(),
        intermediate_generations=cc.inter_gen,
        no_intermediates=cc.n_inter,
        num_simulations=cc.n_sim,

    ).reshape(Dg.shape)

    # Slices to help interpretation
    P_genetic = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=snps.astype(float),
        temporal_distance=np.zeros_like(snps, dtype=float),
        intermediate_generations=cc.inter_gen,
        no_intermediates=cc.n_inter,
        num_simulations=cc.n_sim,
    )
    P_temporal = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=np.zeros_like(days, dtype=float),
        temporal_distance=days.astype(float),
        intermediate_generations=cc.inter_gen,
        no_intermediates=cc.n_inter,
        num_simulations=cc.n_sim,
    )

    # Heatmaps
    fig = heatmap(P_joint, x=snps, y=days, xlabel="Genetic distance (SNPs)", ylabel="Temporal distance (days)",
                  title="")
    save_figure(fig, figs_dir / "sm1_joint_probability_heatmap", formats)
    plt.close(fig)

    # Line plots for sanity
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snps, P_genetic)
    ax.set_xlabel("Genetic distance (SNPs)")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figs_dir / "sm1_prob_vs_snp", formats)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(days, P_temporal)
    ax.set_xlabel("Temporal distance (days)")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    save_figure(fig, figs_dir / "sm1_prob_vs_days", formats)
    plt.close(fig)

    # --- C) Sanity table (handy for Methods / Supplement)
    sanity_rows = []
    for s in [0, 1, 2, 5, 10]:
        for t in [0, 3, 7, 14, 21]:
            p = float(estimate_linkage_probabilities(
                toit=toit,
                genetic_distance=np.array([float(s)]),
                temporal_distance=np.array([float(t)]),
                intermediate_generations=cc.inter_gen,
                no_intermediates=cc.n_inter,
                num_simulations=cc.n_sim,
            )[0])
            sanity_rows.append({"SNPs": s, "DeltaDays": t, "P_mech": p})
    pd.DataFrame(sanity_rows).to_csv(tabs_dir / "mechanism_sanity_table.csv", index=False)

    print(f"Saved characterisation figures to: {figs_dir}")
    print(f"Saved sanity table to: {tabs_dir / 'mechanism_sanity_table.csv'}")


if __name__ == "__main__":
    main()

