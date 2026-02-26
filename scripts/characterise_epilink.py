#!/usr/bin/env python3
"""
scripts/characterise_epilink.py

Characterise epilink 

Outputs
---------------------
tables/supplementary/
  - mechanism_samples.parquet
  - mechanism_probability_surface.parquet
  - mechanism_prob_vs_snp.parquet
  - mechanism_prob_vs_days.parquet
  - mechanism_sanity_table.parquet
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from epilink import (
    TOIT,
    InfectiousnessParams,
    estimate_linkage_probabilities
)

from utils import *

@dataclass
class Cfg:
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
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    param_cfg = load_yaml(Path(args.defaults))

    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))

    ensure_dirs(tabs_dir)
    cfg = Cfg(
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
        k_inc=cfg.k_inc,
        scale_inc=cfg.scale_inc,
        k_E=cfg.k_E,
        mu=cfg.mu,
        k_I=cfg.k_I,
        alpha=cfg.alpha,
    )

    toit = TOIT(
        params=params,
        rng_seed=cfg.rng_seed,
        subs_rate=cfg.subs_rate,
        relax_rate=cfg.relax_rate,
        subs_rate_sigma=cfg.subs_rate_sigma,
        gen_len=cfg.gen_length,
    )

    # --- A) Timing priors: TOIT and generation time
    toit_samples = toit.rvs(cfg.n_sim)
    gen_time_samples = toit.generation_time(cfg.n_sim)

    samples_df = pd.DataFrame({
        "sample_type": (["toit"] * len(toit_samples)) + (["generation_time"] * len(gen_time_samples)),
        "value": np.concatenate([toit_samples, gen_time_samples]).astype(float),
    })
    samples_df.to_parquet(tabs_dir / "mechanism_samples.parquet", index=False)

    # --- B) Plausibility surfaces: genetic-only, temporal-only, joint
    snps = np.arange(0, cfg.max_snp + 1, cfg.snp_step)
    days = np.arange(0, cfg.max_days + 1, cfg.day_step)
    Dg, Dt = np.meshgrid(snps.astype(float), days.astype(float))

    # Genetic plausibility at Dt=0: treat temporal_distance as fixed
    # Temporal synchrony at Dg=0: treat genetic_distance as fixed
    # Joint: both varying
    P_joint = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=Dg.ravel(),
        temporal_distance=Dt.ravel(),
        intermediate_generations=cfg.inter_gen,
        no_intermediates=cfg.n_inter,
        num_simulations=cfg.n_sim,

    ).reshape(Dg.shape)

    # Slices to help interpretation
    P_genetic = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=snps.astype(float),
        temporal_distance=np.zeros_like(snps, dtype=float),
        intermediate_generations=cfg.inter_gen,
        no_intermediates=cfg.n_inter,
        num_simulations=cfg.n_sim,
    )
    P_temporal = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=np.zeros_like(days, dtype=float),
        temporal_distance=days.astype(float),
        intermediate_generations=cfg.inter_gen,
        no_intermediates=cfg.n_inter,
        num_simulations=cfg.n_sim,
    )

    surface_df = pd.DataFrame({
        "snp": Dg.ravel().astype(int),
        "days": Dt.ravel().astype(int),
        "probability": P_joint.ravel().astype(float),
    })
    surface_df.to_parquet(tabs_dir / "mechanism_probability_surface.parquet", index=False)

    prob_vs_snp_df = pd.DataFrame({
        "snp": snps.astype(int),
        "probability": P_genetic.astype(float),
    })
    prob_vs_snp_df.to_parquet(tabs_dir / "mechanism_prob_vs_snp.parquet", index=False)

    prob_vs_days_df = pd.DataFrame({
        "days": days.astype(int),
        "probability": P_temporal.astype(float),
    })
    prob_vs_days_df.to_parquet(tabs_dir / "mechanism_prob_vs_days.parquet", index=False)

    # --- C) Sanity table (handy for Methods / Supplement)
    sanity_rows = []
    for s in [0, 1, 2, 5, 10]:
        for t in [0, 3, 7, 14, 21]:
            p = float(estimate_linkage_probabilities(
                toit=toit,
                genetic_distance=np.array([float(s)]),
                temporal_distance=np.array([float(t)]),
                intermediate_generations=cfg.inter_gen,
                no_intermediates=cfg.n_inter,
                num_simulations=cfg.n_sim,
            )[0])
            sanity_rows.append({"SNPs": s, "DeltaDays": t, "P_mech": p})
    pd.DataFrame(sanity_rows).to_parquet(tabs_dir / "mechanism_sanity_table.parquet", index=False)

    print(f"Saved mechanism tables to: {tabs_dir}")


if __name__ == "__main__":
    main()
