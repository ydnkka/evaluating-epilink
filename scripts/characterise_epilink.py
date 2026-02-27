#!/usr/bin/env python3
"""
scripts/characterise_epilink.py

Characterise epilink 

Outputs
---------------------
tables/supplementary/
  - characteristic_samples.parquet
  - characteristic_probability_surface.parquet
  - characteristic_prob_vs_snp.parquet
  - characteristic_prob_vs_days.parquet
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from numpy.random import default_rng

import numpy as np
import pandas as pd

from epilink import (
    TOIT,
    MolecularClock,
    InfectiousnessParams,
    linkage_probability
)

from utils import *

@dataclass
class Cfg:
    rng_seed: int
    incubation_shape: float
    incubation_scale: float
    latent_shape: float
    symptomatic_rate: float
    symptomatic_shape: float
    rel_presymptomatic_infectiousness: float
    subs_rate: float
    relax_rate: bool
    subs_rate_sigma: float
    gen_length: int
    num_simulations: int
    intermediate_generations: tuple[int, int]
    intermediate_hosts: int
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
        rng_seed=int(deep_get(param_cfg, ["rng_seed"], 42)),
        incubation_shape=float(deep_get(param_cfg, ["infectiousness_params", "incubation_shape"], 5.807)),
        incubation_scale=float(deep_get(param_cfg, ["infectiousness_params", "incubation_scale"], 0.948)),
        latent_shape=float(deep_get(param_cfg, ["infectiousness_params", "latent_shape"], 3.38)),
        symptomatic_rate=float(deep_get(param_cfg, ["infectiousness_params", "symptomatic_rate"], 0.37)),
        symptomatic_shape=float(deep_get(param_cfg, ["infectiousness_params", "symptomatic_shape"], 1.0)),
        rel_presymptomatic_infectiousness=float(
            deep_get(param_cfg, ["infectiousness_params", "rel_presymptomatic_infectiousness"], 2.29)
        ),

        subs_rate=float(deep_get(param_cfg, ["clock", "subs_rate"], 0.001)),
        relax_rate=bool(deep_get(param_cfg, ["clock", "relax_rate"], True)),
        subs_rate_sigma=float(deep_get(param_cfg, ["clock", "subs_rate_sigma"], 0.33)),
        gen_length=int(deep_get(param_cfg, ["clock", "gen_length"], 5)),

        num_simulations=int(deep_get(param_cfg, ["inference", "num_simulations"], 10_000)),
        intermediate_generations=(deep_get(param_cfg, ["inference", "intermediate_generations"], (0,1))),
        intermediate_hosts=int(deep_get(param_cfg, ["inference", "intermediate_hosts"], 10)),

        max_snp=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "max_snp"], 10)),
        snp_step=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "step"], 1)),
        max_days=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "max_days"], 21)),
        day_step=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "step"], 1)),
    )

    rng = default_rng(cfg.rng_seed)

    params = InfectiousnessParams(
        incubation_shape=cfg.incubation_shape,
        incubation_scale=cfg.incubation_scale,
        latent_shape=cfg.latent_shape,
        symptomatic_rate=cfg.symptomatic_rate,
        symptomatic_shape=cfg.symptomatic_shape,
        rel_presymptomatic_infectiousness=cfg.rel_presymptomatic_infectiousness,
    )

    toit = TOIT(
        params=params,
        rng=rng,
    )

    clock = MolecularClock(
        subs_rate=cfg.subs_rate,
        relax_rate=cfg.relax_rate,
        subs_rate_sigma=cfg.subs_rate_sigma,
        gen_len=cfg.gen_length,
        rng=rng,
    )

    # --- A) Timing priors: TOIT and generation time
    toit_samples = toit.rvs(cfg.num_simulations)
    gen_time_samples = toit.generation_time(cfg.num_simulations)

    samples_df = pd.DataFrame({
        "sample_type": (["toit"] * len(toit_samples)) + (["generation_time"] * len(gen_time_samples)),
        "value": np.concatenate([toit_samples, gen_time_samples]).astype(float),
    })
    samples_df.to_parquet(tabs_dir / "characteristic_samples.parquet", index=False)

    # --- B) Plausibility surfaces: genetic-only, temporal-only, joint
    snps = np.arange(0, cfg.max_snp + 1, cfg.snp_step)
    days = np.arange(0, cfg.max_days + 1, cfg.day_step)
    Dg, Dt = np.meshgrid(snps.astype(float), days.astype(float))

    # Genetic plausibility at Dt=0: treat temporal_distance as fixed
    # Temporal synchrony at Dg=0: treat genetic_distance as fixed
    # Joint: both varying
    P_joint = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=Dg.ravel(),
        temporal_distance=Dt.ravel(),
        intermediate_generations=cfg.intermediate_generations,
        intermediate_hosts=cfg.intermediate_hosts,
        num_simulations=cfg.num_simulations,

    ).reshape(Dg.shape)

    # Slices to help interpretation
    P_genetic = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        temporal_distance=np.zeros_like(snps, dtype=float),
        intermediate_generations=cfg.intermediate_generations,
        intermediate_hosts=cfg.intermediate_hosts,
        num_simulations=cfg.num_simulations,
    )
    P_temporal = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=np.zeros_like(days, dtype=float),
        temporal_distance=days.astype(float),
        intermediate_generations=cfg.intermediate_generations,
        intermediate_hosts=cfg.intermediate_hosts,
        num_simulations=cfg.num_simulations,
    )

    surface_df = pd.DataFrame({
        "snp": Dg.ravel().astype(int),
        "days": Dt.ravel().astype(int),
        "probability": P_joint.ravel().astype(float),
    })
    surface_df.to_parquet(tabs_dir / "characteristic_probability_surface.parquet", index=False)

    prob_vs_snp_df = pd.DataFrame({
        "snp": snps.astype(int),
        "probability": P_genetic.astype(float),
    })
    prob_vs_snp_df.to_parquet(tabs_dir / "characteristic_prob_vs_snp.parquet", index=False)

    prob_vs_days_df = pd.DataFrame({
        "days": days.astype(int),
        "probability": P_temporal.astype(float),
    })
    prob_vs_days_df.to_parquet(tabs_dir / "characteristic_prob_vs_days.parquet", index=False)

    print(f"Saved characteristic tables to: {tabs_dir}")


if __name__ == "__main__":
    main()
