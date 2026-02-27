#!/usr/bin/env python3
"""
scripts/characterise_epilink.py

Characterise epilink 

Outputs
---------------------
tables/supplementary/
  - characteristic_samples.parquet
  - characteristic_sample_summary.parquet
  - characteristic_stage_samples.parquet
  - characteristic_toit_grid.parquet
  - characteristic_tost_grid.parquet
  - characteristic_presymptomatic_fraction.parquet
  - characteristic_model_params.parquet
  - characteristic_clock_rate_samples.parquet
  - characteristic_clock_rate_summary.parquet
  - characteristic_expected_mutations.parquet
  - characteristic_temporal_linkage.parquet
  - characteristic_genetic_linkage.parquet
  - characteristic_genetic_scenarios.parquet
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
    TOST,
    MolecularClock,
    InfectiousnessParams,
    linkage_probability,
    temporal_linkage_probability,
    genetic_linkage_probability,
    presymptomatic_fraction,
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
    intermediate_generations: tuple[int, ...]
    intermediate_hosts: int
    max_snp: int
    snp_step: int
    max_days: int
    day_step: int
    toit_max_days: float
    toit_day_step: float
    tost_min_days: float
    tost_max_days: float
    tost_day_step: float
    expected_mutation_max_days: int
    expected_mutation_day_step: int


def summarize_samples(values: np.ndarray, label: str) -> dict[str, float | str | int]:
    arr = np.asarray(values, dtype=float)
    quantiles = np.quantile(arr, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "sample_type": label,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q025": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q975": float(quantiles[4]),
    }


def build_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive.")
    return np.arange(start, stop + step * 0.5, step, dtype=float)


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

    tabs_dir = tabs_dir / "characterise_epilink"

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
        intermediate_generations=tuple(
            int(x) for x in deep_get(param_cfg, ["inference", "intermediate_generations"], (0, 1))
        ),
        intermediate_hosts=int(deep_get(param_cfg, ["inference", "intermediate_hosts"], 10)),

        max_snp=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "max_snp"], 10)),
        snp_step=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "step"], 1)),
        max_days=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "max_days"], 21)),
        day_step=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "step"], 1)),
        toit_max_days=float(
            deep_get(param_cfg, ["characterisation", "toit_grid", "max_days"], 60.0)
        ),
        toit_day_step=float(
            deep_get(param_cfg, ["characterisation", "toit_grid", "step"], 0.1)
        ),
        tost_min_days=float(
            deep_get(param_cfg, ["characterisation", "tost_grid", "min_days"], -30.0)
        ),
        tost_max_days=float(
            deep_get(param_cfg, ["characterisation", "tost_grid", "max_days"], 30.0)
        ),
        tost_day_step=float(
            deep_get(param_cfg, ["characterisation", "tost_grid", "step"], 0.1)
        ),
        expected_mutation_max_days=int(
            deep_get(param_cfg, ["characterisation", "expected_mutations_grid", "max_days"], 60)
        ),
        expected_mutation_day_step=int(
            deep_get(param_cfg, ["characterisation", "expected_mutations_grid", "step"], 1)
        ),
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

    # --- C) TOIT/TOST grids, stage samples, and summary statistics
    toit_grid = build_grid(0.0, cfg.toit_max_days, cfg.toit_day_step)
    toit_grid_df = pd.DataFrame({
        "days": toit_grid.astype(float),
        "pdf": toit.pdf(toit_grid).astype(float),
        "cdf": toit.cdf(toit_grid).astype(float),
    })
    toit_grid_df.to_parquet(tabs_dir / "characteristic_toit_grid.parquet", index=False)

    tost = TOST(params=params, rng=rng)
    tost_grid = build_grid(cfg.tost_min_days, cfg.tost_max_days, cfg.tost_day_step)
    tost_grid_df = pd.DataFrame({
        "days": tost_grid.astype(float),
        "pdf": tost.pdf(tost_grid).astype(float),
        "cdf": tost.cdf(tost_grid).astype(float),
    })
    tost_grid_df.to_parquet(tabs_dir / "characteristic_tost_grid.parquet", index=False)

    latent_samples = toit.sample_latent(cfg.num_simulations)
    presymptomatic_samples = toit.sample_presymptomatic(cfg.num_simulations)
    symptomatic_samples = toit.sample_symptomatic(cfg.num_simulations)
    incubation_samples = toit.sample_incubation(cfg.num_simulations)

    stage_samples_df = pd.DataFrame({
        "stage": (
            ["latent"] * len(latent_samples)
            + ["presymptomatic"] * len(presymptomatic_samples)
            + ["symptomatic"] * len(symptomatic_samples)
            + ["incubation"] * len(incubation_samples)
        ),
        "value": np.concatenate(
            [latent_samples, presymptomatic_samples, symptomatic_samples, incubation_samples]
        ).astype(float),
    })
    stage_samples_df.to_parquet(tabs_dir / "characteristic_stage_samples.parquet", index=False)

    tost_samples = tost.rvs(cfg.num_simulations)
    presymp_fraction_value = float(presymptomatic_fraction(params))
    sample_summary = [
        summarize_samples(toit_samples, "toit"),
        summarize_samples(gen_time_samples, "generation_time"),
        summarize_samples(tost_samples, "tost"),
        summarize_samples(latent_samples, "latent"),
        summarize_samples(presymptomatic_samples, "presymptomatic"),
        summarize_samples(symptomatic_samples, "symptomatic"),
        summarize_samples(incubation_samples, "incubation"),
        summarize_samples(np.array([presymp_fraction_value]), "presymptomatic_fraction"),
    ]
    sample_summary_df = pd.DataFrame(sample_summary)
    sample_summary_df.to_parquet(tabs_dir / "characteristic_sample_summary.parquet", index=False)

    presymp_fraction_df = pd.DataFrame({
        "fraction": [presymp_fraction_value],
    })
    presymp_fraction_df.to_parquet(
        tabs_dir / "characteristic_presymptomatic_fraction.parquet", index=False
    )

    model_params_df = pd.DataFrame(
        [
            {"parameter": "incubation_shape", "value": str(cfg.incubation_shape)},
            {"parameter": "incubation_scale", "value": str(cfg.incubation_scale)},
            {"parameter": "latent_shape", "value": str(cfg.latent_shape)},
            {"parameter": "symptomatic_rate", "value": str(cfg.symptomatic_rate)},
            {"parameter": "symptomatic_shape", "value": str(cfg.symptomatic_shape)},
            {"parameter": "rel_presymptomatic_infectiousness", "value": str(cfg.rel_presymptomatic_infectiousness)},
            {"parameter": "subs_rate", "value": str(cfg.subs_rate)},
            {"parameter": "relax_rate", "value": str(cfg.relax_rate)},
            {"parameter": "subs_rate_sigma", "value": str(cfg.subs_rate_sigma)},
            {"parameter": "gen_length", "value": str(cfg.gen_length)},
            {"parameter": "num_simulations", "value": str(cfg.num_simulations)},
            {"parameter": "intermediate_generations", "value": str(cfg.intermediate_generations)},
            {"parameter": "intermediate_hosts", "value": str(cfg.intermediate_hosts)},
        ]
    )
    model_params_df.to_parquet(tabs_dir / "characteristic_model_params.parquet", index=False)

    # --- D) Molecular clock diagnostics
    clock_rates_per_day = clock.sample_clock_rate_per_day(size=cfg.num_simulations)
    clock_rates_per_site_year = (clock_rates_per_day * 365.0) / cfg.gen_length
    clock_rates_df = pd.DataFrame({
        "rate_per_day": clock_rates_per_day.astype(float),
        "rate_per_site_year": clock_rates_per_site_year.astype(float),
    })
    clock_rates_df.to_parquet(tabs_dir / "characteristic_clock_rate_samples.parquet", index=False)

    clock_rate_summary_df = pd.DataFrame([
        summarize_samples(clock_rates_per_day, "rate_per_day"),
        summarize_samples(clock_rates_per_site_year, "rate_per_site_year"),
    ])
    clock_rate_summary_df.to_parquet(
        tabs_dir / "characteristic_clock_rate_summary.parquet", index=False
    )

    expected_days = np.arange(
        0,
        cfg.expected_mutation_max_days + 1,
        cfg.expected_mutation_day_step,
        dtype=float,
    )
    expected_mutations = clock.expected_mutations(expected_days)
    expected_mutations_df = pd.DataFrame({
        "days": expected_days.astype(int),
        "expected_mutations": expected_mutations.astype(float),
    })
    expected_mutations_df.to_parquet(
        tabs_dir / "characteristic_expected_mutations.parquet", index=False
    )

    # --- E) Temporal-only and genetic-only linkage components
    temporal_days = np.arange(-cfg.max_days, cfg.max_days + 1, cfg.day_step, dtype=float)
    temporal_only = temporal_linkage_probability(
        temporal_distance=temporal_days,
        toit=toit,
        num_simulations=cfg.num_simulations,
    )
    temporal_df = pd.DataFrame({
        "days": temporal_days.astype(int),
        "probability": temporal_only.astype(float),
    })
    temporal_df.to_parquet(tabs_dir / "characteristic_temporal_linkage.parquet", index=False)

    genetic_relative = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=cfg.intermediate_generations,
        kind="relative",
    )
    genetic_raw = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=cfg.intermediate_generations,
        kind="raw",
    )
    genetic_normalized = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=cfg.intermediate_generations,
        kind="normalized",
    )
    genetic_df = pd.DataFrame({
        "snp": snps.astype(int),
        "relative": genetic_relative.astype(float),
        "raw": genetic_raw.astype(float),
        "normalized": genetic_normalized.astype(float),
    })
    genetic_df.to_parquet(tabs_dir / "characteristic_genetic_linkage.parquet", index=False)

    genetic_raw_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=None,
        kind="raw",
    )
    genetic_relative_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=None,
        kind="relative",
    )
    genetic_normalized_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.num_simulations,
        intermediate_hosts=cfg.intermediate_hosts,
        intermediate_generations=None,
        kind="normalized",
    )

    scenario_rows = []
    m_vals = np.arange(0, cfg.intermediate_hosts + 1, dtype=int)
    for idx, snp in enumerate(snps.astype(int)):
        for m in m_vals:
            scenario_rows.append({
                "snp": int(snp),
                "m": int(m),
                "raw": float(genetic_raw_all[idx, m]),
                "relative": float(genetic_relative_all[idx, m]),
                "normalized": float(genetic_normalized_all[idx, m]),
            })
    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_parquet(tabs_dir / "characteristic_genetic_scenarios.parquet", index=False)

    print(f"Saved characteristic tables to: {tabs_dir}")


if __name__ == "__main__":
    main()
