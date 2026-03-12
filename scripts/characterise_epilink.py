#!/usr/bin/env python3
"""
scripts/characterise_epilink.py

Characterise epilink 

Outputs
---------------------
tables/supplementary/characterise_epilink/
  - OUT_PREFIX_samples.parquet
  - OUT_PREFIX_sample_summary.parquet
  - OUT_PREFIX_stage_samples.parquet
  - OUT_PREFIX_toit_grid.parquet
  - OUT_PREFIX_tost_grid.parquet
  - OUT_PREFIX_presymptomatic_fraction.parquet
  - OUT_PREFIX_clock_rate_samples.parquet
  - OUT_PREFIX_clock_rate_summary.parquet
  - OUT_PREFIX_temporal_linkage.parquet
  - OUT_PREFIX_genetic_linkage.parquet
  - OUT_PREFIX_genetic_scenarios.parquet
  - OUT_PREFIX_probability_surface.parquet

Use ``--out-prefix`` to control the file name prefix (default: ``characteristic``).
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

@dataclass(frozen=True)
class ParamConfig:
    rng_seed: int
    toit_cfg: dict[str, float]
    clock_cfg: dict[str, float|int]
    inference_cfg: dict[str, int|tuple[int, ...]]
    max_snp: int
    snp_step: int
    max_days: int
    day_step: int
    toit_max_days: float
    toit_day_step: float
    tost_min_days: float
    tost_max_days: float
    tost_day_step: float

def parse_configs(param_yaml: Path):
    param_cfg = load_yaml(param_yaml)
    inference_cfg = deep_get(param_cfg, ["inference"], {})
    inference_cfg["intermediate_generations"] = tuple(inference_cfg["intermediate_generations"])

    return  ParamConfig(
        rng_seed=int(deep_get(param_cfg, ["rng_seed"], 42)),
        toit_cfg=deep_get(param_cfg, ["infectiousness_params"], {}),
        clock_cfg=deep_get(param_cfg, ["clock"], {}),
        inference_cfg=inference_cfg,
        max_snp=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "max_snp"], 10)),
        snp_step=int(deep_get(param_cfg, ["characterisation", "genetic_distance_grid", "step"], 1)),
        max_days=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "max_days"], 21)),
        day_step=int(deep_get(param_cfg, ["characterisation", "temporal_distance_grid", "step"], 1)),
        toit_max_days=float(deep_get(param_cfg, ["characterisation", "toit_grid", "max_days"], 60.0)),
        toit_day_step=float(deep_get(param_cfg, ["characterisation", "toit_grid", "step"], 0.1)),
        tost_min_days=float(deep_get(param_cfg, ["characterisation", "tost_grid", "min_days"], -30.0)),
        tost_max_days=float(deep_get(param_cfg, ["characterisation", "tost_grid", "max_days"], 30.0)),
        tost_day_step=float(deep_get(param_cfg, ["characterisation", "tost_grid", "step"], 0.1)),
    )



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
    parser.add_argument("--out-prefix", default="characteristic")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    out_prefix = args.out_prefix
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables"], "../tables"))
    tabs_dir = tabs_dir / "characterise_epilink"

    ensure_dirs(tabs_dir)

    def table_path(stem: str) -> Path:
        return tabs_dir / f"{out_prefix}_{stem}.parquet"

    cfg = parse_configs(Path(args.defaults))

    rng = default_rng(cfg.rng_seed)
    params = InfectiousnessParams(**cfg.toit_cfg)
    toit = TOIT(params=params, rng=rng)
    clock = MolecularClock(**cfg.clock_cfg, rng=rng)

    # --- A) Timing priors: TOIT and generation time
    toit_samples = toit.rvs(cfg.inference_cfg["num_simulations"])
    gen_time_samples = toit.generation_time(cfg.inference_cfg["num_simulations"])

    samples_df = pd.DataFrame({
        "sample_type": (["toit"] * len(toit_samples)) + (["generation_time"] * len(gen_time_samples)),
        "value": np.concatenate([toit_samples, gen_time_samples]).astype(float),
    })
    samples_df.to_parquet(table_path("samples"), index=False)

    # --- B) Plausibility surface
    snps = np.arange(0, cfg.max_snp + 1, cfg.snp_step)
    days = np.arange(0, cfg.max_days + 1, cfg.day_step)
    Dg, Dt = np.meshgrid(snps.astype(float), days.astype(float))

    P_joint = linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=Dg.ravel(),
        temporal_distance=Dt.ravel(),
        **cfg.inference_cfg

    ).reshape(Dg.shape)

    surface_df = pd.DataFrame({
        "snp": Dg.ravel().astype(int),
        "days": Dt.ravel().astype(int),
        "probability": P_joint.ravel().astype(float),
    })
    surface_df.to_parquet(table_path("probability_surface"), index=False)

    # --- C) TOIT/TOST grids, stage samples, and summary statistics
    toit_grid = build_grid(0.0, cfg.toit_max_days, cfg.toit_day_step)
    toit_grid_df = pd.DataFrame({
        "days": toit_grid.astype(float),
        "pdf": toit.pdf(toit_grid).astype(float),
        "cdf": toit.cdf(toit_grid).astype(float),
    })
    toit_grid_df.to_parquet(table_path("toit_grid"), index=False)

    tost = TOST(params=params, rng=rng)
    tost_grid = build_grid(cfg.tost_min_days, cfg.tost_max_days, cfg.tost_day_step)
    tost_grid_df = pd.DataFrame({
        "days": tost_grid.astype(float),
        "pdf": tost.pdf(tost_grid).astype(float),
        "cdf": tost.cdf(tost_grid).astype(float),
    })
    tost_grid_df.to_parquet(table_path("tost_grid"), index=False)

    latent_samples = toit.sample_latent(cfg.inference_cfg["num_simulations"])
    presymptomatic_samples = toit.sample_presymptomatic(cfg.inference_cfg["num_simulations"])
    symptomatic_samples = toit.sample_symptomatic(cfg.inference_cfg["num_simulations"])
    incubation_samples = toit.sample_incubation(cfg.inference_cfg["num_simulations"])

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
    stage_samples_df.to_parquet(table_path("stage_samples"), index=False)

    tost_samples = tost.rvs(cfg.inference_cfg["num_simulations"])
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
    sample_summary_df.to_parquet(table_path("sample_summary"), index=False)

    presymp_fraction_df = pd.DataFrame({
        "fraction": [presymp_fraction_value],
    })
    presymp_fraction_df.to_parquet(
        table_path("presymptomatic_fraction"), index=False
    )

    # --- D) Molecular clock diagnostics
    clock_rates_per_day = clock.sample_clock_rate_per_day(size=cfg.inference_cfg["num_simulations"])
    clock_rates_per_site_year = (clock_rates_per_day * 365.0) / cfg.clock_cfg["gen_len"]
    clock_rates_df = pd.DataFrame({
        "rate_per_day": clock_rates_per_day.astype(float),
        "rate_per_site_year": clock_rates_per_site_year.astype(float),
    })
    clock_rates_df.to_parquet(table_path("clock_rate_samples"), index=False)

    clock_rate_summary_df = pd.DataFrame([
        summarize_samples(clock_rates_per_day, "rate_per_day"),
        summarize_samples(clock_rates_per_site_year, "rate_per_site_year"),
    ])
    clock_rate_summary_df.to_parquet(
        table_path("clock_rate_summary"), index=False
    )

    # --- E) Temporal-only and genetic-only linkage components
    temporal_only = temporal_linkage_probability(
        temporal_distance=days,
        toit=toit,
        num_simulations=cfg.inference_cfg["num_simulations"],
    )
    temporal_df = pd.DataFrame({
        "days": days.astype(int),
        "probability": temporal_only.astype(float),
    })
    temporal_df.to_parquet(table_path("temporal_linkage"), index=False)

    genetic_raw = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        **cfg.inference_cfg,
        kind="raw",
    )
    genetic_normalized = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        **cfg.inference_cfg,
        kind="normalized",
    )
    genetic_df = pd.DataFrame({
        "snp": snps.astype(int),
        "raw": genetic_raw.astype(float),
        "normalized": genetic_normalized.astype(float),
    })
    genetic_df.to_parquet(table_path("genetic_linkage"), index=False)

    genetic_raw_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.inference_cfg["num_simulations"],
        intermediate_hosts=cfg.inference_cfg["intermediate_hosts"],
        intermediate_generations=None,
        kind="raw",
    )
    genetic_relative_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.inference_cfg["num_simulations"],
        intermediate_hosts=cfg.inference_cfg["intermediate_hosts"],
        intermediate_generations=None,
        kind="relative",
    )
    genetic_normalized_all = genetic_linkage_probability(
        toit=toit,
        clock=clock,
        genetic_distance=snps.astype(float),
        num_simulations=cfg.inference_cfg["num_simulations"],
        intermediate_hosts=cfg.inference_cfg["intermediate_hosts"],
        intermediate_generations=None,
        kind="normalized",
    )

    scenario_rows = []
    m_vals = np.arange(0, cfg.inference_cfg["intermediate_hosts"] + 1, dtype=int)
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
    scenarios_df.to_parquet(table_path("genetic_scenarios"), index=False)

    print(f"Saved characteristic tables to: {tabs_dir}")


if __name__ == "__main__":
    main()
