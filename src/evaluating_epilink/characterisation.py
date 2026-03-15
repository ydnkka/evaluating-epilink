"""Characterise the timing and genetic components of the epilink model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import config_value, ensure_directories, load_merged_config, resolve_path
from .epilink_adapter import (
    build_inference_kwargs,
    build_model_components,
    build_symptom_onset_profile,
    estimate_genetic_scores,
    estimate_linkage_scores,
    estimate_presymptomatic_fraction,
    estimate_temporal_scores,
)
from .execution import finish_stage_run, start_stage_run


@dataclass(frozen=True)
class CharacterisationConfig:
    output_prefix: str
    num_simulations: int
    max_intermediate_hosts: int
    max_snp: int
    snp_step: int
    max_days: int
    day_step: int
    toit_max_days: float
    toit_day_step: float
    tost_min_days: float
    tost_max_days: float
    tost_day_step: float


def parse_characterisation_config(config: dict) -> CharacterisationConfig:
    """Extract the subset of config values used in the model characterisation workflow."""

    inference_kwargs = build_inference_kwargs(config)
    return CharacterisationConfig(
        output_prefix=str(config_value(config, ["characterisation", "output_prefix"], "characteristic")),
        num_simulations=int(inference_kwargs["num_simulations"]),
        max_intermediate_hosts=int(inference_kwargs["max_intermediate_hosts"]),
        max_snp=int(config_value(config, ["characterisation", "genetic_distance_grid", "max_snp"], 10)),
        snp_step=int(config_value(config, ["characterisation", "genetic_distance_grid", "step"], 1)),
        max_days=int(config_value(config, ["characterisation", "temporal_distance_grid", "max_days"], 15)),
        day_step=int(config_value(config, ["characterisation", "temporal_distance_grid", "step"], 1)),
        toit_max_days=float(config_value(config, ["characterisation", "toit_grid", "max_days"], 60.0)),
        toit_day_step=float(config_value(config, ["characterisation", "toit_grid", "step"], 0.1)),
        tost_min_days=float(config_value(config, ["characterisation", "tost_grid", "min_days"], -30.0)),
        tost_max_days=float(config_value(config, ["characterisation", "tost_grid", "max_days"], 30.0)),
        tost_day_step=float(config_value(config, ["characterisation", "tost_grid", "step"], 0.1)),
    )


def summarize_samples(values: np.ndarray, sample_type: str) -> dict[str, float | str | int]:
    """Summarise a sampled distribution with central tendency and quantiles."""

    sample_array = np.asarray(values, dtype=float)
    quantiles = np.quantile(sample_array, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "sample_type": sample_type,
        "n": int(sample_array.size),
        "mean": float(np.mean(sample_array)),
        "sd": float(np.std(sample_array, ddof=1)) if sample_array.size > 1 else 0.0,
        "q025": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q975": float(quantiles[4]),
    }


def build_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Create a numeric grid including the stop value when it falls on the step."""

    if step <= 0:
        raise ValueError("step must be positive.")
    return np.arange(start, stop + step * 0.5, step, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/characterisation.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("characterisation", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    run_config = parse_characterisation_config(study_config)
    results_dir = resolve_path(config_value(study_config, ["paths", "results", "characterisation"]))
    ensure_directories(results_dir)

    def table_path(stem: str) -> str:
        return str(results_dir / f"{run_config.output_prefix}_{stem}.parquet")

    model_components = build_model_components(study_config)
    transmission_profile = model_components.transmission_profile
    molecular_clock = model_components.molecular_clock
    symptom_onset_profile = build_symptom_onset_profile(study_config, rng=model_components.rng)

    toit_samples = transmission_profile.rvs(run_config.num_simulations)
    generation_time_samples = transmission_profile.sample_generation_intervals(run_config.num_simulations)

    pd.DataFrame(
        {
            "sample_type": (["toit"] * len(toit_samples)) + (["generation_time"] * len(generation_time_samples)),
            "value": np.concatenate([toit_samples, generation_time_samples]).astype(float),
        }
    ).to_parquet(table_path("samples"), index=False)

    snps = np.arange(0, run_config.max_snp + 1, run_config.snp_step)
    days = np.arange(0, run_config.max_days + 1, run_config.day_step)
    snp_grid, day_grid = np.meshgrid(snps.astype(float), days.astype(float))
    probability_surface = estimate_linkage_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snp_grid.ravel(),
        temporal_distance=day_grid.ravel(),
        config=study_config,
    ).reshape(snp_grid.shape)
    pd.DataFrame(
        {
            "snp": snp_grid.ravel().astype(int),
            "days": day_grid.ravel().astype(int),
            "probability": probability_surface.ravel().astype(float),
        }
    ).to_parquet(table_path("probability_surface"), index=False)

    toit_grid = build_grid(0.0, run_config.toit_max_days, run_config.toit_day_step)
    pd.DataFrame(
        {
            "days": toit_grid.astype(float),
            "pdf": transmission_profile.pdf(toit_grid).astype(float),
            "cdf": transmission_profile.cdf(toit_grid).astype(float),
        }
    ).to_parquet(table_path("toit_grid"), index=False)

    tost_grid = build_grid(run_config.tost_min_days, run_config.tost_max_days, run_config.tost_day_step)
    pd.DataFrame(
        {
            "days": tost_grid.astype(float),
            "pdf": symptom_onset_profile.pdf(tost_grid).astype(float),
            "cdf": symptom_onset_profile.cdf(tost_grid).astype(float),
        }
    ).to_parquet(table_path("tost_grid"), index=False)

    latent_samples = transmission_profile.sample_latent_periods(run_config.num_simulations)
    presymptomatic_samples = transmission_profile.sample_presymptomatic_periods(run_config.num_simulations)
    symptomatic_samples = transmission_profile.sample_symptomatic_periods(run_config.num_simulations)
    incubation_samples = transmission_profile.sample_incubation_periods(run_config.num_simulations)
    pd.DataFrame(
        {
            "stage": (
                ["latent"] * len(latent_samples)
                + ["presymptomatic"] * len(presymptomatic_samples)
                + ["symptomatic"] * len(symptomatic_samples)
                + ["incubation"] * len(incubation_samples)
            ),
            "value": np.concatenate(
                [latent_samples, presymptomatic_samples, symptomatic_samples, incubation_samples]
            ).astype(float),
        }
    ).to_parquet(table_path("stage_samples"), index=False)

    tost_samples = symptom_onset_profile.rvs(run_config.num_simulations)
    presymptomatic_fraction = estimate_presymptomatic_fraction(study_config)
    sample_summary = [
        summarize_samples(toit_samples, "toit"),
        summarize_samples(generation_time_samples, "generation_time"),
        summarize_samples(tost_samples, "tost"),
        summarize_samples(latent_samples, "latent"),
        summarize_samples(presymptomatic_samples, "presymptomatic"),
        summarize_samples(symptomatic_samples, "symptomatic"),
        summarize_samples(incubation_samples, "incubation"),
        summarize_samples(np.array([presymptomatic_fraction]), "presymptomatic_fraction"),
    ]
    pd.DataFrame(sample_summary).to_parquet(table_path("sample_summary"), index=False)
    pd.DataFrame({"fraction": [presymptomatic_fraction]}).to_parquet(
        table_path("presymptomatic_fraction"),
        index=False,
    )

    clock_rates_per_day = molecular_clock.sample_substitution_rate_per_day(size=run_config.num_simulations)
    clock_rates_per_site_year = (clock_rates_per_day * 365.0) / molecular_clock.genome_length
    pd.DataFrame(
        {
            "rate_per_day": clock_rates_per_day.astype(float),
            "rate_per_site_year": clock_rates_per_site_year.astype(float),
        }
    ).to_parquet(table_path("clock_rate_samples"), index=False)
    pd.DataFrame(
        [
            summarize_samples(clock_rates_per_day, "rate_per_day"),
            summarize_samples(clock_rates_per_site_year, "rate_per_site_year"),
        ]
    ).to_parquet(table_path("clock_rate_summary"), index=False)

    temporal_only = estimate_temporal_scores(
        transmission_profile,
        temporal_distance=days,
        config=study_config,
    )
    pd.DataFrame({"days": days.astype(int), "probability": temporal_only.astype(float)}).to_parquet(
        table_path("temporal_linkage"),
        index=False,
    )

    genetic_raw = estimate_genetic_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snps.astype(float),
        config=study_config,
        output_mode="raw",
    )
    genetic_normalized = estimate_genetic_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snps.astype(float),
        config=study_config,
        output_mode="normalized",
    )
    pd.DataFrame(
        {
            "snp": snps.astype(int),
            "raw": genetic_raw.astype(float),
            "normalized": genetic_normalized.astype(float),
        }
    ).to_parquet(table_path("genetic_linkage"), index=False)

    genetic_raw_all = estimate_genetic_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snps.astype(float),
        config=study_config,
        output_mode="raw",
        include_all_intermediate_counts=True,
    )
    genetic_relative_all = estimate_genetic_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snps.astype(float),
        config=study_config,
        output_mode="relative",
        include_all_intermediate_counts=True,
    )
    genetic_normalized_all = estimate_genetic_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snps.astype(float),
        config=study_config,
        output_mode="normalized",
        include_all_intermediate_counts=True,
    )

    scenario_rows = []
    intermediate_counts = np.arange(0, run_config.max_intermediate_hosts + 1, dtype=int)
    for snp_index, snp_value in enumerate(snps.astype(int)):
        for intermediate_count in intermediate_counts:
            scenario_rows.append(
                {
                    "snp": int(snp_value),
                    "m": int(intermediate_count),
                    "raw": float(genetic_raw_all[snp_index, intermediate_count]),
                    "relative": float(genetic_relative_all[snp_index, intermediate_count]),
                    "normalized": float(genetic_normalized_all[snp_index, intermediate_count]),
                }
            )
    pd.DataFrame(scenario_rows).to_parquet(table_path("genetic_scenarios"), index=False)

    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "characterisation.json",
        config=study_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
        },
        outputs={"results_dir": str(results_dir)},
        summary={
            "output_prefix": run_config.output_prefix,
            "num_simulations": run_config.num_simulations,
            "max_intermediate_hosts": run_config.max_intermediate_hosts,
            "num_snp_grid_points": len(snps),
            "num_temporal_grid_points": len(days),
        },
    )
    print(f"Saved characteristic tables to: {results_dir}")


if __name__ == "__main__":
    main()
