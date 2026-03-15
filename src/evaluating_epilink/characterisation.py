"""Characterise the joint epilink score across genetic and temporal gaps."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import config_value, ensure_directories, load_merged_config, resolve_path
from .epilink_adapter import (
    build_inference_kwargs,
    build_model_components,
    estimate_linkage_scores,
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
    slice_snps: tuple[int, ...]
    robustness_intermediate_counts: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceSpecification:
    scenario: str
    label: str
    included_intermediate_counts: tuple[int, ...]


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
        slice_snps=tuple(
            int(value) for value in config_value(config, ["characterisation", "slice_snps"], [0, 1, 2, 5])
        ),
        robustness_intermediate_counts=tuple(
            int(value)
            for value in config_value(
                config,
                ["characterisation", "robustness", "included_intermediate_counts"],
                [0, 1, 2],
            )
        ),
    )


def override_intermediate_counts(config: dict, counts: tuple[int, ...]) -> dict:
    """Clone the config and replace the included intermediate counts."""

    updated = deepcopy(config)
    updated.setdefault("inference", {})
    updated["inference"]["included_intermediate_counts"] = list(counts)
    return updated


def format_intermediate_label(counts: tuple[int, ...]) -> str:
    """Create a short human-readable label for an intermediate-host setting."""

    if counts == (0,):
        return "Direct only (M=0)"
    if counts and counts == tuple(range(max(counts) + 1)):
        upper_bound = max(counts)
        if upper_bound == 1:
            return "Allow 1 intermediate (M<=1)"
        return f"Allow up to {upper_bound} intermediates (M<={upper_bound})"
    values = ", ".join(str(value) for value in counts)
    return f"M in {{{values}}}"


def build_surface_specs(default_counts: tuple[int, ...], robustness_counts: tuple[int, ...]) -> list[SurfaceSpecification]:
    """Return the unique model variants to evaluate for the manuscript figure."""

    specs = [
        SurfaceSpecification(
            scenario="default",
            label=format_intermediate_label(default_counts),
            included_intermediate_counts=default_counts,
        )
    ]
    if robustness_counts != default_counts:
        specs.append(
            SurfaceSpecification(
                scenario="robustness",
                label=format_intermediate_label(robustness_counts),
                included_intermediate_counts=robustness_counts,
            )
        )
    return specs


def build_probability_surface(
    transmission_profile,
    molecular_clock,
    *,
    config: dict,
    snps: np.ndarray,
    days: np.ndarray,
    spec: SurfaceSpecification,
) -> pd.DataFrame:
    """Evaluate P(link) over a SNP-by-time grid for one model variant."""

    scenario_config = override_intermediate_counts(config, spec.included_intermediate_counts)
    snp_grid, day_grid = np.meshgrid(snps.astype(float), days.astype(float))
    probabilities = estimate_linkage_scores(
        transmission_profile,
        molecular_clock,
        genetic_distance=snp_grid.ravel(),
        temporal_distance=day_grid.ravel(),
        config=scenario_config,
    ).reshape(snp_grid.shape)
    return pd.DataFrame(
        {
            "scenario": spec.scenario,
            "label": spec.label,
            "included_intermediate_counts": [",".join(str(value) for value in spec.included_intermediate_counts)]
            * snp_grid.size,
            "snp": snp_grid.ravel().astype(int),
            "days": day_grid.ravel().astype(int),
            "probability": probabilities.ravel().astype(float),
        }
    )


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

    snps = np.arange(0, run_config.max_snp + 1, run_config.snp_step)
    days = np.arange(0, run_config.max_days + 1, run_config.day_step)
    default_counts = tuple(build_inference_kwargs(study_config)["included_intermediate_counts"])
    surface_specs = build_surface_specs(default_counts, run_config.robustness_intermediate_counts)
    probability_surface = pd.concat(
        [
            build_probability_surface(
                transmission_profile,
                molecular_clock,
                config=study_config,
                snps=snps,
                days=days,
                spec=spec,
            )
            for spec in surface_specs
        ],
        ignore_index=True,
    )
    probability_surface.to_parquet(table_path("probability_surface"), index=False)

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
            "slice_snps": list(run_config.slice_snps),
            "surface_variants": [
                {
                    "scenario": spec.scenario,
                    "label": spec.label,
                    "included_intermediate_counts": list(spec.included_intermediate_counts),
                }
                for spec in surface_specs
            ],
        },
    )
    print(f"Saved characteristic tables to: {results_dir}")


if __name__ == "__main__":
    main()
