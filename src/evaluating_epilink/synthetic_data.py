#!/usr/bin/env python3
"""Generate synthetic datasets for the manuscript sensitivity analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import networkx as nx

from .config import config_value, deep_merge, ensure_directories, load_merged_config, load_yaml, resolve_path
from .epilink_adapter import build_pairwise_table, build_model_components, simulate_epidemic_tree, simulate_genomic_outputs
from .execution import finish_stage_run, start_stage_run


def summarise_data(
    df: pd.DataFrame,
    scenario: str,
    description: str,
    metrics: list[str],
    params_row: dict,
    related_col: str = "Related",
) -> pd.DataFrame:
    """
    Long table:
      Scenario | Description | (params...) | Metric | Group | N | Mean | SD
    """
    out = []
    base = {"Scenario": scenario, "Description": description, **params_row}

    for metric in metrics:
        for rel_val, rel_name in [(0, "Unrelated"), (1, "Related")]:
            x = df.loc[df[related_col] == rel_val, metric].astype(float)
            x = x[np.isfinite(x)]
            out.append({
                **base,
                "Metric": metric,
                "Group": rel_name,
                "N": int(x.shape[0]),
                "Mean Dist": float(x.mean()) if len(x) else np.nan,
                "SD Dist": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            })

    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/synthetic.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("synthetic_data", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    scenario_config_path = config_value(study_config, ["scenario_config"], "configs/scenarios/synthetic.yaml")
    scenario_config = load_yaml(scenario_config_path)

    processed_dir = resolve_path(config_value(study_config, ["paths", "data", "processed", "synthetic"]))
    summary_dir = resolve_path(config_value(study_config, ["paths", "results", "scovmod"]))
    ensure_directories(processed_dir, summary_dir)

    tree_path = resolve_path(config_value(study_config, ["backbone", "tree_gml"]))
    base_tree = nx.read_gml(tree_path)

    scenarios = config_value(scenario_config, ["scenarios"], None)
    if not isinstance(scenarios, dict) or len(scenarios) == 0:
        raise ValueError("Synthetic scenario config must define a `scenarios:` mapping.")

    baseline = scenarios["baseline"]
    summary_rows = []

    for scenario_name, scenario_patch in scenarios.items():
        scenario_config_merged = deep_merge(study_config, baseline)
        if scenario_name != "baseline":
            scenario_config_merged = deep_merge(scenario_config_merged, scenario_patch)

        print(
            f"\n>>> Simulating scenario: {scenario_name} | "
            f"{scenario_config_merged.get('description', '')}"
        )
        scenario_dir = processed_dir / f"scenario={scenario_name}"
        ensure_directories(scenario_dir)

        model_components = build_model_components(scenario_config_merged)
        populated_tree = simulate_epidemic_tree(
            model_components.transmission_profile,
            base_tree,
            scenario_config_merged,
        )
        genomic_outputs = simulate_genomic_outputs(model_components.molecular_clock, populated_tree)
        pairwise = build_pairwise_table(genomic_outputs["packed"], populated_tree)

        pairwise.to_parquet(scenario_dir / "pairwise.parquet", index=False)

        sampled_pairs = pairwise.loc[pairwise["Sampled"]].copy()

        metrics = ["TemporalDist", "PoissonDist", "LinearDist"]
        natural_history_config = config_value(scenario_config_merged, ["model", "natural_history"], {})
        surveillance_config = config_value(scenario_config_merged, ["surveillance"], {})
        sampling_delay_config = config_value(surveillance_config, ["sampling_delay"], {})
        clock_config = config_value(scenario_config_merged, ["model", "molecular_clock"], {})

        params_row: dict[str, Any] = {
            "incubation_shape": float(natural_history_config.get("incubation_shape", 5.807)),
            "incubation_scale": float(natural_history_config.get("incubation_scale", 0.948)),
            "latent_shape": float(natural_history_config.get("latent_shape", 3.38)),
            "symptomatic_rate": float(natural_history_config.get("symptomatic_rate", 0.37)),
            "symptomatic_shape": float(natural_history_config.get("symptomatic_shape", 1.0)),
            "rel_presymptomatic_infectiousness": float(
                natural_history_config.get("rel_presymptomatic_infectiousness", 2.29)
            ),
            "prop_sampled": float(surveillance_config.get("sampled_fraction", 1.0)),
            "sampling_shape": float(sampling_delay_config.get("shape", 3.0)),
            "sampling_scale": float(sampling_delay_config.get("scale", 1.0)),
            "substitution_rate": float(clock_config.get("substitution_rate", 0.001)),
            "use_relaxed_clock": bool(clock_config.get("use_relaxed_clock", True)),
            "relaxed_clock_sigma": float(clock_config.get("relaxed_clock_sigma", 0.33)),
        }

        dist_summary = summarise_data(
            df=sampled_pairs,
            scenario=scenario_name,
            description=scenario_config_merged.get("description", ""),
            metrics=metrics,
            params_row=params_row,
        )

        summary_rows.append(dist_summary)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    scenario_summary_path = summary_dir / "scenario_summary.parquet"
    summary_df.to_parquet(scenario_summary_path, index=False)

    manifest_config = deep_merge(study_config, {"scenario_definitions": scenario_config})
    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "synthetic_data.json",
        config=manifest_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "scenario_config": resolve_path(scenario_config_path),
            "tree_path": str(tree_path),
        },
        outputs={
            "processed_dir": str(processed_dir),
            "scenario_summary_path": str(scenario_summary_path),
        },
        summary={
            "num_scenarios": len(scenarios),
            "scenario_names": list(scenarios.keys()),
        },
    )

    print(f"\nSaved datasets to: {processed_dir}")
    print(f"Saved scenario summary to: {scenario_summary_path}")


if __name__ == "__main__":
    main()
