#!/usr/bin/env python3
"""Benchmark pairwise discrimination on synthetic datasets."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .config import config_value, ensure_directories, load_merged_config, load_yaml, resolve_path
from .epilink_adapter import build_model_components, estimate_linkage_scores
from .execution import finish_stage_run, start_stage_run
from .logistic import predict_logistic_scores


def evaluate(y: np.ndarray, score: np.ndarray, is_prob: bool) -> dict[str, float]:
    out = {"ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "PR_AUC": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "Brier": float(brier_score_loss(y, score)) if is_prob else np.nan}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/synthetic.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("pairwise_benchmark", cli_args=vars(args))

    study_config = load_merged_config(args.base_config, args.study_config)
    scenario_config_path = config_value(study_config, ["scenario_config"], "configs/scenarios/synthetic.yaml")
    scenario_config = load_yaml(scenario_config_path)

    processed_dir = resolve_path(config_value(study_config, ["paths", "data", "processed", "synthetic"]))
    results_dir = resolve_path(config_value(study_config, ["paths", "results", "discrimination"]))
    ensure_directories(results_dir)

    model_components = build_model_components(study_config)
    rng_seed = int(config_value(study_config, ["project", "rng_seed"], 12345))
    scenarios = config_value(scenario_config, ["scenarios"], {})
    training_fractions = config_value(study_config, ["pairwise_benchmark", "logistic_training_fractions"], [0.1, 1.0])

    rows = []
    for scenario_name in scenarios.keys():
        print(f">>> Evaluating: {scenario_name}")
        scenario_dir = processed_dir / f"scenario={scenario_name}"
        pairwise_path = scenario_dir / "pairwise.parquet"
        if not pairwise_path.exists():
            continue

        scored_pairs = pd.read_parquet(pairwise_path)
        scored_pairs = scored_pairs.loc[scored_pairs["Sampled"]].copy()

        scored_pairs["ProbLinearDist"] = estimate_linkage_scores(
            model_components.transmission_profile,
            model_components.molecular_clock,
            genetic_distance=scored_pairs["LinearDist"].values,
            temporal_distance=scored_pairs["TemporalDist"].values,
            config=study_config,
        )

        scored_pairs["ProbPoissonDist"] = estimate_linkage_scores(
            model_components.transmission_profile,
            model_components.molecular_clock,
            genetic_distance=scored_pairs["PoissonDist"].values,
            temporal_distance=scored_pairs["TemporalDist"].values,
            config=study_config,
        )

        y = scored_pairs["Related"].astype(int).values

        for training_fraction, distance_column in product(training_fractions, ("LinearDist", "PoissonDist")):
            feature_matrix = scored_pairs[["TemporalDist", distance_column]].values
            output_column = f"Logit{distance_column}{int(training_fraction * 100)}"
            scored_pairs[output_column] = predict_logistic_scores(
                feature_matrix,
                y,
                training_fraction=float(training_fraction),
                rng_seed=rng_seed,
            )

        scored_pairs["LinearDistScore"] = 1.0 / (scored_pairs["LinearDist"] + 1.0)
        scored_pairs["PoissonDistScore"] = 1.0 / (scored_pairs["PoissonDist"] + 1.0)

        scored_pairs.to_parquet(scenario_dir / "pairwise_scored.parquet", index=False)

        models = [
            ("LinearDistScore", False),
            ("PoissonDistScore", False),
            ("ProbLinearDist", True),
            ("ProbPoissonDist", True),
            ("LogitLinearDist10", True),
            ("LogitPoissonDist10", True),
            ("LogitLinearDist100", True),
            ("LogitPoissonDist100", True),
        ]

        for model_name, is_probability in models:
            if model_name not in scored_pairs.columns or scored_pairs[model_name].isna().all():
                continue
            met = evaluate(y, scored_pairs[model_name].values, is_prob=is_probability)
            row = {
                "Scenario": scenario_name,
                "Model": model_name,
                "N_pairs": len(scored_pairs),
                "Prevalence": float(y.mean()),
                **met,
            }
            rows.append(row)

    output_frame = pd.DataFrame(rows)
    metrics_path = results_dir / "discrimination_metrics.parquet"
    output_frame.to_parquet(metrics_path, index=False)

    manifest_config = study_config | {"scenario_definitions": scenario_config}
    evaluated_scenarios = sorted(output_frame["Scenario"].unique().tolist()) if not output_frame.empty else []
    finish_stage_run(
        stage_run,
        resolve_path(config_value(study_config, ["paths", "results", "manifests"])) / "pairwise_benchmark.json",
        config=manifest_config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "scenario_config": resolve_path(scenario_config_path),
            "processed_dir": str(processed_dir),
        },
        outputs={
            "results_dir": str(results_dir),
            "metrics_path": str(metrics_path),
        },
        summary={
            "num_configured_scenarios": len(scenarios),
            "evaluated_scenarios": evaluated_scenarios,
            "training_fractions": list(training_fractions),
        },
    )
    print(f"Saved evaluation metrics to: {metrics_path}")

if __name__ == "__main__":
    main()
