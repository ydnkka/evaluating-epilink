#!/usr/bin/env python3
"""
scripts/pairwise_discrimination.py

Consume saved synthetic datasets (pairwise.parquet per scenario) and evaluate
edge “meaningfulness” via binary classification metrics.

This is deliberately separate from dataset generation so that:
- simulation outputs are stable
- edge evaluation and clustering consume identical inputs

Config
------
config/paths.yaml
config/generate_datasets.yaml
config/default_parameters.yaml

Outputs
-------
tables/supplementary/discrimination/
  - discrimination_metrics.parquet
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
from numpy.random import default_rng
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from epilink import (
    TOIT,
    MolecularClock,
    InfectiousnessParams,
    linkage_probability
)

from utils import *


def evaluate(y: np.ndarray, score: np.ndarray, is_prob: bool) -> dict[str, float]:
    out = {"ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "PR_AUC": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "Brier": float(brier_score_loss(y, score)) if is_prob else np.nan}
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--datasets_cfg", default="../config/generate_datasets.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    datasets_cfg = load_yaml(Path(args.datasets))
    defaults_cfg = load_yaml(Path(args.defaults))

    processed_dir = Path(
        deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic")
    )

    tabs_dir = Path(
        deep_get(paths_cfg, ["outputs", "tables"], "../tables")
    )
    tabs_dir = tabs_dir / "discrimination"
    ensure_dirs(tabs_dir)

    rng_seed = int(deep_get(defaults_cfg, ["rng_seed"], 12345))
    toit_cfg = deep_get(defaults_cfg, ["infectiousness_params"], {})
    clock_cfg = deep_get(defaults_cfg, ["clock"], {})
    inference_cfg = deep_get(defaults_cfg, ["inference"], {})
    inference_cfg["intermediate_generations"] = tuple(inference_cfg["intermediate_generations"])

    rng = default_rng(rng_seed)

    params = InfectiousnessParams(**toit_cfg)
    toit = TOIT(params=params,rng=rng)
    clock = MolecularClock(**clock_cfg)

    scenarios = deep_get(datasets_cfg, ["scenarios"], {})

    rows = []
    for scen in scenarios.keys():
        print(f">>> Evaluating: {scen}")
        sc_dir = processed_dir / f"scenario={scen}"
        pw_path = sc_dir / "pairwise.parquet"
        if not pw_path.exists():
            continue

        df = pd.read_parquet(pw_path)
        df = df[df["Sampled"]].copy()

        # Model probability
        df["ProbLinearDist"] = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=df["LinearDist"].values,
            temporal_distance=df["TemporalDist"].values,
            **inference_cfg
        )

        df["ProbPoissonDist"] = linkage_probability(
            toit=toit,
            clock=clock,
            genetic_distance=df["PoissonDist"].values,
            temporal_distance=df["TemporalDist"].values,
            **inference_cfg
        )

        # Logistic regression
        y = df["Related"].astype(int).values

        for p, dist_col in product((0.1, 1.0), ("LinearDist", "PoissonDist")):
            X = df[["TemporalDist", dist_col]].values
            col = f"Logit{dist_col}{int(p * 100)}"

            clf = LogisticRegression(solver="lbfgs", max_iter=200)
            if p == 1.0:
                clf.fit(X, y)
                df[col] = clf.predict_proba(X)[:, 1]
            else:
                X_tr, _, y_tr, _ = train_test_split(X, y, train_size=p, stratify=y, random_state=rng_seed)
                clf.fit(X_tr, y_tr)
                df[col] = clf.predict_proba(X)[:, 1]

        df["LinearDistScore"] = 1.0 / (df["LinearDist"] + 1.0)
        df["PoissonDistScore"] = 1.0 / (df["PoissonDist"] + 1.0)

        df.to_parquet(sc_dir / "pairwise.parquet", index=False)

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

        for m, is_prob in models:
            if m not in df.columns or df[m].isna().all():
                continue
            met = evaluate(y, df[m].values, is_prob=is_prob)
            row = {
                "Scenario": scen,
                "Model": m,
                "N_pairs": len(df),
                "Prevalence": float(y.mean()),
                **met,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(tabs_dir / "discrimination_metrics.parquet", index=False)
    print(f"Saved evaluation metrics to: {tabs_dir / 'discrimination_metrics.parquet'}")

if __name__ == "__main__":
    main()
