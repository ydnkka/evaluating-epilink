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
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss

from epilink import (
    TOIT,
    InfectiousnessParams,
    estimate_linkage_probabilities
)

from utils import *

set_seaborn_paper_context(font_scale=1.2)

MODELS = {
    "LinearDistScore": "Lin–Score",
    "PoissonDistScore": "Pois–Score",
    "MechProbLinearDist": "Lin–Mech",
    "MechProbPoissonDist": "Pois–Mech",
    "LogitProbLinearDist_0.1": "Lin–Logit(0.1)",
    "LogitProbLinearDist_1.0": "Lin–Logit(1)",
    "LogitProbPoissonDist_0.1": "Pois–Logit(0.1)",
    "LogitProbPoissonDist_1.0": "Pois–Logit(1)",
}

SCENARIOS = {
    "baseline": "Baseline",
    "surveillance_moderate": "Surveillance (moderate)",
    "surveillance_severe": "Surveillance (severe)",

    # Low evolutionary signal scenarios
    "low_clock_signal": "Low clock signal",
    "low_k_inc": "Low incubation shape",
    "low_scale_inc": "Low incubation scale",
    # High evolutionary signal scenarios
    "high_clock_signal": "High clock signal",
    "high_k_inc": "High incubation shape",
    "high_scale_inc": "High incubation scale",

    "relaxed_clock": "Relaxed clock",
    "adversarial": "Adversarial",
}

def safe_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(log_loss(y, p))

def evaluate(y: np.ndarray, score: np.ndarray, is_prob: bool) -> Dict[str, float]:
    out = {"ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "PR_AUC": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
           "LogLoss": safe_log_loss(y, score) if is_prob else np.nan,
           "Brier": float(brier_score_loss(y, score)) if is_prob else np.nan}
    return out


def plot_weight_distributions(df: pd.DataFrame, out_path: Path, scenario: str) -> None:
    # Quick diagnostic plot: related vs unrelated score distributions

    # Pair models two-by-two from the flat list
    model_pairs = [
        ("LinearDistScore", "PoissonDistScore"),
        ("MechProbLinearDist", "MechProbPoissonDist"),
        ("LogitProbLinearDist_0.1", "LogitProbPoissonDist_0.1"),
        ("LogitProbLinearDist_1.0", "LogitProbPoissonDist_1.0"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for idx, (m1, m2) in enumerate(model_pairs):
        ax = axes[idx]
        data = []
        labels = []

        for label, related_val in [("Unrelated", 0), ("Related", 1)]:
            for m in [m1, m2]:
                values = df.loc[df["Related"] == related_val, m].dropna().values
                data.append(values)
                labels.append(f"{MODELS[m]}\n{label}")

        ax.boxplot(data, patch_artist=True)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("Edge weight")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{SCENARIOS[scenario]}: effect of deterministic (Lin) vs stochastic (Pois)\ngenetic distances "
                 "on edge weight distributions")

    fig.tight_layout()
    save_figure(fig, out_path, ["png"])
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/generate_datasets.yaml")
    parser.add_argument("--defaults", default="../config/default_parameters.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))
    defaults_cfg = load_yaml(Path(args.defaults))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    figs_dir = figs_dir / "edge_eval"
    ensure_dirs(processed_dir, figs_dir, tabs_dir)

    rng_seed = int(deep_get(defaults_cfg, ["toit", "rng_seed"], 12345))
    params_cfg = deep_get(defaults_cfg, ["toit", "infectiousness_params"], {})
    evol_cfg = deep_get(defaults_cfg, ["toit", "evolution"], {})
    inference_cfg = deep_get(defaults_cfg, ["inference"], {})

    params = InfectiousnessParams(**params_cfg)
    toit = TOIT(
        params=params,
        rng_seed=rng_seed,
        subs_rate=float(evol_cfg["subs_rate"]),
        relax_rate=bool(evol_cfg["relax_rate"]),
        subs_rate_sigma=float(evol_cfg["subs_rate_sigma"]),
        gen_len=int(evol_cfg["gen_length"]),
    )

    scenarios = deep_get(scenarios_cfg, ["scenarios"], {})

    rows = []
    for scen in scenarios.keys():
        print(f">>> Evaluating: {scen}")
        sc_dir = processed_dir / f"scenario={scen}"
        pw_path = sc_dir / "pairwise.parquet"
        if not pw_path.exists():
            continue

        df = pd.read_parquet(pw_path)
        df = df[df["Sampled"]].copy()

        # Mechanistic probability
        df["MechProbLinearDist"] = estimate_linkage_probabilities(
            toit=toit,
            genetic_distance=df["LinearDist"].values,
            temporal_distance=df["TemporalDist"].values,
            intermediate_generations = tuple(inference_cfg["inter_generations"]),
            no_intermediates = int(inference_cfg["num_intermediates"]),
            num_simulations = int(inference_cfg["num_simulations"]),
        )

        df["MechProbPoissonDist"] = estimate_linkage_probabilities(
            toit=toit,
            genetic_distance=df["PoissonDist"].values,
            temporal_distance=df["TemporalDist"].values,
            intermediate_generations=tuple(inference_cfg["inter_generations"]),
            no_intermediates=int(inference_cfg["num_intermediates"]),
            num_simulations=int(inference_cfg["num_simulations"]),
        )

        # Logistic regression
        y = df["Related"].astype(int).values

        for p, dist_col in product((0.1, 1.0), ("LinearDist", "PoissonDist")):
            X = df[["TemporalDist", dist_col]].values
            col = f"LogitProb{dist_col}_{p}"
            try:
                clf = LogisticRegression(solver="lbfgs", max_iter=200)
                if p == 1.0:
                    clf.fit(X, y)
                    df[col] = clf.predict_proba(X)[:, 1]
                else:
                    X_tr, _, y_tr, _ = train_test_split(X, y, train_size=p, stratify=y, random_state=rng_seed)
                    clf.fit(X_tr, y_tr)
                    df[col] = clf.predict_proba(X)[:, 1]
            except Exception:
                df[col] = np.nan

        df["LinearDistScore"] = 1.0 / (df["LinearDist"] + 1.0)
        df["PoissonDistScore"] = 1.0 / (df["PoissonDist"] + 1.0)

        df.to_parquet(sc_dir / "pairwise_eval.parquet", index=False)

        models = [
            ("LinearDistScore", False),
            ("PoissonDistScore", False),
            ("MechProbLinearDist", True),
            ("MechProbPoissonDist", True),
            ("LogitProbLinearDist_0.1", True),
            ("LogitProbPoissonDist_0.1", True),
            ("LogitProbLinearDist_1.0", True),
            ("LogitProbPoissonDist_1.0", True),
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

        plot_weight_distributions(df, figs_dir / f"sm6_{scen}_boxplots", scenario=scen)

    out = pd.DataFrame(rows)
    out.to_csv(tabs_dir / "edge_eval_metrics.csv", index=False)
    print(f"Saved edge evaluation metrics to: {tabs_dir / 'edge_eval_metrics.csv'}")

if __name__ == "__main__":
    main()