#!/usr/bin/env python3
"""
scripts/generate_datasets.py

Generate synthetic datasets across scenarios defined by varying epidemiological parameters and sampling schemes.:

This script should run ONCE to produce datasets reused by:
  - scripts/pairwise_discrimination.py
  - scripts/run_clustering.py

Config
------
config/paths.yaml
config/generate_datasets.yaml

Outputs
-------
data/processed/synthetic/scenario=<name>/
  - pairwise.parquet
tables/supplementary/
  - scenario_summary.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
from numpy.random import default_rng
import pandas as pd

import networkx as nx

from epilink import (
    TOIT,
    MolecularClock,
    InfectiousnessParams,
    populate_epidemic_data,
    simulate_genomic_data,
    generate_pairwise_data,
)

from utils import *


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
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/generate_datasets.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))

    processed_dir = Path(
        deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic")
    )
    tabs_dir = Path(
        deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary")
    )
    tabs_dir = tabs_dir / "scovmod"
    ensure_dirs(tabs_dir)

    tree_path = Path(deep_get(scenarios_cfg, ["backbone", "tree_gml"],
                              ".../data/processed/synthetic/scovmod/scovmod_tree.gml"))
    rng_seed = int(deep_get(scenarios_cfg, ["backbone", "rng_seed"], 12345))
    gen_len = int(deep_get(scenarios_cfg, ["backbone", "gen_len"], 29903))

    base_tree = nx.read_gml(tree_path)

    scenarios = deep_get(scenarios_cfg, ["scenarios"], None)
    if not isinstance(scenarios, dict) or len(scenarios) == 0:
        raise ValueError("generate_datasets.yaml must define a `scenarios:` mapping.")

    # Defaults are applied by copying the baseline scenario and updating with any scenario-specific patches.
    baseline = scenarios["baseline"]

    summary_rows = []

    for name, patch in scenarios.items():
        cfg = baseline.copy()
        if isinstance(patch, dict):
            cfg.update(patch)

        print(f"\n>>> Simulating scenario: {name} | {cfg.get('description','')}")
        sc_dir = processed_dir / f"scenario={name}"
        ensure_dirs(sc_dir)

        rng = default_rng(cfg.rng_seed)

        params = InfectiousnessParams(
            incubation_shape=float(cfg["incubation_shape"]),
            incubation_scale=float(cfg["incubation_scale"]),
            latent_shape=float(cfg["latent_shape"]),
            symptomatic_rate=float(cfg["symptomatic_rate"]),
            symptomatic_shape=float(cfg["symptomatic_shape"]),
            rel_presymptomatic_infectiousness=float(cfg["rel_presymptomatic_infectiousness"]),
        )

        toit = TOIT(params=params,rng=rng)

        clock = MolecularClock(
            subs_rate=float(cfg["subs_rate"]),
            relax_rate=bool(cfg["relax_rate"]),
            subs_rate_sigma=float(cfg["subs_rate_sigma"]),
            gen_len=gen_len,
            rng=rng
        )

        populated_tree = populate_epidemic_data(
            toit=toit,
            tree=base_tree,
            prop_sampled=float(cfg["prop_sampled"]),
            sampling_shape=float(cfg["sampling_shape"]),
            sampling_scale=float(cfg["sampling_scale"]),
        )

        gen_results = simulate_genomic_data(
            clock=clock,
            tree=populated_tree
        )

        pairwise = generate_pairwise_data(
            packed_genomic_data=gen_results["packed"],
            tree=populated_tree,
        )

        pairwise.to_parquet(sc_dir / "pairwise.parquet", index=False)

        df = pairwise.copy()
        sampled = df[df["Sampled"]].copy()

        metrics = ["TemporalDist", "PoissonDist", "LinearDist"]

        params_row = {
            "incubation_shape": float(cfg["incubation_shape"]),
            "incubation_scale": float(cfg["incubation_scale"]),
            "latent_shape": float(cfg["latent_shape"]),
            "symptomatic_rate": float(cfg["symptomatic_rate"]),
            "symptomatic_shape": float(cfg["symptomatic_shape"]),
            "rel_presymptomatic_infectiousness": float(cfg["rel_presymptomatic_infectiousness"]),
            "prop_sampled": float(cfg["prop_sampled"]),
            "sampling_shape": float(cfg["sampling_shape"]),
            "sampling_scale": float(cfg["sampling_scale"]),
            "subs_rate": float(cfg["subs_rate"]),
            "relax_rate": bool(cfg["relax_rate"]),
            "subs_rate_sigma": float(cfg["subs_rate_sigma"]),
        }

        dist_summary = summarise_data(
            df=sampled,
            scenario=name,
            description=cfg.get("description", ""),
            metrics=metrics,
            params_row=params_row,
        )

        summary_rows.append(dist_summary)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df.to_parquet(tabs_dir / "scenario_summary.parquet", index=False)

    print(f"\nSaved datasets to: {processed_dir}")
    print(f"Saved scenario summary to: {tabs_dir / 'scenario_summary.parquet'}")


if __name__ == "__main__":
    main()
