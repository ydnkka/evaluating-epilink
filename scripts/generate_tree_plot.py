#!/usr/bin/env python3
"""
scripts/generate_tree_plot.py

Generate supplementary figures from transmission-tree outputs.

Outputs
---------------------
figures/supplementary/scovmod/
  - OUT_PREFIX_tree_overview.(png|pdf)

Use ``--out-prefix`` to match the output prefix used by generate_tree.py.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import *


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_parquet(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_formats(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=int)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    arr = np.sort(arr)
    unique, first_idx = np.unique(arr, return_index=True)
    ccdf_vals = (arr.size - first_idx) / arr.size
    return unique, ccdf_vals


def tree_depths(tree: nx.DiGraph) -> np.ndarray:
    roots = [n for n, d in tree.in_degree(tree.nodes) if d == 0]
    if not roots:
        raise ValueError("Tree has no roots (in-degree 0).")
    depths: dict[object, int] = {}
    for root in roots:
        for node, dist in nx.single_source_shortest_path_length(tree, root).items():
            if node not in depths or dist < depths[node]:
                depths[node] = dist
    return np.asarray(list(depths.values()), dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--out-prefix", default="scovmod_tree")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    tabs_dir = Path(
        deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary")
    )
    tabs_dir = tabs_dir / "scovmod"

    figs_dir = Path(
        deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary")
    )
    figs_dir = figs_dir / "scovmod"
    ensure_dirs(figs_dir)

    processed_dir = Path(
        deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic")
    )
    processed_dir = processed_dir / "scovmod"

    out_prefix = args.out_prefix

    def table_path(stem: str) -> Path:
        return tabs_dir / f"{out_prefix}_{stem}.parquet"

    def figure_base(stem: str) -> Path:
        return figs_dir / f"{out_prefix}_{stem}"

    formats = parse_formats(args.formats)
    set_seaborn_paper_context()

    summary_df = read_table(table_path("summary"))
    degree_df = read_table(table_path("degree_distributions"))
    component_df = read_table(table_path("component_sizes"))

    gml_path = processed_dir / f"{out_prefix}.gml"
    if not gml_path.exists():
        raise FileNotFoundError(f"Missing GML tree: {gml_path}")
    tree = nx.read_gml(gml_path)
    tree_out_deg = np.array([d for _, d in tree.out_degree(tree.nodes)], dtype=int)
    tree_depth = tree_depths(tree)

    heterogeneity_path = processed_dir / f"{out_prefix}_tree_heterogeneity.json"
    heterogeneity = read_json(heterogeneity_path)

    summary_final = summary_df.loc[summary_df["label"] == "final_tree"]
    summary_final = summary_final.iloc[0] if not summary_final.empty else None

    palette = {
        "raw": "#4c78a8",
        "cleaned": "#59a14f",
        "selected_component": "#f28e2b",
        "final_tree": "#b07aa1",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # A) Component size distribution (raw)
    comp_sizes = component_df["component_size"].to_numpy(dtype=int)
    x_comp, y_comp = ccdf(comp_sizes)
    axes[0, 0].plot(x_comp, y_comp, color=palette["raw"], lw=2)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Component size (nodes)")
    axes[0, 0].set_ylabel("CCDF")
    axes[0, 0].set_title("Component size distribution")

    # B) Out-degree CCDF across construction steps
    for label in ["raw", "cleaned", "selected_component"]:
        subset = degree_df.loc[(degree_df["graph"] == label) & (degree_df["degree_type"] == "out"), "value"]
        vals = subset.to_numpy(dtype=int)
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        x, y = ccdf(vals)
        axes[0, 1].plot(x, y, lw=2, color=palette[label], label=label)

    tree_nonzero = tree_out_deg[tree_out_deg > 0]
    if tree_nonzero.size > 0:
        x_tree, y_tree = ccdf(tree_nonzero)
        axes[0, 1].plot(x_tree, y_tree, lw=2, color=palette["final_tree"], label="final_tree")

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Out-degree (offspring)")
    axes[0, 1].set_ylabel("CCDF")
    axes[0, 1].set_title("Out-degree tail")
    axes[0, 1].legend(frameon=False)

    # C) Depth distribution
    depth_counts = pd.Series(tree_depth).value_counts().sort_index()
    axes[1, 0].bar(depth_counts.index, depth_counts.values, color="#2ca02c", width=0.9)
    axes[1, 0].set_xlabel("Depth from root")
    axes[1, 0].set_ylabel("Nodes")
    axes[1, 0].set_title("Tree depth profile")
    axes[1, 0].set_yscale("log")

    if summary_final is not None:
        text = (
            f"n={int(summary_final['n_nodes']):,}\n"
            f"edges={int(summary_final['n_edges']):,}\n"
            f"max out-degree={int(summary_final['max_out_degree'])}\n"
            f"mean out-degree={summary_final['mean_out_degree']:.2f}"
        )
        axes[1, 0].text(
            0.98,
            0.98,
            text,
            transform=axes[1, 0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.8},
        )

    # D) Offspring distribution with NB fit
    x_max = int(np.quantile(tree_out_deg, 0.99))
    x_max = max(20, min(x_max, int(tree_out_deg.max())))
    bins = np.arange(-0.5, x_max + 1.5, 1.0)
    sns.histplot(
        tree_out_deg,
        bins=bins,
        stat="probability",
        color=palette["final_tree"],
        ax=axes[1, 1],
    )

    mean_rt = float(heterogeneity.get("meanRt", np.nan))
    disp_k = float(heterogeneity.get("disp_k", np.nan))
    if np.isfinite(mean_rt) and np.isfinite(disp_k) and disp_k > 0:
        x_vals = np.arange(0, x_max + 1, dtype=int)
        p = disp_k / (disp_k + mean_rt)
        pmf = stats.nbinom.pmf(x_vals, disp_k, p)
        pmf = np.clip(pmf, 1e-12, None)
        axes[1, 1].plot(x_vals, pmf, color="#1f1f1f", lw=2, label="NB fit")
        axes[1, 1].legend(frameon=False)

    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Offspring count (out-degree)")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_title("Offspring distribution")

    if heterogeneity:
        text = (
            f"R={mean_rt:.2f}\n"
            f"k={disp_k:.2f}\n"
            f"Zero={heterogeneity.get('pct_zero_transmitters', np.nan):.1f}%\n"
            f"80% by {heterogeneity.get('prop_80_percent_transmitters', np.nan) * 100:.1f}%"
        )
        axes[1, 1].text(
            0.98,
            0.98,
            text,
            transform=axes[1, 1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.8},
        )

    fig.tight_layout()
    save_figure(fig, figure_base("tree_overview"), formats)
    plt.close(fig)

    print(f"Saved supplementary figures to: {figs_dir}")


if __name__ == "__main__":
    main()
