#!/usr/bin/env python3
"""
scripts/sparsify_effects.py

Edge-retention and pipeline-speed diagnostics for graph sparsification.

Key questions:
  1) How much total edge weight ("probability mass") is retained?
  2) How many edges survive?
  3) How does the pipeline runtime scale (network constriction + community detection)?

Config
------
config/paths.yaml
config/clustering.yaml

Outputs
-------
tables/supplementary/
  - sparsify_edge_retention.parquet

Notes
-----.
- For fair scaling comparisons across thresholds, we keep a fixed vertex set (ref_nodes),
  adding isolates back in after sparsification.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import pandas as pd
import igraph as ig

from utils import *


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt

def sparsify_edges(df: pd.DataFrame, min_w: float, weight_col: str) -> pd.DataFrame:
    """Filter edges by min edge weight."""
    min_w = float(min_w)
    if min_w <= 0:
        return df
    return df.loc[df[weight_col] >= min_w]

def total_edge_weight(df: pd.DataFrame, weight_col: str) -> float:
    return float(df[weight_col].sum()) if len(df) else 0.0


# -----------------------------
# igraph + Leiden timing
# -----------------------------

def _as_str_id_series(x: pd.Series) -> pd.Series:
    # igraph vertex "name" is typically string; converting ensures consistent set comparisons
    return x.astype(str)


def build_igraph_from_pairwise(
    df: pd.DataFrame,
    weight_col: str,
    vertices: Optional[pd.Index] = None,
) -> ig.Graph:
    """
    Build an undirected weighted igraph from a pairwise table with columns:
      - NodeA, NodeB
      - weight_col

    vertices:
      Optional fixed vertex set to retain isolates (recommended for fair scaling).
      If provided, should be the *string* case_id values.
    """

    # Ensure consistent id types for igraph
    a = _as_str_id_series(df["NodeA"])
    b = _as_str_id_series(df["NodeB"])
    w = df[weight_col].to_numpy()

    edges = list(zip(a.tolist(), b.tolist(), w.tolist()))

    g = ig.Graph.TupleList(
        edges=edges,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs=[weight_col],
    )

    if vertices is not None:
        current = set(g.vs["case_id"])
        missing = set(vertices) - current
        if missing:
            miss = list(missing)
            g.add_vertices(miss)
            # Ensure attribute exists for all vertices
            if "case_id" not in g.vs.attributes():
                g.vs["case_id"] = g.vs["name"]  # fallback
            # Assign case_id for newly added vertices
            g.vs.select(case_id_in=miss)["case_id"] = miss

    return g


def timed_igraph_and_leiden(
    df_w: pd.DataFrame,
    weight_col: str,
    vertices_str: pd.Index,
    gamma: float,
) -> tuple[float, float]:

    g, t_build = timed(build_igraph_from_pairwise, df_w, weight_col, vertices_str)
    def _run_leiden():
        return g.community_leiden(
            weights=weight_col,
            resolution=float(gamma),
            n_iterations=-1,  # until convergence
        )

    _, t_leiden = timed(_run_leiden)

    return float(t_build), float(t_leiden)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    parser.add_argument("--scenario", default="baseline", help="Scenario subdir name, e.g. baseline")
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Leiden resolution_parameter for timing diagnostics"
    )

    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(
        deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic")
    )
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", ], "../tables"))
    tabs_dir = tabs_dir / "sparsify"
    ensure_dirs(tabs_dir)

    min_ws = list(deep_get(clus_cfg, ["network", "min_edge_weights"], [0.0, 0.0001, 0.001, 0.01, 0.1]))
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["ProbLinearDist"]))
    sc_dir = processed_dir / f"scenario={args.scenario}"
    df = pd.read_parquet(sc_dir / "pairwise.parquet")

    weight_col = weight_columns[0]

    # --- reference threshold: smallest minw supplied ---
    ref_minw = float(min(min_ws))
    df_ref = sparsify_edges(df, ref_minw, weight_col)

    # Reference node set (for fair comparisons): nodes present at reference threshold
    ref_nodes = pd.Index(pd.unique(df_ref[["NodeA", "NodeB"]].values.ravel()))
    ref_nodes_str = ref_nodes.astype(str)

    # Reference total weight and edge count
    w_ref = total_edge_weight(df_ref, weight_col)
    m_ref = int(len(df_ref)) if len(df_ref) else 0

    retention_rows: list[dict[str, Any]] = []

    for minw in min_ws:
        minw = float(minw)

        # 1) sparsify
        df_w, t_sparsify = timed(sparsify_edges, df, minw, weight_col)

        # Retention metrics
        w_w = total_edge_weight(df_w, weight_col)
        m_w = int(len(df_w))

        row: dict[str, Any] = {
            "min_edge_weight": float(minw),
            "edge_retention_frac": float(m_w / m_ref) if m_ref > 0 else np.nan,
            "weight_retention_frac": float(w_w / w_ref) if w_ref > 0 else np.nan,
        }

        t_build, t_leiden = timed_igraph_and_leiden(
            df_w=df_w,
            weight_col=weight_col,
            vertices_str=ref_nodes_str,
            gamma=args.gamma
        )
        # Pipeline = sparsify + igraph build + Leiden
        row["t_pipeline_s"] = float(t_sparsify + t_build + t_leiden)

        retention_rows.append(row)

    retention_df = pd.DataFrame(retention_rows).sort_values("min_edge_weight").reset_index(drop=True)

    # ---- Save tables ----
    retention_df.to_parquet(tabs_dir / "sparsify_edge_retention.parquet", index=False)
    print(f"Saved tables to: {tabs_dir}")


if __name__ == "__main__":
    main()
