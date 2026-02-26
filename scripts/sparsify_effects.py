#!/usr/bin/env python3
"""
scripts/sparsify_effects.py

Edge-, neighbourhood-, and computational-scaling diagnostics for graph sparsification.

Key questions (pre-community-detection):
  1) How much total edge weight ("probability mass") is retained?
  2) How many edges survive?
  3) How much does node strength (weighted degree) distort?
  4) How does the graph fragment (components / giant component)?
  5) How does runtime (and approximate Python-allocated memory) scale with retained edges,
     including Leiden runtime on an equivalent igraph representation?

Inputs
------
A *pairwise* DataFrame stored as parquet:
  - NodeA, NodeB
  - weight_col (e.g. "MechProbLinearDist")
  - Related (optional; not used here)

Outputs
-------
tables/supplementary/
  - sparsify_edge_retention.parquet
  - sparsify_node_strength_distortion.parquet
  - sparsify_components_summary.parquet

Notes
-----
- NetworkX graph construction is used for component diagnostics (connected components).
- igraph is used for Leiden timing diagnostics (more representative for your pipeline).
- For fair scaling comparisons across thresholds, we keep a fixed vertex set (ref_nodes),
  adding isolates back in after sparsification.

Dependencies
------------
pandas, numpy, matplotlib, networkx, igraph, pyyaml
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
import gc
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig

from utils import *

# -----------------------------
# Timing helpers
# -----------------------------

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


# -----------------------------
# Sparsification
# -----------------------------

def sparsify_edges(df: pd.DataFrame, min_w: float, weight_col: str) -> pd.DataFrame:
    """Filter edges by min edge weight."""
    min_w = float(min_w)
    if min_w <= 0:
        return df
    # no copy: downstream functions do not mutate df_w
    return df.loc[df[weight_col] >= min_w]


def total_edge_weight(df: pd.DataFrame, weight_col: str) -> float:
    return float(df[weight_col].sum()) if len(df) else 0.0


# -----------------------------
# NetworkX diagnostics
# -----------------------------

def build_nx_graph(df: pd.DataFrame, weight_col: str) -> nx.Graph:
    """Undirected weighted graph with edge attribute 'weight'."""
    g = nx.Graph()
    if len(df):
        edges = df[["NodeA", "NodeB", weight_col]].to_records(index=False).tolist()
        g.add_weighted_edges_from(edges, weight="weight")
    return g


def graph_summary(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else np.nan

    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": np.nan,
            "n_components": 0,
            "giant_component_size": 0,
            "giant_component_frac": np.nan,
        }

    comps = list(nx.connected_components(g))
    sizes = np.array([len(c) for c in comps], dtype=int) if comps else np.array([], dtype=int)
    giant = int(sizes.max()) if sizes.size else 0
    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(density),
        "n_components": int(len(comps)),
        "giant_component_size": int(giant),
        "giant_component_frac": float(giant / n) if n > 0 else np.nan,
    }


# -----------------------------
# Node strength distortion
# -----------------------------

def node_strengths_from_edges(
    df: pd.DataFrame,
    weight_col: str,
    nodes: Optional[pd.Index] = None
) -> pd.Series:
    """Weighted degree ("strength") per node from edge list."""
    if len(df) == 0:
        s = pd.Series(dtype=float)
    else:
        a = df[["NodeA", weight_col]].rename(columns={"NodeA": "node", weight_col: "w"})
        b = df[["NodeB", weight_col]].rename(columns={"NodeB": "node", weight_col: "w"})
        s = pd.concat([a, b], ignore_index=True).groupby("node")["w"].sum()

    if nodes is not None:
        s = s.reindex(nodes, fill_value=0.0)
    return s


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
) -> Dict[str, Any]:
    # --- graph build ---
    g, t_build = timed(build_igraph_from_pairwise, df_w, weight_col, vertices_str)

    # --- Leiden ---
    # python-igraph Leiden API uses:
    #   weights=..., resolution_parameter=..., objective_function=..., n_iterations=..., seed=...
    def _run_leiden():
        return g.community_leiden(
            weights=weight_col,
            resolution=float(gamma),
            n_iterations=-1,  # until convergence
        )

    part, t_leiden = timed(_run_leiden)
    clusters = [c for c in part if len(c) >= 2]

    return {
        "ig_n_vertices": int(g.vcount()),
        "ig_n_edges": int(g.ecount()),
        "t_igraph_build_s": float(t_build),
        "t_leiden_s": float(t_leiden),
        "n_clusters": int(len(clusters)),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    parser.add_argument("--scenario", default="baseline", help="Scenario subdir name, e.g. baseline")
    parser.add_argument("--gamma", type=float, default=0.5, help="Leiden resolution_parameter for timing diagnostics")

    # Plot options
    parser.add_argument("--log-runtime", action="store_true", help="Use log scale for runtime plots")

    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    ensure_dirs(tabs_dir, processed_dir)

    min_ws = list(deep_get(clus_cfg, ["network", "min_edge_weights"], [0.0, 0.0001, 0.001, 0.01, 0.1]))
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["MechProbLinearDist"]))
    sc_dir = processed_dir / f"scenario={args.scenario}"
    df = pd.read_parquet(sc_dir / "pairwise_eval.parquet")

    weight_col = weight_columns[0]

    # --- reference threshold: smallest minw supplied ---
    ref_minw = float(min(min_ws))
    df_ref = sparsify_edges(df, ref_minw, weight_col)

    # Reference node set (for fair comparisons): nodes present at reference threshold
    ref_nodes = pd.Index(pd.unique(df_ref[["NodeA", "NodeB"]].values.ravel()))
    ref_nodes_str = ref_nodes.astype(str)

    # Reference strengths, total weight, and edge count
    ref_strength = node_strengths_from_edges(df_ref, weight_col, nodes=ref_nodes)
    w_ref = total_edge_weight(df_ref, weight_col)
    m_ref = int(len(df_ref)) if len(df_ref) else 0

    # Start memory tracking (Python allocations)
    tracemalloc.start()

    retention_rows: List[Dict[str, Any]] = []
    strength_rows: List[pd.DataFrame] = []
    components_rows: List[Dict[str, Any]] = []

    for minw in min_ws:
        tracemalloc.reset_peak()
        minw = float(minw)

        # 1) sparsify
        df_w, t_sparsify = timed(sparsify_edges, df, minw, weight_col)

        # 2) strengths
        s_w, t_strength = timed(node_strengths_from_edges, df_w, weight_col, ref_nodes)

        # 3) NX graph build
        g_nx, t_build_nx = timed(build_nx_graph, df_w, weight_col)
        # Keep isolates for consistent component counts across thresholds
        g_nx.add_nodes_from(ref_nodes.tolist())

        # 4) components summary
        comp, t_summary = timed(graph_summary, g_nx)

        # 5) memory snapshot
        gc.collect()
        _, peak_mem = tracemalloc.get_traced_memory()
        peak_mb = peak_mem / (1024 ** 2)

        # Retention metrics
        w_w = total_edge_weight(df_w, weight_col)
        m_w = int(len(df_w))

        row: Dict[str, Any] = {
            "min_edge_weight": float(minw),
            "n_edges": int(m_w),
            "edge_retention_frac": float(m_w / m_ref) if m_ref > 0 else np.nan,
            "total_weight": float(w_w),
            "weight_retention_frac": float(w_w / w_ref) if w_ref > 0 else np.nan,

            # Scaling (NX diagnostics)
            "t_sparsify_s": float(t_sparsify),
            "t_strength_s": float(t_strength),
            "t_build_nx_s": float(t_build_nx),
            "t_components_s": float(t_summary),
            "t_total_nx_s": float(t_sparsify + t_strength + t_build_nx + t_summary),
            "peak_tracemalloc_mb": float(peak_mb),
        }

        leiden_row = timed_igraph_and_leiden(
            df_w=df_w,
            weight_col=weight_col,
            vertices_str=ref_nodes_str,
            gamma=args.gamma
        )
        row.update(leiden_row)

        retention_rows.append(row)

        # Strength distortion
        eps = 1e-12
        log_ratio = np.log((s_w + eps) / (ref_strength + eps))
        abs_log_ratio = np.abs(log_ratio)

        strength_rows.append(pd.DataFrame({
            "min_edge_weight": float(minw),
            "case_id": ref_nodes.to_numpy(),
            "strength_ref": ref_strength.values,
            "strength_cur": s_w.values,
            "log_ratio": np.asarray(log_ratio),
            "abs_log_ratio": np.asarray(abs_log_ratio),
        }))

        # Component summary
        components_rows.append({**comp, "min_edge_weight": float(minw)})

    retention_df = pd.DataFrame(retention_rows).sort_values("min_edge_weight").reset_index(drop=True)
    strength_df = pd.concat(strength_rows, ignore_index=True)
    components_df = pd.DataFrame(components_rows).sort_values("min_edge_weight").reset_index(drop=True)

    # ---- Save tables ----
    retention_df.to_parquet(tabs_dir / "sparsify_edge_retention.parquet", index=False)
    strength_df.to_parquet(tabs_dir / "sparsify_node_strength_distortion.parquet", index=False)
    components_df.to_parquet(tabs_dir / "sparsify_components_summary.parquet", index=False)

    print(f"Saved tables to: {tabs_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
