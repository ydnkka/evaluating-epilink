#!/usr/bin/env python3
"""
generate_tree.py

Reconstruct a (rooted, acyclic) transmission tree from SCoVMod-style outputs.

What this script does
---------------------
1) Parse ragged SCoVMod CSVs:
   - Infection history: (TimeStep, Location, [Ids])
   - Transmission events: (TimeStep, Location, [Ids]) for newly exposed individuals

2) Build a directed "raw" transmission network:
   - For each exposed person at (t, location), choose one infector at random from
     the infectious pool at the same (t, location), excluding self-loops.

3) Clean anomalies:
   - Resolve reinfections / multiple parents: keep only the earliest incoming edge
     (minimum timeStep) per node.

4) Select one connected component (introduction) near a target size.

5) Enforce a tree structure via Maximum Spanning Arborescence (MSA):
   - Use negative timeStep as a weight so earlier edges are preferred in cycles.

6) Save outputs:
   - Tree as GML (and optional edge list CSV)
   - Summary statistics (CSV)
   - Figures (component sizes, degree distributions, etc.)

Notes
-----
- This repo is for the paper; the modelling framework lives in `epilink`.
- The tree reconstruction is intended as a realistic synthetic backbone for experiments.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng

from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

from utils import *
from transmission_heterogeneity import heterogeneity


@dataclass
class TreeConfig:
    rng_seed: int
    infection_history_file: str
    transmission_events_file: str
    target_component_size: int
    save_formats: list[str]


@dataclass
class PathsConfig:
    scovmod_output_dir: Path
    processed_dir: Path
    figures_out_dir: Path
    tables_out_dir: Path


def parse_configs(
    paths_yaml: Path,
    scovmod_yaml: Path,
) -> tuple[PathsConfig, TreeConfig]:
    paths_cfg = load_yaml(paths_yaml)
    scovmod_cfg = load_yaml(scovmod_yaml)

    # Paths
    scovmod_output = Path(deep_get(paths_cfg, ["data", "raw", "scovmod_output"], "../data/raw/scovmod_output"))
    processed = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    figures_supp = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    tables_supp = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))

    paths = PathsConfig(
        scovmod_output_dir=scovmod_output,
        processed_dir=processed,
        figures_out_dir=figures_supp / "scovmod",
        tables_out_dir=tables_supp,
    )

    # Tree settings
    tree_cfg = TreeConfig(
        rng_seed=int(deep_get(scovmod_cfg, ["transmission_tree", "rng_seed"], 42)),
        infection_history_file=str(deep_get(scovmod_cfg, ["transmission_tree", "infection_history_file"],
                                            "InfectedIndividuals.1.csv")),
        transmission_events_file=str(deep_get(scovmod_cfg, ["transmission_tree", "transmission_events_file"],
                                              "TransmissionEvents.1.csv")),
        target_component_size=int(deep_get(scovmod_cfg, ["transmission_tree", "target_component_size"], 5000)),
        save_formats=list(deep_get(scovmod_cfg, ["save_formats"], ["png", "pdf"]))
    )

    return paths, tree_cfg


# -----------------------------
# SCoVMod parsing and tree building
# -----------------------------

def parse_scovmod_outputs(filepath: Path) -> pd.DataFrame:
    """
    Parse SCoVMod CSVs where columns 2 onwards represent a list of IDs.

    Why custom parsing:
    - SCoVMod outputs may place ragged lists across multiple CSV columns.
    - Standard pandas parsing struggles with these variable-length rows.

    Returns
    -------
    DataFrame with columns:
      - TimeStep (int)
      - Location (int)
      - Ids (List[int])
    """
    data = []

    with filepath.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        _ = next(reader, None)  # skip header

        for row in reader:
            if not row:
                continue

            time_step = int(row[0])
            location = int(row[1])

            # Join the remainder and evaluate as a Python list
            raw_list_str = ",".join(row[2:])
            ids = list(set(ast.literal_eval(raw_list_str)))  # de-duplicate

            data.append((time_step, location, ids))

    return pd.DataFrame(data, columns=["TimeStep", "Location", "Ids"])


def build_transmission_network(
    trans_events_df: pd.DataFrame,
    infect_hist_df: pd.DataFrame,
    rng_seed: int = 12345,
) -> nx.DiGraph:
    """
    Construct a directed transmission network from:
      - transmission events (who becomes exposed at time t in location L)
      - infection history (who is infectious at time t in location L)

    Core logic:
    - For each infectee in the transmission events row (t, L),
      sample one infector uniformly from the infectious pool at (t, L),
      excluding self.

    Output:
    - nx.DiGraph with edge attributes: timeStep, location
    """
    rng = default_rng(rng_seed)

    # Build fast lookup: (TimeStep, Location) -> List[IDs]
    infection_lookup = {(int(row.TimeStep), int(row.Location)): row.Ids for row in infect_hist_df.itertuples()}

    edges = []

    for row in trans_events_df.itertuples():
        t = int(row.TimeStep)
        loc = int(row.Location)
        exposed = row.Ids

        potential = infection_lookup.get((t, loc), [])
        if not potential:
            continue

        # Convert to numpy array once for faster filtering
        potential = np.asarray(potential, dtype=int)

        for infectee in exposed:
            infectee = int(infectee)
            valid = potential[potential != infectee]
            if valid.size == 0:
                continue

            infector = int(rng.choice(valid))
            edges.append((infector, infectee, {"timeStep": t, "location": loc}))

    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def remove_reinfections(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Resolve nodes with multiple parents by keeping the earliest incoming edge.

    Interpretation:
    - In-degree > 1 in raw reconstruction can reflect reinfections, ambiguity,
      or multiple plausible infectors in the same time/location.
    - For a single-outbreak synthetic backbone, we retain only the earliest
      inferred event as a pragmatic simplification.
    """
    cleaned = graph.copy()
    nodes_multi = [n for n, d in cleaned.in_degree(cleaned.nodes) if d > 1]

    for node in nodes_multi:
        in_edges = list(cleaned.in_edges(node, data=True))
        in_edges_sorted = sorted(in_edges, key=lambda x: x[2].get("timeStep", np.inf))
        # Keep earliest, remove the rest
        for u, v, _ in in_edges_sorted[1:]:
            cleaned.remove_edge(u, v)

    return cleaned


def select_target_component(graph: nx.DiGraph, target_size: int) -> nx.DiGraph:
    """
    Select a weakly-connected component with size closest to target_size.
    """
    comps = list(nx.weakly_connected_components(graph))
    comps.sort(key=len, reverse=True)
    selected = min(comps, key=lambda c: abs(len(c) - target_size))

    return nx.DiGraph(graph.subgraph(selected).copy())


def build_msa_tree(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute a Maximum Spanning Arborescence (MSA) to enforce a rooted, acyclic tree.

    Weights:
    - weight = -timeStep so that earlier transmission edges are preferred
      when the algorithm must break cycles.

    Note:
    - networkx returns a branching/arborescence (directed spanning tree per component).
    - On a connected component, this yields one arborescence.
    """
    weighted = graph.copy()
    for u, v, d in weighted.edges(data=True):
        d["weight"] = -int(d.get("timeStep", 0))

    msa = maximum_spanning_arborescence(weighted, attr="weight", preserve_attrs=True)
    # Ensure type is DiGraph
    return nx.DiGraph(msa)


# -----------------------------
# Figures and summaries
# -----------------------------


def plot_component_size_distribution(graph: nx.DiGraph) -> plt.Figure:
    comps = list(nx.weakly_connected_components(graph))
    sizes = np.array([len(c) for c in comps], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Log-spaced bins help with heavy-tailed component distributions
    bin_edges = np.unique(np.logspace(0, np.log10(sizes.max()), num=min(100, sizes.max()), dtype=int))
    ax.hist(sizes, bins=bin_edges, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Component size (log10 scale)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of connected component sizes")
    ax.grid(True, alpha=0.3)
    return fig


def plot_degree_distributions(graph: nx.DiGraph, title: str) -> plt.Figure:
    in_deg = np.array([d for _, d in graph.in_degree(graph.nodes)], dtype=int)
    out_deg = np.array([d for _, d in graph.out_degree(graph.nodes)], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(in_deg, bins=np.arange(in_deg.max() + 2) - 0.5, density=True)
    axes[0].set_xlabel("In-degree")
    axes[0].set_ylabel("Density")
    axes[0].set_title("In-degree distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(out_deg, bins=np.arange(out_deg.max() + 2) - 0.5, density=True)
    axes[1].set_xlabel("Out-degree")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Out-degree distribution")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_edge_timestep_distribution(graph: nx.DiGraph, title: str) -> plt.Figure:
    ts = [d.get("timeStep", np.nan) for _, _, d in graph.edges(data=True)]
    ts = np.array([t for t in ts if not pd.isna(t)], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    if ts.size > 0:
        ax.hist(ts, bins=50, density=True)
    ax.set_xlabel("Edge timeStep")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def summarise_graph(graph: nx.DiGraph, label: str) -> Dict[str, Any]:
    in_degs = np.array([d for _, d in graph.in_degree(graph.nodes)], dtype=int)
    out_degs = np.array([d for _, d in graph.out_degree(graph.nodes)], dtype=int)

    summary = {
        "label": label,
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_components": int(nx.number_weakly_connected_components(graph)),
        "max_in_degree": int(in_degs.max()) if in_degs.size else 0,
        "max_out_degree": int(out_degs.max()) if out_degs.size else 0,
        "mean_out_degree": float(out_degs.mean()) if out_degs.size else 0.0,
        "prop_in_degree_gt1": float(np.mean(in_degs > 1)) if in_degs.size else 0.0,
        "prop_out_degree_ge10": float(np.mean(out_degs >= 10)) if out_degs.size else 0.0,
    }
    return summary


# -----------------------------
# Main execution
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transmission tree from SCoVMod outputs.")
    parser.add_argument("--paths", type=str, default="../config/paths.yaml", help="Path to config/paths.yaml")
    parser.add_argument("--simulation", type=str, default="../config/scovmod.yaml", help="Path to config/scovmod.yaml")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="scovmod_tree",
        help="Prefix used for saved outputs (GML, summaries, figures).",
    )
    args = parser.parse_args()

    paths, tree_cfg = parse_configs(
        paths_yaml=Path(args.paths),
        scovmod_yaml=Path(args.simulation),
    )

    ensure_dirs(paths.processed_dir, paths.figures_out_dir, paths.tables_out_dir)

    # Inputs (SCoVMod output dir + file names from config)
    infection_path = paths.scovmod_output_dir / tree_cfg.infection_history_file
    transmission_path = paths.scovmod_output_dir / tree_cfg.transmission_events_file

    if not infection_path.exists():
        raise FileNotFoundError(f"Missing infection history file: {infection_path}")
    if not transmission_path.exists():
        raise FileNotFoundError(f"Missing transmission events file: {transmission_path}")

    # Parse
    infect_df = parse_scovmod_outputs(infection_path)
    trans_df = parse_scovmod_outputs(transmission_path)

    print(f"Loaded {len(infect_df):,} infection-history records and {len(trans_df):,} transmission-event records.")

    # Build raw network
    raw_G = build_transmission_network(trans_df, infect_df, rng_seed=tree_cfg.rng_seed)
    print(f"Raw network: {raw_G.number_of_nodes():,} nodes, {raw_G.number_of_edges():,} edges.")

    # Figures on raw network
    fig = plot_component_size_distribution(raw_G)
    save_figure(fig, paths.figures_out_dir / "scovmod_tree_raw_component_sizes", tree_cfg.save_formats)
    plt.close(fig)

    fig = plot_degree_distributions(raw_G, title="Raw network degree distributions")
    save_figure(fig, paths.figures_out_dir / "scovmod_tree_raw_degrees", tree_cfg.save_formats)
    plt.close(fig)

    fig = plot_edge_timestep_distribution(raw_G, title="Raw network edge timeStep distribution")
    save_figure(fig, paths.figures_out_dir / "scovmod_tree_raw_edge_timesteps", tree_cfg.save_formats)
    plt.close(fig)

    # Clean multiple parents
    clean_G = remove_reinfections(raw_G)
    print(
        f"After resolving multiple parents: {clean_G.number_of_nodes():,} nodes, {clean_G.number_of_edges():,} edges "
        f"({raw_G.number_of_edges() - clean_G.number_of_edges():,} edges removed)."
    )

    fig = plot_degree_distributions(clean_G, title="Cleaned network degree distributions")
    save_figure(fig, paths.figures_out_dir / "scovmod_tree_clean_degrees", tree_cfg.save_formats)
    plt.close(fig)

    # Select target component
    comp_G = select_target_component(
        clean_G,
        target_size=tree_cfg.target_component_size
    )
    print(f"Selected component: {comp_G.number_of_nodes():,} nodes, {comp_G.number_of_edges():,} edges.")

    fig = plot_degree_distributions(comp_G, title="Selected component degree distributions")
    save_figure(fig, paths.figures_out_dir / "scovmod_tree_component_degrees", tree_cfg.save_formats)
    plt.close(fig)

    # Build tree (MSA)
    tree_G = build_msa_tree(comp_G)
    print(f"MSA tree: {tree_G.number_of_nodes():,} nodes, {tree_G.number_of_edges():,} edges.")

    # Save graph
    gml_path = paths.processed_dir / f"scovmod_tree.gml"
    nx.write_gml(tree_G, gml_path)
    print(f"Saved tree to: {gml_path}")

    # Save edge list for interoperability (handy for igraph / R)
    edges_csv = paths.processed_dir / f"scovmod_tree_edges.parquet"
    edge_rows = []
    for u, v, d in tree_G.edges(data=True):
        edge_rows.append({"infector": u, "infectee": v, "timeStep": d.get("timeStep"), "location": d.get("location")})
    pd.DataFrame(edge_rows).to_parquet(edges_csv, index=False)
    print(f"Saved edge list to: {edges_csv}")

    offspring_counts = np.array(list(dict(clean_G.out_degree(clean_G.nodes)).values()))
    results = heterogeneity(offspring_counts)
    heterogeneity_path = paths.processed_dir / f"scovmod_tree_heterogeneity.json"
    heterogeneity_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save summary stats
    summaries = [
        summarise_graph(raw_G, "raw"),
        summarise_graph(clean_G, "cleaned"),
        summarise_graph(comp_G, "selected_component"),
        summarise_graph(tree_G, "final_tree"),
    ]

    summary_df = pd.DataFrame(summaries)
    summary_csv = paths.tables_out_dir / f"scovmod_tree_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to: {summary_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
