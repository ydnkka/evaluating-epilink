"""Render manuscript and supplementary figures from workflow outputs."""

from __future__ import annotations

import argparse

from ..config import config_value, ensure_directories, load_merged_config, resolve_path
from ..execution import finish_stage_run, start_stage_run
from .boston import render_boston
from .characterisation import render_model_characterisation
from .common import set_plot_theme
from .scovmod import render_tree_figure
from .synthetic import render_discrimination_and_clustering
from .temporal import render_temporal_stability


def main() -> None:
    """Render all manuscript figures from workflow outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--study-config", default="configs/studies/figures.yaml")
    args = parser.parse_args()
    stage_run = start_stage_run("figures", cli_args=vars(args))

    config = load_merged_config(args.base_config, args.study_config)
    figure_formats = tuple(str(value) for value in config_value(config, ["render", "formats"], ["pdf"]))
    font_scale = float(config_value(config, ["render", "font_scale"], 1.5))

    figures_dir = resolve_path(config_value(config, ["paths", "results", "figures"]))
    processed_synthetic_dir = resolve_path(config_value(config, ["paths", "data", "processed", "synthetic"]))
    results_root = figures_dir.parent
    ensure_directories(figures_dir)
    set_plot_theme(font_scale=font_scale)

    render_model_characterisation(results_root, figures_dir, figure_formats)
    render_tree_figure(results_root, processed_synthetic_dir, figures_dir, figure_formats)
    render_discrimination_and_clustering(results_root, figures_dir, figure_formats)
    render_temporal_stability(results_root, figures_dir, figure_formats)
    render_boston(results_root, figures_dir, figure_formats)

    finish_stage_run(
        stage_run,
        resolve_path(config_value(config, ["paths", "results", "manifests"])) / "figures.json",
        config=config,
        inputs={
            "base_config": resolve_path(args.base_config),
            "study_config": resolve_path(args.study_config),
            "results_root": str(results_root),
            "processed_synthetic_dir": str(processed_synthetic_dir),
        },
        outputs={"figures_dir": str(figures_dir)},
        summary={
            "formats": list(figure_formats),
            "num_figure_groups": 5,
        },
    )


if __name__ == "__main__":
    main()
