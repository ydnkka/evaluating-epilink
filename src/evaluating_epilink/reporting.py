"""Aggregate stage manifests into a compact pipeline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import config_value, ensure_directories, load_yaml, resolve_path
from .execution import utc_now_iso

PIPELINE_STAGE_ORDER = [
    "characterisation",
    "scovmod",
    "synthetic_data",
    "pairwise_benchmark",
    "synthetic_clustering",
    "clustering_evaluation",
    "sparsify_analysis",
    "temporal_stability",
    "boston",
    "figures",
]


def load_stage_manifests(manifests_dir: Path) -> dict[str, dict[str, Any]]:
    """Load stage manifests keyed by stage name."""

    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        if manifest_path.name == "pipeline_report.json":
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_name = str(manifest.get("stage_name", manifest_path.stem))
        manifests[stage_name] = manifest
    return manifests


def build_pipeline_report(
    manifests: dict[str, dict[str, Any]],
    *,
    expected_stages: list[str] | None = None,
) -> dict[str, Any]:
    """Build a normalized pipeline report from per-stage manifests."""

    ordered_stage_names = list(expected_stages or PIPELINE_STAGE_ORDER)
    remaining_stage_names = sorted(set(manifests) - set(ordered_stage_names))
    stage_rows: list[dict[str, Any]] = []
    total_duration_seconds = 0.0

    for stage_name in ordered_stage_names + remaining_stage_names:
        manifest = manifests.get(stage_name)
        if manifest is None:
            stage_rows.append(
                {
                    "stage_name": stage_name,
                    "status": "missing",
                    "duration_seconds": None,
                    "created_at_utc": None,
                    "output_keys": [],
                }
            )
            continue

        execution = manifest.get("execution", {})
        duration_seconds = execution.get("duration_seconds")
        if isinstance(duration_seconds, (int, float)):
            total_duration_seconds += float(duration_seconds)

        stage_rows.append(
            {
                "stage_name": stage_name,
                "status": manifest.get("status", "unknown"),
                "duration_seconds": duration_seconds,
                "created_at_utc": manifest.get("created_at_utc"),
                "output_keys": sorted((manifest.get("outputs") or {}).keys()),
                "summary": manifest.get("summary", {}),
            }
        )

    completed_stages = sum(1 for stage_row in stage_rows if stage_row["status"] == "completed")
    missing_stages = sum(1 for stage_row in stage_rows if stage_row["status"] == "missing")
    return {
        "generated_at_utc": utc_now_iso(),
        "num_expected_stages": len(ordered_stage_names),
        "num_completed_stages": completed_stages,
        "num_missing_stages": missing_stages,
        "total_duration_seconds": round(total_duration_seconds, 3),
        "stages": stage_rows,
    }


def render_pipeline_report_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary of the pipeline report."""

    lines = [
        "# Pipeline Report",
        "",
        f"- Generated at (UTC): {report['generated_at_utc']}",
        f"- Expected stages: {report['num_expected_stages']}",
        f"- Completed stages: {report['num_completed_stages']}",
        f"- Missing stages: {report['num_missing_stages']}",
        f"- Total tracked duration (s): {report['total_duration_seconds']}",
        "",
        "| Stage | Status | Duration (s) | Outputs |",
        "| --- | --- | ---: | --- |",
    ]
    for stage in report["stages"]:
        duration = stage["duration_seconds"]
        duration_text = f"{float(duration):.3f}" if isinstance(duration, (int, float)) else ""
        output_keys = ", ".join(stage["output_keys"])
        lines.append(
            f"| {stage['stage_name']} | {stage['status']} | {duration_text} | {output_keys} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_pipeline_report(
    manifests_dir: Path,
    *,
    expected_stages: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write JSON and Markdown pipeline reports beside the stage manifests."""

    manifests = load_stage_manifests(manifests_dir)
    report = build_pipeline_report(manifests, expected_stages=expected_stages)

    json_path = manifests_dir / "pipeline_report.json"
    markdown_path = manifests_dir / "pipeline_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_pipeline_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    """Generate a compact pipeline execution report from stage manifests."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config)
    manifests_dir = resolve_path(config_value(base_config, ["paths", "results", "manifests"]))
    ensure_directories(manifests_dir)

    json_path, markdown_path = write_pipeline_report(manifests_dir)
    print(f"[report] Wrote {json_path}")
    print(f"[report] Wrote {markdown_path}")


if __name__ == "__main__":
    main()
