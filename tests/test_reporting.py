from __future__ import annotations

from evaluating_epilink.reporting import build_pipeline_report, render_pipeline_report_markdown


def test_build_pipeline_report_marks_missing_expected_stages() -> None:
    manifests = {
        "characterisation": {
            "stage_name": "characterisation",
            "status": "completed",
            "execution": {"duration_seconds": 1.25},
            "outputs": {"results_dir": "results/characterise_epilink"},
        },
        "scovmod": {
            "stage_name": "scovmod",
            "status": "completed",
            "execution": {"duration_seconds": 2.75},
            "outputs": {"tree_path": "data/processed/synthetic/scovmod/scovmod_tree.gml"},
        },
    }

    report = build_pipeline_report(
        manifests,
        expected_stages=["characterisation", "scovmod", "figures"],
    )

    assert report["num_completed_stages"] == 2
    assert report["num_missing_stages"] == 1
    assert report["total_duration_seconds"] == 4.0
    assert report["stages"][2]["stage_name"] == "figures"
    assert report["stages"][2]["status"] == "missing"

    markdown = render_pipeline_report_markdown(report)

    assert "| characterisation | completed | 1.250 | results_dir |" in markdown
    assert "| figures | missing |  |  |" in markdown
