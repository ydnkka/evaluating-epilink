from __future__ import annotations

import json
from pathlib import Path

from evaluating_epilink.execution import finish_stage_run, start_stage_run


def test_finish_stage_run_writes_standardized_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "stage.json"
    stage_run = start_stage_run("demo_stage", cli_args={"base_config": "configs/base.yaml"})

    finish_stage_run(
        stage_run,
        manifest_path,
        config={"project": {"rng_seed": 1}},
        inputs={"base_config": "configs/base.yaml"},
        outputs={"results_dir": tmp_path / "results"},
        summary={"num_rows": 3},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["stage_name"] == "demo_stage"
    assert manifest["status"] == "completed"
    assert manifest["inputs"]["base_config"] == "configs/base.yaml"
    assert manifest["summary"]["num_rows"] == 3
    assert manifest["outputs"]["results_dir"] == str((tmp_path / "results").resolve())
    assert manifest["execution"]["command"]
    assert manifest["execution"]["duration_seconds"] >= 0.0
