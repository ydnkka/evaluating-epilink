from __future__ import annotations

from pathlib import Path

from evaluating_epilink.config import load_merged_config


def test_load_merged_config_merges_nested_mappings(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    override_config = tmp_path / "override.yaml"

    base_config.write_text(
        "model:\n  clock:\n    rate: 1\nproject:\n  rng_seed: 1\n",
        encoding="utf-8",
    )
    override_config.write_text(
        "model:\n  clock:\n    sigma: 2\nproject:\n  label: test\n",
        encoding="utf-8",
    )

    merged = load_merged_config(base_config, override_config)

    assert merged["model"]["clock"]["rate"] == 1
    assert merged["model"]["clock"]["sigma"] == 2
    assert merged["project"]["rng_seed"] == 1
    assert merged["project"]["label"] == "test"
