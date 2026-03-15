"""Shared helpers for consistent workflow execution tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import platform
import socket
import sys
import time
from typing import Any, Mapping

from .manifests import write_manifest


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-serializable data."""

    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return str(value)


@dataclass(frozen=True)
class StageRun:
    """Structured metadata for a single workflow stage execution."""

    stage_name: str
    cli_args: dict[str, Any]
    started_at_utc: str
    started_monotonic: float


def start_stage_run(stage_name: str, *, cli_args: Mapping[str, Any] | None = None) -> StageRun:
    """Emit a standard start message and capture runtime metadata."""

    stage_run = StageRun(
        stage_name=stage_name,
        cli_args=dict(cli_args or {}),
        started_at_utc=utc_now_iso(),
        started_monotonic=time.perf_counter(),
    )
    print(f"[{stage_name}] Starting")
    return stage_run


def finish_stage_run(
    stage_run: StageRun,
    manifest_path: str | Path,
    *,
    config: dict[str, Any],
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a standardized manifest and emit a standard completion message."""

    finished_at_utc = utc_now_iso()
    duration_seconds = round(time.perf_counter() - stage_run.started_monotonic, 3)

    payload: dict[str, Any] = {
        "status": "completed",
        "execution": {
            "started_at_utc": stage_run.started_at_utc,
            "finished_at_utc": finished_at_utc,
            "duration_seconds": duration_seconds,
            "command": sys.argv,
            "cli_args": _json_safe(stage_run.cli_args),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "working_directory": str(Path.cwd()),
        },
    }
    if inputs:
        payload["inputs"] = _json_safe(inputs)
    if outputs:
        payload["outputs"] = _json_safe(outputs)
    if summary:
        payload["summary"] = _json_safe(summary)
    if extra_metadata:
        payload.update(_json_safe(extra_metadata))

    write_manifest(
        manifest_path,
        stage_name=stage_run.stage_name,
        config=config,
        extra_metadata=payload,
    )
    print(f"[{stage_run.stage_name}] Completed in {duration_seconds:.3f}s")
