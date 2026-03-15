"""Manifest helpers for reproducible workflow runs."""

from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import project_root, resolve_path


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def config_digest(config: dict[str, Any]) -> str:
    """Create a short stable digest for a merged configuration."""

    encoded = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def write_manifest(
    manifest_path: str | Path,
    *,
    stage_name: str,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write a small machine-readable manifest for a workflow stage."""

    repository_root = project_root()
    resolved_manifest_path = resolve_path(manifest_path)
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "stage_name": stage_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "git_revision": _git_revision(repository_root),
        "config_digest": config_digest(config),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    resolved_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
