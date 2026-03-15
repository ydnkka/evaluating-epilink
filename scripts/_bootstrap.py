"""Ensure local package imports work from script entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    """Add the local ``src`` directory to ``sys.path`` and return the repo root."""

    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return repository_root
