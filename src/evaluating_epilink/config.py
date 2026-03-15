"""Configuration helpers for reproducible analysis workflows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root for the companion analysis repo."""

    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the repository root."""

    path = Path(path_like)
    if path.is_absolute():
        return path
    base = project_root() if root is None else root
    return (base / path).resolve()


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty mapping when the document is empty."""

    resolved_path = resolve_path(path_like)
    with resolved_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping in {resolved_path}, found {type(loaded).__name__}.")
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_merged_config(*path_likes: str | Path) -> dict[str, Any]:
    """Load and recursively merge multiple YAML configuration files."""

    merged: dict[str, Any] = {}
    for path_like in path_likes:
        merged = deep_merge(merged, load_yaml(path_like))
    return merged


def config_value(config: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    """Safely read a nested configuration value."""

    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def ensure_directories(*path_likes: str | Path) -> None:
    """Create one or more directories if they do not already exist."""

    for path_like in path_likes:
        resolve_path(path_like).mkdir(parents=True, exist_ok=True)
