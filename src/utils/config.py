"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return loaded


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_extends(
    path: str | Path, _seen: set[Path] | None = None
) -> dict[str, Any]:
    """Load YAML and recursively merge an optional relative ``extends`` file."""
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"Config inheritance cycle detected at {config_path}")
    seen.add(config_path)

    config = load_yaml(config_path)
    parent_value = config.pop("extends", None)
    if parent_value is None:
        return config
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise ValueError("extends must be a non-empty path string")
    parent_path = (config_path.parent / parent_value).resolve()
    parent = load_yaml_with_extends(parent_path, seen)
    return _deep_merge(parent, config)


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be a mapping")
    return value
