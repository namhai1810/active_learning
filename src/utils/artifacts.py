"""Strict readers for research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return value


def read_jsonl(
    path: str | Path, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    resolved = Path(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} in {resolved} is not an object")
            records.append(value)
    if not records and not allow_empty:
        raise ValueError(f"No records found in {resolved}")
    return records
