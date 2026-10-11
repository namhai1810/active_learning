"""Read-only checkpoint identity and local-content fingerprinting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def hash_local_path(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.exists():
        return None
    digest = hashlib.sha256()
    paths = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in paths:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def checkpoint_descriptor(model: Mapping[str, Any]) -> dict[str, Any]:
    name = str(model.get("name_or_path", "Qwen/Qwen3-1.7B"))
    adapter = model.get("adapter_path")
    return {
        "name_or_path": name,
        "adapter_path": adapter,
        "model_local_sha256": hash_local_path(name),
        "adapter_local_sha256": hash_local_path(adapter),
    }
