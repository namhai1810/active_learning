"""Fail-fast continuity checks for PEFT adapters used in closed loops."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def _canonical_key(value: str) -> str:
    # PEFT's in-memory state may include the active adapter name while the
    # serialized safetensors state omits it.
    return value.replace(".default.", ".")


def _canonical_state(value: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, tensor in value.items():
        canonical = _canonical_key(str(key))
        if canonical in state:
            raise ValueError(f"Duplicate canonical adapter tensor: {canonical}")
        state[canonical] = tensor
    return state


def _tensor_digest(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key, tensor in sorted(_canonical_state(state).items()):
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_artifact_sha256(adapter_dir: str | Path) -> str:
    """Hash the adapter config and weights as one immutable checkpoint ID."""
    directory = Path(adapter_dir)
    files = [
        path
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
        if (path := directory / name).is_file()
    ]
    if not files or not any(path.name.startswith("adapter_model.") for path in files):
        raise FileNotFoundError(f"No serialized adapter found beneath {directory}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_adapter_state(adapter_dir: str | Path) -> dict[str, Any]:
    directory = Path(adapter_dir)
    safe_path = directory / "adapter_model.safetensors"
    if safe_path.is_file():
        from safetensors.torch import load_file

        return dict(load_file(safe_path, device="cpu"))
    binary_path = directory / "adapter_model.bin"
    if not binary_path.is_file():
        raise FileNotFoundError(f"No adapter weights found beneath {directory}")
    import torch

    try:
        value = torch.load(binary_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        value = torch.load(binary_path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("Serialized adapter state must be a mapping")
    return value


def compare_adapter_states(
    loaded: Mapping[str, Any],
    serialized: Mapping[str, Any],
    *,
    tolerance: float = 5e-4,
) -> dict[str, Any]:
    """Compare logical adapter states, allowing only dtype-rounding noise."""
    import torch

    left = _canonical_state(loaded)
    right = _canonical_state(serialized)
    missing = sorted(set(right) - set(left))
    extra = sorted(set(left) - set(right))
    shape_mismatches: list[str] = []
    maximum_difference = 0.0
    for key in sorted(set(left) & set(right)):
        if tuple(left[key].shape) != tuple(right[key].shape):
            shape_mismatches.append(key)
            continue
        difference = (
            left[key].detach().to(device="cpu", dtype=torch.float32)
            - right[key].detach().to(device="cpu", dtype=torch.float32)
        ).abs()
        if difference.numel():
            maximum_difference = max(maximum_difference, float(difference.max().item()))
    matched = (
        not missing
        and not extra
        and not shape_mismatches
        and maximum_difference <= tolerance
    )
    return {
        "matched": matched,
        "tensor_count": len(right),
        "missing_keys": missing,
        "extra_keys": extra,
        "shape_mismatches": shape_mismatches,
        "maximum_absolute_difference": maximum_difference,
        "tolerance": tolerance,
        "loaded_tensor_sha256": _tensor_digest(loaded),
        "serialized_tensor_sha256": _tensor_digest(serialized),
    }


def audit_loaded_adapter(model: Any, adapter_dir: str | Path) -> dict[str, Any]:
    """Assert that a PEFT model contains the exact requested parent adapter."""
    from peft import get_peft_model_state_dict

    serialized = _read_adapter_state(adapter_dir)
    loaded = get_peft_model_state_dict(model)
    audit = compare_adapter_states(loaded, serialized)
    audit["artifact_sha256"] = adapter_artifact_sha256(adapter_dir)
    audit["adapter_dir"] = str(Path(adapter_dir).resolve())
    if not audit["matched"]:
        raise RuntimeError(
            "Loaded PEFT adapter differs from its serialized parent checkpoint: "
            f"{audit}"
        )
    return audit
