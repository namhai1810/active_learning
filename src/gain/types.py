"""Serializable types for targeted pilot gain measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PilotGainObservation:
    hypothesis_id: str
    pilot_index: int
    pilot_seed: int
    starting_checkpoint: dict[str, Any]
    train_ids: list[str]
    eval_ids: list[str]
    training_config: dict[str, Any]
    risk_before: float
    risk_after: float
    gain: float
    train_loss: float | None
    wall_clock_seconds: float
    gpu_seconds: float | None
    adapter_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GainEstimate:
    hypothesis_id: str
    m: int
    delta_hat: float | None
    radius: float | None
    lower_bound: float | None
    upper_bound: float | None
    M: float
    delta: float
    K: int
    T: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
