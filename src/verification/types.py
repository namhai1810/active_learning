"""Verification result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    hypothesis_id: str
    n: int
    nu_hat: float | None
    confidence_radius: float | None
    lower_bound: float | None
    threshold: float
    certified: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
