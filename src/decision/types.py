"""Serializable types for cost-aware decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionCandidate:
    hypothesis_id: str
    delta_hat: float
    confidence_radius: float
    cost: float
    lambda_cost: float
    estimated_net_value: float
    optimistic_net_value: float
    cost_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionDecision:
    action: str
    selected_hypothesis_id: str | None
    max_optimistic_value: float | None
    delta_hat: float | None
    confidence_radius: float | None
    cost: float | None
    lambda_cost: float
    estimated_net_value: float | None
    optimistic_net_value: float | None
    reason: str
    cost_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
