"""Types returned by the controlled ablation strategy layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.decision.types import DecisionCandidate, InterventionDecision


@dataclass(frozen=True)
class AblationOutcome:
    variant: str
    accepted_hypothesis_ids: tuple[str, ...]
    candidates: tuple[DecisionCandidate, ...]
    decision: InterventionDecision
