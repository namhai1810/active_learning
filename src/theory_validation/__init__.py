"""Offline empirical checks for verification, gain UCB, and stopping."""

from src.theory_validation.gain_coverage import evaluate_gain_coverage
from src.theory_validation.stopping import audit_safe_stopping
from src.theory_validation.verification import evaluate_verification_confidence

__all__ = (
    "audit_safe_stopping",
    "evaluate_gain_coverage",
    "evaluate_verification_confidence",
)
