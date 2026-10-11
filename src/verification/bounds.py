"""Pure concentration-bound calculations for binary verification probes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.verification.types import VerificationResult


def empirical_mean(outcomes: Sequence[int]) -> float:
    _validate_outcomes(outcomes)
    if not outcomes:
        raise ValueError("At least one binary outcome is required")
    return sum(outcomes) / len(outcomes)


def confidence_radius(n: int, K: int, T: int, delta: float) -> float:
    """K-by-T union-adjusted radius, conditional on per-round validity.

    Adaptive choice of a hypothesis/round is covered by the union only if each
    constituent bound remains valid conditional on all earlier-round history.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer")
    if isinstance(T, bool) or not isinstance(T, int) or T <= 0:
        raise ValueError("T must be a positive integer")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    return math.sqrt(math.log(4 * K * T / delta) / (2 * n))


def lower_confidence_bound(nu_hat: float, radius: float) -> float:
    return nu_hat - radius


def is_certified(lower_bound: float, threshold: float) -> bool:
    return lower_bound >= threshold


def _validate_outcomes(outcomes: Sequence[int]) -> None:
    invalid = [value for value in outcomes if type(value) is not int or value not in (0, 1)]
    if invalid:
        raise ValueError(f"Verification outcomes must be binary 0/1: {invalid}")


def verify_binary_outcomes(
    hypothesis_id: str,
    outcomes: Sequence[int],
    *,
    delta: float,
    tau_v: float,
    K: int,
    T: int,
) -> VerificationResult:
    _validate_outcomes(outcomes)
    if not 0.0 <= tau_v <= 1.0:
        raise ValueError("tau_v must be in [0, 1]")
    # Validate run-level theory parameters even when no probes are available.
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    if (
        isinstance(K, bool)
        or not isinstance(K, int)
        or K <= 0
        or isinstance(T, bool)
        or not isinstance(T, int)
        or T <= 0
    ):
        raise ValueError("K and T must be positive integers")

    n = len(outcomes)
    if n == 0:
        return VerificationResult(
            hypothesis_id=hypothesis_id,
            n=0,
            nu_hat=None,
            confidence_radius=None,
            lower_bound=None,
            threshold=tau_v,
            certified=False,
            status="insufficient_probes",
        )

    estimate = empirical_mean(outcomes)
    radius = confidence_radius(n, K, T, delta)
    lower_bound = lower_confidence_bound(estimate, radius)
    return VerificationResult(
        hypothesis_id=hypothesis_id,
        n=n,
        nu_hat=estimate,
        confidence_radius=radius,
        lower_bound=lower_bound,
        threshold=tau_v,
        certified=is_certified(lower_bound, tau_v),
        status="ok",
    )
