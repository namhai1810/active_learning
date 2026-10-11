"""Bounded pilot-gain estimates and confidence radii."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.gain.types import GainEstimate


def _validate_bound(M: float, tolerance: float = 0.0) -> None:
    if not math.isfinite(M) or M <= 0:
        raise ValueError("M must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")


def validate_gain(gain: float, M: float, tolerance: float = 1e-12) -> float:
    """Reject a bound violation without clipping the observation."""
    _validate_bound(M, tolerance)
    if not math.isfinite(gain):
        raise ValueError("Pilot gain must be finite")
    if abs(gain) > M + tolerance:
        raise ValueError(
            f"Pilot gain {gain} violates configured bound |X| <= {M} "
            f"(tolerance={tolerance})"
        )
    return gain


def risk_reduction(
    risk_before: float,
    risk_after: float,
    M: float,
    tolerance: float = 1e-12,
) -> float:
    for name, value in (("risk_before", risk_before), ("risk_after", risk_after)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return validate_gain(risk_before - risk_after, M, tolerance)


def empirical_mean_gain(gains: Sequence[float]) -> float:
    if not gains:
        raise ValueError("At least one pilot gain is required")
    return sum(gains) / len(gains)


def gain_confidence_radius(m: int, M: float, K: int, T: int, delta: float) -> float:
    """Return the predeclared K-by-T union-adjusted Hoeffding radius.

    Under adaptive rounds this is valid only when the observations used at each
    selected round satisfy the bounded conditional concentration assumption
    given the preceding history. The arithmetic alone does not establish that
    generated data or related training runs are conditionally independent.
    """
    if isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")
    if isinstance(K, bool) or K <= 0 or isinstance(T, bool) or T <= 0:
        raise ValueError("K and T must be positive integers")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    _validate_bound(M)
    return M * math.sqrt(2.0 * math.log(2.0 * K * T / delta) / m)


def estimate_gain(
    hypothesis_id: str,
    gains: Sequence[float],
    *,
    M: float,
    delta: float,
    K: int,
    T: int,
    tolerance: float = 1e-12,
) -> GainEstimate:
    """Estimate mean gain; return an explicit invalid state for zero pilots."""
    if not hypothesis_id:
        raise ValueError("hypothesis_id must be non-empty")
    _validate_bound(M, tolerance)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive")
    if not gains:
        return GainEstimate(
            hypothesis_id=hypothesis_id,
            m=0,
            delta_hat=None,
            radius=None,
            lower_bound=None,
            upper_bound=None,
            M=M,
            delta=delta,
            K=K,
            T=T,
            status="insufficient_pilots",
        )

    checked = [validate_gain(value, M, tolerance) for value in gains]
    mean = empirical_mean_gain(checked)
    radius = gain_confidence_radius(len(checked), M, K, T, delta)
    return GainEstimate(
        hypothesis_id=hypothesis_id,
        m=len(checked),
        delta_hat=mean,
        radius=radius,
        lower_bound=mean - radius,
        upper_bound=mean + radius,
        M=M,
        delta=delta,
        K=K,
        T=T,
        status="estimated",
    )
