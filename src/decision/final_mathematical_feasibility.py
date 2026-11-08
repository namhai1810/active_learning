"""Mathematical primitives for the final V7 stopping-feasibility audit.

The functions in this module are deliberately offline and side-effect free.
They do not alter the historical V7 controller.  A bound returned here is a
conditional fixed-shard bound only when the corresponding independence
assumption is supplied; cached metadata alone cannot prove that assumption.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.decision.cost_model import CostRates, CostUsage, compute_cost
from src.gain.range_aware import validate_fixed_shard_pilot_observations


@dataclass(frozen=True)
class GainBoundComparison:
    m: int
    delta_hat: float
    sample_variance: float
    baseline_accuracy: float
    support_lower: float
    support_upper: float
    support_width: float
    corrected_alpha: float
    hoeffding_radius: float
    empirical_bernstein_radius: float
    hoeffding_gain_ucb: float
    empirical_bernstein_gain_ucb: float
    conditional_independence_proved: bool
    population_unbiasedness_claimed: bool
    certificate_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostLowerBoundComponent:
    stage: str
    lower_bound: float
    included: bool
    predecision_proven: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullRoundCostLowerBound:
    currency: str
    total: float
    conditional_on_executable_intervention: bool
    components: tuple[CostLowerBoundComponent, ...]
    usage: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "total": self.total,
            "conditional_on_executable_intervention": (
                self.conditional_on_executable_intervention
            ),
            "components": [item.to_dict() for item in self.components],
            "usage": dict(self.usage),
        }


def familywise_alpha(*, delta: float, K: int, T: int) -> float:
    """Bonferroni allocation for one method over K actions and T rounds."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive")
    return delta / (K * T)


def range_aware_hoeffding_radius(
    m: int, *, delta: float, K: int, T: int, width: float = 1.0
) -> float:
    if m <= 0:
        raise ValueError("m must be positive")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be finite and positive")
    return width * math.sqrt(math.log(2.0 * K * T / delta) / (2.0 * m))


def empirical_bernstein_radius(
    values: Sequence[float],
    *,
    delta: float,
    K: int,
    T: int,
    width: float = 1.0,
) -> float:
    """Finite-sample empirical-Bernstein radius for bounded observations.

    This uses the common sample-variance form

      sqrt(2 s^2 log(2KT/delta) / m)
        + 7 width log(2KT/delta) / (3(m-1)).

    It is intentionally not advertised as tighter than Hoeffding: at m=32
    the finite-sample correction can dominate when the sample variance is low.
    """
    m = len(values)
    if m < 2:
        raise ValueError("empirical Bernstein requires at least two observations")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be finite and positive")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) for value in checked):
        raise ValueError("observations must be finite")
    variance = statistics.variance(checked)
    log_term = math.log(2.0 * K * T / delta)
    return math.sqrt(2.0 * variance * log_term / m) + (
        7.0 * width * log_term / (3.0 * (m - 1))
    )


def compare_gain_bounds(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_pilots: int,
    delta: float,
    K: int,
    T: int,
) -> tuple[GainBoundComparison, dict[str, Any]]:
    """Compare range-aware Hoeffding and empirical Bernstein on cached pilots."""
    audit = validate_fixed_shard_pilot_observations(
        observations, required_pilots=required_pilots
    )
    values = [float(item["gain"]) for item in observations]
    if len(values) != required_pilots:
        raise ValueError("a complete pilot set is required")
    mean = statistics.fmean(values)
    variance = statistics.variance(values)
    support_lower = float(audit["support_lower"])
    support_upper = float(audit["support_upper"])
    width = float(audit["support_width"])
    hoeffding = range_aware_hoeffding_radius(
        len(values), delta=delta, K=K, T=T, width=width
    )
    bernstein = empirical_bernstein_radius(
        values, delta=delta, K=K, T=T, width=width
    )

    # Intersect each probabilistic UCB with the exact deterministic support.
    # This is not tuning and cannot invalidate coverage.
    comparison = GainBoundComparison(
        m=len(values),
        delta_hat=mean,
        sample_variance=variance,
        baseline_accuracy=float(audit["baseline_accuracy"]),
        support_lower=support_lower,
        support_upper=support_upper,
        support_width=width,
        corrected_alpha=familywise_alpha(delta=delta, K=K, T=T),
        hoeffding_radius=hoeffding,
        empirical_bernstein_radius=bernstein,
        hoeffding_gain_ucb=min(support_upper, mean + hoeffding),
        empirical_bernstein_gain_ucb=min(support_upper, mean + bernstein),
        conditional_independence_proved=bool(
            audit["conditional_independence_proved"]
        ),
        population_unbiasedness_claimed=bool(
            audit["population_unbiasedness_claimed"]
        ),
        certificate_scope=(
            "conditional_fixed_shard_training_randomization; mechanical checks "
            "pass, conditional independence is not proved by artifacts"
        ),
    )
    return comparison, audit


def protocol_full_round_cost_lower_bound(
    *,
    rates: CostRates,
    currency: str,
    acquired_examples: int,
    stopping_examples: int,
) -> FullRoundCostLowerBound:
    """Build a predecision lower bound for an executable V7 intervention.

    Variable discovery/teacher/verification/generation work is assigned the
    universally valid lower bound zero.  Historical means and future realised
    costs are never used.  Selection, one LoRA run, and the minimum validation
    work are deterministic once a 32-example intervention is executable.
    GPU-time rates contribute zero because the frozen protocol has projections,
    not a proven positive runtime lower bound.
    """
    if acquired_examples <= 0 or stopping_examples <= 0:
        raise ValueError("acquired_examples and stopping_examples must be positive")

    selection_usage = CostUsage(
        evaluation_runs=2,
        evaluation_items=2 * acquired_examples,
    )
    training_usage = CostUsage(lora_training_runs=1)
    validation_usage = CostUsage(
        evaluation_runs=2,
        evaluation_items=2 * stopping_examples,
    )
    selection = compute_cost(selection_usage, rates, currency=currency)
    training = compute_cost(training_usage, rates, currency=currency)
    validation = compute_cost(validation_usage, rates, currency=currency)
    components = (
        CostLowerBoundComponent(
            "discovery",
            0.0,
            False,
            True,
            "candidate/state dependent; no positive token or GPU-time minimum is frozen",
        ),
        CostLowerBoundComponent(
            "verification",
            0.0,
            False,
            True,
            "may be skipped when no candidates exist; adaptive probe count is state dependent",
        ),
        CostLowerBoundComponent(
            "generation_and_qc",
            0.0,
            False,
            True,
            "generation is skipped when no active hypothesis remains",
        ),
        CostLowerBoundComponent(
            "soft_pair_selection",
            selection.total,
            True,
            True,
            "two scoring passes over at least the acquired examples are mandatory",
        ),
        CostLowerBoundComponent(
            "main_lora_training",
            training.total,
            True,
            True,
            "one fixed per-run charge; runtime lower bound remains zero",
        ),
        CostLowerBoundComponent(
            "epoch_validation_and_best_restore_check",
            validation.total,
            True,
            True,
            "at least one epoch validation plus one restored-checkpoint evaluation",
        ),
        CostLowerBoundComponent(
            "pilot_estimation",
            0.0,
            False,
            True,
            "post-acquisition estimator overhead is not mandatory in the target V7 round action",
        ),
    )
    total = math.fsum(item.lower_bound for item in components if item.included)
    usage = selection_usage.plus(training_usage).plus(validation_usage)
    return FullRoundCostLowerBound(
        currency=currency,
        total=total,
        conditional_on_executable_intervention=True,
        components=components,
        usage=asdict(usage),
    )


def net_ucb(*, gain_ucb: float, rho: float, lambda_cost: float, cost_lcb: float) -> float:
    values = (gain_ucb, rho, lambda_cost, cost_lcb)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("net-UCB inputs must be finite")
    if rho < 0 or lambda_cost < 0 or cost_lcb < 0:
        raise ValueError("rho, lambda_cost, and cost_lcb must be non-negative")
    return float(gain_ucb) + float(rho) - float(lambda_cost) * float(cost_lcb)


def maximum_rho_for_stop(
    *, gain_ucb: float, lambda_cost: float, cost_lcb: float
) -> float:
    """Largest non-negative drift allowance compatible with U_net <= 0.

    A negative result means STOP is impossible even at rho=0.
    """
    return float(lambda_cost) * float(cost_lcb) - float(gain_ucb)


def consecutive_gain_drifts(round_means: Mapping[int, float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for current in sorted(round_means):
        previous = current - 1
        if previous not in round_means:
            continue
        signed = float(round_means[current]) - float(round_means[previous])
        rows.append(
            {
                "from_round": float(previous),
                "to_round": float(current),
                "signed_drift": signed,
                "absolute_drift": abs(signed),
            }
        )
    return rows
