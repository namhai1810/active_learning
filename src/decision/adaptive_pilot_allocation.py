"""ARCHIVED offline v4 adaptive-pilot experiment.

Retained at its original import path for historical reproducibility. It is not
part of the V7 runtime or the current next-round expected-net-gain objective.
See ``archive/v7_cost_aware_experiments/README.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log, sqrt
from random import Random
from statistics import fmean
from typing import Any, Optional, Sequence

from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config


@dataclass(frozen=True)
class FrozenPilotCostProjection:
    """Ex-ante cost of one pilot in the configured monetary currency.

    The projection is frozen from the training recipe and cost config before
    any pilot outcome or runtime is revealed. Measured pilot timings are not
    inputs to this object; they are reserved for retrospective reporting.
    """

    currency: str
    training_examples: int
    gain_evaluation_examples: int
    gain_evaluation_passes: int
    stopping_validation_examples: int
    max_epochs: int
    projected_training_gpu_seconds: float
    projected_evaluation_gpu_seconds: float
    training_gpu_second_cost: float
    evaluation_gpu_second_cost: float
    frozen_pilot_gpu_second_cost: float
    projected_training_cost: float
    projected_evaluation_cost: float
    projected_total_cost_per_pilot: float
    source: str

    def __post_init__(self) -> None:
        counts = (
            self.training_examples,
            self.gain_evaluation_examples,
            self.gain_evaluation_passes,
            self.stopping_validation_examples,
            self.max_epochs,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("frozen pilot projection counts must be positive")
        values = (
            self.projected_training_gpu_seconds,
            self.projected_evaluation_gpu_seconds,
            self.training_gpu_second_cost,
            self.evaluation_gpu_second_cost,
            self.frozen_pilot_gpu_second_cost,
            self.projected_training_cost,
            self.projected_evaluation_cost,
            self.projected_total_cost_per_pilot,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("frozen pilot projection values must be finite and non-negative")
        if abs(
            self.projected_total_cost_per_pilot
            - self.projected_training_cost
            - self.projected_evaluation_cost
        ) > 1e-9:
            raise ValueError("pilot total cost must equal training plus evaluation cost")
        if not self.currency.strip() or not self.source.strip():
            raise ValueError("frozen pilot projection requires currency and source")

    def utility_penalty(self, *, pilots: int, lambda_cost: float) -> float:
        if pilots < 0 or lambda_cost < 0:
            raise ValueError("pilots and lambda_cost must be non-negative")
        return pilots * self.projected_total_cost_per_pilot * lambda_cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def freeze_pilot_cost_projection(
    cost_config: dict[str, Any],
    *,
    training_examples: int,
    gain_evaluation_examples: int,
    gain_evaluation_passes: int,
    stopping_validation_examples: int,
    max_epochs: int,
    expected_currency: str,
) -> FrozenPilotCostProjection:
    """Build one immutable pilot projection from the existing cost model."""

    rates, currency = cost_rates_from_config(cost_config)
    if currency != expected_currency:
        raise ValueError(
            "pilot cost currency differs from lambda calibration currency: "
            f"{currency!r} != {expected_currency!r}"
        )
    raw_projections = cost_config.get("projections")
    if not isinstance(raw_projections, dict):
        raise ValueError("cost_aware.cost.projections must be a mapping")

    def projection(name: str) -> float:
        raw = raw_projections.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"missing numeric frozen cost projection: {name}")
        value = float(raw)
        if not isfinite(value) or value < 0:
            raise ValueError(f"invalid frozen cost projection: {name}")
        return value

    training_gpu_seconds = (
        projection("lora_training_gpu_seconds_fixed")
        + training_examples * projection("lora_training_gpu_seconds_per_item")
    )
    evaluation_items = (
        gain_evaluation_passes * gain_evaluation_examples
        + max_epochs * stopping_validation_examples
    )
    evaluation_runs = gain_evaluation_passes + max_epochs
    evaluation_gpu_seconds = (
        projection("evaluation_gpu_seconds_fixed")
        + evaluation_items * projection("evaluation_gpu_seconds_per_item")
    )
    training = compute_cost(
        CostUsage(
            lora_training_runs=1,
            lora_training_gpu_seconds=training_gpu_seconds,
        ),
        rates,
        currency=currency,
    )
    evaluation = compute_cost(
        CostUsage(
            evaluation_runs=evaluation_runs,
            evaluation_items=evaluation_items,
            evaluation_gpu_seconds=evaluation_gpu_seconds,
        ),
        rates,
        currency=currency,
    )
    total_gpu_seconds = training_gpu_seconds + evaluation_gpu_seconds
    gpu_cost = (
        training_gpu_seconds * rates.lora_training_per_gpu_hour / 3600.0
        + evaluation_gpu_seconds * rates.evaluation_per_gpu_hour / 3600.0
    )
    effective_gpu_second_cost = (
        gpu_cost / total_gpu_seconds if total_gpu_seconds > 0 else 0.0
    )
    return FrozenPilotCostProjection(
        currency=currency,
        training_examples=training_examples,
        gain_evaluation_examples=gain_evaluation_examples,
        gain_evaluation_passes=gain_evaluation_passes,
        stopping_validation_examples=stopping_validation_examples,
        max_epochs=max_epochs,
        projected_training_gpu_seconds=training_gpu_seconds,
        projected_evaluation_gpu_seconds=evaluation_gpu_seconds,
        training_gpu_second_cost=rates.lora_training_per_gpu_hour / 3600.0,
        evaluation_gpu_second_cost=rates.evaluation_per_gpu_hour / 3600.0,
        frozen_pilot_gpu_second_cost=effective_gpu_second_cost,
        projected_training_cost=training.total,
        projected_evaluation_cost=evaluation.total,
        projected_total_cost_per_pilot=training.total + evaluation.total,
        source="resolved_run_config.cost_aware.cost",
    )


@dataclass(frozen=True)
class PilotObservation:
    pilot_index: int
    gain: float
    baseline_accuracy: float
    gpu_seconds: Optional[float]
    eval_ids_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pilot_index < 0:
            raise ValueError("pilot_index must be non-negative")
        if not 0 <= self.baseline_accuracy <= 1:
            raise ValueError("baseline_accuracy must be in [0, 1]")
        lower, upper = -self.baseline_accuracy, 1 - self.baseline_accuracy
        if not isfinite(self.gain) or not lower - 1e-12 <= self.gain <= upper + 1e-12:
            raise ValueError("pilot gain violates range-aware support")
        if self.gpu_seconds is not None and (
            not isfinite(self.gpu_seconds) or self.gpu_seconds < 0
        ):
            raise ValueError("gpu_seconds must be finite and non-negative")


@dataclass(frozen=True)
class AdaptivePilotConfig:
    schedule: tuple[int, ...] = (4, 8, 16, 32)
    delta: float = 0.05
    K: int = 1
    T: int = 7
    lambda_cost: float = 0.2
    bootstrap_draws: int = 512

    def __post_init__(self) -> None:
        if not self.schedule or self.schedule[0] <= 0:
            raise ValueError("pilot schedule must be non-empty and positive")
        if tuple(sorted(set(self.schedule))) != self.schedule:
            raise ValueError("pilot schedule must be strictly increasing")
        if not 0 < self.delta < 1 or self.K <= 0 or self.T <= 0:
            raise ValueError("invalid confidence accounting parameters")
        if self.lambda_cost < 0 or self.bootstrap_draws <= 0:
            raise ValueError("invalid lambda/bootstrap_draws")

    @property
    def delta_per_look(self) -> float:
        return self.delta / len(self.schedule)

    def radius(self, m: int) -> float:
        """Range-width-one Hoeffding radius with Bonferroni alpha spending."""
        return sqrt(log(2 * self.K * self.T / self.delta_per_look) / (2 * m))

    def fixed_radius(self, m: int) -> float:
        """Original one-look v2.1 comparator radius."""
        return sqrt(log(2 * self.K * self.T / self.delta) / (2 * m))


@dataclass(frozen=True)
class PilotLook:
    m: int
    delta_look: float
    delta_hat: float
    confidence_radius: float
    gain_lcb: float
    gain_ucb: float
    net_lcb: float
    net_ucb: float
    stop_certified: bool
    continue_certified: bool
    next_m: Optional[int]
    stop_reachable_within_cap: Optional[bool]
    bootstrap_stop_probability: Optional[float]
    projected_information_cost: Optional[float]
    expected_information_benefit: Optional[float]
    allocation_action: str
    allocation_reason: str
    voi_target_m: Optional[int] = None
    projected_information_cost_currency: Optional[str] = None
    projected_information_utility_penalty: Optional[float] = None
    value_of_information: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllocationResult:
    strategy: str
    pilot_cap: int
    pilots_used: int
    action: str
    certification: str
    reason: str
    final_delta_hat: Optional[float]
    final_radius: Optional[float]
    final_gain_ucb: Optional[float]
    final_net_ucb: Optional[float]
    pilot_gpu_seconds_used: float
    pilot_gpu_seconds_available: float
    pilot_gpu_seconds_saved: float
    pilot_gpu_hours_saved: float
    projected_decision_cost: Optional[float]
    evidence_gate: Optional[str]
    confidence_valid: bool
    looks: tuple[PilotLook, ...]
    frozen_pilot_gpu_second_cost: Optional[float] = None
    pilot_cost_currency: Optional[str] = None
    projected_pilot_cost_per_pilot: Optional[float] = None
    projected_pilot_utility_penalty_per_pilot: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["looks"] = [look.to_dict() for look in self.looks]
        return payload


def _validate_sequence(
    observations: Sequence[PilotObservation],
    cap: int,
) -> tuple[PilotObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda item: item.pilot_index))
    if len(ordered) < cap:
        raise ValueError(f"need {cap} cached pilots, found {len(ordered)}")
    indices = [item.pilot_index for item in ordered[:cap]]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate pilot indices")
    baselines = {round(item.baseline_accuracy, 12) for item in ordered[:cap]}
    if len(baselines) != 1:
        raise ValueError("range-aware simulation requires one fixed baseline")
    hashes = {item.eval_ids_hash for item in ordered[:cap] if item.eval_ids_hash}
    if len(hashes) > 1:
        raise ValueError("pilots do not share one fixed evaluation shard")
    return ordered


def _gpu_seconds(items: Sequence[PilotObservation]) -> float:
    return sum(item.gpu_seconds or 0.0 for item in items)


def _bootstrap_stop_probability(
    revealed: Sequence[PilotObservation],
    *,
    next_m: int,
    next_radius: float,
    decision_cost: float,
    lambda_cost: float,
    lower_support: float,
    upper_support: float,
    draws: int,
    random_seed: int,
) -> float:
    """Heuristic probability using revealed observations only.

    This empirical bootstrap guides allocation and has no role in the STOP
    certificate or its confidence guarantee.
    """
    rng = Random(random_seed)
    gains = [item.gain for item in revealed]
    additional = next_m - len(gains)
    certified = 0
    for _ in range(draws):
        future = [gains[rng.randrange(len(gains))] for _ in range(additional)]
        mean = fmean(gains + future)
        gain_ucb = min(upper_support, mean + next_radius)
        if gain_ucb - lambda_cost * decision_cost <= 0:
            certified += 1
    return certified / draws


def simulate_adaptive_allocation(
    observations: Sequence[PilotObservation],
    *,
    decision_cost: Optional[float],
    pilot_cost_projection: Optional[FrozenPilotCostProjection],
    config: AdaptivePilotConfig,
    random_seed: int,
    evidence_gate: Optional[str] = None,
) -> AllocationResult:
    cap = config.schedule[-1]
    if evidence_gate:
        available_seconds = _gpu_seconds(observations[:cap])
        return AllocationResult(
            strategy="adaptive",
            pilot_cap=cap,
            pilots_used=0,
            action="continue",
            certification="uncertified_continue",
            reason="evidence_gate_skip_pilot_sampling",
            final_delta_hat=None,
            final_radius=None,
            final_gain_ucb=None,
            final_net_ucb=None,
            pilot_gpu_seconds_used=0.0,
            pilot_gpu_seconds_available=available_seconds,
            pilot_gpu_seconds_saved=available_seconds,
            pilot_gpu_hours_saved=available_seconds / 3600,
            projected_decision_cost=decision_cost,
            evidence_gate=evidence_gate,
            confidence_valid=True,
            looks=(),
            frozen_pilot_gpu_second_cost=(
                pilot_cost_projection.frozen_pilot_gpu_second_cost
                if pilot_cost_projection is not None else None
            ),
            pilot_cost_currency=(
                pilot_cost_projection.currency
                if pilot_cost_projection is not None else None
            ),
            projected_pilot_cost_per_pilot=(
                pilot_cost_projection.projected_total_cost_per_pilot
                if pilot_cost_projection is not None else None
            ),
            projected_pilot_utility_penalty_per_pilot=(
                config.lambda_cost * pilot_cost_projection.projected_total_cost_per_pilot
                if pilot_cost_projection is not None else None
            ),
        )

    if decision_cost is None or decision_cost < 0:
        available_seconds = _gpu_seconds(observations[:cap])
        return AllocationResult(
            strategy="adaptive",
            pilot_cap=cap,
            pilots_used=0,
            action="continue",
            certification="uncertified_continue",
            reason="missing_projected_decision_cost",
            final_delta_hat=None,
            final_radius=None,
            final_gain_ucb=None,
            final_net_ucb=None,
            pilot_gpu_seconds_used=0.0,
            pilot_gpu_seconds_available=available_seconds,
            pilot_gpu_seconds_saved=available_seconds,
            pilot_gpu_hours_saved=available_seconds / 3600,
            projected_decision_cost=decision_cost,
            evidence_gate=None,
            confidence_valid=False,
            looks=(),
            frozen_pilot_gpu_second_cost=(
                pilot_cost_projection.frozen_pilot_gpu_second_cost
                if pilot_cost_projection is not None else None
            ),
            pilot_cost_currency=(
                pilot_cost_projection.currency
                if pilot_cost_projection is not None else None
            ),
            projected_pilot_cost_per_pilot=(
                pilot_cost_projection.projected_total_cost_per_pilot
                if pilot_cost_projection is not None else None
            ),
            projected_pilot_utility_penalty_per_pilot=(
                config.lambda_cost * pilot_cost_projection.projected_total_cost_per_pilot
                if pilot_cost_projection is not None else None
            ),
        )

    ordered = _validate_sequence(observations, cap)
    available_seconds = _gpu_seconds(ordered[:cap])

    baseline = ordered[0].baseline_accuracy
    lower_support, upper_support = -baseline, 1 - baseline
    looks: list[PilotLook] = []
    final_reason = "pilot_cap_reached_without_stop_certificate"
    final_certification = "uncertified_continue"

    for look_index, m in enumerate(config.schedule):
        revealed = ordered[:m]
        delta_hat = fmean(item.gain for item in revealed)
        radius = config.radius(m)
        gain_lcb = max(lower_support, delta_hat - radius)
        gain_ucb = min(upper_support, delta_hat + radius)
        net_lcb = gain_lcb - config.lambda_cost * decision_cost
        net_ucb = gain_ucb - config.lambda_cost * decision_cost
        stop_certified = net_ucb <= 0
        continue_certified = net_lcb > 0
        next_m = config.schedule[look_index + 1] if look_index + 1 < len(config.schedule) else None

        if stop_certified:
            looks.append(
                PilotLook(
                    m, config.delta_per_look, delta_hat, radius, gain_lcb,
                    gain_ucb, net_lcb, net_ucb, True, False, next_m, None,
                    None, None, None, "stop", "nonpositive_valid_net_ucb",
                )
            )
            used_seconds = _gpu_seconds(revealed)
            return AllocationResult(
                "adaptive", cap, m, "stop", "certified_stop",
                "nonpositive_valid_net_ucb", delta_hat, radius, gain_ucb,
                net_ucb, used_seconds, available_seconds,
                available_seconds - used_seconds,
                (available_seconds - used_seconds) / 3600, decision_cost,
                None, True, tuple(looks),
                frozen_pilot_gpu_second_cost=(
                    pilot_cost_projection.frozen_pilot_gpu_second_cost
                    if pilot_cost_projection is not None else None
                ),
                pilot_cost_currency=(
                    pilot_cost_projection.currency
                    if pilot_cost_projection is not None else None
                ),
                projected_pilot_cost_per_pilot=(
                    pilot_cost_projection.projected_total_cost_per_pilot
                    if pilot_cost_projection is not None else None
                ),
                projected_pilot_utility_penalty_per_pilot=(
                    config.lambda_cost
                    * pilot_cost_projection.projected_total_cost_per_pilot
                    if pilot_cost_projection is not None else None
                ),
            )

        if continue_certified:
            looks.append(
                PilotLook(
                    m, config.delta_per_look, delta_hat, radius, gain_lcb,
                    gain_ucb, net_lcb, net_ucb, False, True, next_m, None,
                    None, None, None, "continue", "positive_valid_net_lcb",
                )
            )
            final_reason = "positive_valid_net_lcb"
            final_certification = "certified_continue"
            break

        if next_m is None:
            looks.append(
                PilotLook(
                    m, config.delta_per_look, delta_hat, radius, gain_lcb,
                    gain_ucb, net_lcb, net_ucb, False, False, None, None,
                    None, None, None, "continue",
                    "pilot_cap_reached_without_stop_certificate",
                )
            )
            break

        target_m = cap
        target_radius = config.radius(target_m)
        remaining_to_cap = target_m - m
        min_next_mean = (
            m * delta_hat + remaining_to_cap * lower_support
        ) / target_m
        min_next_ucb = min(upper_support, min_next_mean + target_radius)
        stop_reachable = (
            min_next_ucb - config.lambda_cost * decision_cost <= 0
        )
        if not stop_reachable:
            looks.append(
                PilotLook(
                    m, config.delta_per_look, delta_hat, radius, gain_lcb,
                    gain_ucb, net_lcb, net_ucb, False, False, next_m, False,
                    0.0, None, 0.0, "continue",
                    "additional_pilots_cannot_certify_stop_within_cap",
                )
            )
            final_reason = "additional_pilots_cannot_certify_stop_within_cap"
            break

        if pilot_cost_projection is None:
            looks.append(
                PilotLook(
                    m, config.delta_per_look, delta_hat, radius, gain_lcb,
                    gain_ucb, net_lcb, net_ucb, False, False, next_m, True,
                    None, None, None, "continue",
                    "missing_predecision_pilot_cost_projection",
                )
            )
            final_reason = "missing_predecision_pilot_cost_projection"
            break

        # Look ahead over declared future checkpoints using only the revealed
        # gains. Costs are frozen ex ante; cached runtimes never enter VOI.
        candidates: list[tuple[float, int, float, float, float, float]] = []
        for target_offset, candidate_m in enumerate(
            config.schedule[look_index + 1 :]
        ):
            candidate_probability = _bootstrap_stop_probability(
                revealed,
                next_m=candidate_m,
                next_radius=config.radius(candidate_m),
                decision_cost=decision_cost,
                lambda_cost=config.lambda_cost,
                lower_support=lower_support,
                upper_support=upper_support,
                draws=config.bootstrap_draws,
                random_seed=random_seed + 1009 * m + 7919 * target_offset,
            )
            candidate_cost = (
                candidate_m - m
            ) * pilot_cost_projection.projected_total_cost_per_pilot
            candidate_penalty = config.lambda_cost * candidate_cost
            candidate_benefit = (
                candidate_probability * config.lambda_cost * decision_cost
            )
            candidates.append(
                (
                    candidate_benefit - candidate_penalty,
                    candidate_m,
                    candidate_probability,
                    candidate_cost,
                    candidate_penalty,
                    candidate_benefit,
                )
            )
        (
            value_of_information,
            voi_target_m,
            probability,
            projected_information_cost,
            information_penalty,
            expected_benefit,
        ) = max(candidates, key=lambda item: (item[0], -item[1]))
        allocate = value_of_information > 0
        reason = (
            "expected_information_benefit_exceeds_projected_pilot_cost"
            if allocate
            else "expected_information_benefit_not_above_projected_pilot_cost"
        )
        looks.append(
            PilotLook(
                m, config.delta_per_look, delta_hat, radius, gain_lcb,
                gain_ucb, net_lcb, net_ucb, False, False, next_m, True,
                probability, projected_information_cost, expected_benefit,
                "sample_more" if allocate else "continue", reason,
                voi_target_m=voi_target_m,
                projected_information_cost_currency=pilot_cost_projection.currency,
                projected_information_utility_penalty=information_penalty,
                value_of_information=value_of_information,
            )
        )
        if not allocate:
            final_reason = reason
            break

    final = looks[-1]
    used = final.m
    used_seconds = _gpu_seconds(ordered[:used])
    return AllocationResult(
        strategy="adaptive",
        pilot_cap=cap,
        pilots_used=used,
        action="continue",
        certification=final_certification,
        reason=final_reason,
        final_delta_hat=final.delta_hat,
        final_radius=final.confidence_radius,
        final_gain_ucb=final.gain_ucb,
        final_net_ucb=final.net_ucb,
        pilot_gpu_seconds_used=used_seconds,
        pilot_gpu_seconds_available=available_seconds,
        pilot_gpu_seconds_saved=available_seconds - used_seconds,
        pilot_gpu_hours_saved=(available_seconds - used_seconds) / 3600,
        projected_decision_cost=decision_cost,
        evidence_gate=None,
        confidence_valid=True,
        looks=tuple(looks),
        frozen_pilot_gpu_second_cost=(
            pilot_cost_projection.frozen_pilot_gpu_second_cost
            if pilot_cost_projection is not None else None
        ),
        pilot_cost_currency=(
            pilot_cost_projection.currency
            if pilot_cost_projection is not None else None
        ),
        projected_pilot_cost_per_pilot=(
            pilot_cost_projection.projected_total_cost_per_pilot
            if pilot_cost_projection is not None else None
        ),
        projected_pilot_utility_penalty_per_pilot=(
            config.lambda_cost * pilot_cost_projection.projected_total_cost_per_pilot
            if pilot_cost_projection is not None else None
        ),
    )


def simulate_fixed_32(
    observations: Sequence[PilotObservation],
    *,
    decision_cost: Optional[float],
    config: AdaptivePilotConfig,
    evidence_gate: Optional[str] = None,
) -> AllocationResult:
    cap = 32
    if evidence_gate or decision_cost is None:
        available_seconds = _gpu_seconds(observations[:cap])
        reason = "evidence_gate_skip_pilot_sampling" if evidence_gate else "missing_projected_decision_cost"
        return AllocationResult(
            "fixed_32", cap, 0, "continue", "uncertified_continue",
            reason, None, None, None, None, 0.0, available_seconds,
            available_seconds, available_seconds / 3600, decision_cost,
            evidence_gate, decision_cost is not None, (),
        )
    ordered = _validate_sequence(observations, cap)
    available_seconds = _gpu_seconds(ordered[:cap])
    baseline = ordered[0].baseline_accuracy
    lower_support, upper_support = -baseline, 1 - baseline
    delta_hat = fmean(item.gain for item in ordered[:cap])
    radius = config.fixed_radius(cap)
    gain_lcb = max(lower_support, delta_hat - radius)
    gain_ucb = min(upper_support, delta_hat + radius)
    net_lcb = gain_lcb - config.lambda_cost * decision_cost
    net_ucb = gain_ucb - config.lambda_cost * decision_cost
    if net_ucb <= 0:
        action, certification, reason = "stop", "certified_stop", "nonpositive_valid_net_ucb"
    elif net_lcb > 0:
        action, certification, reason = "continue", "certified_continue", "positive_valid_net_lcb"
    else:
        action, certification, reason = "continue", "uncertified_continue", "no_stop_certificate_at_fixed_32"
    look = PilotLook(
        cap, config.delta, delta_hat, radius, gain_lcb, gain_ucb,
        net_lcb, net_ucb, net_ucb <= 0, net_lcb > 0, None, None,
        None, None, None, action, reason,
    )
    return AllocationResult(
        "fixed_32", cap, cap, action, certification, reason, delta_hat,
        radius, gain_ucb, net_ucb, available_seconds, available_seconds,
        0.0, 0.0, decision_cost, evidence_gate, True, (look,),
    )


def simulate_no_pilots(
    observations: Sequence[PilotObservation],
    *,
    decision_cost: Optional[float],
    evidence_gate: Optional[str] = None,
) -> AllocationResult:
    available_seconds = _gpu_seconds(observations[:32])
    return AllocationResult(
        "no_pilots_continue", 0, 0, "continue", "uncertified_continue",
        "no_pilot_information_by_design", None, None, None, None, 0.0,
        available_seconds, available_seconds, available_seconds / 3600,
        decision_cost, evidence_gate, True, (),
    )

