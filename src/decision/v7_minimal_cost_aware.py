"""Minimal cost-aware controller for the unchanged V7 mixed-pair action.

The appendix theorem is indexed by a hypothesis-specific intervention.  V7's
proven selector instead builds one mixed-hypothesis batch.  This module keeps
that action intact and applies the same bounded-gain UCB arithmetic to the
round policy itself.  It deliberately does not claim the appendix theorem.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.decision.cost_aware import (
    compute_net_value,
    compute_optimistic_net_value,
)
from src.gain.types import GainEstimate


MIXED_POLICY_ID = "v7_soft_pair_mixed_round_policy"


@dataclass(frozen=True)
class V7MinimalCostAwareConfig:
    enabled: bool
    start_round: int
    required_pilots: int
    pilot_eval_size: int
    M: float
    delta: float
    K: int
    T: int
    lambda_cost: float
    calibration_reference_gain: float | None
    calibration_reference_cost: float | None
    calibration_currency: str | None
    pilot_seed: int
    preserve_pilot_adapters: bool

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "V7MinimalCostAwareConfig":
        raw: Mapping[str, Any] = value or {}
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            start_round=int(raw.get("start_round", 2)),
            required_pilots=int(raw.get("required_pilots", 4)),
            pilot_eval_size=int(raw.get("pilot_eval_size", 64)),
            M=float(raw.get("M", 1.0)),
            delta=float(raw.get("delta", 0.05)),
            K=int(raw.get("K", 1)),
            T=int(raw.get("T", 1)),
            lambda_cost=float(raw.get("lambda_cost", 0.001)),
            calibration_reference_gain=(
                None
                if raw.get("calibration_reference_gain") is None
                else float(raw["calibration_reference_gain"])
            ),
            calibration_reference_cost=(
                None
                if raw.get("calibration_reference_cost") is None
                else float(raw["calibration_reference_cost"])
            ),
            calibration_currency=(
                None
                if raw.get("calibration_currency") is None
                else str(raw["calibration_currency"]).strip()
            ),
            pilot_seed=int(raw.get("pilot_seed", 42)),
            preserve_pilot_adapters=bool(
                raw.get("preserve_pilot_adapters", False)
            ),
        )
        if config.start_round != 2:
            raise ValueError("Minimal V7 Cost-Aware must preserve round 1")
        if config.required_pilots <= 0 or config.pilot_eval_size <= 0:
            raise ValueError("Cost-Aware pilot sizes must be positive")
        if not math.isfinite(config.M) or config.M <= 0:
            raise ValueError("cost_aware.M must be finite and positive")
        if not 0.0 < config.delta < 1.0:
            raise ValueError("cost_aware.delta must be in (0, 1)")
        if config.K != 1:
            raise ValueError("Mixed-policy controller has exactly K=1 action")
        if config.T <= 0:
            raise ValueError("cost_aware.T must be positive")
        if not math.isfinite(config.lambda_cost) or config.lambda_cost < 0:
            raise ValueError("cost_aware.lambda_cost must be non-negative")
        if config.enabled:
            if (
                config.calibration_reference_gain is None
                or config.calibration_reference_cost is None
                or not config.calibration_currency
            ):
                raise ValueError(
                    "Enabled Cost-Aware requires an explicit gain/cost calibration"
                )
            if (
                not math.isfinite(config.calibration_reference_gain)
                or config.calibration_reference_gain <= 0
                or config.calibration_reference_gain > config.M
            ):
                raise ValueError(
                    "calibration_reference_gain must be in (0, M]"
                )
            if (
                not math.isfinite(config.calibration_reference_cost)
                or config.calibration_reference_cost <= 0
            ):
                raise ValueError(
                    "calibration_reference_cost must be finite and positive"
                )
            calibrated_lambda = (
                config.calibration_reference_gain
                / config.calibration_reference_cost
            )
            if not math.isclose(
                config.lambda_cost,
                calibrated_lambda,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "lambda_cost must equal calibration_reference_gain / "
                    "calibration_reference_cost"
                )
            raw_cost = raw.get("cost")
            if not isinstance(raw_cost, Mapping):
                raise ValueError("Enabled Cost-Aware requires a cost mapping")
            if str(raw_cost.get("currency", "")).strip() != config.calibration_currency:
                raise ValueError(
                    "calibration_currency must match cost.currency"
                )
        return config


@dataclass(frozen=True)
class MixedPolicyDecision:
    action: str
    reason: str
    policy_id: str
    delta_hat: float | None
    confidence_radius: float | None
    cost: float
    lambda_cost: float
    estimated_net_value: float | None
    raw_gain_ucb: float | None
    bounded_gain_ucb: float | None
    raw_optimistic_net_value: float | None
    optimistic_net_value: float | None
    gain_upper_bound_M: float
    cost_penalty: float
    required_pilots: int
    observed_pilots: int
    certified_hypotheses: tuple[str, ...]
    unresolved_hypotheses: tuple[str, ...]
    refuted_hypotheses: tuple[str, ...]
    gain_cost_stop_evaluated: bool
    appendix_theorem_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predeclare_pilot_eval_plan(
    report_ids: Sequence[str],
    *,
    decision_rounds: Sequence[int],
    pilots_per_round: int,
    eval_size: int,
    seed: int,
) -> dict[str, Any]:
    """Partition pilot-eval IDs before model outcomes are observed."""
    ids = [str(item) for item in report_ids]
    rounds = [int(item) for item in decision_rounds]
    if len(ids) != len(set(ids)):
        raise ValueError("Pilot-eval source IDs must be unique")
    if not rounds or len(rounds) != len(set(rounds)) or any(item <= 1 for item in rounds):
        raise ValueError("Decision rounds must be unique integers after round 1")
    if pilots_per_round <= 0 or eval_size <= 0:
        raise ValueError("Pilot count and eval size must be positive")
    required = len(rounds) * pilots_per_round * eval_size
    if required >= len(ids):
        raise ValueError(
            "Pilot-eval plan must leave a non-empty held-out gain-report set"
        )
    ordered = sorted(
        ids,
        key=lambda item: hashlib.sha256(
            f"{seed}:v7-cost-aware-pilot-eval:{item}".encode("utf-8")
        ).hexdigest(),
    )
    cursor = 0
    shards: dict[str, list[list[str]]] = {}
    for round_index in rounds:
        current: list[list[str]] = []
        for _ in range(pilots_per_round):
            current.append(ordered[cursor : cursor + eval_size])
            cursor += eval_size
        shards[str(round_index)] = current
    pilot_ids = ordered[:cursor]
    held_out = ordered[cursor:]
    return {
        "policy_id": MIXED_POLICY_ID,
        "selection_is_outcome_independent": True,
        "seed": seed,
        "decision_rounds": rounds,
        "pilots_per_round": pilots_per_round,
        "eval_size": eval_size,
        "source_ids": ids,
        "pilot_ids": pilot_ids,
        "held_out_gain_report_ids": held_out,
        "round_shards": shards,
        "pairwise_disjoint": len(pilot_ids) == len(set(pilot_ids)),
    }


def _state_ids(
    states: Mapping[str, Mapping[str, Any]], state: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(hypothesis_id)
            for hypothesis_id, payload in states.items()
            if payload.get("state") == state
        )
    )


def decide_mixed_policy_round(
    *,
    states: Mapping[str, Mapping[str, Any]],
    estimate: GainEstimate,
    required_pilots: int,
    cost: float,
    lambda_cost: float,
) -> MixedPolicyDecision:
    """Apply an evidence gate before the mixed-policy gain-cost rule.

    Unresolved hypotheses and incomplete pilots force CONTINUE rather than a
    gain-cost STOP.  Operational absence of active hypotheses/data is handled
    by the unchanged V7 loop before this controller is called.
    """
    if required_pilots <= 0:
        raise ValueError("required_pilots must be positive")
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and non-negative")
    if not math.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if estimate.hypothesis_id != MIXED_POLICY_ID:
        raise ValueError("Gain estimate does not describe the V7 mixed policy")
    certified = _state_ids(states, "certified")
    unresolved = _state_ids(states, "unresolved")
    refuted = _state_ids(states, "refuted")
    known = set(certified) | set(unresolved) | set(refuted)
    if known != {str(item) for item in states}:
        raise ValueError("Hypothesis states must be certified/unresolved/refuted")

    common = {
        "policy_id": MIXED_POLICY_ID,
        "cost": float(cost),
        "lambda_cost": float(lambda_cost),
        "gain_upper_bound_M": float(estimate.M),
        "cost_penalty": float(lambda_cost * cost),
        "required_pilots": required_pilots,
        "observed_pilots": estimate.m,
        "certified_hypotheses": certified,
        "unresolved_hypotheses": unresolved,
        "refuted_hypotheses": refuted,
    }
    if unresolved:
        return MixedPolicyDecision(
            action="continue",
            reason="unresolved_hypotheses_require_more_evidence",
            delta_hat=estimate.delta_hat,
            confidence_radius=estimate.radius,
            estimated_net_value=None,
            raw_gain_ucb=None,
            bounded_gain_ucb=None,
            raw_optimistic_net_value=None,
            optimistic_net_value=None,
            gain_cost_stop_evaluated=False,
            **common,
        )
    if not certified:
        return MixedPolicyDecision(
            action="continue",
            reason="no_certified_hypothesis_but_evidence_not_exhausted",
            delta_hat=estimate.delta_hat,
            confidence_radius=estimate.radius,
            estimated_net_value=None,
            raw_gain_ucb=None,
            bounded_gain_ucb=None,
            raw_optimistic_net_value=None,
            optimistic_net_value=None,
            gain_cost_stop_evaluated=False,
            **common,
        )
    if (
        estimate.status != "estimated"
        or estimate.m != required_pilots
        or estimate.delta_hat is None
        or estimate.radius is None
    ):
        return MixedPolicyDecision(
            action="continue",
            reason="insufficient_pilot_evidence",
            delta_hat=estimate.delta_hat,
            confidence_radius=estimate.radius,
            estimated_net_value=None,
            raw_gain_ucb=None,
            bounded_gain_ucb=None,
            raw_optimistic_net_value=None,
            optimistic_net_value=None,
            gain_cost_stop_evaluated=False,
            **common,
        )
    estimated_net = compute_net_value(
        estimate.delta_hat, cost, lambda_cost
    )
    raw_gain_ucb = estimate.delta_hat + estimate.radius
    # Since every gain observation lies in [-M, M], its population mean also
    # satisfies Delta <= M. Intersecting the Hoeffding upper interval with this
    # known support bound tightens the UCB without changing its coverage.
    bounded_gain_ucb = min(estimate.M, raw_gain_ucb)
    raw_optimistic_net = compute_optimistic_net_value(
        estimate.delta_hat, estimate.radius, cost, lambda_cost
    )
    optimistic_net = bounded_gain_ucb - lambda_cost * cost
    return MixedPolicyDecision(
        action="continue" if optimistic_net > 0.0 else "stop",
        reason=(
            "positive_mixed_policy_optimistic_net_gain"
            if optimistic_net > 0.0
            else "nonpositive_mixed_policy_optimistic_net_gain"
        ),
        delta_hat=estimate.delta_hat,
        confidence_radius=estimate.radius,
        estimated_net_value=estimated_net,
        raw_gain_ucb=raw_gain_ucb,
        bounded_gain_ucb=bounded_gain_ucb,
        raw_optimistic_net_value=raw_optimistic_net,
        optimistic_net_value=optimistic_net,
        gain_cost_stop_evaluated=True,
        **common,
    )

