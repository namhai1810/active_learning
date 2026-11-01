#!/usr/bin/env python3
"""Static, CPU-only audit for the V7 minimal Cost-Aware protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.v7_minimal_cost_aware import (  # noqa: E402
    V7MinimalCostAwareConfig,
    predeclare_pilot_eval_plan,
)
from src.gain.estimation import gain_confidence_radius  # noqa: E402
from src.utils.config import load_yaml_with_extends  # noqa: E402


def audit_protocol() -> dict[str, Any]:
    legacy = load_yaml_with_extends(
        PROJECT_ROOT / "configs/stage2_v7_early_stopping_verification_on.yaml"
    )
    current = load_yaml_with_extends(
        PROJECT_ROOT / "configs/stage2_v7_minimal_cost_aware.yaml"
    )
    policy = V7MinimalCostAwareConfig.from_mapping(current["cost_aware"])
    current_verification = dict(current["verification"])
    legacy_verification = dict(legacy["verification"])
    current_verification.pop("max_rounds")
    legacy_verification.pop("max_rounds")
    plan = predeclare_pilot_eval_plan(
        [f"gain-{index}" for index in range(2850)],
        decision_rounds=range(policy.start_round, current["closed_loop"]["max_rounds"] + 1),
        pilots_per_round=policy.required_pilots,
        eval_size=policy.pilot_eval_size,
        seed=policy.pilot_seed,
    )
    radius = gain_confidence_radius(
        policy.required_pilots,
        policy.M,
        policy.K,
        policy.T,
        policy.delta,
    )
    checks = {
        "training_unchanged": current["training"] == legacy["training"],
        "selector_unchanged": (
            current["closed_loop"]["selection"]
            == legacy["closed_loop"]["selection"]
        ),
        "initial_pool_unchanged": (
            current["closed_loop"]["initial_pool_dir"]
            == legacy["closed_loop"]["initial_pool_dir"]
        ),
        "synthetic_generation_unchanged": (
            current["synthetic_generation"] == legacy["synthetic_generation"]
        ),
        "synthetic_qc_unchanged": (
            current["synthetic_validation"] == legacy["synthetic_validation"]
        ),
        "adaptive_verification_unchanged_except_horizon": (
            current_verification == legacy_verification
        ),
        "round_1_preserved": policy.start_round == 2,
        "budget_is_8x32": (
            current["closed_loop"]["max_rounds"] == 8
            and current["closed_loop"]["per_round_examples"] == 32
            and current["closed_loop"]["total_budget"] == 256
        ),
        "verification_horizon_covers_final_audit": (
            current["verification"]["max_rounds"]
            >= current["closed_loop"]["max_rounds"] + 1
        ),
        "decision_horizon_matches_rounds_2_to_8": policy.T == 7,
        "pilot_plan_disjoint": bool(plan["pairwise_disjoint"]),
        "held_out_gain_report_nonempty": bool(plan["held_out_gain_report_ids"]),
        "mixed_policy_does_not_claim_appendix_theorem": (
            current["cost_aware"]["decision_scope"]
            == "exact_v7_mixed_soft_pair_round_action"
        ),
        "lambda_has_explicit_ex_ante_calibration": (
            policy.calibration_reference_gain is not None
            and policy.calibration_reference_cost is not None
            and policy.lambda_cost
            == policy.calibration_reference_gain
            / policy.calibration_reference_cost
            and policy.calibration_currency
            == current["cost_aware"]["cost"]["currency"]
        ),
    }
    return {
        "status": "ready" if all(checks.values()) else "failed",
        "checks": checks,
        "legacy_config": "configs/stage2_v7_early_stopping_verification_on.yaml",
        "cost_aware_config": "configs/stage2_v7_minimal_cost_aware.yaml",
        "round_1_cost_decision": False,
        "decision_rounds": list(range(2, 9)),
        "gain_ucb": {
            "formula": "min(M, delta_hat + b) - lambda_cost * C",
            "M": policy.M,
            "m": policy.required_pilots,
            "K": policy.K,
            "T": policy.T,
            "delta": policy.delta,
            "radius": radius,
            "raw_radius_unchanged": True,
            "known_gain_support_upper_bound": policy.M,
            "lambda_cost": policy.lambda_cost,
            "calibration_reference_gain": policy.calibration_reference_gain,
            "calibration_reference_cost": policy.calibration_reference_cost,
            "calibration_currency": policy.calibration_currency,
            "maximum_ucb_break_even_cost": policy.M / policy.lambda_cost,
            "support_intersection_preserves_coverage": True,
        },
        "expected_stopping_behavior": (
            "STOP is feasible when lambda*C reaches the bounded gain UCB. "
            "It is not forced and no seed/final-test outcome calibrates lambda."
        ),
        "pilot_eval": {
            "source": 2850,
            "used": len(plan["pilot_ids"]),
            "held_out": len(plan["held_out_gain_report_ids"]),
        },
        "theory_claim": (
            "policy-level Hoeffding UCB only; appendix hypothesis-specific "
            "safe-stopping theorem not claimed"
        ),
    }


def main() -> None:
    result = audit_protocol()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

