#!/usr/bin/env python3
"""CPU-only validation of the minimal V7 Cost-Aware controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.v7_minimal_cost_aware import (  # noqa: E402
    MIXED_POLICY_ID,
    V7MinimalCostAwareConfig,
    decide_mixed_policy_round,
    predeclare_pilot_eval_plan,
)
from src.gain.estimation import estimate_gain  # noqa: E402
from src.utils.config import load_yaml_with_extends  # noqa: E402


def main() -> None:
    config = load_yaml_with_extends(
        PROJECT_ROOT / "configs/stage2_v7_minimal_cost_aware.yaml"
    )
    policy = V7MinimalCostAwareConfig.from_mapping(config["cost_aware"])
    plan = predeclare_pilot_eval_plan(
        [f"gain-{index:04d}" for index in range(2850)],
        decision_rounds=range(2, 9),
        pilots_per_round=policy.required_pilots,
        eval_size=policy.pilot_eval_size,
        seed=policy.pilot_seed,
    )
    complete = estimate_gain(
        MIXED_POLICY_ID,
        [0.0] * policy.required_pilots,
        M=policy.M,
        delta=policy.delta,
        K=policy.K,
        T=policy.T,
    )
    incomplete = estimate_gain(
        MIXED_POLICY_ID,
        [0.0] * (policy.required_pilots - 1),
        M=policy.M,
        delta=policy.delta,
        K=policy.K,
        T=policy.T,
    )
    certified = {"h1": {"state": "certified"}}
    unresolved = {
        "h1": {"state": "certified"},
        "h2": {"state": "unresolved"},
    }
    decisions = {
        "positive": decide_mixed_policy_round(
            states=certified,
            estimate=complete,
            required_pilots=policy.required_pilots,
            cost=1.0,
            lambda_cost=policy.lambda_cost,
        ).to_dict(),
        "nonpositive": decide_mixed_policy_round(
            states=certified,
            estimate=complete,
            required_pilots=policy.required_pilots,
            cost=1_000_000.0,
            lambda_cost=policy.lambda_cost,
        ).to_dict(),
        "unresolved": decide_mixed_policy_round(
            states=unresolved,
            estimate=complete,
            required_pilots=policy.required_pilots,
            cost=1_000_000.0,
            lambda_cost=policy.lambda_cost,
        ).to_dict(),
        "insufficient_pilots": decide_mixed_policy_round(
            states=certified,
            estimate=incomplete,
            required_pilots=policy.required_pilots,
            cost=1_000_000.0,
            lambda_cost=policy.lambda_cost,
        ).to_dict(),
    }
    assert decisions["positive"]["action"] == "continue"
    assert decisions["nonpositive"]["action"] == "stop"
    assert decisions["unresolved"]["action"] == "continue"
    assert decisions["unresolved"]["gain_cost_stop_evaluated"] is False
    assert decisions["insufficient_pilots"]["action"] == "continue"
    assert plan["pairwise_disjoint"] is True
    assert len(plan["held_out_gain_report_ids"]) == 1058
    print(
        json.dumps(
            {
                "status": "pass",
                "config": policy.__dict__,
                "pilot_plan": {
                    "pilot_ids": len(plan["pilot_ids"]),
                    "held_out_gain_report_ids": len(
                        plan["held_out_gain_report_ids"]
                    ),
                },
                "decisions": decisions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

