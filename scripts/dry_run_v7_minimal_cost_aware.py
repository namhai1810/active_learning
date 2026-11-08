#!/usr/bin/env python3
"""CPU-only validation of the minimal V7 Cost-Aware controller."""

from __future__ import annotations

import json
import sys
import tempfile
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
from src.decision.cost_model import CostUsage  # noqa: E402
from src.models.lora_training import EarlyStoppingConfig  # noqa: E402
from src.utils.config import load_yaml_with_extends  # noqa: E402
from scripts.run_stage2_closed_loop import (  # noqa: E402
    _finalize_mixed_policy_round_cost,
    _mixed_policy_round_cost,
)


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
        shared_eval_shard_within_round=(
            policy.shared_eval_shard_within_round
        ),
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
            cost=(complete.radius / policy.lambda_cost) - 0.01,
            lambda_cost=policy.lambda_cost,
        ).to_dict(),
        "nonpositive": decide_mixed_policy_round(
            states=certified,
            estimate=complete,
            required_pilots=policy.required_pilots,
            cost=complete.radius / policy.lambda_cost,
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
    assert decisions["unresolved"]["action"] == "stop"
    assert decisions["unresolved"]["gain_cost_stop_evaluated"] is True
    assert decisions["insufficient_pilots"]["action"] == "continue"
    assert plan["pairwise_disjoint"] is True
    assert len(plan["pilot_ids"]) == 448
    assert len(plan["held_out_gain_report_ids"]) == 2402
    with tempfile.TemporaryDirectory(prefix="v7-cost-v2-dry-run-") as temp_dir:
        ledger = _mixed_policy_round_cost(
            refresh_dir=Path(temp_dir),
            teacher_usage={"input_tokens": 0.0, "output_tokens": 0.0},
            observations=[],
            eligible_examples=0,
            acquired_examples=32,
            stopping_examples=900,
            early_stopping=EarlyStoppingConfig(
                max_epochs=8, patience=2, min_delta=0.001
            ),
            cost_config=config["cost_aware"]["cost"],
        )
        stopped_ledger = _finalize_mixed_policy_round_cost(
            ledger, action="stop"
        )
        continued_ledger = _finalize_mixed_policy_round_cost(
            ledger,
            action="continue",
            actual_main_update_usage=CostUsage(
                lora_training_runs=1,
                lora_training_gpu_seconds=5.0,
                evaluation_runs=4,
                evaluation_items=3600,
                evaluation_gpu_seconds=50.0,
            ),
        )
    assert ledger["incurred_exploration_cost"] >= 0.0
    assert ledger["decision_cost"] > 0.0
    assert (
        stopped_ledger["total_pipeline_cost"]
        == ledger["incurred_exploration_cost"]
    )
    assert (
        continued_ledger["total_pipeline_cost"]
        == ledger["incurred_exploration_cost"]
        + continued_ledger["actual_main_update_cost"]
    )
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
                "cost_accounting": {
                    "incurred_exploration_cost": ledger[
                        "incurred_exploration_cost"
                    ],
                    "decision_cost": ledger["decision_cost"],
                    "stop_total_pipeline_cost": stopped_ledger[
                        "total_pipeline_cost"
                    ],
                    "continue_total_pipeline_cost": continued_ledger[
                        "total_pipeline_cost"
                    ],
                    "incurred_cost_enters_U": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

