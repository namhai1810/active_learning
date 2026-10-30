#!/usr/bin/env python3
"""CPU-only deterministic dry-run of V7 STOP/CONTINUE and cost accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.cost_aware import build_decision_candidates, select_intervention
from src.decision.safe_stopping import cumulative_cost_ledger


def _decision(delta_hat: float, radius: float, cost: float):
    candidates = build_decision_candidates(
        [{"hypothesis_id": "h", "certified": True}],
        [
            {
                "hypothesis_id": "h",
                "m": 4,
                "delta_hat": delta_hat,
                "radius": radius,
                "status": "estimated",
            }
        ],
        {"h": cost},
        0.1,
    )
    return select_intervention(candidates, 0.1).to_dict()


def main() -> None:
    first = _decision(delta_hat=0.4, radius=0.2, cost=1.0)
    if first["action"] != "continue":
        raise RuntimeError("Dry-run CONTINUE branch failed")
    ledger = cumulative_cost_ledger(
        0.0, incurred_exploration=0.25, realized_intervention=0.5
    )
    second = _decision(delta_hat=-0.4, radius=0.2, cost=1.0)
    if second["action"] != "stop":
        raise RuntimeError("Dry-run STOP branch failed")
    ledger = cumulative_cost_ledger(
        ledger["cumulative_after"],
        incurred_exploration=0.25,
        realized_intervention=0.0,
    )
    print(
        json.dumps(
            {
                "device": "cpu_only_no_model_loaded",
                "round_1": first,
                "round_2": second,
                "cost_ledger": ledger,
                "checkpoint_continuity": "covered_by artifact-resume unit test",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
