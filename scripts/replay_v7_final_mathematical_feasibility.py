#!/usr/bin/env python3
"""CPU-only final mathematical-feasibility replay for V7 Cost-Aware stopping.

This script reads frozen seeds 42/43.  It never loads a model, performs
inference, calls a teacher, trains, or writes into historical run directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision.cost_model import cost_rates_from_config
from src.decision.final_mathematical_feasibility import (
    compare_gain_bounds,
    consecutive_gain_drifts,
    maximum_rho_for_stop,
    net_ucb,
    protocol_full_round_cost_lower_bound,
)


DEFAULT_SOURCE = Path(
    "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/stage2_diagnostics/v7_final_mathematical_feasibility"
)
DEFAULT_REPORT = Path(
    "reports/v7_minimal_cost_aware/FINAL_MATHEMATICAL_FEASIBILITY.md"
)
EXPECTED = {
    "required_pilots": 32,
    "delta": 0.05,
    "K": 1,
    "T": 7,
    "lambda_cost": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_v7_minimal_cost_aware.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    from src.utils.config import load_yaml_with_extends

    return load_yaml_with_extends(path)


def protocol(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("cost_aware")
    if not isinstance(value, dict):
        raise ValueError("Missing cost_aware config")
    for key, expected in EXPECTED.items():
        if value.get(key) != expected:
            raise ValueError(
                f"Frozen protocol mismatch for {key}: {value.get(key)!r} != {expected!r}"
            )
    return value


def historical_rounds(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for item in result.get("cost_aware_rounds", []):
        if isinstance(item, dict) and item.get("round") is not None:
            rows[int(item["round"])] = item
    return rows


def checkpoint_accuracy(seed: int, round_index: int) -> tuple[float | None, str | None]:
    path = (
        Path("outputs/stage2_diagnostics/v7_one_step_cost_aware/checkpoint_metrics")
        / f"seed_{seed}"
        / f"round_{round_index:03d}"
        / "metrics.json"
    )
    if not path.is_file():
        return None, None
    payload = read_json(path)
    return float(payload["direct_binary_accuracy"]), str(path)


def actual_component(round_payload: dict[str, Any], name: str) -> float:
    decision_cost = round_payload.get("decision_cost")
    if not isinstance(decision_cost, dict):
        return 0.0
    components = decision_cost.get("components")
    if not isinstance(components, dict):
        return 0.0
    return float(components.get(name, 0.0))


def actual_total(round_payload: dict[str, Any]) -> float:
    decision_cost = round_payload.get("decision_cost")
    if not isinstance(decision_cost, dict):
        return 0.0
    value = decision_cost.get("total_pipeline_cost")
    return 0.0 if value is None else float(value)


def teacher_data_cost(round_payload: dict[str, Any]) -> float:
    return math.fsum(
        actual_component(round_payload, name)
        for name in (
            "llm_inference_input",
            "llm_inference_output",
            "data_generation",
            "data_verification",
        )
    )


def validate_inputs(
    *, source_root: Path, seeds: list[int], config_path: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    cost_aware = protocol(config)
    manifest = Path(config["closed_loop"]["stopping_subset_manifest"])
    manifest_payload = read_json(manifest)
    stopping_count = len(manifest_payload["stopping_eval_ids"])
    checks: list[dict[str, Any]] = []
    for seed in seeds:
        result_path = source_root / f"seed_{seed}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing historical result: {result_path}")
        result = read_json(result_path)
        rounds = historical_rounds(result)
        if sorted(index for index in rounds if 2 <= index <= 8) != list(range(2, 9)):
            raise ValueError(f"Seed {seed} does not contain rounds 2-8")
        for round_index in range(2, 9):
            observations = rounds[round_index].get("pilot_observations") or []
            if not isinstance(observations, list):
                raise ValueError(f"Invalid pilots seed={seed} round={round_index}")
            if len(observations) not in {0, EXPECTED["required_pilots"]}:
                raise ValueError(
                    f"Partial pilot cache seed={seed} round={round_index}: {len(observations)}"
                )
            checks.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "cached_pilots": len(observations),
                    "gain_cost_evaluated": bool(
                        (rounds[round_index].get("decision") or {}).get(
                            "gain_cost_stop_evaluated"
                        )
                    ),
                }
            )
    return {
        "status": "valid",
        "gpu_or_model_work_required": False,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_root": str(source_root),
        "seeds": seeds,
        "stopping_manifest": str(manifest),
        "stopping_manifest_sha256": sha256(manifest),
        "stopping_examples": stopping_count,
        "protocol": {key: cost_aware[key] for key in EXPECTED},
        "round_checks": checks,
    }


def replay(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    config = load_config(args.config)
    cost_aware = protocol(config)
    rates, currency = cost_rates_from_config(cost_aware["cost"])
    cost_lcb = protocol_full_round_cost_lower_bound(
        rates=rates,
        currency=currency,
        acquired_examples=int(config["closed_loop"]["per_round_examples"]),
        stopping_examples=int(validation["stopping_examples"]),
    )
    lambda_cost = float(cost_aware["lambda_cost"])
    penalty = lambda_cost * cost_lcb.total
    rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in args.seeds:
        result_path = args.source_root / f"seed_{seed}" / "result.json"
        result = read_json(result_path)
        rounds = historical_rounds(result)
        bound_by_round: dict[int, dict[str, Any]] = {}
        mean_by_round: dict[int, float] = {}
        r8_accuracy, r8_path = checkpoint_accuracy(seed, 8)

        for round_index in range(2, 9):
            payload = rounds[round_index]
            observations = payload.get("pilot_observations") or []
            decision = payload.get("decision") or {}
            accuracy, accuracy_path = checkpoint_accuracy(seed, round_index)
            base: dict[str, Any] = {
                "seed": seed,
                "round": round_index,
                "budget_before": 32 * (round_index - 1),
                "budget_after": 32 * round_index,
                "m": len(observations),
                "evidence_gate_gain_cost_evaluated": bool(
                    decision.get("gain_cost_stop_evaluated")
                ),
                "cost_lcb": cost_lcb.total,
                "cost_penalty": penalty,
                "accuracy_direct_binary": accuracy,
                "retrospective_regret_vs_r8": (
                    None
                    if accuracy is None or r8_accuracy is None
                    else r8_accuracy - accuracy
                ),
                "accuracy_artifact": accuracy_path,
                "actual_pipeline_cost": actual_total(payload),
                "actual_teacher_generation_verification_cost": teacher_data_cost(payload),
                "actual_lora_training_cost": actual_component(payload, "lora_training"),
                "actual_evaluation_cost": actual_component(payload, "evaluation"),
                "historical_cost_is_not_online_lcb": True,
            }
            if len(observations) != EXPECTED["required_pilots"]:
                base.update(
                    {
                        "gain_status": "not_evaluated_due_to_evidence_gate",
                        "post_acquisition_action": "uncertified_continue",
                        "post_acquisition_reason": "missing_complete_pilot_set",
                    }
                )
                missing.append(
                    {
                        "seed": seed,
                        "round": round_index,
                        "missing": "32 cached pilot observations",
                        "historical_reason": decision.get("reason"),
                    }
                )
                rows.append(base)
                continue

            comparison, audit = compare_gain_bounds(
                observations,
                required_pilots=EXPECTED["required_pilots"],
                delta=EXPECTED["delta"],
                K=EXPECTED["K"],
                T=EXPECTED["T"],
            )
            values = comparison.to_dict()
            h_net = net_ucb(
                gain_ucb=comparison.hoeffding_gain_ucb,
                rho=0.0,
                lambda_cost=lambda_cost,
                cost_lcb=cost_lcb.total,
            )
            eb_net = net_ucb(
                gain_ucb=comparison.empirical_bernstein_gain_ucb,
                rho=0.0,
                lambda_cost=lambda_cost,
                cost_lcb=cost_lcb.total,
            )
            base.update(values)
            maximum_radius_for_stop = penalty - comparison.delta_hat
            base.update(
                {
                    "gain_status": "conditional_bound_assumptions_unproved",
                    "legacy_symmetric_hoeffding_radius": 2.0
                    * comparison.hoeffding_radius,
                    "hoeffding_post_acquisition_net_ucb_rho0": h_net,
                    "empirical_bernstein_post_acquisition_net_ucb_rho0": eb_net,
                    "hoeffding_required_cost_lcb_for_stop_rho0": (
                        comparison.hoeffding_gain_ucb / lambda_cost
                    ),
                    "empirical_bernstein_required_cost_lcb_for_stop_rho0": (
                        comparison.empirical_bernstein_gain_ucb / lambda_cost
                    ),
                    "hoeffding_cost_lcb_gap": max(
                        0.0,
                        comparison.hoeffding_gain_ucb / lambda_cost
                        - cost_lcb.total,
                    ),
                    "maximum_hoeffding_radius_for_stop": maximum_radius_for_stop,
                    "hoeffding_radius_reduction_needed_for_stop": (
                        max(
                            0.0,
                            comparison.hoeffding_radius
                            - maximum_radius_for_stop,
                        )
                        if maximum_radius_for_stop >= 0.0
                        else None
                    ),
                    "mean_gain_excess_over_cost_penalty_at_zero_radius": max(
                        0.0, comparison.delta_hat - penalty
                    ),
                    "empirical_bernstein_cost_lcb_gap": max(
                        0.0,
                        comparison.empirical_bernstein_gain_ucb / lambda_cost
                        - cost_lcb.total,
                    ),
                    "post_acquisition_action": "uncertified_continue",
                    "post_acquisition_reason": (
                        "conditional_independence_not_proved"
                        if h_net > 0 and eb_net > 0
                        else "numeric_stop_not_promoted_without_assumptions"
                    ),
                    "same_round_pilots_available_pre_round": False,
                    "mechanical_pilot_checks_pass": audit["status"]
                    == "mechanically_valid",
                }
            )
            bound_by_round[round_index] = values
            mean_by_round[round_index] = comparison.delta_hat
            rows.append(base)

        for item in consecutive_gain_drifts(mean_by_round):
            drift_rows.append(
                {
                    "seed": seed,
                    "from_round": int(item["from_round"]),
                    "to_round": int(item["to_round"]),
                    "signed_drift": item["signed_drift"],
                    "absolute_drift": item["absolute_drift"],
                    "use": "retrospective_diagnostic_only",
                    "theoretical_rho_certificate": False,
                }
            )

        # Pre-round transfer uses only the preceding round's pilots.  It is
        # still uncertified because no valid cross-round stability radius rho
        # can be calibrated from these two dependent trajectories.
        for row in rows:
            if row["seed"] != seed:
                continue
            round_index = int(row["round"])
            source = bound_by_round.get(round_index - 1)
            row["pre_round_source_round"] = round_index - 1 if source else None
            row["rho_status"] = "unproved_cross_round_stability"
            row["rho_value_used_for_certificate"] = None
            # Before R8 there is still one legal 32-example action.  Budget
            # exhaustion occurs only after that action, never as the R8
            # pre-round decision.
            row["pre_round_decision"] = "uncertified_continue"
            row["pre_round_stop_certified"] = False
            row["pre_round_reason"] = "no_valid_cross_round_rho"
            row["terminal_after_round"] = (
                "budget_stop" if round_index == 8 else None
            )
            if source:
                previous_ucb = float(source["hoeffding_gain_ucb"])
                row["pre_round_hoeffding_gain_ucb_at_rho0"] = previous_ucb
                row["maximum_rho_compatible_with_stop"] = maximum_rho_for_stop(
                    gain_ucb=previous_ucb,
                    lambda_cost=lambda_cost,
                    cost_lcb=cost_lcb.total,
                )
                row["pre_round_net_ucb_at_rho0_diagnostic"] = net_ucb(
                    gain_ucb=previous_ucb,
                    rho=0.0,
                    lambda_cost=lambda_cost,
                    cost_lcb=cost_lcb.total,
                )
            else:
                row["pre_round_hoeffding_gain_ucb_at_rho0"] = None
                row["maximum_rho_compatible_with_stop"] = None
                row["pre_round_net_ucb_at_rho0_diagnostic"] = None

        seed_summaries.append(
            {
                "seed": seed,
                "earliest_certified_stop_round": None,
                "terminal_outcome": "budget_stop_after_round_8",
                "certified_avoidable_budget": 0,
                "certified_avoidable_pipeline_cost": 0.0,
                "certified_avoidable_teacher_cost": 0.0,
                "certified_avoidable_training_cost": 0.0,
                "retrospective_regret_at_valid_outcome": 0.0,
                "r8_accuracy": r8_accuracy,
                "r8_accuracy_artifact": r8_path,
            }
        )

    max_abs_drift = max((row["absolute_drift"] for row in drift_rows), default=None)
    max_positive_drift = max((row["signed_drift"] for row in drift_rows), default=None)
    complete = [row for row in rows if row.get("m") == 32]
    summary = {
        "schema": "v7_final_mathematical_feasibility_v1",
        "decision_target": "pre_round_next_intervention_expected_net_gain",
        "formula": "U_net = gain_UCB + rho - lambda * full_round_cost_LCB",
        "lambda_cost": lambda_cost,
        "delta": EXPECTED["delta"],
        "K": EXPECTED["K"],
        "T": EXPECTED["T"],
        "sealed_test_used_for_decision": False,
        "future_round_metrics_used_for_decision": False,
        "same_round_pilots_available_pre_round": False,
        "cost_lower_bound": cost_lcb.to_dict(),
        "prior_v5_cost_lower_bound": 0.02,
        "cost_lcb_improvement": cost_lcb.total - 0.02,
        "drift": {
            "historical_max_absolute": max_abs_drift,
            "historical_max_positive": max_positive_drift,
            "use": "diagnostic_only",
            "valid_rho_proved": False,
            "reason": (
                "two dependent trajectories with missing pilot rounds cannot "
                "calibrate a future-round stability guarantee"
            ),
        },
        "gain_bounds": {
            "complete_rounds": len(complete),
            "hoeffding_radius": (
                complete[0].get("hoeffding_radius") if complete else None
            ),
            "empirical_bernstein_radius_min": min(
                (row["empirical_bernstein_radius"] for row in complete),
                default=None,
            ),
            "empirical_bernstein_radius_max": max(
                (row["empirical_bernstein_radius"] for row in complete),
                default=None,
            ),
            "empirical_bernstein_tighter_rounds": sum(
                row["empirical_bernstein_radius"] < row["hoeffding_radius"]
                for row in complete
            ),
            "conditional_independence_proved": False,
        },
        "numeric_bottleneck": {
            "post_acquisition_hoeffding_net_ucb_min": min(
                (
                    row["hoeffding_post_acquisition_net_ucb_rho0"]
                    for row in complete
                ),
                default=None,
            ),
            "post_acquisition_hoeffding_net_ucb_max": max(
                (
                    row["hoeffding_post_acquisition_net_ucb_rho0"]
                    for row in complete
                ),
                default=None,
            ),
            "additional_cost_lcb_needed_min": min(
                (row["hoeffding_cost_lcb_gap"] for row in complete),
                default=None,
            ),
            "additional_cost_lcb_needed_max": max(
                (row["hoeffding_cost_lcb_gap"] for row in complete),
                default=None,
            ),
            "radius_reduction_needed_min_when_possible": min(
                (
                    row["hoeffding_radius_reduction_needed_for_stop"]
                    for row in complete
                    if row["hoeffding_radius_reduction_needed_for_stop"] is not None
                ),
                default=None,
            ),
        },
        "earliest_certified_stop_round": None,
        "certified_stop_exists": False,
        "valid_online_outcome": "uncertified_continue_until_budget_stop",
        "seed_summaries": seed_summaries,
        "missing_data": missing,
    }
    return {
        "summary": summary,
        "per_round": rows,
        "drift": drift_rows,
        "provenance": {
            "validation": validation,
            "historical_sources_are_read_only": True,
        },
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["per_round"]
    complete = [row for row in rows if row.get("m") == 32]
    cost = summary["cost_lower_bound"]
    drift = summary["drift"]
    bottleneck = summary["numeric_bottleneck"]
    lines = [
        "# V7 Cost-Aware Stopping — Final Mathematical Feasibility",
        "",
        "> CPU-only offline audit of frozen seeds 42/43. No model inference, teacher call, training, or historical-artifact mutation was performed.",
        "",
        "## Executive result",
        "",
        "No valid pre-round certified STOP was obtained. The valid online classification is `uncertified_continue` through R8, followed by `budget_stop`; budget exhaustion is not a gain–cost certificate.",
        "",
        "The numerical bounds improved, but two assumptions remain unproved: (1) conditional independence/unbiasedness of the 32 fixed-shard pilot training randomizations, and (2) a valid cross-round stability radius `rho` that transfers post-acquisition gain evidence to the next pre-round decision.",
        "",
        f"Even in the post-acquisition `rho=0` diagnostic, Hoeffding Net-UCB stays positive by `{fmt(bottleneck['post_acquisition_hoeffding_net_ucb_min'])}`–`{fmt(bottleneck['post_acquisition_hoeffding_net_ucb_max'])}`. At the frozen `lambda`, the proven cost LCB would need to increase by at least `${fmt(bottleneck['additional_cost_lcb_needed_min'])}` (up to `${fmt(bottleneck['additional_cost_lcb_needed_max'])}` across complete rounds), or the Hoeffding radius would need to shrink by at least `{fmt(bottleneck['radius_reduction_needed_min_when_possible'])}` in rounds where zero-radius STOP is numerically possible.",
        "",
        "## A/B/C before–after summary",
        "",
        "| Side | Before | Final audit | Consequence |",
        "|---|---|---|---|",
        f"| A — gain | Symmetric Hoeffding radius `{fmt(2.0 * complete[0]['hoeffding_radius'])}` | Range-aware Hoeffding `{fmt(complete[0]['hoeffding_radius'])}`; empirical Bernstein `{fmt(summary['gain_bounds']['empirical_bernstein_radius_min'])}`–`{fmt(summary['gain_bounds']['empirical_bernstein_radius_max'])}` | EB is tighter in {summary['gain_bounds']['empirical_bernstein_tighter_rounds']}/{summary['gain_bounds']['complete_rounds']} complete rounds; its finite-sample correction is not automatically beneficial. |",
        f"| B — cost | Prior v5 universally proven lower bound `$0.020000` | Conditional executable-round LCB `${fmt(cost['total'])}` | Larger penalty, but discovery/verification/generation remain zero in the LCB because their positive minimum is state-dependent. |",
        f"| C — cross-round | No next-round transfer certificate | Historical max absolute drift `{fmt(drift['historical_max_absolute'])}` is diagnostic only; valid `rho` remains unavailable | Same-round pilots cannot be used before that round; pre-round decisions remain uncertified. |",
        "",
        "## A. Variance-aware gain bounds",
        "",
        "`X_j = accuracy_after_j - accuracy_before` uses exact support `[-A0, 1-A0]` (width 1). The family-wise allocation is `delta/(K*T) = 0.05/7`. UCBs are intersected with the exact support upper endpoint.",
        "",
        "| Seed | Round | m | Mean gain | Variance | Hoeffding r | EB r | H gain-UCB | EB gain-UCB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round']} | {row['m']} | {fmt(row.get('delta_hat'))} | {fmt(row.get('sample_variance'))} | {fmt(row.get('hoeffding_radius'))} | {fmt(row.get('empirical_bernstein_radius'))} | {fmt(row.get('hoeffding_gain_ucb'))} | {fmt(row.get('empirical_bernstein_gain_ucb'))} |"
        )
    lines.extend(
        [
            "",
            "Mechanical checks confirm one fixed evaluation shard/baseline, the gain identity, bounded support, the same selected batch/checkpoint, and distinct pilot seeds. They do **not** prove conditional independence of deterministic seeds or population-level unbiasedness. Accordingly these are conditional fixed-shard bounds, not a population theorem.",
            "",
            "## B. Protocol-based full-round cost lower bound",
            "",
            f"The audited lower bound is **${fmt(cost['total'])}** per executable 32-example intervention. With `lambda=0.2`, the certified cost penalty is `{fmt(0.2 * cost['total'])}` gain units.",
            "",
            "| Stage | Included USD | In LCB | Why |",
            "|---|---:|:---:|---|",
        ]
    )
    for component in cost["components"]:
        lines.append(
            f"| {component['stage']} | {fmt(component['lower_bound'])} | {'yes' if component['included'] else 'no'} | {component['reason']} |"
        )
    lines.extend(
        [
            "",
            "Historical realised costs are reported separately and never substituted for the online LCB. GPU-time projections are also excluded because a projection is not a proven positive runtime lower bound.",
            "",
            "| Seed | Round | H Net-UCB (rho=0, post-acq) | EB Net-UCB | H required LCB | H LCB gap | Actual pipeline USD |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round']} | {fmt(row.get('hoeffding_post_acquisition_net_ucb_rho0'))} | {fmt(row.get('empirical_bernstein_post_acquisition_net_ucb_rho0'))} | {fmt(row.get('hoeffding_required_cost_lcb_for_stop_rho0'))} | {fmt(row.get('hoeffding_cost_lcb_gap'))} | {fmt(row.get('actual_pipeline_cost'))} |"
        )
    lines.extend(
        [
            "",
            "These Net-UCB values are post-acquisition diagnostics with `rho=0` for the same intervention. They are not promoted to pre-round certificates.",
            "",
            "## C. Cross-round drift and pre-round validity",
            "",
            f"Observed consecutive-round maximum absolute drift is `{fmt(drift['historical_max_absolute'])}` and maximum signed increase is `{fmt(drift['historical_max_positive'])}`. Both are retrospective diagnostics from only two dependent trajectories; neither is a valid `rho` calibration.",
            "",
            "| Seed | Decision round | Prior pilot round | Gain-UCB at rho=0 | Max rho compatible with STOP | Classification |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round']} | {fmt(row.get('pre_round_source_round'), 0)} | {fmt(row.get('pre_round_hoeffding_gain_ucb_at_rho0'))} | {fmt(row.get('maximum_rho_compatible_with_stop'))} | {row.get('pre_round_decision')} |"
        )
    lines.extend(
        [
            "",
            "A negative `max rho compatible with STOP` means even the unjustified optimistic choice `rho=0` would not stop; a non-negative value would still require a proved drift bound no larger than that value.",
            "",
            "## Decision and savings",
            "",
            "| Seed | Earliest certified STOP | Terminal outcome | Certified budget saved | Certified teacher/data cost saved | Certified training cost saved | R8 regret |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for seed in summary["seed_summaries"]:
        lines.append(
            f"| {seed['seed']} | — | {seed['terminal_outcome']} | {seed['certified_avoidable_budget']} | {fmt(seed['certified_avoidable_teacher_cost'])} | {fmt(seed['certified_avoidable_training_cost'])} | {fmt(seed['retrospective_regret_at_valid_outcome'])} |"
        )
    lines.extend(
        [
            "",
            "Because no certified STOP exists, scientifically attributable savings are zero. Per-round R8 regrets and historical costs in the CSV/JSON are retrospective evaluation fields only.",
            "",
            "## Missing evidence and theorem readiness",
            "",
            "Confirmed: frozen `lambda/delta/K/T`, complete 32-pilot caches where the evidence gate ran, fixed baseline/shard, gain identity, exact width-one support, family-wise correction, immutable cost rates, and no sealed/future metric in the decision calculation.",
            "",
            "Not proved: pilot conditional independence/unbiasedness, a population-level fixed-shard guarantee, a pre-round gain estimator, and a valid cross-round stability radius. Rounds with zero cached pilots were intentionally skipped by historical evidence gates and are not reconstructed.",
            "",
            "The implementation and negative feasibility result are sufficient for an experimental/limitations section. They are **not sufficient for a stopping theorem claiming a valid pre-round certified STOP**. Such a theorem needs a predeclared pre-round gain construction and independently justified cross-round stability (or a different theorem that avoids transferring post-acquisition pilots).",
            "",
            "## Provenance",
            "",
            f"- Historical source: `{payload['provenance']['validation']['source_root']}`",
            "- Historical artifacts were read-only.",
            "- Sealed HANS/future-round metrics were not used for decisions.",
            "- Accuracy metrics are retrospective diagnostics only.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validation = validate_inputs(
        source_root=args.source_root, seeds=args.seeds, config_path=args.config
    )
    validation_path = args.output_dir / "validation.json"
    atomic_json(validation_path, validation)
    print(f"[VALID] {validation_path}")
    if args.validate_only:
        return

    payload = replay(args, validation)
    atomic_json(args.output_dir / "replay.json", payload)
    write_csv(args.output_dir / "per_round.csv", payload["per_round"])
    write_csv(args.output_dir / "drift.csv", payload["drift"])
    cost_rows = payload["summary"]["cost_lower_bound"]["components"]
    write_csv(args.output_dir / "cost_lower_bound.csv", cost_rows)
    atomic_text(args.report, render_report(payload))
    print(f"[DONE] {args.output_dir / 'replay.json'}")
    print(f"[DONE] {args.output_dir / 'per_round.csv'}")
    print(f"[DONE] {args.report}")


if __name__ == "__main__":
    main()
