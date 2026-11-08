#!/usr/bin/env python3
"""CPU-only sliding-window replay over frozen V7 pilot trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision.cost_model import cost_rates_from_config
from src.decision.final_mathematical_feasibility import (
    protocol_full_round_cost_lower_bound,
    range_aware_hoeffding_radius,
)
from src.decision.time_adaptive_stopping import (
    assess_time_adaptive_decision,
    historical_window_estimate,
    planned_test_family_size,
    simultaneous_hoeffding_radius,
    weighted_temporal_drift,
)
from src.gain.range_aware import validate_fixed_shard_pilot_observations
from src.utils.config import load_yaml_with_extends


DEFAULT_SOURCE = Path(
    "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2"
)
DEFAULT_OUTPUT = Path("outputs/stage2_diagnostics/v7_time_adaptive_stopping")
DEFAULT_REPORT = Path(
    "reports/v7_minimal_cost_aware/TIME_ADAPTIVE_STOPPING.md"
)
DEFAULT_CONFIG = Path("configs/stage2_v7_minimal_cost_aware.yaml")
FIXED_WINDOWS = tuple(range(1, 8))
DECISION_ROUNDS = tuple(range(2, 9))
DRIFT_SCENARIOS = (0.0, 0.005, 0.01, 0.02)
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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


def round_payloads(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for item in result.get("cost_aware_rounds", []):
        if isinstance(item, dict) and item.get("round") is not None:
            payloads[int(item["round"])] = item
    return payloads


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


def component(payload: dict[str, Any], name: str) -> float:
    cost = payload.get("decision_cost")
    if not isinstance(cost, dict):
        return 0.0
    components = cost.get("components")
    if not isinstance(components, dict):
        return 0.0
    return float(components.get(name, 0.0))


def total_pipeline_cost(payload: dict[str, Any]) -> float:
    cost = payload.get("decision_cost")
    if not isinstance(cost, dict) or cost.get("total_pipeline_cost") is None:
        return 0.0
    return float(cost["total_pipeline_cost"])


def avoided_costs(payloads: dict[int, dict[str, Any]], decision_round: int) -> dict[str, float]:
    future = [payloads[index] for index in range(decision_round, 9)]
    return {
        "posthoc_pipeline_cost_avoided": math.fsum(
            total_pipeline_cost(item) for item in future
        ),
        "posthoc_teacher_data_cost_avoided": math.fsum(
            component(item, name)
            for item in future
            for name in (
                "llm_inference_input",
                "llm_inference_output",
                "data_generation",
                "data_verification",
            )
        ),
        "posthoc_lora_cost_avoided": math.fsum(
            component(item, "lora_training") for item in future
        ),
        "posthoc_evaluation_cost_avoided": math.fsum(
            component(item, "evaluation") for item in future
        ),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_with_extends(args.config)
    cost_aware = protocol(config)
    n_tests = planned_test_family_size(
        decision_rounds=len(DECISION_ROUNDS),
        fixed_windows=len(FIXED_WINDOWS),
        include_cumulative=True,
    )
    stopping_manifest = Path(config["closed_loop"]["stopping_subset_manifest"])
    stopping_examples = len(read_json(stopping_manifest)["stopping_eval_ids"])
    checks: list[dict[str, Any]] = []
    for seed in args.seeds:
        result_path = args.source_root / f"seed_{seed}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing historical result: {result_path}")
        result = read_json(result_path)
        payloads = round_payloads(result)
        if sorted(payloads) != list(range(1, 9)):
            raise ValueError(f"Seed {seed} does not contain rounds 1-8")
        for round_index in range(2, 9):
            observations = payloads[round_index].get("pilot_observations") or []
            if not isinstance(observations, list):
                raise ValueError(f"Invalid pilot cache seed={seed} round={round_index}")
            if len(observations) not in {0, EXPECTED["required_pilots"]}:
                raise ValueError(
                    f"Partial pilot cache seed={seed} round={round_index}: {len(observations)}"
                )
            mechanical = None
            if observations:
                mechanical = validate_fixed_shard_pilot_observations(
                    observations, required_pilots=EXPECTED["required_pilots"]
                )
            checks.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "cached_pilots": len(observations),
                    "mechanical_status": (
                        mechanical["status"] if mechanical is not None else "no_pilots"
                    ),
                    "conditional_independence_proved": False,
                }
            )
    return {
        "status": "valid",
        "cpu_only": True,
        "gpu_or_model_work_required": False,
        "source_root": str(args.source_root),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "stopping_manifest": str(stopping_manifest),
        "stopping_manifest_sha256": sha256(stopping_manifest),
        "stopping_examples": stopping_examples,
        "fixed_windows": list(FIXED_WINDOWS),
        "include_cumulative": True,
        "decision_rounds": list(DECISION_ROUNDS),
        "n_tests": n_tests,
        "bonferroni_alpha_per_test": EXPECTED["delta"] / n_tests,
        "protocol": {key: cost_aware[key] for key in EXPECTED},
        "round_checks": checks,
    }


def replay(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    config = load_yaml_with_extends(args.config)
    cost_aware = protocol(config)
    rates, currency = cost_rates_from_config(cost_aware["cost"])
    cost_lcb = protocol_full_round_cost_lower_bound(
        rates=rates,
        currency=currency,
        acquired_examples=int(config["closed_loop"]["per_round_examples"]),
        stopping_examples=int(validation["stopping_examples"]),
    )
    lambda_cost = float(cost_aware["lambda_cost"])
    n_tests = int(validation["n_tests"])
    baseline_radius = range_aware_hoeffding_radius(
        32,
        delta=EXPECTED["delta"],
        K=EXPECTED["K"],
        T=EXPECTED["T"],
        width=1.0,
    )
    window_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in args.seeds:
        result_path = args.source_root / f"seed_{seed}" / "result.json"
        payloads = round_payloads(read_json(result_path))
        observations_by_round: dict[int, list[float]] = {}
        round_means: dict[int, float] = {}
        for round_index in range(2, 9):
            observations = payloads[round_index].get("pilot_observations") or []
            values = [float(item["gain"]) for item in observations]
            observations_by_round[round_index] = values
            if values:
                round_means[round_index] = statistics.fmean(values)

        for current_round in sorted(round_means):
            previous_round = current_round - 1
            if previous_round not in round_means:
                continue
            signed = round_means[current_round] - round_means[previous_round]
            drift_rows.append(
                {
                    "seed": seed,
                    "from_round": previous_round,
                    "to_round": current_round,
                    "signed_mean_gain_drift": signed,
                    "absolute_mean_gain_drift": abs(signed),
                    "use": "historical_diagnostic_only_not_calibration",
                }
            )

        r8_accuracy, r8_path = checkpoint_accuracy(seed, 8)
        seed_numeric_stops = 0
        seed_min_n = None
        seed_max_n = 0

        for decision_round in DECISION_ROUNDS:
            previous_values = observations_by_round.get(decision_round - 1, [])
            fixed_mean = statistics.fmean(previous_values) if previous_values else None
            fixed_net = (
                fixed_mean + baseline_radius - lambda_cost * cost_lcb.total
                if fixed_mean is not None
                else None
            )
            fixed_rows.append(
                {
                    "seed": seed,
                    "decision_round": decision_round,
                    "source_round": decision_round - 1,
                    "m": len(previous_values),
                    "mean_gain": fixed_mean,
                    "radius": baseline_radius if previous_values else None,
                    "optimistic_net_ucb_rho_eta_zero": fixed_net,
                    "numerical_stop": fixed_net is not None and fixed_net <= 0.0,
                    "certificate": "uncertified_continue",
                    "reason": (
                        "no_previous_round_pilots"
                        if not previous_values
                        else "cross_round_drift_and_policy_mismatch_unknown"
                    ),
                }
            )

            estimators: list[tuple[str, int | None]] = [
                (f"L{length}", length) for length in FIXED_WINDOWS
            ] + [("cumulative", None)]
            for label, length in estimators:
                estimate = historical_window_estimate(
                    observations_by_round,
                    decision_round=decision_round,
                    window_length=length,
                )
                radius = (
                    simultaneous_hoeffding_radius(
                        estimate.n_eff,
                        delta=EXPECTED["delta"],
                        n_tests=n_tests,
                        width=1.0,
                    )
                    if estimate.n_eff
                    else None
                )
                decision = assess_time_adaptive_decision(
                    estimate=estimate,
                    statistical_radius=radius,
                    rho=None,
                    eta=None,
                    cost_lcb=cost_lcb.total,
                    lambda_cost=lambda_cost,
                    sampling_assumptions_valid=False,
                    drift_assumption_valid=False,
                    policy_mismatch_assumption_valid=False,
                    cost_lcb_applicable_predecision=False,
                )
                stop_accuracy, accuracy_path = checkpoint_accuracy(
                    seed, decision_round - 1
                )
                retrospective = avoided_costs(payloads, decision_round)
                regret = (
                    None
                    if stop_accuracy is None or r8_accuracy is None
                    else r8_accuracy - stop_accuracy
                )
                historical_drifts_available = [
                    row["absolute_mean_gain_drift"]
                    for row in drift_rows
                    if row["seed"] == seed and row["to_round"] < decision_round
                ]
                optimistic_stop = (
                    decision.optimistic_net_ucb is not None
                    and decision.optimistic_net_ucb <= 0.0
                )
                if optimistic_stop:
                    seed_numeric_stops += 1
                if estimate.n_eff:
                    seed_min_n = (
                        estimate.n_eff
                        if seed_min_n is None
                        else min(seed_min_n, estimate.n_eff)
                    )
                    seed_max_n = max(seed_max_n, estimate.n_eff)
                row = {
                    "seed": seed,
                    "decision_round": decision_round,
                    "estimator": label,
                    "requested_window_length": length,
                    "calendar_window_start": estimate.calendar_window_start,
                    "calendar_window_end": estimate.calendar_window_end,
                    "calendar_window_length": estimate.calendar_window_length,
                    "calendar_rounds": json.dumps(estimate.calendar_rounds),
                    "observation_counts_by_round": json.dumps(
                        estimate.observation_counts_by_round, sort_keys=True
                    ),
                    "observed_rounds": json.dumps(estimate.observed_rounds),
                    "zero_pilot_rounds": json.dumps(estimate.zero_pilot_rounds),
                    "n_eff": estimate.n_eff,
                    "historical_mean_gain": estimate.mean_gain,
                    "sample_variance": estimate.sample_variance,
                    "statistical_radius": radius,
                    "n_tests": n_tests,
                    "bound_status": "assumption_conditional_diagnostic",
                    "concentration_target": (
                        "observation_weighted_average_of_historical_conditional_means"
                    ),
                    "next_round_target_justified": False,
                    "optimistic_net_ucb_rho_eta_zero": decision.optimistic_net_ucb,
                    "optimistic_numerical_stop": optimistic_stop,
                    "maximum_combined_rho_eta_for_stop": (
                        decision.maximum_combined_rho_eta_for_stop
                    ),
                    "required_cost_lcb_for_stop_rho_eta_zero": (
                        None
                        if estimate.mean_gain is None or radius is None
                        else max(0.0, (estimate.mean_gain + radius) / lambda_cost)
                    ),
                    "classification": decision.certification,
                    "decision_reason": decision.reason,
                    "evidence_issues": json.dumps(decision.evidence_issues),
                    "historical_max_abs_drift_available": (
                        max(historical_drifts_available)
                        if historical_drifts_available
                        else None
                    ),
                    "historical_drift_is_diagnostic_only": True,
                    "cost_lcb": cost_lcb.total,
                    "cost_lcb_conditional_on_executable_intervention": True,
                    "cost_lcb_applicable_predecision_proved": False,
                    "retrospective_checkpoint_accuracy": stop_accuracy,
                    "retrospective_accuracy_regret_vs_r8": regret,
                    "retrospective_accuracy_artifact": accuracy_path,
                    **retrospective,
                }
                window_rows.append(row)

                if estimate.n_eff:
                    for d_value in DRIFT_SCENARIOS:
                        rho = weighted_temporal_drift(
                            estimate.observation_counts_by_round,
                            decision_round=decision_round,
                            per_round_drift=d_value,
                        )
                        gain_ucb = float(estimate.mean_gain) + float(radius) + rho
                        net = gain_ucb - lambda_cost * cost_lcb.total
                        sensitivity_rows.append(
                            {
                                "seed": seed,
                                "decision_round": decision_round,
                                "estimator": label,
                                "requested_window_length": length,
                                "n_eff": estimate.n_eff,
                                "illustrative_d": d_value,
                                "hypothetical_eta": 0.0,
                                "weighted_rho": rho,
                                "gain_ucb": gain_ucb,
                                "net_ucb": net,
                                "numerical_stop": net <= 0.0,
                                "status": "illustrative_sensitivity_not_certificate",
                                "drift_assumption_calibrated": False,
                                "eta_guarantee_proved": False,
                            }
                        )

        seed_summaries.append(
            {
                "seed": seed,
                "window_rows_with_optimistic_numerical_stop": seed_numeric_stops,
                "minimum_effective_pilots": seed_min_n,
                "maximum_effective_pilots": seed_max_n,
                "certified_stop_round": None,
                "terminal_outcome": "budget_stop_after_round_8",
                "r8_accuracy": r8_accuracy,
                "r8_accuracy_artifact": r8_path,
            }
        )

    populated = [row for row in window_rows if row["n_eff"] > 0]
    numeric = [row for row in populated if row["optimistic_numerical_stop"]]
    sensitivity_populated = [row for row in sensitivity_rows if row["n_eff"] > 0]
    radius_by_n: dict[int, float] = {}
    for row in populated:
        radius_by_n[int(row["n_eff"])] = float(row["statistical_radius"])
    summary = {
        "schema": "v7_time_adaptive_stopping_v1",
        "seeds": args.seeds,
        "delta": EXPECTED["delta"],
        "lambda_cost": lambda_cost,
        "fixed_windows": list(FIXED_WINDOWS),
        "cumulative_included": True,
        "decision_rounds": list(DECISION_ROUNDS),
        "n_tests": n_tests,
        "fixed_32_baseline_radius": baseline_radius,
        "time_adaptive_radius_by_effective_n": {
            str(key): value for key, value in sorted(radius_by_n.items())
        },
        "minimum_time_adaptive_radius": min(
            (float(row["statistical_radius"]) for row in populated), default=None
        ),
        "maximum_effective_pilots": max(
            (int(row["n_eff"]) for row in populated), default=0
        ),
        "cost_lcb": cost_lcb.to_dict(),
        "cost_penalty": lambda_cost * cost_lcb.total,
        "optimistic_numerical_stop_count": len(numeric),
        "optimistic_numerical_stop_candidates": [
            {
                "seed": row["seed"],
                "decision_round": row["decision_round"],
                "estimator": row["estimator"],
                "n_eff": row["n_eff"],
                "net_ucb": row["optimistic_net_ucb_rho_eta_zero"],
                "allowable_rho_eta": row["maximum_combined_rho_eta_for_stop"],
                "regret": row["retrospective_accuracy_regret_vs_r8"],
                "posthoc_cost_avoided": row["posthoc_pipeline_cost_avoided"],
            }
            for row in numeric
        ],
        "smallest_positive_allowable_rho_eta": min(
            (
                float(row["maximum_combined_rho_eta_for_stop"])
                for row in numeric
                if float(row["maximum_combined_rho_eta_for_stop"]) >= 0.0
            ),
            default=None,
        ),
        "closest_to_stop": (
            {
                "seed": closest["seed"],
                "decision_round": closest["decision_round"],
                "estimator": closest["estimator"],
                "n_eff": closest["n_eff"],
                "optimistic_net_ucb": closest[
                    "optimistic_net_ucb_rho_eta_zero"
                ],
                "maximum_combined_rho_eta_for_stop": closest[
                    "maximum_combined_rho_eta_for_stop"
                ],
                "required_cost_lcb": closest[
                    "required_cost_lcb_for_stop_rho_eta_zero"
                ],
                "retrospective_regret": closest[
                    "retrospective_accuracy_regret_vs_r8"
                ],
                "posthoc_pipeline_cost_avoided": closest[
                    "posthoc_pipeline_cost_avoided"
                ],
            }
            if (
                closest := min(
                    populated,
                    key=lambda item: float(
                        item["optimistic_net_ucb_rho_eta_zero"]
                    ),
                    default=None,
                )
            )
            is not None
            else None
        ),
        "minimum_required_cost_lcb_for_optimistic_stop": min(
            (
                float(row["required_cost_lcb_for_stop_rho_eta_zero"])
                for row in populated
                if row["required_cost_lcb_for_stop_rho_eta_zero"] is not None
            ),
            default=None,
        ),
        "sensitivity_numerical_stops_by_d": {
            str(d_value): sum(
                row["illustrative_d"] == d_value and row["numerical_stop"]
                for row in sensitivity_populated
            )
            for d_value in DRIFT_SCENARIOS
        },
        "sensitivity_min_net_ucb_by_d": {
            str(d_value): min(
                (
                    float(row["net_ucb"])
                    for row in sensitivity_populated
                    if row["illustrative_d"] == d_value
                ),
                default=None,
            )
            for d_value in DRIFT_SCENARIOS
        },
        "certified_stop_exists": False,
        "valid_online_outcome": "uncertified_continue_until_budget_stop",
        "certificate_blockers": [
            "cross-round pilot conditional independence or martingale assumption unproved",
            "historical-mean to next-round temporal drift bound rho unknown",
            "historical selected-batch to next-policy mismatch eta unknown",
            "cost LCB is conditional on an executable intervention and applicability is not known predecision",
        ],
        "sealed_or_future_accuracy_used_for_decision": False,
        "seed_summaries": seed_summaries,
    }
    return {
        "summary": summary,
        "windows": window_rows,
        "fixed_32_baseline": fixed_rows,
        "drift_sensitivity": sensitivity_rows,
        "historical_drift": drift_rows,
        "provenance": {
            "validation": validation,
            "historical_sources_read_only": True,
            "retrospective_metrics_used_only_after_decisions": True,
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
    windows = payload["windows"]
    numeric = summary["optimistic_numerical_stop_candidates"]
    max_n = summary["maximum_effective_pilots"]
    min_radius = summary["minimum_time_adaptive_radius"]
    baseline = summary["fixed_32_baseline_radius"]
    reduction = baseline - min_radius if min_radius is not None else None
    percent = reduction / baseline * 100.0 if reduction is not None else None
    lines = [
        "# V7 Time-Adaptive Sliding-Window Stopping",
        "",
        "> CPU-only offline replay over frozen seeds 42/43. No GPU, teacher API, model loading, inference, training, or historical-artifact mutation was performed.",
        "",
        "## Executive result",
        "",
        f"Pooling increases the largest effective sample from 32 to **{max_n}** pilots. After simultaneous Bonferroni correction over `7 decisions × (7 windows + cumulative) = {summary['n_tests']}` tests, the smallest radius is **{fmt(min_radius)}**, versus **{fmt(baseline)}** for the existing fixed-32 bound: an absolute reduction of **{fmt(reduction)}** ({fmt(percent, 2)}%).",
        "",
        f"The optimistic `rho=eta=0` diagnostic produces **{summary['optimistic_numerical_stop_count']}** negative Net-UCB window/decision combinations. None is a certified STOP: pooling concentrates around a historical weighted mean, while conditional sampling, next-round drift `rho`, policy mismatch `eta`, and predecision applicability of the conditional cost-LCB remain unproved.",
        "",
        "The valid online conclusion is `uncertified_continue` through R8, followed by `budget_stop`.",
        "",
        "## Radius by effective pilot count",
        "",
        "| Effective pilots | Simultaneous radius | Change vs fixed-32 |",
        "|---:|---:|---:|",
    ]
    for n_value, radius in summary["time_adaptive_radius_by_effective_n"].items():
        lines.append(
            f"| {n_value} | {fmt(radius)} | {fmt(float(radius) - baseline)} |"
        )
    lines.extend(
        [
            "",
            "The L=1 time-adaptive radius is wider than the fixed-32 baseline because the time-adaptive experiment protects the complete family of 56 tests. Larger windows eventually overcome that correction through larger `n_eff`.",
            "",
            "## Historical-window estimates",
            "",
            "| Seed | Pre-round | Window | Calendar rounds | Observed rounds | n_eff | Mean | Radius | Net-UCB rho=eta=0 | Numeric STOP |",
            "|---:|---:|---|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    # Keep the report readable: show unique fixed windows and cumulative, but
    # retain every row in CSV/JSON.
    for row in windows:
        lines.append(
            f"| {row['seed']} | {row['decision_round']} | {row['estimator']} | {row['calendar_rounds']} | {row['observed_rounds']} | {row['n_eff']} | {fmt(row['historical_mean_gain'])} | {fmt(row['statistical_radius'])} | {fmt(row['optimistic_net_ucb_rho_eta_zero'])} | {'yes' if row['optimistic_numerical_stop'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Every row uses only rounds strictly earlier than its decision round. Missing-pilot rounds remain in the calendar window with zero weight. Unequal pilot counts are weighted observation-by-observation.",
            "",
            "## Optimistic numerical STOP candidates",
            "",
            "| Seed | Pre-round | Window | n_eff | Net-UCB | Max rho+eta allowed | Post-hoc regret | Post-hoc cost avoided |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    if numeric:
        for row in numeric:
            lines.append(
                f"| {row['seed']} | {row['decision_round']} | {row['estimator']} | {row['n_eff']} | {fmt(row['net_ucb'])} | {fmt(row['allowable_rho_eta'])} | {fmt(row['regret'])} | {fmt(row['posthoc_cost_avoided'])} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "These are diagnostics, not policy decisions. Accuracy regret and avoided historical costs are computed only after identifying a numerical candidate and never enter the stopping calculation.",
            "",
            "## Drift sensitivity",
            "",
            "| Illustrative d | Numerical STOP combinations | Status |",
            "|---:|---:|---|",
        ]
    )
    for d_value, count in summary["sensitivity_numerical_stops_by_d"].items():
        lines.append(
            f"| {float(d_value):.3f} | {count} (minimum Net-UCB {fmt(summary['sensitivity_min_net_ucb_by_d'][d_value])}) | illustrative, eta=0; not calibrated |"
        )
    lines.extend(
        [
            "",
            "For the stated stability scenario, each observation pays `d × age`; older windows can reduce beta while increasing rho. Therefore adding temporal drift never tightens the total bound relative to the `d=0` diagnostic. Historical maximum drift is reported in CSV/JSON only and is not used to set d.",
            "",
            "## Scientific validity",
            "",
            "The simultaneous radius would cover the observation-weighted average of historical conditional means under bounded width-one observations plus an appropriate conditional-independence or martingale concentration assumption. It does not automatically cover next-round gain.",
            "",
            "A publishable pre-round theorem additionally needs:",
            "",
            "1. a filtration and conditional independence/martingale argument across pilot training runs and rounds;",
            "2. a predeclared temporal stability bound relating each historical conditional mean to the next round;",
            "3. a policy-mismatch bound between historical selected batches and the full next V7 policy;",
            "4. predecision proof that the protocol cost-LCB applies to the candidate intervention;",
            "5. simultaneous coverage for the complete window/decision family without selecting an uncovered minimum.",
            "",
            f"No non-negative combined `rho+eta` allowance is compatible with STOP. The closest case is seed {summary['closest_to_stop']['seed']} before R{summary['closest_to_stop']['decision_round']} with {summary['closest_to_stop']['estimator']}: even at `rho+eta=0`, Net-UCB is **+{fmt(summary['closest_to_stop']['optimistic_net_ucb'])}**, so the formal maximum allowance is **{fmt(summary['closest_to_stop']['maximum_combined_rho_eta_for_stop'])}** (impossible for a non-negative uncertainty term). Its required cost-LCB is **${fmt(summary['closest_to_stop']['required_cost_lcb'])}**, versus the audited conditional **${fmt(summary['cost_lcb']['total'])}**.",
            "",
            "## Paper conclusion",
            "",
            "The result is suitable as a negative feasibility/limitations experiment: historical pooling materially reduces statistical radius, but it does not solve the target-shift problem. Without independently justified rho and eta, it does not support a theorem claiming certified Cost-Aware STOP.",
            "",
            "## Outputs and provenance",
            "",
            "- Full per-window and sensitivity results are stored in JSON/CSV.",
            "- Historical source artifacts were read-only.",
            "- Sealed/future accuracy was not used in any decision calculation.",
            "- `budget_stop` remains distinct from certified Cost-Aware STOP.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validation = validate(args)
    validation_path = args.output_dir / "validation.json"
    atomic_json(validation_path, validation)
    print(f"[VALID] {validation_path}")
    if args.validate_only:
        return

    payload = replay(args, validation)
    atomic_json(args.output_dir / "replay.json", payload)
    write_csv(args.output_dir / "windows.csv", payload["windows"])
    write_csv(args.output_dir / "fixed_32_baseline.csv", payload["fixed_32_baseline"])
    write_csv(args.output_dir / "drift_sensitivity.csv", payload["drift_sensitivity"])
    write_csv(args.output_dir / "historical_drift.csv", payload["historical_drift"])
    atomic_text(args.report, render_report(payload))
    print(f"[DONE] {args.output_dir / 'replay.json'}")
    print(f"[DONE] {args.output_dir / 'windows.csv'}")
    print(f"[DONE] {args.report}")


if __name__ == "__main__":
    main()
