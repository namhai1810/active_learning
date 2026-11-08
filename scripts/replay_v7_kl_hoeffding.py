#!/usr/bin/env python3
"""CPU-only KL-Hoeffding comparison over frozen V7 pilot caches."""

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
)
from src.gain.kl_hoeffding import (
    compare_kl_and_range_aware_hoeffding,
    pre_round_bound_availability,
)
from src.utils.config import load_yaml_with_extends


DEFAULT_SOURCE = Path(
    "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2"
)
DEFAULT_OUTPUT = Path("outputs/stage2_diagnostics/v7_kl_hoeffding")
DEFAULT_REPORT = Path(
    "reports/v7_minimal_cost_aware/KL_HOEFFDING_COMPARISON.md"
)
DEFAULT_CONFIG = Path("configs/stage2_v7_minimal_cost_aware.yaml")
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


def round_payloads(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["round"]): item
        for item in result.get("cost_aware_rounds", [])
        if isinstance(item, dict) and item.get("round") is not None
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_with_extends(args.config)
    protocol(config)
    checks: list[dict[str, Any]] = []
    for seed in args.seeds:
        result_path = args.source_root / f"seed_{seed}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing historical result: {result_path}")
        payloads = round_payloads(read_json(result_path))
        if sorted(payloads) != list(range(1, 9)):
            raise ValueError(f"Seed {seed} does not contain rounds 1-8")
        for round_index in range(2, 9):
            observations = payloads[round_index].get("pilot_observations") or []
            if not isinstance(observations, list):
                raise ValueError(f"Invalid pilots seed={seed} round={round_index}")
            if len(observations) not in {0, EXPECTED["required_pilots"]}:
                raise ValueError(
                    f"Partial pilot set seed={seed} round={round_index}: {len(observations)}"
                )
            checks.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "cached_pilots": len(observations),
                    "eligible_for_comparison": len(observations)
                    == EXPECTED["required_pilots"],
                }
            )
    return {
        "status": "valid",
        "cpu_only": True,
        "gpu_or_model_work_required": False,
        "source_root": str(args.source_root),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "protocol": EXPECTED,
        "same_round_confidence_family": (
            "K=1 mixed-policy action by T=7 rounds within each seed"
        ),
        "conditional_independence_proved": False,
        "round_checks": checks,
    }


def replay(args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    config = load_yaml_with_extends(args.config)
    cost_aware = protocol(config)
    rates, currency = cost_rates_from_config(cost_aware["cost"])
    stopping_manifest = read_json(
        Path(config["closed_loop"]["stopping_subset_manifest"])
    )
    cost_lcb = protocol_full_round_cost_lower_bound(
        rates=rates,
        currency=currency,
        acquired_examples=int(config["closed_loop"]["per_round_examples"]),
        stopping_examples=len(stopping_manifest["stopping_eval_ids"]),
    )
    penalty = EXPECTED["lambda_cost"] * cost_lcb.total
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for seed in args.seeds:
        result_path = args.source_root / f"seed_{seed}" / "result.json"
        payloads = round_payloads(read_json(result_path))
        valid_rounds = {
            round_index
            for round_index in range(2, 9)
            if len(payloads[round_index].get("pilot_observations") or [])
            == EXPECTED["required_pilots"]
        }
        for round_index in range(2, 9):
            observations = payloads[round_index].get("pilot_observations") or []
            if len(observations) != EXPECTED["required_pilots"]:
                missing.append(
                    {
                        "seed": seed,
                        "round": round_index,
                        "cached_pilots": len(observations),
                        "historical_reason": (
                            payloads[round_index].get("decision") or {}
                        ).get("reason"),
                    }
                )
                continue
            comparison, audit = compare_kl_and_range_aware_hoeffding(
                observations,
                required_pilots=EXPECTED["required_pilots"],
                delta=EXPECTED["delta"],
                K=EXPECTED["K"],
                T=EXPECTED["T"],
            )
            hoeffding_net = comparison.hoeffding_gain_ucb - penalty
            kl_net = comparison.kl_gain_ucb - penalty
            same_round_pre = pre_round_bound_availability(
                pilot_source_round=round_index,
                decision_round=round_index,
                cross_round_transfer_valid=False,
            )
            prior_source = round_index - 1
            prior_status = (
                pre_round_bound_availability(
                    pilot_source_round=prior_source,
                    decision_round=round_index,
                    cross_round_transfer_valid=False,
                )
                if prior_source in valid_rounds
                else {
                    "available_predecision": False,
                    "valid_for_next_round_gain": False,
                    "reason": "no_complete_prior_round_pilot_bound",
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "m": comparison.m,
                    "baseline_accuracy": comparison.baseline_accuracy,
                    "pilot_mean_gain": comparison.pilot_mean_gain,
                    "mean_normalized_accuracy": (
                        comparison.mean_normalized_accuracy
                    ),
                    "confidence_budget": comparison.confidence_budget,
                    "hoeffding_radius": comparison.hoeffding_radius,
                    "kl_radius": comparison.kl_radius,
                    "radius_reduction": comparison.radius_reduction,
                    "radius_reduction_percent": (
                        comparison.radius_reduction_percent
                    ),
                    "hoeffding_gain_ucb": comparison.hoeffding_gain_ucb,
                    "kl_gain_ucb": comparison.kl_gain_ucb,
                    "conditional_cost_lcb": cost_lcb.total,
                    "cost_penalty": penalty,
                    "hoeffding_net_ucb": hoeffding_net,
                    "kl_net_ucb": kl_net,
                    "kl_numerical_stop": kl_net <= 0.0,
                    "hoeffding_numerical_stop": hoeffding_net <= 0.0,
                    "same_round_status": (
                        "assumption_conditional_post_acquisition_diagnostic"
                    ),
                    "same_round_certified_stop": False,
                    "conditional_independence_proved": False,
                    "cost_usage_status": (
                        "optimistic_hypothetical_full_round_cost_lcb; "
                        "not incurred exploration cost and not a certificate"
                    ),
                    "same_round_predecision_available": same_round_pre[
                        "available_predecision"
                    ],
                    "same_round_predecision_reason": same_round_pre["reason"],
                    "prior_round_source": (
                        prior_source if prior_source in valid_rounds else None
                    ),
                    "prior_round_available_predecision": prior_status[
                        "available_predecision"
                    ],
                    "prior_round_valid_for_next_gain": prior_status[
                        "valid_for_next_round_gain"
                    ],
                    "prior_round_transfer_reason": prior_status["reason"],
                    "pre_round_certified_stop": False,
                    "normalized_min": audit["normalized_min"],
                    "normalized_max": audit["normalized_max"],
                }
            )

    reductions = [float(row["radius_reduction_percent"]) for row in rows]
    closest = min(rows, key=lambda row: float(row["kl_net_ucb"]))
    closest_required_cost_lcb = (
        float(closest["kl_gain_ucb"]) / EXPECTED["lambda_cost"]
    )
    closest_cost_lcb_gap = closest_required_cost_lcb - cost_lcb.total
    summary = {
        "schema": "v7_kl_hoeffding_comparison_v1",
        "seeds": args.seeds,
        "complete_rounds": len(rows),
        "missing_rounds": missing,
        "protocol": EXPECTED,
        "cost_lcb": cost_lcb.to_dict(),
        "cost_penalty": penalty,
        "cost_label": "optimistic_hypothetical_diagnostic",
        "radius_reduction_percent": {
            "minimum": min(reductions),
            "maximum": max(reductions),
            "mean": statistics.fmean(reductions),
        },
        "kl_radius_minimum": min(float(row["kl_radius"]) for row in rows),
        "kl_radius_maximum": max(float(row["kl_radius"]) for row in rows),
        "hoeffding_radius": rows[0]["hoeffding_radius"],
        "hoeffding_numerical_stop_count": sum(
            bool(row["hoeffding_numerical_stop"]) for row in rows
        ),
        "kl_numerical_stop_count": sum(
            bool(row["kl_numerical_stop"]) for row in rows
        ),
        "same_round_certified_stop_count": 0,
        "pre_round_certified_stop_count": 0,
        "closest_kl_case": {
            key: closest[key]
            for key in (
                "seed",
                "round",
                "pilot_mean_gain",
                "baseline_accuracy",
                "hoeffding_radius",
                "kl_radius",
                "kl_gain_ucb",
                "hoeffding_net_ucb",
                "kl_net_ucb",
                "radius_reduction_percent",
            )
        },
        "closest_required_cost_lcb_for_numerical_stop": (
            closest_required_cost_lcb
        ),
        "closest_cost_lcb_gap": closest_cost_lcb_gap,
        "pre_round_feasibility": (
            "no current-round pilot bound is available before Teacher; prior-round "
            "bounds lack a proved drift/policy-transfer guarantee"
        ),
        "historical_artifacts_mutated": False,
    }
    return {
        "summary": summary,
        "per_round": rows,
        "provenance": {
            "validation": validation,
            "historical_sources_read_only": True,
            "gpu_teacher_inference_training_used": False,
        },
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["per_round"]
    reductions = summary["radius_reduction_percent"]
    closest = summary["closest_kl_case"]
    lines = [
        "# KL-Hoeffding vs Range-Aware Hoeffding for V7",
        "",
        "> CPU-only replay over frozen 32-pilot caches for seeds 42/43. No GPU, Teacher API, inference, training, controller modification, or historical-artifact mutation was performed.",
        "",
        "## Executive result",
        "",
        f"Across **{summary['complete_rounds']}** complete cached rounds, KL radius is `{fmt(summary['kl_radius_minimum'])}–{fmt(summary['kl_radius_maximum'])}` versus the constant Range-Aware Hoeffding radius `{fmt(summary['hoeffding_radius'])}`. Radius reduction is **{fmt(reductions['minimum'], 2)}%–{fmt(reductions['maximum'], 2)}%**, mean **{fmt(reductions['mean'], 2)}%**.",
        "",
        f"Using conditional full-round Cost-LCB `${fmt(summary['cost_lcb']['total'])}` only as an **optimistic hypothetical diagnostic**, Hoeffding produces `{summary['hoeffding_numerical_stop_count']}` numerical STOPs and KL produces `{summary['kl_numerical_stop_count']}`. Certified same-round STOPs: `0`; certified pre-round STOPs: `0`.",
        "",
        "## Per-round same-round conditional diagnostic",
        "",
        "| Seed | Round | Pilot mean gain | Hoeffding radius | KL radius | Reduction % | Hoeffding Net-UCB | KL Net-UCB | Numerical STOP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round']} | {fmt(row['pilot_mean_gain'])} | {fmt(row['hoeffding_radius'])} | {fmt(row['kl_radius'])} | {fmt(row['radius_reduction_percent'], 2)}% | {fmt(row['hoeffding_net_ucb'])} | {fmt(row['kl_net_ucb'])} | {'yes' if row['kl_numerical_stop'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "`Y_j = X_j + baseline_accuracy = accuracy_after_j` was mechanically checked in `[0,1]`. Both methods use `m=32`, `K=1`, `T=7`, `delta=0.05`, and the same `log(2KT/delta)` confidence budget. The KL result is intersected with the natural upper endpoint through `u≤1`; its radius is verified not to exceed Hoeffding/Pinsker radius.",
            "",
            f"The closest KL case is seed {closest['seed']} round {closest['round']}: mean gain `{fmt(closest['pilot_mean_gain'])}`, KL radius `{fmt(closest['kl_radius'])}`, and optimistic hypothetical Net-UCB `{fmt(closest['kl_net_ucb'])}`.",
            "",
            "This is not a post-acquisition certificate. Teacher/verification/generation/selection and pilot costs are already incurred when current-round pilots exist; they are not subtracted again. The `$0.2264` term is shown only as a hypothetical lower bound for a full future executable round.",
            "",
            "## Pre-round feasibility",
            "",
            "Current-round pilots are created only after Teacher, verification, generation, and selected-batch construction. Therefore their KL/Hoeffding bounds are unavailable before the same round and cannot implement a valid pre-round STOP.",
            "",
            "A prior-round bound is available in time but targets the previous checkpoint and selected batch. Without a proved temporal-drift and policy-transfer bound, it is not a valid upper bound for next-round gain. The replay never substitutes current-round gain into a pre-round decision.",
            "",
            "| Comparison | Numerical status | Certified STOP | Reason |",
            "|---|---|:---:|---|",
            "| Same-round KL/Hoeffding | conditional diagnostic | no | conditional independence is unproved; evidence is post-acquisition |",
            "| Pre-round using same-round pilots | unavailable | no | current-round pilots do not exist before Teacher |",
            "| Pre-round using prior-round pilots | uncertified | no | drift and selected-policy transfer are unproved |",
            "",
            "## Bottleneck",
            "",
            f"KL tightens sampling uncertainty, but the closest hypothetical Net-UCB remains `{fmt(closest['kl_net_ucb'])}`. At `lambda=0.2`, that row would require an eligible Cost-LCB of at least `${fmt(summary['closest_required_cost_lcb_for_numerical_stop'])}`, which is `${fmt(summary['closest_cost_lcb_gap'])}` above the frozen `${fmt(summary['cost_lcb']['total'])}` diagnostic. More importantly, no cached metadata proves conditional independence of deterministic pilot seeds, and no cross-round transfer theorem makes the same-round bound available before Teacher. Thus a smaller radius alone cannot create a certified pre-round STOP.",
            "",
            "## Appendix proposition (assumption-conditional)",
            "",
            "**Proposition.** Fix a round after its checkpoint and selected batch are measurable with respect to history `F_t`. Suppose the normalized pilot outcomes `Y_1,…,Y_m∈[0,1]` are conditionally independent given `F_t` and share conditional mean `mu_t`. For each of the predeclared `K×T` tests, let `u_t` be the largest `u∈[mean(Y),1]` satisfying",
            "",
            "`kl(mean(Y) || u) ≤ log(2KT/delta)/m`.",
            "",
            "Then, by the one-sided conditional KL-Hoeffding/Chernoff inequality and a union bound over `K×T`, simultaneously `mu_t ≤ u_t` with failure probability at most `delta/2` (hence at most `delta`). Therefore the conditional fixed-shard gain satisfies `mu_t − A_0 ≤ u_t − A_0`.",
            "",
            "**Proof sketch.** Conditional on `F_t`, Chernoff inversion gives `P(mu_t>u_t | F_t) ≤ exp(-m·kl(mean(Y)||u_t)) = delta/(2KT)` for each test. Sum these probabilities over the predeclared family. Subtracting the fixed baseline `A_0` preserves the inequality.",
            "",
            "The proposition does not establish the assumptions from deterministic cached seeds, does not give a population-HANS guarantee, and does not transfer the bound to a future round or a different selected policy.",
            "",
            "## Missing rounds and provenance",
            "",
        ]
    )
    for item in summary["missing_rounds"]:
        lines.append(
            f"- Seed {item['seed']} R{item['round']}: `{item['cached_pilots']}` pilots; `{item['historical_reason']}`. No data were fabricated."
        )
    lines.extend(
        [
            "",
            "Full per-round JSON/CSV and validation provenance are stored under `outputs/stage2_diagnostics/v7_kl_hoeffding/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validation = validate(args)
    atomic_json(args.output_dir / "validation.json", validation)
    print(f"[VALID] {args.output_dir / 'validation.json'}")
    if args.validate_only:
        return
    payload = replay(args, validation)
    atomic_json(args.output_dir / "replay.json", payload)
    write_csv(args.output_dir / "per_round.csv", payload["per_round"])
    atomic_text(args.report, render_report(payload))
    print(f"[DONE] {args.output_dir / 'replay.json'}")
    print(f"[DONE] {args.output_dir / 'per_round.csv'}")
    print(f"[DONE] {args.report}")


if __name__ == "__main__":
    main()
