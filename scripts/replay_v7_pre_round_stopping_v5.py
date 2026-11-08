#!/usr/bin/env python3
"""ARCHIVED offline replay for V7 Pre-Round Risk-Ceiling Stopping v5.

This program is deliberately independent from the V7 runtime.  It only reads
historical artifacts/manifests and writes a versioned diagnostic report.  A
STOP certificate is emitted only when both inputs below pass their provenance
gates:

  1. a one-sided risk UCB from an independent held-out Bernoulli sample; and
  2. a pre-decision lower bound for every required full-round cost component.

Historical/future-round accuracy and realised costs are used only after a
decision has been replayed, to quantify regret and avoidable resources.

This risk-ceiling objective is not the current V7 next-round expected-gain
objective. Retained only to reproduce the historical diagnostic. See
``archive/v7_cost_aware_experiments/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_COST_COMPONENTS = ("discovery", "verification", "generation", "training")
VALID_LOWER_BOUND_BASES = {
    "deterministic",
    "contractual_minimum",
    "proven_statistical_lower_bound",
}
FORBIDDEN_ONLINE_RISK_ROLES = {
    "sealed_test",
    "final_test",
    "early_stopping",
    "model_selection",
    "training",
    "pilot_eval",
}


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P[X <= k] for X~Binomial(n,p), evaluated stably."""
    if not 0 <= k <= n:
        return 0.0 if k < 0 else 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    return min(1.0, math.exp(_logsumexp(terms)))


def clopper_pearson_upper(errors: int, total: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound on error risk."""
    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("errors/total must describe a non-empty binomial sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if errors == total:
        return 1.0
    lower = errors / total
    upper = 1.0
    # Solve BinomCDF(errors; total, p)=alpha.  CDF is decreasing in p.
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if binomial_cdf(errors, total, middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_records(payload: Any, *, key: str = "records") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(
        key,
        payload.get(
            "rows",
            payload.get("per_round", payload.get("rounds", payload.get("entries", []))),
        ),
    )
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for seed_key, seed_value in value.items():
            if not isinstance(seed_value, Mapping):
                continue
            for round_key, record in seed_value.items():
                if isinstance(record, Mapping):
                    item = dict(record)
                    item.setdefault("seed", int(str(seed_key).replace("seed_", "")))
                    item.setdefault("round", int(str(round_key).replace("round_", "")))
                    records.append(item)
    return records


def _bool(record: Mapping[str, Any], name: str) -> bool:
    return record.get(name) is True


@dataclass(frozen=True)
class RiskEvidence:
    seed: int
    model_round: int
    errors: int | None
    total: int | None
    risk_ucb: float | None
    alpha: float
    valid: bool
    reason: str
    source: str | None


def validate_risk_record(
    record: Mapping[str, Any] | None,
    *,
    seed: int,
    model_round: int,
    pre_round: int,
    alpha: float,
) -> RiskEvidence:
    if record is None:
        return RiskEvidence(seed, model_round, None, None, None, alpha, False,
                            "missing_independent_risk_evidence", None)
    source = str(record.get("source", record.get("artifact", "manifest")))
    role = str(record.get("dataset_role", "")).strip().lower()
    invalid_reasons: list[str] = []
    if role in FORBIDDEN_ONLINE_RISK_ROLES or not role:
        invalid_reasons.append("dataset_role_not_independent_heldout")
    required_true = (
        "independent_of_training",
        "independent_of_pilot_eval",
        "not_used_for_early_stopping",
        "not_used_for_model_selection",
        "not_sealed_test",
        "model_independent_of_heldout",
    )
    invalid_reasons.extend(name for name in required_true if not _bool(record, name))
    available = record.get("available_before_round")
    if not isinstance(available, int) or available > pre_round:
        invalid_reasons.append("risk_evidence_not_available_pre_decision")
    errors = record.get("errors")
    total = record.get("total", record.get("n"))
    if not isinstance(errors, int) or not isinstance(total, int) or total <= 0:
        invalid_reasons.append("missing_binomial_error_counts")
        errors_int = total_int = None
    else:
        errors_int, total_int = errors, total
        if errors < 0 or errors > total:
            invalid_reasons.append("invalid_binomial_error_counts")
    if invalid_reasons:
        return RiskEvidence(seed, model_round, errors_int, total_int, None, alpha,
                            False, ";".join(invalid_reasons), source)
    upper = clopper_pearson_upper(errors_int, total_int, alpha)  # type: ignore[arg-type]
    return RiskEvidence(seed, model_round, errors_int, total_int, upper, alpha,
                        True, "valid_clopper_pearson_risk_ucb", source)


@dataclass(frozen=True)
class CostEvidence:
    seed: int
    pre_round: int
    lower_bound: float | None
    unit: str | None
    valid: bool
    reason: str
    components: dict[str, float]
    source: str | None


def validate_cost_record(
    record: Mapping[str, Any] | None, *, seed: int, pre_round: int
) -> CostEvidence:
    if record is None:
        return CostEvidence(seed, pre_round, None, None, False,
                            "missing_full_round_cost_lower_bound", {}, None)
    source = str(record.get("source", record.get("artifact", "manifest")))
    unit = record.get("unit")
    invalid_reasons: list[str] = []
    if unit not in {"usd", "normalized_cost"}:
        invalid_reasons.append("unsupported_or_missing_cost_unit")
    if not _bool(record, "frozen_pre_decision"):
        invalid_reasons.append("cost_bound_not_frozen_pre_decision")
    available = record.get("available_before_round")
    if not isinstance(available, int) or available > pre_round:
        invalid_reasons.append("cost_bound_not_available_pre_decision")
    components_payload = record.get("components")
    components: dict[str, float] = {}
    if not isinstance(components_payload, Mapping):
        invalid_reasons.append("missing_component_lower_bounds")
    else:
        for name in REQUIRED_COST_COMPONENTS:
            component = components_payload.get(name)
            if not isinstance(component, Mapping):
                invalid_reasons.append(f"missing_{name}_lower_bound")
                continue
            value = component.get("lower_bound")
            basis = component.get("basis")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                invalid_reasons.append(f"invalid_{name}_lower_bound")
                continue
            if basis not in VALID_LOWER_BOUND_BASES:
                invalid_reasons.append(f"unproven_{name}_lower_bound")
                continue
            # An empirical mean/projection is explicitly not a lower bound.
            if component.get("derived_from") in {"empirical_mean", "future_realized_cost", "projection_mean"}:
                invalid_reasons.append(f"forbidden_{name}_bound_source")
                continue
            components[name] = float(value)
    if set(components) != set(REQUIRED_COST_COMPONENTS):
        invalid_reasons.append("incomplete_full_round_cost_bound")
    if invalid_reasons:
        return CostEvidence(seed, pre_round, None, str(unit) if unit else None,
                            False, ";".join(dict.fromkeys(invalid_reasons)),
                            components, source)
    return CostEvidence(seed, pre_round, sum(components.values()), str(unit), True,
                        "valid_full_round_cost_lower_bound", components, source)


@dataclass(frozen=True)
class HistoricalRound:
    seed: int
    round: int
    accuracy: float | None
    teacher_calls: int | None
    teacher_cost: float | None
    total_cost: float | None
    cumulative_budget: int | None
    source: str | None


def _historical_index(payload: Any) -> dict[tuple[int, int], HistoricalRound]:
    indexed: dict[tuple[int, int], HistoricalRound] = {}
    for item in _index_records(payload):
        try:
            seed, round_number = int(item["seed"]), int(item["round"])
        except (KeyError, TypeError, ValueError):
            continue
        def number(name: str) -> float | None:
            value = item.get(name)
            return float(value) if isinstance(value, (int, float)) else None
        teacher_calls = item.get("teacher_calls")
        budget = item.get("cumulative_budget", item.get("budget"))
        indexed[(seed, round_number)] = HistoricalRound(
            seed=seed,
            round=round_number,
            accuracy=number("accuracy"),
            teacher_calls=int(teacher_calls) if isinstance(teacher_calls, int) else None,
            teacher_cost=number("teacher_cost"),
            total_cost=number("total_cost"),
            cumulative_budget=int(budget) if isinstance(budget, int) else None,
            source=str(item.get("source", "historical_manifest")),
        )
    return indexed


def bootstrap_cached_artifacts(
    *,
    seeds: Sequence[int],
    run_root: Path,
    checkpoint_metrics_root: Path,
    evaluation_manifest_root: Path,
    max_round: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], HistoricalRound], list[str]]:
    """Build replay inputs from existing, read-only v2 artifacts.

    The bootstrap is intentionally conservative.  It accepts the fixed
    gain-report set only when its manifest proves disjointness and predeclared
    use.  For cost it uses only the deterministic per-run LoRA tariff; observed
    or projected full-round means never become a lower bound.
    """
    risks: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    historical: dict[tuple[int, int], HistoricalRound] = {}
    issues: list[str] = []
    for seed in seeds:
        result_path = run_root / f"seed_{seed}" / "result.json"
        eval_manifest_path = evaluation_manifest_root / f"seed_{seed}.json"
        if not result_path.exists():
            issues.append(f"seed_{seed}:missing:{result_path}")
            continue
        if not eval_manifest_path.exists():
            issues.append(f"seed_{seed}:missing:{eval_manifest_path}")
            continue
        result = _read_json(result_path)
        manifest = _read_json(eval_manifest_path)
        disjoint = manifest.get("disjointness", {})
        policy = manifest.get("policy", {})
        valid_manifest = (
            manifest.get("authorized_for_inference") is True
            and manifest.get("source_role") == "predeclared_gain_report_only"
            and all(disjoint.get(name) is True for name in ("early_stopping", "pilot_eval", "sealed_final"))
            and policy.get("no_future_round_metrics_for_decision") is True
            and policy.get("no_sealed_test") is True
            and policy.get("not_used_for_controller_tuning") is True
            and not manifest.get("issues")
        )
        round_costs = {
            int(item["round"]): item.get("decision_cost")
            for item in result.get("cost_aware_rounds", [])
            if isinstance(item, Mapping) and "round" in item
        }
        for round_number in range(1, max_round + 1):
            metrics_path = (
                checkpoint_metrics_root / f"seed_{seed}" /
                f"round_{round_number:03d}" / "metrics.json"
            )
            if not metrics_path.exists():
                issues.append(f"seed_{seed}:round_{round_number}:missing:{metrics_path}")
                continue
            metrics = _read_json(metrics_path)
            total = metrics.get("num_examples")
            accuracy = metrics.get("direct_binary_accuracy")
            if not isinstance(total, int) or not isinstance(accuracy, (int, float)):
                issues.append(f"seed_{seed}:round_{round_number}:invalid_checkpoint_metrics")
                continue
            correct = int(round(float(accuracy) * total))
            if abs(correct / total - float(accuracy)) > 1e-10:
                issues.append(f"seed_{seed}:round_{round_number}:non_integral_accuracy")
                continue
            if round_number < max_round:
                risks.append({
                    "seed": seed,
                    "model_round": round_number,
                    "errors": total - correct,
                    "total": total,
                    "dataset_role": "independent_heldout" if valid_manifest else "unverified_heldout",
                    "independent_of_training": valid_manifest,
                    "independent_of_pilot_eval": valid_manifest,
                    "not_used_for_early_stopping": valid_manifest,
                    "not_used_for_model_selection": valid_manifest,
                    "not_sealed_test": valid_manifest,
                    "model_independent_of_heldout": valid_manifest,
                    "available_before_round": round_number + 1,
                    "counterfactual_metric_materialized_offline": True,
                    "source": str(metrics_path),
                })
            decision_cost = round_costs.get(round_number)
            usage = decision_cost.get("usage", {}) if isinstance(decision_cost, Mapping) else {}
            components = decision_cost.get("components", {}) if isinstance(decision_cost, Mapping) else {}
            teacher_cost = None
            if isinstance(components, Mapping):
                input_cost = components.get("llm_inference_input")
                output_cost = components.get("llm_inference_output")
                if isinstance(input_cost, (int, float)) and isinstance(output_cost, (int, float)):
                    teacher_cost = float(input_cost) + float(output_cost)
            historical[(seed, round_number)] = HistoricalRound(
                seed=seed,
                round=round_number,
                accuracy=float(accuracy),
                teacher_calls=None,
                teacher_cost=teacher_cost,
                total_cost=(float(decision_cost["total_pipeline_cost"])
                            if isinstance(decision_cost, Mapping)
                            and isinstance(decision_cost.get("total_pipeline_cost"), (int, float))
                            else None),
                cumulative_budget=32 * round_number,
                source=str(metrics_path),
            )
        for pre_round in range(2, max_round + 1):
            decision_cost = round_costs.get(pre_round)
            rates = decision_cost.get("rates", {}) if isinstance(decision_cost, Mapping) else {}
            frozen = bool(isinstance(decision_cost, Mapping)
                          and decision_cost.get("rates_are_predeclared_assumptions") is True)
            training_minimum = rates.get("lora_training_per_run") if isinstance(rates, Mapping) else None
            if not isinstance(training_minimum, (int, float)) or training_minimum < 0:
                issues.append(f"seed_{seed}:round_{pre_round}:missing_deterministic_lora_run_tariff")
                continue
            costs.append({
                "seed": seed,
                "pre_round": pre_round,
                "unit": "usd",
                "frozen_pre_decision": frozen,
                "available_before_round": pre_round,
                "components": {
                    "discovery": {"lower_bound": 0.0, "basis": "deterministic"},
                    "verification": {"lower_bound": 0.0, "basis": "deterministic"},
                    "generation": {"lower_bound": 0.0, "basis": "deterministic"},
                    "training": {
                        "lower_bound": float(training_minimum),
                        "basis": "contractual_minimum",
                    },
                },
                "bound_scope_note": (
                    "Conservative full-round bound: only the mandatory LoRA run tariff is positive; "
                    "other components retain their universally valid zero lower bound."
                ),
                "source": f"{result_path}#cost_aware_rounds[{pre_round}]",
            })
    return risks, costs, historical, issues


def _sum_optional(values: Iterable[int | float | None]) -> float | None:
    materialized = list(values)
    return sum(float(value) for value in materialized if value is not None) \
        if materialized and all(value is not None for value in materialized) else None


def _reason(risk: RiskEvidence, cost: CostEvidence, pre_value: float | None) -> str:
    if not risk.valid:
        return risk.reason
    if not cost.valid:
        return cost.reason
    return "risk_ucb_minus_cost_lb_nonpositive" if pre_value is not None and pre_value <= 0 \
        else "no_valid_stop_certificate"


def replay(
    *,
    seeds: Sequence[int],
    risk_records: Sequence[Mapping[str, Any]],
    cost_records: Sequence[Mapping[str, Any]],
    historical: Mapping[tuple[int, int], HistoricalRound],
    lambda_cost: float,
    delta: float,
    first_pre_round: int,
    max_round: int,
    max_risk_looks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    risk_index = {(int(r["seed"]), int(r["model_round"])): r for r in risk_records
                  if "seed" in r and "model_round" in r}
    cost_index = {(int(r["seed"]), int(r["pre_round"])): r for r in cost_records
                  if "seed" in r and "pre_round" in r}
    planned_decisions = max_round - first_pre_round + 1
    alpha = delta / (planned_decisions * max_risk_looks)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        certified_at: int | None = None
        for pre_round in range(first_pre_round, max_round + 1):
            model_round = pre_round - 1
            risk = validate_risk_record(
                risk_index.get((seed, model_round)), seed=seed, model_round=model_round,
                pre_round=pre_round, alpha=alpha,
            )
            cost = validate_cost_record(cost_index.get((seed, pre_round)),
                                        seed=seed, pre_round=pre_round)
            pre_value = None
            required_cost_for_stop = None
            cost_certificate_gap = None
            if risk.valid and cost.valid:
                assert risk.risk_ucb is not None and cost.lower_bound is not None
                pre_value = risk.risk_ucb - lambda_cost * cost.lower_bound
                required_cost_for_stop = risk.risk_ucb / lambda_cost
                cost_certificate_gap = max(0.0, required_cost_for_stop - cost.lower_bound)
            decision = "certified_stop" if pre_value is not None and pre_value <= 0 else "uncertified_continue"
            diagnostic_only = certified_at is not None
            if decision == "certified_stop" and certified_at is None:
                certified_at = pre_round
            future = [historical.get((seed, r)) for r in range(pre_round, max_round + 1)]
            future = [record for record in future if record is not None]
            current = historical.get((seed, model_round))
            actual_next_round = historical.get((seed, pre_round))
            final = historical.get((seed, max_round))
            regret = None
            if current and final and current.accuracy is not None and final.accuracy is not None:
                regret = final.accuracy - current.accuracy
            budget_saved = None
            if current and final and current.cumulative_budget is not None and final.cumulative_budget is not None:
                budget_saved = final.cumulative_budget - current.cumulative_budget
            row = {
                "seed": seed,
                "pre_round": pre_round,
                "current_model_round": model_round,
                "risk_errors": risk.errors,
                "risk_total": risk.total,
                "risk_ucb": risk.risk_ucb,
                "risk_alpha": risk.alpha,
                "risk_valid": risk.valid,
                "risk_reason": risk.reason,
                "risk_source": risk.source,
                "full_round_cost_lower_bound": cost.lower_bound,
                "cost_unit": cost.unit,
                "cost_bound_valid": cost.valid,
                "cost_reason": cost.reason,
                "cost_components": cost.components,
                "cost_source": cost.source,
                "lambda": lambda_cost,
                "pre_round_net_ucb": pre_value,
                "required_cost_lower_bound_for_stop": required_cost_for_stop,
                "cost_lower_bound_gap_to_stop": cost_certificate_gap,
                "decision": decision,
                "decision_reason": _reason(risk, cost, pre_value),
                "certificate_valid": decision == "certified_stop" and risk.valid and cost.valid,
                "diagnostic_only_after_counterfactual_stop": diagnostic_only,
                "retrospective_accuracy_current": current.accuracy if current else None,
                "retrospective_accuracy_r8": final.accuracy if final else None,
                "retrospective_accuracy_regret_vs_r8": regret,
                "retrospective_actual_next_round_cost": (
                    actual_next_round.total_cost if actual_next_round else None
                ),
                "retrospective_teacher_calls_avoidable": _sum_optional(r.teacher_calls for r in future),
                "retrospective_teacher_cost_avoidable": _sum_optional(r.teacher_cost for r in future),
                "retrospective_total_cost_avoidable": _sum_optional(r.total_cost for r in future),
                "retrospective_budget_avoidable": budget_saved,
                "retrospective_metrics_used_online": False,
            }
            rows.append(row)
        online = [row for row in rows if row["seed"] == seed and not row["diagnostic_only_after_counterfactual_stop"]]
        stop_row = next((row for row in online if row["decision"] == "certified_stop"), None)
        summaries.append({
            "seed": seed,
            "terminal_status": "certified_stop" if stop_row else "budget_stop",
            "certified_stop_before_round": stop_row["pre_round"] if stop_row else None,
            "stop_reason": stop_row["decision_reason"] if stop_row else "maximum_budget_reached_without_certificate",
            "budget_stop_is_certified_safe_stop": False,
            "teacher_calls_avoidable": stop_row["retrospective_teacher_calls_avoidable"] if stop_row else 0,
            "teacher_cost_avoidable": stop_row["retrospective_teacher_cost_avoidable"] if stop_row else 0.0,
            "total_cost_avoidable": stop_row["retrospective_total_cost_avoidable"] if stop_row else 0.0,
            "budget_avoidable": stop_row["retrospective_budget_avoidable"] if stop_row else 0,
            "accuracy_regret_vs_r8": stop_row["retrospective_accuracy_regret_vs_r8"] if stop_row else 0.0,
            "missing_risk_rounds": [row["pre_round"] for row in online if not row["risk_valid"]],
            "missing_cost_bound_rounds": [row["pre_round"] for row in online if not row["cost_bound_valid"]],
        })
    return rows, summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]],
                  *, delta: float, lambda_cost: float, max_risk_looks: int) -> str:
    lines = [
        "# V7.1 Pre-Round Cost-Aware Stopping v5 — Offline Replay",
        "",
        "> Offline diagnostic only. Historical artifacts are read-only; retrospective accuracy and realised costs never enter the online decision.",
        "",
        "## Protocol",
        "",
        f"- Decision: `STOP` iff `U_R(theta_t) - {lambda_cost} * C_lower(t+1) <= 0`.",
        f"- Family confidence budget: delta={delta}, planned risk looks per decision={max_risk_looks}.",
        "- `U_R` is a one-sided Clopper–Pearson risk bound after provenance validation.",
        "- Full-round lower bound must cover discovery, verification, generation and training.",
        "- Empirical/projected mean cost is rejected as a lower confidence bound.",
        "",
        "## Seed summary",
        "",
        "| Seed | Terminal status | Certified before round | Reason | Calls avoidable* | Budget avoidable* | Accuracy regret vs R8* |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['seed']} | {summary['terminal_status']} | "
            f"{_fmt(summary['certified_stop_before_round'])} | {summary['stop_reason']} | "
            f"{_fmt(summary['teacher_calls_avoidable'])} | {_fmt(summary['budget_avoidable'])} | "
            f"{_fmt(summary['accuracy_regret_vs_r8'])} |"
        )
    lines += [
        "",
        "*Retrospective diagnostics only; these values are not used by the certificate.*",
        "",
        "## Per-round replay",
        "",
        "| Seed | Before R | Risk UCB | Cost LB | Required LB for STOP | LB gap | Actual cost* | Net UCB | Decision | Reason |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['pre_round']} | {_fmt(row['risk_ucb'])} | "
            f"{_fmt(row['full_round_cost_lower_bound'])} | "
            f"{_fmt(row['required_cost_lower_bound_for_stop'])} | "
            f"{_fmt(row['cost_lower_bound_gap_to_stop'])} | "
            f"{_fmt(row['retrospective_actual_next_round_cost'])} | "
            f"{_fmt(row['pre_round_net_ucb'])} | {row['decision']} | {row['decision_reason']} |"
        )
    lines += [
        "",
        "## Validity interpretation",
        "",
        "- Missing or model-selected validation data yields `uncertified_continue`, never STOP.",
        "- Missing/unproven full-round cost lower bounds yield `uncertified_continue`.",
        "- Reaching R8 is reported as `budget_stop`, not as a statistically certified STOP.",
        "- Rows after the first counterfactual STOP are diagnostic-only and are not an online trajectory.",
        "- Reusing a held-out set is accepted only when the manifest attests that training and model selection are independent of it and confidence is allocated across all planned looks.",
        "- The R8 comparison is an offline oracle-style diagnostic and provides no stopping guarantee.",
        "",
        "## Minimal V7.1 conclusion rule",
        "",
        "Add no theorem or runtime change unless the audit supplies both valid evidence streams. If either stream fails its provenance gate, the scientifically correct result is `uncertified_continue`.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--risk-manifest", type=Path,
                        default=Path("configs/v7_pre_round_v5/risk_evidence.json"))
    parser.add_argument("--cost-bound-manifest", type=Path,
                        default=Path("configs/v7_pre_round_v5/full_round_cost_lower_bounds.json"))
    parser.add_argument("--historical-manifest", type=Path,
                        default=Path("configs/v7_pre_round_v5/historical_trajectory_index.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/stage2_diagnostics/v7_pre_round_stopping_v5"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/v7_minimal_cost_aware/PRE_ROUND_STOPPING_V5.md"))
    parser.add_argument("--lambda-cost", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--first-pre-round", type=int, default=2)
    parser.add_argument("--max-round", type=int, default=8)
    parser.add_argument("--max-risk-looks", type=int, default=1)
    parser.add_argument("--bootstrap-cached-artifacts", action="store_true",
                        help="Read fixed held-out metrics and frozen tariffs from cached v2 outputs")
    parser.add_argument("--cached-run-root", type=Path,
                        default=Path("outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2"))
    parser.add_argument("--checkpoint-metrics-root", type=Path,
                        default=Path("outputs/stage2_diagnostics/v7_one_step_cost_aware/checkpoint_metrics"))
    parser.add_argument("--evaluation-manifest-root", type=Path,
                        default=Path("outputs/stage2_diagnostics/v7_one_step_cost_aware/evaluation_manifests"))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_risk_looks < 1:
        raise ValueError("--max-risk-looks must be positive")
    risk_payload = _read_json(args.risk_manifest) if args.risk_manifest.exists() else {"records": []}
    cost_payload = _read_json(args.cost_bound_manifest) if args.cost_bound_manifest.exists() else {"records": []}
    historical_payload = _read_json(args.historical_manifest) if args.historical_manifest.exists() else {"records": []}
    bootstrap_issues: list[str] = []
    risk_records = _index_records(risk_payload)
    cost_records = _index_records(cost_payload)
    historical = _historical_index(historical_payload)
    if args.bootstrap_cached_artifacts:
        cached_risks, cached_costs, cached_historical, bootstrap_issues = bootstrap_cached_artifacts(
            seeds=args.seeds,
            run_root=args.cached_run_root,
            checkpoint_metrics_root=args.checkpoint_metrics_root,
            evaluation_manifest_root=args.evaluation_manifest_root,
            max_round=args.max_round,
        )
        if not risk_records:
            risk_records = cached_risks
        if not cost_records:
            cost_records = cached_costs
        if not historical:
            historical = cached_historical
    rows, summaries = replay(
        seeds=args.seeds,
        risk_records=risk_records,
        cost_records=cost_records,
        historical=historical,
        lambda_cost=args.lambda_cost,
        delta=args.delta,
        first_pre_round=args.first_pre_round,
        max_round=args.max_round,
        max_risk_looks=args.max_risk_looks,
    )
    validation = {
        "protocol": "v7.1_pre_round_cost_aware_stopping_v5",
        "read_only": True,
        "risk_manifest_exists": args.risk_manifest.exists(),
        "cost_bound_manifest_exists": args.cost_bound_manifest.exists(),
        "historical_manifest_exists": args.historical_manifest.exists(),
        "seeds": args.seeds,
        "valid_risk_rows": sum(bool(row["risk_valid"]) for row in rows),
        "valid_cost_bound_rows": sum(bool(row["cost_bound_valid"]) for row in rows),
        "certified_stop_rows": sum(bool(row["certificate_valid"]) for row in rows),
        "bootstrap_cached_artifacts": args.bootstrap_cached_artifacts,
        "bootstrap_issues": bootstrap_issues,
        "risk_evidence_rows": len(risk_records),
        "cost_bound_rows": len(cost_records),
        "historical_rows": len(historical),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    payload = {
        "protocol": validation["protocol"],
        "parameters": {
            "lambda": args.lambda_cost,
            "delta": args.delta,
            "first_pre_round": args.first_pre_round,
            "max_round": args.max_round,
            "max_risk_looks": args.max_risk_looks,
        },
        "validation": validation,
        "summaries": summaries,
        "rows": rows,
    }
    (args.output_dir / "replay.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "per_round.csv", rows)
    _write_csv(args.output_dir / "per_seed.csv", summaries)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(rows, summaries, delta=args.delta,
                                         lambda_cost=args.lambda_cost,
                                         max_risk_looks=args.max_risk_looks), encoding="utf-8")
    print(f"[DONE] {args.output_dir / 'replay.json'}")
    print(f"[DONE] {args.report}")


if __name__ == "__main__":
    main()
