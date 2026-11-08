#!/usr/bin/env python3
"""CPU-only Incremental Cost-Aware Accounting & Decision Replay v3.

Historical artifacts are read-only. Outputs are versioned under a new
diagnostic root. The online replay uses projected decision cost only; incurred
exploration and actual future costs are retrospective reporting fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decision.incremental_cost_accounting import (  # noqa: E402
    CostPartition,
    ReplayInputs,
    audit_evaluation_rate_modes,
    replay_incremental_decision,
)
from src.utils.config import load_yaml_with_extends  # noqa: E402


SCHEMA = "v7_incremental_cost_aware_replay_v3"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _first(mapping: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _number(mapping.get(key))
        if value is not None:
            return value
    return None


def _find_key(value: Any, names: set[str]) -> Optional[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in names:
                return child
        for child in value.values():
            found = _find_key(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, names)
            if found is not None:
                return found
    return None


def _find_number(value: Any, *names: str) -> Optional[float]:
    return _number(_find_key(value, {name.lower() for name in names}))


def _contains_material_key(value: Any, fragments: Sequence[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if all(fragment in lowered for fragment in fragments) and child not in (None, {}, [], 0, 0.0):
                return True
            if _contains_material_key(child, fragments):
                return True
    elif isinstance(value, list):
        return any(_contains_material_key(child, fragments) for child in value)
    return False


def _decision_records(run_dir: Path, result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for row in result.get("cost_aware_rounds", []):
        round_index = row.get("round", row.get("round_index"))
        if round_index is not None:
            records[int(round_index)] = row
    for path in sorted((run_dir / "cost_aware").glob("round_*/decision.json")):
        payload = read_json(path)
        round_index = payload.get("round", payload.get("round_index"))
        if round_index is None:
            round_index = int(path.parent.name.rsplit("_", 1)[-1])
        records[int(round_index)] = payload
    return records


def _range_records(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get("rounds", payload.get("rows", payload.get("comparisons", [])))
    return {
        int(row.get("round", row.get("round_index"))): row
        for row in rows
        if row.get("round", row.get("round_index")) is not None
    }


def _one_step_records(path: Path) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return {
        (int(row["seed"]), int(row["round_index"]), str(row["policy"])): row
        for row in payload.get("rows", [])
        if row.get("observed_gain") is not None
    }


def _cost_payload(decision_record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = decision_record.get("decision_cost", {})
    return value if isinstance(value, Mapping) else {}


def _cost_partition(
    decision_record: Mapping[str, Any],
    result: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> CostPartition:
    cost = _cost_payload(decision_record)
    exploration = _first(cost, "incurred_exploration_cost", "exploration_cost")
    decision = _first(cost, "decision_cost", "projected_decision_cost")
    actual = _first(cost, "actual_main_update_cost", "actual_intervention_cost")
    total = _first(cost, "total_pipeline_cost")
    modeled_usd = _first(cost, "modeled_usd_cost", "usd_cost", "total_usd")
    unit = str(cost.get("cost_unit", cost.get("unit", "normalized_cost_units")))
    initial_available = (
        _contains_material_key(result, ("initial", "pool", "cost"))
        or _contains_material_key(source_result, ("initial", "pool", "cost"))
    )
    round_one_available = (
        _contains_material_key(result, ("round", "1", "cost"))
        or _contains_material_key(source_result, ("round", "1", "cost"))
        or _contains_material_key(result, ("round_one", "cost"))
        or _contains_material_key(source_result, ("round_one", "cost"))
    )
    missing: list[str] = []
    for name, value in (
        ("exploration_cost", exploration),
        ("decision_cost", decision),
        ("actual_intervention_cost", actual),
        ("total_pipeline_cost", total),
    ):
        if value is None:
            missing.append(name)
    return CostPartition(
        exploration_cost=exploration,
        decision_cost=decision,
        actual_intervention_cost=actual,
        total_pipeline_cost=total,
        modeled_usd_cost=modeled_usd,
        cost_unit=unit,
        shared_initial_pool_cost_available=initial_available,
        round_one_cost_available=round_one_available,
        missing_components=tuple(missing),
    )


def _sum_field(rows: Iterable[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [_number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _pilot_breakdown(run_dir: Path, round_index: int) -> dict[str, Any]:
    path = run_dir / "cost_aware" / f"round_{round_index:03d}" / "pilot_observations.jsonl"
    if not path.exists():
        return {
            "pilot_observations": 0,
            "pilot_gpu_seconds": None,
            "pilot_training_gpu_seconds": None,
            "pilot_evaluation_gpu_seconds": None,
            "pilot_wall_seconds": None,
            "pilot_gpu_double_count_check": "missing_pilot_observations",
        }
    rows = read_jsonl(path)
    total = _sum_field(rows, "gpu_seconds")
    training = _sum_field(rows, "pilot_training_gpu_seconds")
    evaluation = _sum_field(rows, "pilot_evaluation_gpu_seconds")
    status = "insufficient_subcomponents"
    if total is not None and training is not None and evaluation is not None:
        status = "consistent_total_equals_training_plus_evaluation" if math.isclose(
            total, training + evaluation, rel_tol=1e-6, abs_tol=1e-6
        ) else "mismatch_total_vs_training_plus_evaluation"
    return {
        "pilot_observations": len(rows),
        "pilot_gpu_seconds": total,
        "pilot_training_gpu_seconds": training,
        "pilot_evaluation_gpu_seconds": evaluation,
        "pilot_wall_seconds": _sum_field(rows, "wall_clock_seconds"),
        "pilot_gpu_double_count_check": status,
    }


def _usage_breakdown(
    decision_record: Mapping[str, Any],
    run_dir: Path,
    round_index: int,
) -> dict[str, Any]:
    cost = _cost_payload(decision_record)
    usage = cost.get("usage", {}) if isinstance(cost.get("usage"), Mapping) else {}
    phases = cost.get("phases", {}) if isinstance(cost.get("phases"), Mapping) else {}
    breakdown = {
        "teacher_input_tokens": _find_number(usage, "teacher_input_tokens", "prompt_tokens"),
        "teacher_output_tokens": _find_number(usage, "teacher_output_tokens", "completion_tokens"),
        "teacher_total_tokens": _find_number(usage, "teacher_total_tokens", "total_tokens"),
        "verification_tokens": _find_number(usage, "verification_tokens"),
        "generation_tokens": _find_number(usage, "generation_tokens", "synthetic_generation_tokens"),
        "verification_cost": _find_number(phases, "verification_cost"),
        "generation_qc_cost": _find_number(phases, "generation_qc_cost", "synthetic_generation_qc_cost"),
        "main_training_gpu_seconds": _find_number(phases, "main_training_gpu_seconds", "training_gpu_seconds"),
        "epoch_validation_gpu_seconds": _find_number(phases, "epoch_validation_gpu_seconds", "validation_gpu_seconds"),
        "evaluation_gpu_seconds": _find_number(phases, "evaluation_gpu_seconds"),
        "projected_main_training_cost": _find_number(phases, "projected_main_training_cost"),
        "projected_epoch_validation_cost": _find_number(phases, "projected_epoch_validation_cost"),
        "actual_main_training_cost": _find_number(phases, "actual_main_training_cost"),
        "actual_validation_cost": _find_number(phases, "actual_validation_cost"),
        "modeled_usd_cost": _find_number(cost, "modeled_usd_cost", "usd_cost", "total_usd"),
    }
    breakdown.update(_pilot_breakdown(run_dir, round_index))
    return breakdown


def _collect_rate_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    rates: dict[str, Any] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if isinstance(child, (int, float)) and not isinstance(child, bool) and (
                    any(marker in lowered for marker in ("rate", "per_", "cost_per", "usd"))
                    or lowered in {
                        "evaluation_per_item",
                        "evaluation_per_run",
                        "evaluation_gpu_second",
                        "evaluation_gpu_hour",
                    }
                ):
                    rates[path] = child
                visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(config)
    return rates


def _flat_rate_names(rates: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for path, value in rates.items():
        flattened[path.rsplit(".", 1)[-1]] = value
    return flattened


def _decision_values(
    policy: str,
    decision_record: Mapping[str, Any],
    range_record: Optional[Mapping[str, Any]],
) -> tuple[Optional[float], Optional[float], str, str, Optional[float]]:
    decision = decision_record.get("decision", decision_record)
    gain = decision_record.get("gain_estimate", {})
    selected: Mapping[str, Any] = decision
    if policy == "v2_1_range_aware" and range_record is not None:
        selected = range_record.get("range_aware", range_record)
    delta_hat = _first(selected, "delta_hat", "estimated_gain")
    if delta_hat is None:
        delta_hat = _first(decision, "delta_hat", "estimated_gain")
    if delta_hat is None and isinstance(gain, Mapping):
        delta_hat = _first(gain, "delta_hat", "mean_gain")
    radius = _first(selected, "confidence_radius", "radius")
    action = str(selected.get("action", decision.get("action", "continue"))).lower()
    reason = str(selected.get("reason", decision.get("reason", "unavailable")))
    prior_net = _first(selected, "net_ucb", "optimistic_net_value", "max_optimistic_value")
    return delta_hat, radius, action, reason, prior_net


def _evidence_gate(reason: str) -> Optional[str]:
    lowered = reason.lower()
    if any(token in lowered for token in ("no_certified", "no_eligible", "insufficient_evidence", "evidence_not_exhausted")):
        return reason
    return None


def _resource_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # One row per round is sufficient because policy variants share costs.
    per_round: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        per_round.setdefault((int(row["seed"]), int(row["round_index"])), row)
    selected = list(per_round.values())
    exploration = [row.get("exploration_cost") for row in selected if row.get("exploration_cost") is not None]
    actual = [row.get("actual_intervention_cost") for row in selected if row.get("actual_intervention_cost") is not None]
    return {
        "rounds": len(selected),
        "total_exploration_overhead": sum(exploration) if exploration else None,
        "total_actual_intervention_cost": sum(actual) if actual else None,
        "partial_total_cost_rounds": sum(bool(row.get("partial_total_cost")) for row in selected),
        "teacher_total_tokens": sum(row.get("teacher_total_tokens") or 0 for row in selected),
        "pilot_gpu_seconds": sum(row.get("pilot_gpu_seconds") or 0 for row in selected),
        "main_training_gpu_seconds": sum(row.get("main_training_gpu_seconds") or 0 for row in selected),
        "epoch_validation_gpu_seconds": sum(row.get("epoch_validation_gpu_seconds") or 0 for row in selected),
        "evaluation_gpu_seconds": sum(row.get("evaluation_gpu_seconds") or 0 for row in selected),
    }


def _sensitivity_rows(
    base_rows: Sequence[Mapping[str, Any]],
    lambda_values: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    groups = sorted({(int(row["seed"]), str(row["policy"])) for row in base_rows})
    for seed, policy in groups:
        trajectory = sorted(
            (row for row in base_rows if int(row["seed"]) == seed and str(row["policy"]) == policy),
            key=lambda row: int(row["round_index"]),
        )
        for lambda_cost in lambda_values:
            first_stop: Optional[int] = None
            avoidable = 0.0
            for base in trajectory:
                counterfactual = first_stop is not None
                gain_ucb = base.get("gain_ucb")
                decision_cost = base.get("decision_cost")
                gate = base.get("evidence_gate")
                net_ucb = (
                    float(gain_ucb) - lambda_cost * float(decision_cost)
                    if gain_ucb is not None and decision_cost is not None
                    else None
                )
                if gate:
                    action = base["recorded_action"]
                    reason = base["recorded_reason"]
                elif net_ucb is None:
                    action = base["recorded_action"]
                    reason = "insufficient_evidence_preserve_recorded_action"
                else:
                    action = "continue" if net_ucb > 0 else "stop"
                    reason = "positive_incremental_net_ucb" if action == "continue" else "nonpositive_incremental_net_ucb"
                if first_stop is None and action == "stop":
                    first_stop = int(base["round_index"])
                if first_stop is not None and int(base["round_index"]) >= first_stop and decision_cost is not None:
                    avoidable += float(decision_cost)
                observed_net = (
                    float(base["observed_one_step_gain"])
                    - lambda_cost * float(base["actual_intervention_cost"])
                    if base.get("observed_one_step_gain") is not None
                    and base.get("actual_intervention_cost") is not None
                    else None
                )
                rows.append(
                    {
                        "seed": seed,
                        "policy": policy,
                        "round_index": int(base["round_index"]),
                        "lambda_cost": lambda_cost,
                        "net_ucb": net_ucb,
                        "action": action,
                        "reason": reason,
                        "counterfactual_only": counterfactual,
                        "observed_one_step_net_gain": observed_net,
                        "decision_cost": decision_cost,
                    }
                )
            summaries.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "lambda_cost": lambda_cost,
                    "first_counterfactual_stop_round": first_stop,
                    "estimated_avoidable_decision_cost": avoidable if first_stop is not None else 0.0,
                    "budget_exhaustion_counted_as_stop": False,
                }
            )
    return rows, summaries


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    totals: Mapping[str, Any],
    totals_by_seed: Mapping[str, Mapping[str, Any]],
    audit_issues: Sequence[str],
) -> str:
    lines = [
        "# V7 Incremental Cost-Aware Accounting & Decision Replay v3",
        "",
        "> CPU-only retrospective analysis. Historical trajectories and v2/v2.1 protocols are unchanged.",
        "",
        "## Cost semantics",
        "",
        "- `exploration_cost`: already incurred teacher/verification/generation/pilot overhead; never used in online J/U.",
        "- `decision_cost`: projected avoidable main LoRA training plus epoch validation, known before decision.",
        "- `total_pipeline_cost`: retrospective pipeline reporting; marked partial when initial-pool or round-1 cost is unavailable.",
        "- Online replay: `J = delta_hat - lambda * decision_cost`; `U = Gain_UCB - lambda * decision_cost`.",
        "",
        "## Artifact validation",
        "",
        "| Seed | Status | Missing artifacts |",
        "|---:|---|---|",
    ]
    for item in validation:
        lines.append(f"| {item['seed']} | {item['status']} | {', '.join(item.get('missing', [])) or 'none'} |")
    lines += [
        "",
        "## Per-round replay",
        "",
        "| Seed | R | Policy | Delta | Radius | Gain-UCB | Decision cost | Est. net | Net-UCB | Action | Same as v2 | Observed net | Partial total |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round_index']} | {row['policy']} | {_fmt(row.get('delta_hat'))} | "
            f"{_fmt(row.get('confidence_radius'))} | {_fmt(row.get('gain_ucb'))} | {_fmt(row.get('decision_cost'))} | "
            f"{_fmt(row.get('estimated_net_gain'))} | {_fmt(row.get('net_ucb'))} | {row.get('replay_action')} | "
            f"{row.get('replay_matches_recorded')} | {_fmt(row.get('observed_one_step_net_gain'))} | {row.get('partial_total_cost')} |"
        )
    lines += [
        "",
        "## Per-round resource and cost breakdown",
        "",
        "Costs retain their artifact unit; modeled USD is never merged with normalized decision cost. GPU fields are measured seconds when available.",
        "",
        "| Seed | R | Teacher tokens | Verification cost | Generation/QC cost | Pilot train GPU s | Pilot eval GPU s | Main train GPU s | Validation GPU s | Evaluation GPU s | Exploration cost | Projected decision cost | Actual intervention | Total pipeline | Modeled USD | Unit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    cost_rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        cost_rows.setdefault((int(row["seed"]), int(row["round_index"])), row)
    for (seed, round_index), row in sorted(cost_rows.items()):
        lines.append(
            f"| {seed} | {round_index} | {row.get('teacher_total_tokens') or 'N/A'} | "
            f"{_fmt(row.get('verification_cost'))} | {_fmt(row.get('generation_qc_cost'))} | "
            f"{_fmt(row.get('pilot_training_gpu_seconds'))} | {_fmt(row.get('pilot_evaluation_gpu_seconds'))} | "
            f"{_fmt(row.get('main_training_gpu_seconds'))} | {_fmt(row.get('epoch_validation_gpu_seconds'))} | "
            f"{_fmt(row.get('evaluation_gpu_seconds'))} | {_fmt(row.get('exploration_cost'))} | "
            f"{_fmt(row.get('decision_cost'))} | {_fmt(row.get('actual_intervention_cost'))} | "
            f"{_fmt(row.get('total_pipeline_cost'))} | {_fmt(row.get('modeled_usd_cost'))} | "
            f"{row.get('cost_unit', 'unknown')} |"
        )
    lines += [
        "",
        "## Aggregate resources",
        "",
        f"- Total exploration overhead (available round records): `{_fmt(totals.get('total_exploration_overhead'))}`.",
        f"- Total actual intervention cost: `{_fmt(totals.get('total_actual_intervention_cost'))}`.",
        f"- Teacher tokens recorded: `{totals.get('teacher_total_tokens')}`.",
        f"- Pilot GPU seconds recorded: `{_fmt(totals.get('pilot_gpu_seconds'))}`.",
        f"- Partial-total rounds: `{totals.get('partial_total_cost_rounds')}/{totals.get('rounds')}`.",
        "",
        "| Seed | Exploration overhead | Actual intervention cost | Teacher tokens | Pilot GPU s | Partial rounds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, summary in sorted(totals_by_seed.items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {seed} | {_fmt(summary.get('total_exploration_overhead'))} | "
            f"{_fmt(summary.get('total_actual_intervention_cost'))} | "
            f"{summary.get('teacher_total_tokens')} | {_fmt(summary.get('pilot_gpu_seconds'))} | "
            f"{summary.get('partial_total_cost_rounds')}/{summary.get('rounds')} |"
        )
    unchanged = [row.get("cost_model_numerically_unchanged") for row in rows]
    known_unchanged = [value for value in unchanged if value is not None]
    lines += [
        "",
        (
            f"Numerical replay consistency with saved v2/v2.1 Net-UCB: "
            f"`{sum(value is True for value in known_unchanged)}/{len(known_unchanged)}` comparable rows unchanged. "
            "No change is claimed for rows without a saved comparable Net-UCB."
        ),
        "",
        "## Lambda sensitivity (diagnostic only)",
        "",
        "| Seed | Policy | Lambda | First counterfactual STOP | Estimated avoidable decision cost |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['seed']} | {row['policy']} | {row['lambda_cost']} | "
            f"{row.get('first_counterfactual_stop_round') or 'none'} | "
            f"{_fmt(row.get('estimated_avoidable_decision_cost'))} |"
        )
    lines += [
        "",
        "## Audit findings",
        "",
    ]
    lines.extend(f"- {issue}" for issue in audit_issues)
    if not audit_issues:
        lines.append("- No mechanical accounting inconsistency detected in available fields.")
    lines += [
        "",
        "## Appendix assumptions requiring clarification",
        "",
        "- The stopping objective must define decision cost as avoidable marginal cost, excluding sunk exploration.",
        "- Total pipeline cost must include exploration and pilot overhead, but it is not the penalty in the online stopping inequality.",
        "- Actual future intervention cost is retrospective only and cannot enter an online decision.",
        "- Evaluation per-item, per-run and GPU-time charging are additive only when the frozen cost config explicitly says so.",
        "- Rounds after the first sensitivity STOP are counterfactual diagnostics, not a valid online trajectory.",
        "- Mixed-policy intervention and fixed-shard pilots do not establish a population-level theorem guarantee.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v7_incremental_cost_aware_replay_v3.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    source_specs = {int(item["seed"]): item for item in config["sources"]}
    seeds = args.seeds or sorted(source_specs)
    rounds = [int(value) for value in config["protocol"]["rounds"]]
    lambda_cost = float(config["protocol"]["lambda_cost"])
    lambda_values = args.lambda_values or [float(value) for value in config["sensitivity"]["lambda_values"]]
    output_root = Path(config["outputs"]["root"])
    report_path = Path(config["outputs"]["report"])
    one_step = _one_step_records(Path(config["inputs"]["one_step_benchmark"]))
    output_root.mkdir(parents=True, exist_ok=True)

    validation: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    audit_issues: list[str] = []
    rate_snapshots: dict[str, Any] = {}

    for seed in seeds:
        source_spec = source_specs.get(seed)
        if source_spec is None:
            validation.append({"seed": seed, "status": "missing", "missing": ["source_spec"]})
            continue
        run_dir = Path(source_spec["run_dir"])
        result_path = run_dir / "result.json"
        missing: list[str] = []
        if not result_path.exists():
            missing.append(str(result_path))
        range_path = Path(source_spec["range_replay"])
        if not range_path.exists():
            missing.append(str(range_path))
        if missing:
            validation.append({"seed": seed, "status": "missing", "missing": missing})
            continue

        result = read_json(result_path)
        source_dir = Path(source_spec["source_run_dir"]) if source_spec.get("source_run_dir") else run_dir
        source_result_path = source_dir / "result.json"
        source_result = read_json(source_result_path) if source_result_path.exists() else {}
        run_config_path = run_dir / "config.yaml"
        if not run_config_path.exists():
            run_config_path = source_dir / "config.yaml"
        if run_config_path.exists():
            resolved_config = load_yaml_with_extends(run_config_path)
            rate_snapshot = _collect_rate_snapshot(resolved_config)
            rate_snapshots[str(seed)] = rate_snapshot
            explicit_additive = bool(_find_key(resolved_config, {"evaluation_costs_are_additive", "additive_evaluation_charging"}))
            for issue in audit_evaluation_rate_modes(
                _flat_rate_names(rate_snapshot),
                explicit_additive_policy=explicit_additive,
            ):
                audit_issues.append(f"seed_{seed}:{issue}")
        else:
            missing.append("resolved_run_config")

        decisions = _decision_records(run_dir, result)
        range_records = _range_records(range_path)
        absent_rounds = [round_index for round_index in rounds if round_index not in decisions]
        if absent_rounds:
            missing.append("decision_rounds:" + ",".join(str(value) for value in absent_rounds))
        validation.append({"seed": seed, "status": "ready" if not missing else "partial", "missing": missing})

        for round_index in rounds:
            decision_record = decisions.get(round_index)
            if decision_record is None:
                continue
            costs = _cost_partition(decision_record, result, source_result)
            breakdown = _usage_breakdown(decision_record, run_dir, round_index)
            if costs.internally_consistent is False:
                audit_issues.append(f"seed_{seed}_round_{round_index}:total_pipeline_cost_below_exploration_plus_actual")
            if breakdown["pilot_gpu_double_count_check"] == "mismatch_total_vs_training_plus_evaluation":
                audit_issues.append(f"seed_{seed}_round_{round_index}:pilot_gpu_total_mismatch")

            for policy in config["protocol"]["policies"]:
                delta_hat, radius, action, reason, prior_net = _decision_values(
                    policy,
                    decision_record,
                    range_records.get(round_index),
                )
                observed = one_step.get((seed, round_index, policy), {})
                replay = replay_incremental_decision(
                    ReplayInputs(
                        seed=seed,
                        round_index=round_index,
                        policy=policy,
                        delta_hat=delta_hat,
                        confidence_radius=radius,
                        support_upper=float(config["protocol"]["support_upper"]),
                        lambda_cost=lambda_cost,
                        costs=costs,
                        recorded_action=action,
                        recorded_reason=reason,
                        evidence_gate=_evidence_gate(reason),
                        observed_one_step_gain=_number(observed.get("observed_gain")),
                    )
                )
                row = replay.to_dict()
                row.update(costs.to_dict())
                row.update(breakdown)
                row["prior_net_ucb"] = prior_net
                row["cost_model_numerically_unchanged"] = (
                    math.isclose(prior_net, replay.net_ucb, rel_tol=1e-9, abs_tol=1e-9)
                    if prior_net is not None and replay.net_ucb is not None
                    else None
                )
                row["decision_cost_known_before_decision"] = costs.decision_cost is not None
                row["decision_cost_provenance"] = "predecision_decision_artifact"
                row["actual_future_cost_used_online"] = False
                row["post_stop_online_valid"] = True
                replay_rows.append(row)

    write_json(output_root / "validation.json", {"schema": SCHEMA, "seeds": validation, "rate_snapshots": rate_snapshots})
    if args.validate_only:
        print(f"Wrote {output_root / 'validation.json'}")
        return

    sensitivity_rows, sensitivity_summary = _sensitivity_rows(replay_rows, lambda_values)
    totals = _resource_totals(replay_rows)
    totals_by_seed = {
        str(seed): _resource_totals(
            [row for row in replay_rows if int(row["seed"]) == seed]
        )
        for seed in seeds
        if any(int(row["seed"]) == seed for row in replay_rows)
    }
    payload = {
        "schema": SCHEMA,
        "protocol": config["protocol"],
        "cost_semantics": {
            "online_penalty": "projected_incremental_decision_cost_only",
            "sunk_exploration_used_online": False,
            "actual_future_cost_used_online": False,
        },
        "validation": validation,
        "resource_totals": totals,
        "resource_totals_by_seed": totals_by_seed,
        "audit_issues": sorted(set(audit_issues)),
        "rows": replay_rows,
        "sensitivity_summary": sensitivity_summary,
    }
    write_json(output_root / "replay.json", payload)
    write_csv(output_root / "replay.csv", replay_rows)
    write_json(output_root / "sensitivity.json", {"schema": SCHEMA, "rows": sensitivity_rows, "summary": sensitivity_summary})
    write_csv(output_root / "sensitivity.csv", sensitivity_rows)
    for seed in seeds:
        seed_rows = [row for row in replay_rows if int(row["seed"]) == seed]
        write_json(output_root / f"seed_{seed}" / "replay.json", {"schema": SCHEMA, "seed": seed, "rows": seed_rows})
        write_csv(output_root / f"seed_{seed}" / "replay.csv", seed_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            replay_rows,
            validation,
            sensitivity_summary,
            totals,
            totals_by_seed,
            sorted(set(audit_issues)),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_root / 'replay.json'}")
    print(f"Wrote {output_root / 'replay.csv'}")
    print(f"Wrote {output_root / 'sensitivity.json'}")
    print(f"Wrote {output_root / 'sensitivity.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
