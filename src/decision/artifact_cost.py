"""Build candidate cost ledgers from measured round artifacts and projections."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config


_PROJECTION_FIELDS = {
    "lora_training_gpu_seconds_fixed",
    "lora_training_gpu_seconds_per_item",
    "evaluation_gpu_seconds_fixed",
    "evaluation_gpu_seconds_per_item",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            result.append(value)
    return result


def _number(mapping: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"cost.projections.{key} must be numeric")
    result = float(value)
    if result < 0:
        raise ValueError(f"cost.projections.{key} must be non-negative")
    return result


def _projections(cost_config: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = cost_config.get("projections", {})
    if not isinstance(raw, Mapping):
        raise ValueError("cost.projections must be a mapping")
    unknown = sorted(set(raw) - _PROJECTION_FIELDS)
    if unknown:
        raise ValueError(f"Unknown cost projection fields: {unknown}")
    return raw


def _contains_id(value: Any, hypothesis_id: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_id(item, hypothesis_id) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_id(item, hypothesis_id) for item in value)
    return value == hypothesis_id


def _allocated_teacher_usage(
    run_dir: Path, hypothesis_ids: Sequence[str]
) -> dict[str, tuple[float, float]]:
    allocated = {item: [0.0, 0.0] for item in hypothesis_ids}
    logs: list[dict[str, Any]] = []
    for path in sorted((run_dir / "teacher_calls").glob("*.jsonl")):
        logs.extend(_jsonl(path))
    for record in logs:
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_tokens = _number(usage, "input_tokens")
        output_tokens = _number(usage, "output_tokens")
        matched = [item for item in hypothesis_ids if _contains_id(record, item)]
        recipients = matched or list(hypothesis_ids)
        if not recipients:
            continue
        for item in recipients:
            allocated[item][0] += input_tokens / len(recipients)
            allocated[item][1] += output_tokens / len(recipients)
    return {key: (value[0], value[1]) for key, value in allocated.items()}


def build_round_incurred_cost(
    *,
    run_dir: str | Path,
    pilot_observations: Sequence[Mapping[str, Any]],
    cost_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Account once for every exploration resource, including STOP/no-candidate rounds."""
    root = Path(run_dir)
    rates, currency = cost_rates_from_config(cost_config)
    _projections(cost_config)
    input_tokens = 0.0
    output_tokens = 0.0
    for path in sorted((root / "teacher_calls").glob("*.jsonl")):
        for record in _jsonl(path):
            usage = record.get("usage")
            if isinstance(usage, Mapping):
                input_tokens += _number(usage, "input_tokens")
                output_tokens += _number(usage, "output_tokens")
    generated = len(_jsonl(root / "synthetic" / "generated.jsonl"))
    probes = _jsonl(root / "verification" / "generated_probes.jsonl")
    probe_predictions = _jsonl(root / "verification" / "probes.jsonl")
    round_predictions = _jsonl(root / "predictions.jsonl")
    pilot_gpu = sum(float(row.get("gpu_seconds") or 0.0) for row in pilot_observations)
    pilot_eval_items = sum(
        2 * len(row.get("eval_ids", []))
        for row in pilot_observations
        if isinstance(row.get("eval_ids"), list)
    )
    usage = CostUsage(
        llm_input_tokens=input_tokens,
        llm_output_tokens=output_tokens,
        generated_items=generated,
        verified_items=generated + len(probes),
        lora_training_runs=len(pilot_observations),
        lora_training_gpu_seconds=pilot_gpu,
        evaluation_runs=(
            2 * len(pilot_observations)
            + len({str(row.get("hypothesis_id", "")) for row in probe_predictions})
            + (1 if any("prediction" in row for row in round_predictions) else 0)
        ),
        evaluation_items=(
            pilot_eval_items
            + len(probe_predictions)
            + sum("prediction" in row for row in round_predictions)
        ),
    )
    result = compute_cost(usage, rates, currency=currency).to_dict()
    result["measurement_status"] = {
        "teacher_tokens": "measured",
        "pilot_training_gpu_seconds": "measured_when_available",
        "rates": "configured_assumptions",
    }
    return result


def build_artifact_costs(
    *,
    run_dir: str | Path,
    hypothesis_ids: Sequence[str],
    full_training_counts: Mapping[str, int],
    evaluation_counts: Mapping[str, int],
    pilot_observations: Sequence[Mapping[str, Any]],
    cost_config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Compute ``C(h)`` for every candidate from measured and projected usage.

    Teacher usage and completed pilot GPU/evaluation usage are measured. Future
    full-training and evaluation usage must be configured projections because
    committing those resources before selection would invalidate the decision.
    """
    root = Path(run_dir)
    ids = [str(item) for item in hypothesis_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis_ids must be unique")
    rates, currency = cost_rates_from_config(cost_config)
    projections_raw = _projections(cost_config)
    tokens = _allocated_teacher_usage(root, ids)
    generated_rows = _jsonl(root / "synthetic" / "generated.jsonl")
    verification_rows = _jsonl(root / "verification" / "generated_probes.jsonl")
    verification_predictions = _jsonl(root / "verification" / "probes.jsonl")
    round_predictions = _jsonl(root / "predictions.jsonl")
    shared_eval_items = sum("prediction" in row for row in round_predictions)

    costs: dict[str, float] = {}
    breakdowns: dict[str, dict[str, Any]] = {}
    for hypothesis_id in ids:
        full_count = int(full_training_counts.get(hypothesis_id, 0))
        eval_count = int(evaluation_counts.get(hypothesis_id, 0))
        if full_count < 0 or eval_count < 0:
            raise ValueError("Candidate training/evaluation counts cannot be negative")
        generated_count = sum(
            row.get("hypothesis_id") == hypothesis_id for row in generated_rows
        )
        probe_count = sum(
            row.get("hypothesis_id") == hypothesis_id for row in verification_rows
        )
        probe_eval_count = sum(
            row.get("hypothesis_id") == hypothesis_id
            for row in verification_predictions
        )
        pilots = [
            row
            for row in pilot_observations
            if row.get("hypothesis_id") == hypothesis_id
        ]
        pilot_gpu = sum(float(row.get("gpu_seconds") or 0.0) for row in pilots)
        pilot_eval_items = sum(
            2 * len(row.get("eval_ids", []))
            for row in pilots
            if isinstance(row.get("eval_ids"), list)
        )
        input_tokens, output_tokens = tokens.get(hypothesis_id, (0.0, 0.0))
        projected_train_gpu = (
            _number(projections_raw, "lora_training_gpu_seconds_fixed")
            + full_count
            * _number(projections_raw, "lora_training_gpu_seconds_per_item")
            if full_count
            else 0.0
        )
        projected_eval_gpu = (
            _number(projections_raw, "evaluation_gpu_seconds_fixed")
            + eval_count
            * _number(projections_raw, "evaluation_gpu_seconds_per_item")
            if eval_count
            else 0.0
        )
        incurred_usage = CostUsage(
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            generated_items=generated_count,
            verified_items=generated_count + probe_count,
            lora_training_runs=len(pilots),
            lora_training_gpu_seconds=pilot_gpu,
            evaluation_runs=(
                2 * len(pilots)
                + (1 if probe_eval_count else 0)
                + (1 / len(ids) if ids and shared_eval_items else 0)
            ),
            evaluation_items=(
                pilot_eval_items
                + probe_eval_count
                + (shared_eval_items / len(ids) if ids else 0)
            ),
        )
        projected_usage = CostUsage(
            lora_training_runs=1 if full_count else 0,
            lora_training_gpu_seconds=projected_train_gpu,
            evaluation_runs=2 if eval_count else 0,
            evaluation_items=2 * eval_count,
            evaluation_gpu_seconds=projected_eval_gpu,
        )
        usage = incurred_usage.plus(projected_usage)
        incurred = compute_cost(incurred_usage, rates, currency=currency)
        projected = compute_cost(projected_usage, rates, currency=currency)
        breakdown = compute_cost(usage, rates, currency=currency)
        payload = breakdown.to_dict()
        payload["phases"] = {
            "incurred_exploration": incurred.to_dict(),
            "projected_intervention": projected.to_dict(),
        }
        payload["measurement_status"] = {
            "teacher_tokens": "measured",
            "pilot_training_gpu_seconds": "measured_when_available",
            "full_training_gpu_seconds": "projected",
            "full_evaluation_gpu_seconds": "projected",
            "rates": "configured_assumptions",
        }
        costs[hypothesis_id] = breakdown.total
        breakdowns[hypothesis_id] = payload
    return costs, breakdowns
