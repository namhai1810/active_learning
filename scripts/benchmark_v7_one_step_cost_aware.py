#!/usr/bin/env python3
"""Build the CPU-only one-step decision benchmark for V7 Cost-Aware.

This script never loads a model.  It validates artifact provenance, prepares a
strict held-out evaluation manifest/inference plan, and combines saved metrics
with v2/v2.1 decision records.  Missing evidence is reported, never inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_yaml_with_extends
from src.decision.one_step_cost_aware_benchmark import (
    DecisionInputs,
    ObservedUpdate,
    assess_one_step,
    cost_accounting_row,
)


SCHEMA = "v7_one_step_cost_aware_benchmark_v1"
PILOT_ID_KEYS = {
    "evaluation_ids",
    "eval_ids",
    "pilot_eval_ids",
    "gain_eval_ids",
    "pilot_ids",
}


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


def hash_local_path(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        example_id = row.get("id", row.get("example_id"))
        if example_id is not None:
            result.add(str(example_id))
    return result


def _collect_id_lists(value: Any) -> set[str]:
    """Collect only explicitly named evaluation-ID lists; hashes are ignored."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in PILOT_ID_KEYS and isinstance(child, list):
                if all(isinstance(item, (str, int)) for item in child):
                    found.update(str(item) for item in child)
            else:
                found.update(_collect_id_lists(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_id_lists(child))
    return found


def _pilot_partitions(plan: Any) -> tuple[set[str], set[str]]:
    """Return (pilot-eval IDs, disjoint held-out gain-report IDs)."""
    if not isinstance(plan, Mapping):
        return set(), set()
    pilot_ids = set(str(value) for value in plan.get("pilot_ids", []) if isinstance(value, (str, int)))
    round_shards = plan.get("round_shards", {})
    if isinstance(round_shards, Mapping):
        for shards in round_shards.values():
            if isinstance(shards, list):
                for shard in shards:
                    if isinstance(shard, list):
                        pilot_ids.update(str(value) for value in shard if isinstance(value, (str, int)))
    pilot_ids.update(_collect_id_lists(plan))
    held_out = {
        str(value)
        for value in plan.get("held_out_gain_report_ids", [])
        if isinstance(value, (str, int))
    }
    return pilot_ids, held_out


def _existing(*paths: Path) -> Optional[Path]:
    return next((path for path in paths if path.exists()), None)


def _load_run_config(run_dir: Path, source_dir: Path) -> tuple[dict[str, Any], Path]:
    path = _existing(run_dir / "config.yaml", source_dir / "config.yaml")
    if path is None:
        raise FileNotFoundError("No resolved run config exists in run or replay source")
    return load_yaml_with_extends(path), path


def _source_dir(result: Mapping[str, Any], run_dir: Path) -> Path:
    replay = result.get("replay", {}) if isinstance(result.get("replay"), Mapping) else {}
    candidate = replay.get("source_run_dir") or result.get("source_run_dir")
    return Path(str(candidate)) if candidate else run_dir


def _checkpoint_for_round(
    round_index: int,
    run_dir: Path,
    source_dir: Path,
    result: Mapping[str, Any],
) -> Optional[Path]:
    candidates = [
        run_dir / "training" / f"round_{round_index:03d}" / "adapter",
        source_dir / "training" / f"round_{round_index:03d}" / "adapter",
    ]
    replay = result.get("replay", {}) if isinstance(result.get("replay"), Mapping) else {}
    saved = replay.get("source_round_checkpoints", {})
    if not saved:
        saved = result.get("source_round_checkpoints", {})
    if isinstance(saved, Mapping):
        value = saved.get(str(round_index), saved.get(round_index))
        if value:
            candidates.append(Path(str(value)))
    return _existing(*candidates)


def _pilot_plan(result: Mapping[str, Any], run_dir: Path, source_dir: Path) -> Any:
    path = _existing(
        run_dir / "pilot_eval_plan.json",
        run_dir / "cost_aware" / "pilot_eval_plan.json",
        source_dir / "pilot_eval_plan.json",
        source_dir / "cost_aware" / "pilot_eval_plan.json",
    )
    if path is None:
        for key in ("pilot_eval_plan_path", "gain_eval_plan_path"):
            candidate = result.get(key)
            if candidate and Path(str(candidate)).exists():
                path = Path(str(candidate))
                break
    if path:
        return read_json(path)
    return result.get("pilot_eval_plan")


def _all_stopping_ids(*roots: Path) -> set[str]:
    values: set[str] = set()
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        patterns = (
            "round_*/stopping_predictions.jsonl",
            "round_*/validation_predictions.jsonl",
            "round_*/early_stopping_predictions.jsonl",
        )
        for pattern in patterns:
            for path in sorted((root / "training").glob(pattern)):
                values.update(_ids(read_jsonl(path)))
    return values


def _make_manifest(
    *,
    seed: int,
    run_dir: Path,
    source_dir: Path,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    minimum_examples: int,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    issues: list[str] = []
    gain_predictions = _existing(
        run_dir / "gain_report_predictions.jsonl",
        run_dir / "gain_report" / "predictions.jsonl",
        run_dir / "evaluation" / "gain_report_predictions.jsonl",
        source_dir / "gain_report_predictions.jsonl",
        source_dir / "gain_report" / "predictions.jsonl",
        source_dir / "evaluation" / "gain_report_predictions.jsonl",
    )
    cached_gain_report_ids = _ids(read_jsonl(gain_predictions)) if gain_predictions else set()

    closed_loop = config.get("closed_loop", {})
    shared_value = closed_loop.get("shared_predictions_path") if isinstance(closed_loop, Mapping) else None
    shared_path = Path(str(shared_value)) if shared_value else None
    if shared_path is None or not shared_path.exists():
        return None, ["missing_shared_prediction_source"]
    source_ids = _ids(read_jsonl(shared_path))

    pilot_ids, predeclared_heldout_ids = _pilot_partitions(
        _pilot_plan(result, run_dir, source_dir)
    )
    if not pilot_ids:
        issues.append("pilot_eval_ids_unavailable_fail_closed")
    if not predeclared_heldout_ids:
        issues.append("held_out_gain_report_ids_unavailable_fail_closed")
    candidate_ids = set(predeclared_heldout_ids)
    early_ids = _all_stopping_ids(run_dir, source_dir)
    sealed_cfg = config.get("stage2_benchmark", {})
    sealed_value = sealed_cfg.get("sealed_final_test_path") if isinstance(sealed_cfg, Mapping) else None
    sealed_path = Path(str(sealed_value)) if sealed_value else None
    sealed_ids = _ids(read_jsonl(sealed_path)) if sealed_path and sealed_path.exists() else set()
    if sealed_value and not sealed_ids:
        issues.append("sealed_final_ids_unavailable_fail_closed")

    allowed = sorted(candidate_ids - pilot_ids - early_ids - sealed_ids)
    missing_source = set(allowed) - source_ids
    if missing_source:
        issues.append(f"evaluation_ids_missing_from_source:{len(missing_source)}")
    allowed = sorted(set(allowed) & source_ids)
    if len(allowed) < minimum_examples:
        issues.append(f"insufficient_disjoint_evaluation_examples:{len(allowed)}<{minimum_examples}")
    if not pilot_ids or not predeclared_heldout_ids:
        allowed = []

    manifest: dict[str, Any] = {
        "schema": "v7_one_step_evaluation_manifest_v1",
        "seed": seed,
        "authorized_for_inference": not issues,
        "source_role": "predeclared_gain_report_only",
        "source_path": str(shared_path),
        "source_hash": hash_local_path(shared_path),
        "candidate_predictions_path": str(gain_predictions) if gain_predictions else None,
        "candidate_predictions_hash": hash_local_path(gain_predictions) if gain_predictions else None,
        "candidate_predictions_cached_ids": len(cached_gain_report_ids),
        "evaluation_ids": allowed,
        "evaluation_ids_hash": _json_hash(allowed),
        "num_examples": len(allowed),
        "excluded": {
            "pilot_eval": {"count": len(pilot_ids), "ids_hash": _json_hash(sorted(pilot_ids))},
            "early_stopping": {"count": len(early_ids), "ids_hash": _json_hash(sorted(early_ids))},
            "sealed_final": {"count": len(sealed_ids), "ids_hash": _json_hash(sorted(sealed_ids))},
        },
        "disjointness": {
            "pilot_eval": not bool(set(allowed) & pilot_ids),
            "early_stopping": not bool(set(allowed) & early_ids),
            "sealed_final": not bool(set(allowed) & sealed_ids),
        },
        "issues": issues,
        "policy": {
            "no_future_round_metrics_for_decision": True,
            "no_sealed_test": True,
            "not_used_for_controller_tuning": True,
        },
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    return manifest, issues


def _build_plan(
    seed: int,
    run_dir: Path,
    source_dir: Path,
    result: Mapping[str, Any],
    manifest: Optional[Mapping[str, Any]],
    rounds: Sequence[int],
    config_path: Optional[Path],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for round_index in sorted({round_number for t in rounds for round_number in (t - 1, t)}):
        checkpoint = _checkpoint_for_round(round_index, run_dir, source_dir, result)
        entries.append(
            {
                "round": round_index,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_hash": hash_local_path(checkpoint) if checkpoint else None,
                "status": "ready" if checkpoint else "missing_checkpoint",
            }
        )
    authorized = bool(manifest and manifest.get("authorized_for_inference")) and all(
        entry["status"] == "ready" for entry in entries
    )
    plan = {
        "schema": "v7_one_step_inference_plan_v1",
        "seed": seed,
        "authorized": authorized,
        "evaluation_manifest_hash": manifest.get("manifest_hash") if manifest else None,
        "resolved_run_config_path": str(config_path) if config_path else None,
        "checkpoints": entries,
        "notes": "Inference only; no training, teacher, verification, pilots, or controller updates.",
    }
    plan["plan_hash"] = _json_hash(plan)
    return plan


def _first(mapping: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return float(value)
    return None


def _range_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    rows = payload.get("rounds", payload.get("rows", payload.get("comparisons", [])))
    return {int(row.get("round", row.get("round_index"))): row for row in rows}


def _decision_rows(result: Mapping[str, Any], run_dir: Path) -> dict[int, Mapping[str, Any]]:
    rows = result.get("cost_aware_rounds", [])
    indexed = {int(row.get("round", row.get("round_index"))): row for row in rows}
    for path in sorted((run_dir / "cost_aware").glob("round_*/decision.json")):
        payload = read_json(path)
        round_index = payload.get("round", payload.get("round_index"))
        if round_index is None:
            try:
                round_index = int(path.parent.name.split("_")[-1])
            except ValueError:
                continue
        indexed.setdefault(int(round_index), payload)
    return indexed


def _state_counts(row: Mapping[str, Any]) -> tuple[Optional[int], Optional[int]]:
    state_map = row.get("hypothesis_states")
    if isinstance(state_map, Mapping):
        states = [str(value.get("state", value) if isinstance(value, Mapping) else value).lower() for value in state_map.values()]
        return states.count("certified"), states.count("unresolved")
    counts = row.get("hypothesis_state_counts")
    if isinstance(counts, Mapping):
        certified = counts.get("certified")
        unresolved = counts.get("unresolved")
        return (
            int(certified) if certified is not None else None,
            int(unresolved) if unresolved is not None else None,
        )
    certified = row.get("certified_hypotheses")
    unresolved = row.get("unresolved_hypotheses")
    certified_count = len(certified) if isinstance(certified, list) else (int(certified) if certified is not None else None)
    unresolved_count = len(unresolved) if isinstance(unresolved, list) else (int(unresolved) if unresolved is not None else None)
    return certified_count, unresolved_count


def _evidence_gate(reason: str) -> bool:
    text = reason.lower()
    return any(token in text for token in ("no_eligible", "no_certified", "insufficient_evidence", "evidence_gate"))


def _metrics_path(metrics_root: Path, seed: int, round_index: int) -> Path:
    return metrics_root / f"seed_{seed}" / f"round_{round_index:03d}" / "metrics.json"


def _binary_label(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"entailment", "entails", "e"}:
            return 0
        if normalized in {"neutral", "contradiction", "non-entailment", "nonentailment", "n", "c"}:
            return 1
    if value in (0, False):
        return 0
    if value in (1, 2, True):
        return 1
    raise ValueError(f"unsupported binary NLI label: {value!r}")


def _cached_prediction(row: Mapping[str, Any], rule: str) -> int:
    direct_keys = ("direct_binary_prediction", "binary_prediction")
    three_keys = ("three_way_binary_prediction", "prediction_binary")
    keys = direct_keys if rule == "direct" else three_keys
    for key in keys:
        if key in row:
            return _binary_label(row[key])
    probabilities = row.get("probabilities", row.get("probs"))
    if isinstance(probabilities, Mapping):
        entailment = float(probabilities["entailment"])
        neutral = float(probabilities["neutral"])
        contradiction = float(probabilities["contradiction"])
    else:
        entailment = row.get("p_entailment")
        neutral = row.get("p_neutral")
        contradiction = row.get("p_contradiction")
        if entailment is None or neutral is None or contradiction is None:
            if "prediction" in row:
                return _binary_label(row["prediction"])
            raise ValueError("cached row lacks auditable E/N/C probabilities or binary prediction")
        entailment, neutral, contradiction = float(entailment), float(neutral), float(contradiction)
    if rule == "direct":
        return int(neutral + contradiction > entailment)
    return int(max(range(3), key=lambda index: (entailment, neutral, contradiction)[index]) != 0)


def _cached_round_predictions(round_index: int, *roots: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "training" / f"round_{round_index:03d}" / "gain_report_predictions.jsonl",
                root / "training" / f"round_{round_index:03d}" / "evaluation" / "gain_report_predictions.jsonl",
                root / "gain_report" / f"round_{round_index:03d}" / "predictions.jsonl",
            ]
        )
        if round_index == 8:
            candidates.append(root / "gain_report_predictions.jsonl")
    return _existing(*candidates)


def _reuse_cached_checkpoint_metrics(
    *,
    seed: int,
    run_dir: Path,
    source_dir: Path,
    manifest: Mapping[str, Any],
    plan_entries: Mapping[int, Mapping[str, Any]],
    metrics_root: Path,
) -> None:
    """Derive diagnostic metrics from saved per-round predictions, if present."""
    wanted_ids = [str(value) for value in manifest.get("evaluation_ids", [])]
    for round_index, plan_entry in plan_entries.items():
        metrics_path = _metrics_path(metrics_root, seed, round_index)
        if metrics_path.exists() or plan_entry.get("status") != "ready":
            continue
        predictions_path = _cached_round_predictions(round_index, run_dir, source_dir)
        if predictions_path is None:
            continue
        indexed = {str(row.get("id", row.get("example_id"))): row for row in read_jsonl(predictions_path)}
        if any(example_id not in indexed for example_id in wanted_ids):
            continue
        rows = [indexed[example_id] for example_id in wanted_ids]
        try:
            gold = [_binary_label(row.get("gold_binary", row.get("gold", row.get("gold_label", row.get("label"))))) for row in rows]
            direct = [_cached_prediction(row, "direct") for row in rows]
            three = [_cached_prediction(row, "three") for row in rows]
        except (KeyError, TypeError, ValueError):
            continue
        metrics = {
            "schema": "v7_one_step_checkpoint_metric_v1",
            "seed": seed,
            "round": round_index,
            "num_examples": len(rows),
            "primary_rule": "direct_binary",
            "overall_accuracy": sum(int(left == right) for left, right in zip(direct, gold)) / len(rows),
            "direct_binary_accuracy": sum(int(left == right) for left, right in zip(direct, gold)) / len(rows),
            "three_way_binary_accuracy": sum(int(left == right) for left, right in zip(three, gold)) / len(rows),
            "prediction_disagreement_count": sum(int(left != right) for left, right in zip(direct, three)),
            "evaluation_manifest_hash": manifest["manifest_hash"],
            "evaluation_ids_hash": manifest["evaluation_ids_hash"],
            "checkpoint_path": plan_entry["checkpoint_path"],
            "checkpoint_hash": plan_entry["checkpoint_hash"],
            "derived_from_cached_predictions": str(predictions_path),
            "derived_from_cached_predictions_hash": hash_local_path(predictions_path),
            "observed_gain_role": "single_training_realization_not_expected_gain",
        }
        write_json(metrics_path, metrics)


def _observed_update(
    metrics_root: Path,
    seed: int,
    round_index: int,
    manifest_hash: Optional[str],
    expected_before_hash: Optional[str],
    expected_after_hash: Optional[str],
) -> ObservedUpdate:
    before_path = _metrics_path(metrics_root, seed, round_index - 1)
    after_path = _metrics_path(metrics_root, seed, round_index)
    if not before_path.exists() or not after_path.exists():
        return ObservedUpdate(
            accuracy_before=None,
            accuracy_after=None,
            evaluation_manifest_hash=manifest_hash,
            checkpoint_before_hash=None,
            checkpoint_after_hash=None,
            num_examples=None,
        )
    before, after = read_json(before_path), read_json(after_path)
    if before.get("evaluation_manifest_hash") != after.get("evaluation_manifest_hash"):
        return ObservedUpdate(None, None, manifest_hash, None, None, None)
    if manifest_hash and before.get("evaluation_manifest_hash") != manifest_hash:
        return ObservedUpdate(None, None, manifest_hash, None, None, None)
    if expected_before_hash and before.get("checkpoint_hash") != expected_before_hash:
        return ObservedUpdate(None, None, manifest_hash, None, None, None)
    if expected_after_hash and after.get("checkpoint_hash") != expected_after_hash:
        return ObservedUpdate(None, None, manifest_hash, None, None, None)
    before_count = int(before.get("num_examples", 0))
    after_count = int(after.get("num_examples", 0))
    if before_count <= 0 or before_count != after_count:
        return ObservedUpdate(None, None, manifest_hash, None, None, None)
    return ObservedUpdate(
        accuracy_before=float(before["overall_accuracy"]),
        accuracy_after=float(after["overall_accuracy"]),
        evaluation_manifest_hash=str(before.get("evaluation_manifest_hash")),
        checkpoint_before_hash=before.get("checkpoint_hash"),
        checkpoint_after_hash=after.get("checkpoint_hash"),
        num_examples=before_count,
    )


def _protocol_input(
    *,
    seed: int,
    round_index: int,
    policy: str,
    decision_row: Mapping[str, Any],
    range_row: Optional[Mapping[str, Any]],
) -> Optional[DecisionInputs]:
    decision = decision_row.get("decision", decision_row)
    gain = decision_row.get("gain_estimate", {})
    cost = decision_row.get("decision_cost", {})
    selected = decision
    if policy == "v2_1_range_aware":
        if range_row is None:
            return None
        selected = range_row.get("range_aware", range_row)
    delta_hat = _first(selected, "delta_hat", "estimated_gain")
    if delta_hat is None:
        delta_hat = _first(decision, "delta_hat", "estimated_gain")
    if delta_hat is None and isinstance(gain, Mapping):
        delta_hat = _first(gain, "delta_hat", "mean_gain")
    radius = _first(selected, "confidence_radius", "radius")
    action = str(selected.get("action", decision.get("action", "unavailable"))).lower()
    reason = str(selected.get("reason", decision.get("reason", "unavailable")))
    projected = _first(cost, "decision_cost", "projected_decision_cost") if isinstance(cost, Mapping) else None
    if projected is None:
        projected = _first(decision, "cost", "decision_cost")
    lambda_cost = _first(decision, "lambda_cost")
    actual_main_cost = _first(cost, "actual_main_update_cost") if isinstance(cost, Mapping) else None
    certified, unresolved = _state_counts(decision_row)
    return DecisionInputs(
        seed=seed,
        round_index=round_index,
        policy=policy,
        delta_hat=delta_hat,
        radius=radius,
        lambda_cost=float(lambda_cost if lambda_cost is not None else 0.2),
        decision_cost=projected,
        actual_main_update_cost=actual_main_cost,
        support_upper=1.0,
        action=action,
        reason=reason,
        certified_hypotheses=certified,
        unresolved_hypotheses=unresolved,
        evidence_gate=None if not _evidence_gate(reason) else reason,
    )


def _numeric_from(value: Any, keys: Sequence[str]) -> Optional[float]:
    if not isinstance(value, Mapping):
        return None
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    for child in value.values():
        found = _numeric_from(child, keys)
        if found is not None:
            return found
    return None


def _cost_details(decision_row: Mapping[str, Any]) -> dict[str, Optional[float]]:
    cost = decision_row.get("decision_cost", {})
    pilot = cost.get("pilot_overhead", cost.get("pilots", {})) if isinstance(cost, Mapping) else {}
    actual = cost.get("actual_main_update", {}) if isinstance(cost, Mapping) else {}
    total = cost.get("total_pipeline", {}) if isinstance(cost, Mapping) else {}
    return {
        "pilot_overhead_cost": _numeric_from(pilot, ("modeled_cost", "cost", "usd_cost")),
        "pilot_overhead_gpu_seconds": _numeric_from(pilot, ("gpu_seconds", "gpu_time_seconds")),
        "pilot_overhead_wall_seconds": _numeric_from(pilot, ("wall_seconds", "wall_time_seconds")),
        "actual_main_update_gpu_seconds": _numeric_from(actual, ("gpu_seconds", "gpu_time_seconds")),
        "actual_main_update_wall_seconds": _numeric_from(actual, ("wall_seconds", "wall_time_seconds")),
        "total_pipeline_gpu_seconds": _numeric_from(total, ("gpu_seconds", "gpu_time_seconds")),
        "total_pipeline_wall_seconds": _numeric_from(total, ("wall_seconds", "wall_time_seconds")),
    }


def _flatten(seed: int, assessment: Mapping[str, Any], costs: Mapping[str, Any]) -> dict[str, Any]:
    row = {"seed": seed, **assessment}
    row.update(costs)
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _render_report(rows: Sequence[Mapping[str, Any]], validations: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# V7 One-Step Cost-Aware Decision Benchmark",
        "",
        "> Retrospective diagnostic only. The controller decisions are unchanged; no future-round or sealed-test metric is used as online evidence.",
        "",
        "## Validation and data availability",
        "",
        "| Seed | Manifest | Inference authorized | Evaluation examples | Issues |",
        "|---:|---|---|---:|---|",
    ]
    for item in validations:
        manifest = item.get("manifest") or {}
        lines.append(
            f"| {item['seed']} | {'ready' if manifest else 'missing'} | "
            f"{manifest.get('authorized_for_inference', False)} | {manifest.get('num_examples', 0)} | "
            f"{'; '.join(item.get('issues', [])) or 'none'} |"
        )
    lines += [
        "",
        "## Per-round one-step comparison",
        "",
        "Observed gain is one realized main update on one fixed held-out shard; it is not the expected gain parameter and does not prove a confidence theorem.",
        "",
        "| Seed | Round | Policy | Decision | Est. gain | Est. net | Obs. gain | Obs. net | Gain-UCB | Net-UCB | Gap | Covered | One-step alignment | Evidence |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round_index']} | {row['policy']} | {row['action']} | "
            f"{_number(row.get('delta_hat'))} | {_number(row.get('estimated_net_gain'))} | "
            f"{_number(row.get('observed_gain'))} | {_number(row.get('observed_net_gain'))} | "
            f"{_number(row.get('gain_ucb'))} | {_number(row.get('net_ucb'))} | "
            f"{_number(row.get('confidence_gap'))} | {row.get('observed_below_gain_ucb')} | "
            f"{('aligned' if row.get('one_step_action_alignment') else 'misaligned') if row.get('one_step_action_alignment') is not None else 'N/A'} | "
            f"{row.get('evidence_gate') or 'pass'} |"
        )
    lines += [
        "",
        "## Cost-accounting scope",
        "",
        "Projected decision cost, actual main-update cost, pilot/exploration overhead, and total pipeline cost are reported as separate scopes. They are not summed in this benchmark, because the cumulative scopes overlap.",
        "",
        "| Seed | Round | Projected decision | Actual main update | Pilot overhead | Incurred exploration | Total pipeline |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cost_rows: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for row in rows:
        cost_rows.setdefault((row.get("seed"), row.get("round_index")), row)
    for (seed, round_index), row in sorted(cost_rows.items()):
        lines.append(
            f"| {seed} | {round_index} | {_number(row.get('projected_decision_cost'))} | "
            f"{_number(row.get('actual_main_update_cost'))} | "
            f"{_number(row.get('pilot_overhead_cost'))} | "
            f"{_number(row.get('incurred_exploration_cost'))} | "
            f"{_number(row.get('total_pipeline_cost'))} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Pilot estimates use a fixed shard and baseline; observations may not satisfy conditional independence or unbiased population sampling.",
        "- The V7 intervention may mix hypotheses, so a per-hypothesis theorem is not claimed for the mixed-policy update.",
        "- Confidence coverage of one realized update is descriptive, not validation of a population-level guarantee.",
        "- Missing concrete pilot-evaluation IDs fails closed: supplemental inference is not authorized until disjointness can be verified.",
        "- Maximum-budget termination is not counted as a Cost-Aware STOP or claimed budget saving.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v7_one_step_cost_aware_benchmark.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = read_json(args.config)
    source_specs = {int(item["seed"]): item for item in spec.get("sources", [])}
    seeds = args.seeds or sorted(source_specs)
    rounds = [int(value) for value in spec["protocol"]["rounds"]]
    metrics_root = Path(spec["evaluation"]["checkpoint_metrics_root"])
    output_root = metrics_root.parent
    report_path = Path("reports/v7_minimal_cost_aware/ONE_STEP_COST_AWARE_BENCHMARK.md")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation_manifests").mkdir(parents=True, exist_ok=True)
    validations: list[dict[str, Any]] = []
    seed_context: dict[int, tuple[Path, Path, Mapping[str, Any]]] = {}

    for seed in seeds:
        source_spec = source_specs.get(seed)
        if source_spec is None:
            validations.append({"seed": seed, "manifest": None, "issues": ["seed_not_declared_in_config"]})
            continue
        run_dir = Path(source_spec["run_dir"])
        result_path = run_dir / "result.json"
        if not result_path.exists():
            validations.append({"seed": seed, "manifest": None, "issues": ["missing_seed_result"]})
            continue
        result = read_json(result_path)
        configured_source = source_spec.get("source_run_dir")
        source_dir = Path(configured_source) if configured_source else _source_dir(result, run_dir)
        try:
            run_config, config_path = _load_run_config(run_dir, source_dir)
            manifest, issues = _make_manifest(
                seed=seed,
                run_dir=run_dir,
                source_dir=source_dir,
                result=result,
                config=run_config,
                minimum_examples=int(spec["evaluation"]["minimum_examples"]),
            )
        except (FileNotFoundError, KeyError, ValueError) as error:
            manifest, issues, config_path = None, [f"validation_error:{error}"], None
        if manifest is not None:
            manifest_path = output_root / "evaluation_manifests" / f"seed_{seed}.json"
            write_json(manifest_path, manifest)
        plan = _build_plan(seed, run_dir, source_dir, result, manifest, rounds, config_path)
        plan_path = output_root / f"inference_plan_seed_{seed}.json"
        write_json(plan_path, plan)
        validations.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "source_dir": str(source_dir),
                "config_path": str(config_path) if config_path else None,
                "manifest": manifest,
                "inference_plan": plan,
                "issues": issues,
            }
        )
        seed_context[seed] = (run_dir, source_dir, result)

    write_json(output_root / "validation.json", {"schema": SCHEMA, "seeds": validations})
    if args.validate_only:
        print(f"Wrote {output_root / 'validation.json'}")
        return

    rows: list[dict[str, Any]] = []
    validation_by_seed = {int(row["seed"]): row for row in validations}
    for seed, (run_dir, source_dir, result) in seed_context.items():
        decision_rows = _decision_rows(result, run_dir)
        range_value = source_specs.get(seed, {}).get("range_replay")
        range_path = Path(range_value) if range_value else Path("__missing_range_replay__")
        range_rows = _range_rows(range_path)
        manifest = validation_by_seed[seed].get("manifest") or {}
        manifest_hash = manifest.get("manifest_hash")
        plan_entries = {
            int(entry["round"]): entry
            for entry in validation_by_seed[seed].get("inference_plan", {}).get("checkpoints", [])
        }
        if manifest.get("authorized_for_inference"):
            _reuse_cached_checkpoint_metrics(
                seed=seed,
                run_dir=run_dir,
                source_dir=source_dir,
                manifest=manifest,
                plan_entries=plan_entries,
                metrics_root=metrics_root,
            )
        for round_index in rounds:
            decision_row = decision_rows.get(round_index)
            if decision_row is None:
                for policy in ("v2_hoeffding", "v2_1_range_aware"):
                    rows.append({"seed": seed, "round_index": round_index, "policy": policy, "status": "missing_decision_record"})
                continue
            decision_cost_payload = decision_row.get("decision_cost", {})
            costs = cost_accounting_row(decision_cost_payload)
            costs.update(_cost_details(decision_row))
            observed = _observed_update(
                metrics_root,
                seed,
                round_index,
                manifest_hash,
                plan_entries.get(round_index - 1, {}).get("checkpoint_hash"),
                plan_entries.get(round_index, {}).get("checkpoint_hash"),
            )
            for policy in ("v2_hoeffding", "v2_1_range_aware"):
                inputs = _protocol_input(
                    seed=seed,
                    round_index=round_index,
                    policy=policy,
                    decision_row=decision_row,
                    range_row=range_rows.get(round_index),
                )
                if inputs is None:
                    rows.append({"seed": seed, "round_index": round_index, "policy": policy, "status": "missing_policy_replay"})
                    continue
                assessment = assess_one_step(inputs, observed)
                rows.append(_flatten(seed, asdict(assessment), costs))

    payload = {
        "schema": SCHEMA,
        "definition": {
            "observed_gain": "accuracy_after_one_main_update - accuracy_before_on_same_heldout_shard",
            "observed_gain_estimand": "single_training_realization_not_expected_gain",
            "future_round_metrics_used": False,
            "sealed_test_used": False,
        },
        "validations": validations,
        "rows": rows,
    }
    write_json(output_root / "benchmark.json", payload)
    _write_csv(output_root / "benchmark.csv", rows)
    for seed in seeds:
        seed_rows = [row for row in rows if row.get("seed") == seed]
        write_json(output_root / f"seed_{seed}" / "benchmark.json", {"schema": SCHEMA, "seed": seed, "rows": seed_rows})
        _write_csv(output_root / f"seed_{seed}" / "benchmark.csv", seed_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(rows, validations), encoding="utf-8")
    print(f"Wrote {output_root / 'benchmark.json'}")
    print(f"Wrote {output_root / 'benchmark.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
