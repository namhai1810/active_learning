#!/usr/bin/env python3
"""CPU-only retrospective Correct Stopping benchmark for V7 v2/v2.1.

The script reads immutable historical artifacts and writes to a separate
diagnostic output.  It never imports torch, loads a model, or invokes teacher,
training, evaluation, or inference code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decision.correct_stopping_benchmark import (  # noqa: E402
    BenchmarkResult,
    CostSnapshot,
    DecisionEvidence,
    RoundRecord,
    Trajectory,
    benchmark_trajectory,
)


ROUND_KEYS = ("round", "round_index", "round_number")
OFFLINE_ACCURACY_KEYS = (
    "full_hans_accuracy",
    "hans_accuracy",
    "hans_overall_accuracy",
    "hans_three_way_accuracy",
    "three_way_hans_accuracy",
)
VALIDATION_ACCURACY_KEYS = (
    "round_validation_accuracy",
    "stopping_accuracy",
    "restored_validation_accuracy",
)
CUMULATIVE_COST_KEYS = (
    "cumulative_total_pipeline_cost",
    "total_pipeline_cost_cumulative",
    "cumulative_pipeline_modeled_usd",
    "cumulative_modeled_usd",
)
PILOT_COST_KEYS = (
    "cumulative_pilot_modeled_usd",
    "pilot_modeled_usd_cumulative",
)
MEASURED_KEYS = (
    "teacher_tokens_cumulative",
    "teacher_input_tokens_cumulative",
    "teacher_output_tokens_cumulative",
    "teacher_seconds_cumulative",
    "verification_seconds_cumulative",
    "generation_seconds_cumulative",
    "pilot_gpu_seconds_cumulative",
    "pilot_wall_seconds_cumulative",
    "training_gpu_seconds_cumulative",
    "evaluation_gpu_seconds_cumulative",
    "gpu_seconds_cumulative",
    "wall_seconds_cumulative",
)
PILOT_MEASURED_KEYS = (
    "pilot_gpu_seconds_cumulative",
    "pilot_wall_seconds_cumulative",
    "pilot_teacher_tokens_cumulative",
)
JSON_BASENAMES = {
    "result.json",
    "summary.json",
    "metrics.json",
    "manifest.json",
    "decision.json",
    "cost_aware_decision.json",
    "resource_accounting.json",
    "training_summary.json",
}
POLICY_NAMES = (
    "v2_hoeffding",
    "v2_1_range_aware",
    "v7_no_stopping",
    "validation_patience",
    "retrospective_oracle",
)
ROUND_PATH_PATTERN = re.compile(r"(?:^|[/_-])round[_-]?(\d{1,3})(?:$|[/_.-])")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _accuracy(value: Any) -> Optional[float]:
    number = _finite_number(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return number if 0.0 <= number <= 1.0 else None


def _round_from(node: Mapping[str, Any], inherited: Optional[int]) -> Optional[int]:
    values = []
    for key in ROUND_KEYS:
        value = node.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    if values and len(set(values)) != 1:
        return None
    return values[0] if values else inherited


def _walk(
    value: Any,
    *,
    path: Tuple[str, ...] = (),
    inherited_round: Optional[int] = None,
) -> Iterator[Tuple[Tuple[str, ...], Mapping[str, Any], Optional[int]]]:
    if isinstance(value, dict):
        current_round = _round_from(value, inherited_round)
        yield path, value, current_round
        for key, child in value.items():
            yield from _walk(
                child,
                path=path + (str(key),),
                inherited_round=current_round,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(
                child,
                path=path + (str(index),),
                inherited_round=inherited_round,
            )


class Evidence:
    def __init__(self, seed: int, t_max: int, budget_max: int) -> None:
        self.seed = seed
        self.t_max = t_max
        self.budget_max = budget_max
        self.rounds: Dict[int, Dict[str, Any]] = {
            index: {
                "budget": index * (budget_max // t_max),
                "offline_accuracy": None,
                "validation_accuracy": None,
                "total_modeled_usd": None,
                "pilot_modeled_usd": None,
                "measured": {},
                "pilot_measured": {},
                "decisions": {},
                "provenance": {},
            }
            for index in range(1, t_max + 1)
        }
        self.issues: List[str] = []
        self.sources: Dict[str, str] = {}
        self.ambiguous: set[Tuple[int, str]] = set()

    def set_scalar(self, round_index: int, key: str, value: Any, source: str) -> None:
        if round_index not in self.rounds:
            self.issues.append(f"round_out_of_range:{round_index}:{source}")
            return
        token = (round_index, key)
        if token in self.ambiguous:
            return
        current = self.rounds[round_index][key]
        if current is None:
            self.rounds[round_index][key] = value
            self.rounds[round_index]["provenance"][key] = source
        elif current != value:
            self.rounds[round_index][key] = None
            self.ambiguous.add(token)
            self.issues.append(f"ambiguous:{round_index}:{key}:{source}")

    def set_measured(
        self, round_index: int, namespace: str, key: str, value: float, source: str
    ) -> None:
        if round_index not in self.rounds:
            return
        token = (round_index, f"{namespace}.{key}")
        if token in self.ambiguous:
            return
        target = self.rounds[round_index][namespace]
        if key in target and target[key] != value:
            target.pop(key, None)
            self.ambiguous.add(token)
            self.issues.append(f"ambiguous:{round_index}:{namespace}.{key}:{source}")
            return
        target[key] = value
        self.rounds[round_index]["provenance"][f"{namespace}.{key}"] = source

    def set_decision(
        self,
        round_index: int,
        policy: str,
        decision: Mapping[str, Any],
        source: str,
    ) -> None:
        if round_index not in self.rounds:
            return
        token = (round_index, f"decision.{policy}")
        if token in self.ambiguous:
            return
        current = self.rounds[round_index]["decisions"].get(policy)
        normalized = dict(decision)
        if current is not None and current != normalized:
            self.rounds[round_index]["decisions"].pop(policy, None)
            self.ambiguous.add(token)
            self.issues.append(f"ambiguous:{round_index}:decision.{policy}:{source}")
            return
        self.rounds[round_index]["decisions"][policy] = normalized
        self.rounds[round_index]["provenance"][f"decision.{policy}"] = source


def _path_text(path: Sequence[str]) -> str:
    return "/".join(path).lower()


def _round_from_source(source: str) -> Optional[int]:
    matches = ROUND_PATH_PATTERN.findall(source.lower())
    return int(matches[-1]) if matches else None


def _certified_stop(action: str, reason: str, explicit: Any = None) -> bool:
    if isinstance(explicit, bool):
        return explicit and action == "stop"
    lowered = reason.lower()
    excluded = ("budget", "maximum", "missing", "evidence", "candidate", "operational")
    gain_cost = ("gain_cost", "gain-cost", "net_gain", "net-gain")
    return action == "stop" and any(token in lowered for token in gain_cost) and not any(
        token in lowered for token in excluded
    )


def _decision_from_node(node: Mapping[str, Any], prefix: str = "") -> Optional[Dict[str, Any]]:
    action = node.get(prefix + "action")
    reason = node.get(prefix + "reason")
    if action not in {"continue", "stop"} or not isinstance(reason, str):
        return None
    radius = None
    for key in (
        prefix + "confidence_radius",
        prefix + "radius",
        prefix + "hoeffding_radius",
    ):
        if key in node:
            radius = _finite_number(node[key])
            break
    explicit = node.get(prefix + "certified_gain_cost_stop")
    return {
        "action": action,
        "reason": reason,
        "confidence_radius": radius,
        "certified_gain_cost_stop": _certified_stop(action, reason, explicit),
    }


def _ingest_payload(evidence: Evidence, payload: Any, source: str) -> None:
    # The closed-loop result stores the predeclared 900-example stopping
    # subset under ``learning_curve[].metrics.overall_accuracy``.  Keep this
    # explicitly in the validation namespace: it must never be promoted to
    # the retrospective full-HANS quality trajectory.
    if isinstance(payload, dict) and isinstance(payload.get("learning_curve"), list):
        for item in payload["learning_curve"]:
            if not isinstance(item, dict):
                continue
            round_index = item.get("round")
            metrics = item.get("metrics")
            if (
                isinstance(round_index, int)
                and 1 <= round_index <= evidence.t_max
                and isinstance(metrics, dict)
            ):
                value = _accuracy(metrics.get("overall_accuracy"))
                if value is not None:
                    evidence.set_scalar(
                        round_index,
                        "validation_accuracy",
                        value,
                        f"{source}#learning_curve/{round_index}/metrics/overall_accuracy",
                    )

        # Only the terminal checkpoint is evaluated on the sealed full HANS
        # set in the historical run.  Recording A_T is useful evidence, but
        # absent A_1..A_(T-1) remain absent and therefore block the oracle.
        prediction_rules = payload.get("hans_prediction_rules")
        if isinstance(prediction_rules, dict):
            three_way = prediction_rules.get("three_way_argmax_to_binary")
            if isinstance(three_way, dict):
                value = _accuracy(three_way.get("overall_accuracy"))
                if value is not None:
                    evidence.set_scalar(
                        evidence.t_max,
                        "offline_accuracy",
                        value,
                        f"{source}#hans_prediction_rules/three_way_argmax_to_binary/overall_accuracy",
                    )

    for path, node, round_index in _walk(
        payload, inherited_round=_round_from_source(source)
    ):
        if round_index is None:
            continue
        path_text = _path_text(path)
        for key in OFFLINE_ACCURACY_KEYS:
            if key in node and not any(
                token in path_text for token in ("validation", "stopping", "gain_eval")
            ):
                value = _accuracy(node[key])
                if value is not None:
                    evidence.set_scalar(round_index, "offline_accuracy", value, f"{source}#{'/'.join(path)}/{key}")
        for key in VALIDATION_ACCURACY_KEYS:
            if key in node:
                value = _accuracy(node[key])
                if value is not None:
                    evidence.set_scalar(round_index, "validation_accuracy", value, f"{source}#{'/'.join(path)}/{key}")
        for key in CUMULATIVE_COST_KEYS:
            if key in node:
                value = _finite_number(node[key])
                if value is not None and value >= 0:
                    evidence.set_scalar(round_index, "total_modeled_usd", value, f"{source}#{'/'.join(path)}/{key}")
        for key in PILOT_COST_KEYS:
            if key in node:
                value = _finite_number(node[key])
                if value is not None and value >= 0:
                    evidence.set_scalar(round_index, "pilot_modeled_usd", value, f"{source}#{'/'.join(path)}/{key}")
        for key in MEASURED_KEYS:
            if key in node:
                value = _finite_number(node[key])
                if value is not None and value >= 0:
                    evidence.set_measured(round_index, "measured", key, value, f"{source}#{'/'.join(path)}/{key}")
        for key in PILOT_MEASURED_KEYS:
            if key in node:
                value = _finite_number(node[key])
                if value is not None and value >= 0:
                    evidence.set_measured(round_index, "pilot_measured", key, value, f"{source}#{'/'.join(path)}/{key}")

        if "decision" in path_text:
            policy = "v2_1_range_aware" if "range_aware" in path_text else "v2_hoeffding"
            decision = _decision_from_node(node)
            if decision is not None:
                evidence.set_decision(round_index, policy, decision, f"{source}#{'/'.join(path)}")


def _ingest_range_replay(evidence: Evidence, payload: Any, source: str) -> None:
    for path, node, round_index in _walk(
        payload, inherited_round=_round_from_source(source)
    ):
        if round_index is None:
            continue
        old = None
        new = None
        # Current replay schema uses nested ``old`` and ``range_aware``
        # objects.  Retain support for the earlier flat diagnostic schema
        # below so historical artifacts remain readable.
        nested_old = node.get("old")
        if isinstance(nested_old, dict):
            old = _decision_from_node(nested_old)
        nested_new = node.get("range_aware")
        if isinstance(nested_new, dict):
            new = _decision_from_node(nested_new)
        for prefix in ("old_", "legacy_", "v2_"):
            if old is None:
                old = _decision_from_node(node, prefix)
            if old is not None:
                break
        for prefix in ("new_", "range_aware_", "v2_1_"):
            if new is None:
                new = _decision_from_node(node, prefix)
            if new is not None:
                break
        if old is not None:
            evidence.set_decision(round_index, "v2_hoeffding", old, f"{source}#{'/'.join(path)}")
        if new is not None:
            evidence.set_decision(round_index, "v2_1_range_aware", new, f"{source}#{'/'.join(path)}")


def _relevant_json_files(run_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in run_dir.rglob("*.json"):
        if path.name in JSON_BASENAMES or "decision" in path.name or "resource" in path.name:
            if path.is_file() and path.stat().st_size <= 50 * 1024 * 1024:
                files.append(path)
    return sorted(set(files))


def _resolve_run_dir(source: Mapping[str, Any]) -> Optional[Path]:
    explicit = source.get("run_dir")
    if explicit:
        path = REPO_ROOT / str(explicit)
        return path if path.is_dir() else None
    candidates = [REPO_ROOT / str(item) for item in source.get("run_candidates", [])]
    existing = [item for item in candidates if item.is_dir()]
    if len(existing) > 1:
        raise RuntimeError(
            "multiple run candidates exist; set run_dir explicitly: "
            + ", ".join(str(item) for item in existing)
        )
    return existing[0] if existing else None


def _load_normalized(path: Path) -> Trajectory:
    payload = _json(path)
    rounds = []
    for item in payload["rounds"]:
        decisions = {
            key: DecisionEvidence(**value)
            for key, value in item.get("decisions", {}).items()
        }
        cost_payload = item.get("cost", {})
        rounds.append(
            RoundRecord(
                round_index=int(item["round_index"]),
                budget=int(item["budget"]),
                offline_accuracy=item.get("offline_accuracy"),
                validation_accuracy=item.get("validation_accuracy"),
                cost=CostSnapshot(
                    total_modeled_usd=cost_payload.get("total_modeled_usd"),
                    measured=cost_payload.get("measured", {}),
                    pilot_modeled_usd=cost_payload.get("pilot_modeled_usd"),
                    pilot_measured=cost_payload.get("pilot_measured", {}),
                ),
                decisions=decisions,
            )
        )
    return Trajectory(
        seed=int(payload["seed"]),
        rounds=tuple(rounds),
        t_max=int(payload.get("t_max", 8)),
        budget_max=int(payload.get("budget_max", 256)),
    )


def _trajectory_from_evidence(evidence: Evidence) -> Trajectory:
    rounds = []
    for index in range(1, evidence.t_max + 1):
        item = evidence.rounds[index]
        decisions = {
            key: DecisionEvidence(**value) for key, value in item["decisions"].items()
        }
        rounds.append(
            RoundRecord(
                round_index=index,
                budget=int(item["budget"]),
                offline_accuracy=item["offline_accuracy"],
                validation_accuracy=item["validation_accuracy"],
                cost=CostSnapshot(
                    total_modeled_usd=item["total_modeled_usd"],
                    measured=item["measured"],
                    pilot_modeled_usd=item["pilot_modeled_usd"],
                    pilot_measured=item["pilot_measured"],
                ),
                decisions=decisions,
            )
        )
    return Trajectory(
        seed=evidence.seed,
        rounds=tuple(rounds),
        t_max=evidence.t_max,
        budget_max=evidence.budget_max,
    )


def _normalized_payload(trajectory: Trajectory, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "v7_correct_stopping_trajectory_v1",
        "seed": trajectory.seed,
        "t_max": trajectory.t_max,
        "budget_max": trajectory.budget_max,
        "rounds": [
            {
                "round_index": item.round_index,
                "budget": item.budget,
                "offline_accuracy": item.offline_accuracy,
                "validation_accuracy": item.validation_accuracy,
                "cost": asdict(item.cost),
                "decisions": {key: asdict(value) for key, value in item.decisions.items()},
            }
            for item in trajectory.rounds
        ],
        "provenance": dict(provenance),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _missing_result(seed: int, policy: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        seed=seed,
        policy=policy,
        status="incomplete_evidence",
        classification=None,
        decision_round=None,
        deployed_round=None,
        budget=None,
        decision_reason=reason,
        confidence_radius=None,
        certified_online_stop=False,
        a_star=None,
        a_stop=None,
        quality_regret=None,
        epsilon_optimal=None,
        oracle_round=None,
        full_modeled_usd=None,
        stopped_modeled_usd=None,
        cost_saving_ratio=None,
        avoidable_modeled_usd=None,
        pilot_overhead_modeled_usd=None,
        measured_full={},
        measured_stopped={},
        measured_saving_ratio={},
        pilot_measured={},
        missing_evidence=(reason,),
    )


def _historical_result_summary(payload: Any) -> Dict[str, Any]:
    """Extract descriptive evidence without filling benchmark-only gaps."""

    if not isinstance(payload, dict):
        return {}
    summary: Dict[str, Any] = {}
    curve = []
    for item in payload.get("learning_curve", []):
        if not isinstance(item, dict) or not isinstance(item.get("metrics"), dict):
            continue
        round_index = item.get("round")
        value = _accuracy(item["metrics"].get("overall_accuracy"))
        if isinstance(round_index, int) and round_index >= 1 and value is not None:
            curve.append({"round": round_index, "accuracy": value})
    summary["validation_curve"] = sorted(curve, key=lambda item: item["round"])

    rules = payload.get("hans_prediction_rules", {})
    if isinstance(rules, dict):
        for source_key, target_key in (
            ("three_way_argmax_to_binary", "final_hans_three_way"),
            ("direct_binary_probability_aggregation", "final_hans_direct_binary"),
        ):
            node = rules.get(source_key)
            if isinstance(node, dict):
                value = _accuracy(node.get("overall_accuracy"))
                if value is not None:
                    summary[target_key] = value

    partial_cost = _finite_number(payload.get("total_pipeline_cost_rounds_2_plus"))
    if partial_cost is not None and partial_cost >= 0:
        summary["rounds_2_plus_pipeline_usd"] = partial_cost
    per_round_cost: Dict[str, float] = {}
    for item in payload.get("cost_aware_rounds", []):
        if not isinstance(item, dict) or not isinstance(item.get("decision_cost"), dict):
            continue
        round_index = item.get("round")
        value = _finite_number(item["decision_cost"].get("total_pipeline_cost"))
        if isinstance(round_index, int) and value is not None and value >= 0:
            per_round_cost[str(round_index)] = value
    summary["per_round_pipeline_usd"] = per_round_cost

    current = payload.get("resource_usage", {})
    reused = payload.get("reused_source_resource_usage_full_8_round_historical") or {}
    teacher_tokens = 0.0
    teacher_tokens_known = False
    for resource in (current, reused):
        if not isinstance(resource, dict) or not isinstance(resource.get("teacher"), dict):
            continue
        teacher = resource["teacher"]
        for key in ("input_tokens", "output_tokens"):
            value = _finite_number(teacher.get(key))
            if value is not None:
                teacher_tokens += value
                teacher_tokens_known = True
    if teacher_tokens_known:
        summary["teacher_tokens_observed"] = teacher_tokens
    if isinstance(current, dict):
        pilot_wall = _finite_number(current.get("cost_aware_pilot_wall_seconds"))
        if pilot_wall is not None and pilot_wall >= 0:
            summary["pilot_wall_seconds_observed"] = pilot_wall
        summary["resource_scope"] = current.get("scope")
    return summary


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    results: Sequence[BenchmarkResult],
    source_audits: Sequence[Mapping[str, Any]],
) -> None:
    benchmark = config["benchmark"]
    lines = [
        "# V7 Cost-Aware Protocol v2.1 — Correct Stopping Retrospective Benchmark",
        "",
        "> CPU-only artifact benchmark. No model, teacher, training, evaluation, or inference is run.",
        "> Offline full-trajectory quality is used only by the retrospective evaluator/oracle, never by an online policy.",
        "",
        "## Frozen definition",
        "",
        f"- `T={benchmark['t_max']}`, `budget_max={benchmark['budget_max']}`, `epsilon={benchmark['epsilon']}`.",
        "- `A_star=max_t A_t`; `quality_regret=A_star-A_stop`; epsilon-optimal iff regret <= epsilon.",
        "- Primary scalar cost saving uses cumulative **total modeled USD** only.",
        "- Teacher tokens, phase seconds, GPU-time and wall time remain separate measured counters.",
        "- Budget exhaustion is not a certified gain-cost STOP.",
        "",
        "## Observed historical evidence",
        "",
        "| Seed | Validation accuracy by round (1→8) | Final HANS Three-way | Final HANS Direct Binary | Modeled USD, rounds 2–8 only | Teacher tokens observed | Pilot wall time |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for audit in source_audits:
        summary = audit.get("artifact_summary", {})
        if not summary:
            lines.append(f"| {audit['seed']} | — | — | — | — | — | — |")
            continue
        curve = ", ".join(
            f"R{item['round']}={100.0 * item['accuracy']:.2f}%"
            for item in summary.get("validation_curve", [])
        ) or "—"
        pilot_seconds = summary.get("pilot_wall_seconds_observed")
        pilot_text = (
            f"{pilot_seconds / 3600.0:.2f} h" if pilot_seconds is not None else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(audit["seed"]),
                    curve,
                    (
                        f"{100.0 * summary['final_hans_three_way']:.2f}%"
                        if "final_hans_three_way" in summary
                        else "—"
                    ),
                    (
                        f"{100.0 * summary['final_hans_direct_binary']:.2f}%"
                        if "final_hans_direct_binary" in summary
                        else "—"
                    ),
                    _fmt(summary.get("rounds_2_plus_pipeline_usd")),
                    _fmt(summary.get("teacher_tokens_observed"), 0),
                    pilot_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The USD totals above are descriptive partial totals from rounds 2–8. Historical artifacts explicitly exclude the shared initial-pool/round-1 cost, so these values are not used as `C_full` or as the denominator of the primary saving ratio.",
            "",
            "### Validation-patience counterfactual cost evidence",
            "",
            "| Seed | Decision/deployed round | Later-round modeled USD that would be avoided | Scope |",
            "|---:|---|---:|---|",
        ]
    )
    validation_results = {
        item.seed: item for item in results if item.policy == "validation_patience"
    }
    for audit in source_audits:
        summary = audit.get("artifact_summary", {})
        result = validation_results.get(int(audit["seed"]))
        costs = summary.get("per_round_pipeline_usd", {})
        if result is None or result.decision_round is None or not costs:
            lines.append(f"| {audit['seed']} | — | — | missing artifacts |")
            continue
        avoidable = sum(
            float(value)
            for round_text, value in costs.items()
            if int(round_text) > result.decision_round
        )
        lines.append(
            f"| {audit['seed']} | {result.decision_round}/{result.deployed_round} | "
            f"{avoidable:.6f} | exact later-round subtotal; not a full-pipeline ratio |"
        )
    lines.extend(
        [
            "",
            "This validation-patience row is a counterfactual based only on the saved validation trajectory. Its final-test regret remains unknown because earlier full-HANS evaluations were not saved.",
            "",
            "## Per-seed policy comparison",
            "",
            "| Seed | Policy | Status | Class | Decision/deployed round | Budget | Regret | Epsilon-optimal | Cost saving | Pilot USD | Radius | Reason |",
            "|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in results:
        rounds = f"{_fmt(item.decision_round, 0)}/{_fmt(item.deployed_round, 0)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.seed),
                    f"`{item.policy}`",
                    f"`{item.status}`",
                    f"`{item.classification}`" if item.classification else "—",
                    rounds,
                    _fmt(item.budget, 0),
                    _fmt(item.quality_regret),
                    _fmt(item.epsilon_optimal),
                    _fmt(item.cost_saving_ratio),
                    _fmt(item.pilot_overhead_modeled_usd),
                    _fmt(item.confidence_radius),
                    f"`{item.decision_reason}`" if item.decision_reason else "—",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Modeled USD and avoidable cost",
            "",
            "| Seed | Policy | Full USD | Stopped USD | Avoidable USD | Saving ratio | Pilot overhead USD |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.seed),
                    f"`{item.policy}`",
                    _fmt(item.full_modeled_usd),
                    _fmt(item.stopped_modeled_usd),
                    _fmt(item.avoidable_modeled_usd),
                    _fmt(item.cost_saving_ratio),
                    _fmt(item.pilot_overhead_modeled_usd),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Measured resources (kept dimensioned)",
            "",
            "| Seed | Policy | Full counters | Stopped counters | Per-counter saving ratios | Pilot counters |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for item in results:
        encoded = [
            json.dumps(dict(item.measured_full), sort_keys=True),
            json.dumps(dict(item.measured_stopped), sort_keys=True),
            json.dumps(dict(item.measured_saving_ratio), sort_keys=True),
            json.dumps(dict(item.pilot_measured), sort_keys=True),
        ]
        encoded = [value.replace("|", "\\|") for value in encoded]
        lines.append(
            f"| {item.seed} | `{item.policy}` | `{encoded[0]}` | `{encoded[1]}` | "
            f"`{encoded[2]}` | `{encoded[3]}` |"
        )
    lines.extend(
        [
            "",
            "## Data completeness",
            "",
            "| Seed | Run source | Range replay | Issues |",
            "|---:|---|---|---|",
        ]
    )
    for audit in source_audits:
        issues = "; ".join(audit.get("issues", [])) or "none"
        lines.append(
            f"| {audit['seed']} | `{audit.get('run_dir', 'missing')}` | "
            f"`{audit.get('range_aware_replay', 'missing')}` | {issues} |"
        )
    incomplete = [item for item in results if item.status != "complete"]
    lines.extend(["", "## Missing evidence", ""])
    if not incomplete:
        lines.append("All requested policy rows have complete per-round offline quality and cumulative modeled cost.")
    else:
        for item in incomplete:
            lines.append(
                f"- Seed {item.seed}, `{item.policy}`: "
                + (", ".join(item.missing_evidence) or item.decision_reason or "unknown")
            )
        lines.append("")
        lines.append(
            "No missing full-trajectory accuracy is replaced with validation accuracy, and no missing cost is inferred from unrelated counters."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `good_stop`: an online policy stops before T and its deployed checkpoint is epsilon-optimal.",
            "- `premature_stop`: an online policy stops before T with regret above epsilon.",
            "- `unnecessary_continue`: no early stop, while the retrospective epsilon oracle identifies a cheaper earlier round.",
            "- `budget_exhausted`: no early stop and the retrospective oracle also requires round T.",
            "- `oracle_reference`: evaluation-only oracle row; it is not classified as an online stopping policy.",
            "- The retrospective oracle is an evaluation reference, not a statistically certified stopping policy.",
            "- Validation patience consumes only current/past validation values; its deployed checkpoint may be earlier than its decision/cost round.",
            "",
            "## Cost accounting",
            "",
            "The primary `cost_saving_ratio` is computed only when every round provides a cumulative, full-pipeline modeled USD total including teacher, verification, generation, pilots, training and evaluation overhead. Measured resource savings are emitted per counter in JSON/CSV and are never summed across units. Pilot overhead is reported separately but remains included in total pipeline cost.",
            "",
            "## Statistical scope",
            "",
            "This retrospective benchmark does not establish an online stopping guarantee, does not validate conditional independence/unbiasedness of pilot observations, and does not authorize tuning lambda, epsilon, m, or confidence levels against sealed or future-round outcomes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/v7_correct_stopping_benchmark.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage2_diagnostics/v7_correct_stopping_benchmark",
    )
    parser.add_argument(
        "--report",
        default="reports/v7_minimal_cost_aware/CORRECT_STOPPING_BENCHMARK.md",
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write normalized evidence/audit without computing policy classifications",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = REPO_ROOT / args.config
    config = _json(config_path)
    benchmark = config["benchmark"]
    output_dir = REPO_ROOT / args.output_dir
    report_path = REPO_ROOT / args.report
    if output_dir.exists() and not args.resume:
        raise RuntimeError(f"output exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.seeds or [int(item["seed"]) for item in config["sources"]])

    source_audits: List[Dict[str, Any]] = []
    trajectories: List[Trajectory] = []
    missing_sources: Dict[int, str] = {}
    normalized_payloads: Dict[int, Dict[str, Any]] = {}
    source_hashes: Dict[str, str] = {str(config_path.relative_to(REPO_ROOT)): _sha256(config_path)}
    for source in config["sources"]:
        seed = int(source["seed"])
        if seed not in requested:
            continue
        audit: Dict[str, Any] = {"seed": seed, "issues": []}
        normalized = source.get("normalized_artifact")
        normalized_path = REPO_ROOT / str(normalized) if normalized else None
        if normalized_path is not None and normalized_path.is_file():
            trajectory = _load_normalized(normalized_path)
            audit["run_dir"] = str(normalized_path.relative_to(REPO_ROOT))
            source_hashes[str(normalized_path.relative_to(REPO_ROOT))] = _sha256(normalized_path)
        else:
            run_dir = _resolve_run_dir(source)
            if run_dir is None:
                audit["issues"].append("missing_run_artifacts")
                missing_sources[seed] = "missing_run_artifacts"
                source_audits.append(audit)
                continue
            audit["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
            root_result = run_dir / "result.json"
            if root_result.is_file():
                audit["artifact_summary"] = _historical_result_summary(
                    _json(root_result)
                )
            evidence = Evidence(
                seed,
                int(benchmark["t_max"]),
                int(benchmark["budget_max"]),
            )
            for artifact in _relevant_json_files(run_dir):
                relative = str(artifact.relative_to(REPO_ROOT))
                try:
                    payload = _json(artifact)
                except (json.JSONDecodeError, OSError) as error:
                    evidence.issues.append(f"unreadable:{relative}:{type(error).__name__}")
                    continue
                source_hashes[relative] = _sha256(artifact)
                _ingest_payload(evidence, payload, relative)
            replay = source.get("range_aware_replay")
            replay_path = REPO_ROOT / str(replay) if replay else None
            if replay_path is not None and replay_path.is_file():
                relative = str(replay_path.relative_to(REPO_ROOT))
                source_hashes[relative] = _sha256(replay_path)
                _ingest_range_replay(evidence, _json(replay_path), relative)
                audit["range_aware_replay"] = relative
            else:
                audit["range_aware_replay"] = "missing"
                evidence.issues.append("missing_range_aware_replay")
            audit["issues"].extend(evidence.issues)
            trajectory = _trajectory_from_evidence(evidence)
        if trajectory.seed != seed:
            raise RuntimeError(f"seed mismatch: config={seed}, artifact={trajectory.seed}")
        normalized_payloads[seed] = _normalized_payload(trajectory, audit)
        trajectories.append(trajectory)
        source_audits.append(audit)

    manifest_path = output_dir / "source_manifest.json"
    manifest = {"schema": "v7_correct_stopping_sources_v1", "hashes": source_hashes}
    if args.resume and manifest_path.is_file():
        previous = _json(manifest_path)
        if previous != manifest:
            raise RuntimeError("resume source hashes differ; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for seed, normalized_payload in sorted(normalized_payloads.items()):
        normalized_output = evidence_dir / f"seed_{seed}.json"
        normalized_output.write_text(
            json.dumps(normalized_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.audit_only:
        print(f"Wrote evidence audit to {evidence_dir}")
        return

    results: List[BenchmarkResult] = []
    for trajectory in trajectories:
        results.extend(
            benchmark_trajectory(
                trajectory,
                epsilon=float(benchmark["epsilon"]),
                patience_rounds=int(benchmark["validation_patience_rounds"]),
                validation_min_delta=float(benchmark["validation_min_delta"]),
            )
        )
    for seed, reason in sorted(missing_sources.items()):
        results.extend(_missing_result(seed, policy, reason) for policy in POLICY_NAMES)
    results.sort(key=lambda item: (item.seed, POLICY_NAMES.index(item.policy)))
    payload = {
        "schema": "v7_correct_stopping_benchmark_v1",
        "benchmark": benchmark,
        "results": [asdict(item) for item in results],
        "source_audits": source_audits,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "policy",
            "status",
            "classification",
            "decision_round",
            "deployed_round",
            "budget",
            "quality_regret",
            "epsilon_optimal",
            "cost_saving_ratio",
            "full_modeled_usd",
            "stopped_modeled_usd",
            "avoidable_modeled_usd",
            "pilot_overhead_modeled_usd",
            "measured_full",
            "measured_stopped",
            "measured_saving_ratio",
            "pilot_measured",
            "confidence_radius",
            "decision_reason",
            "certified_online_stop",
            "missing_evidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = asdict(item)
            row["missing_evidence"] = ";".join(item.missing_evidence)
            for key in (
                "measured_full",
                "measured_stopped",
                "measured_saving_ratio",
                "pilot_measured",
            ):
                row[key] = json.dumps(row[key], sort_keys=True)
            writer.writerow({key: row.get(key) for key in fieldnames})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report_path,
        config=config,
        results=results,
        source_audits=source_audits,
    )
    print(f"Wrote {output_dir / 'benchmark.json'}")
    print(f"Wrote {output_dir / 'benchmark.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
