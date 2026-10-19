#!/usr/bin/env python3
"""Generate new controlled probes, query the student, and compute an LCB."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION
from src.data.pipeline_split import (
    materialize_role,
    records_by_id,
    reconstruct_five_way_split,
    validate_manifest,
)
from src.hypotheses.types import FailureHypothesis
from src.models.qwen_nli import QwenNLIStudent
from src.teacher import PersistentJSONTeacherCache, build_teacher
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed
from src.verification.bounds import verify_binary_outcomes
from src.verification.generated_probes import (
    ControlledProbeConfig,
    ControlledVerificationProbeGenerator,
    TeacherVerificationProbeValidator,
    finalize_valid_probe_sets,
    predict_probe_sets,
)


CERTIFICATION_METRICS = {"strict_joint", "target_failure"}
METRIC_DEFINITIONS = {
    "target_failure": (
        "target has the predeclared wrong binary HANS label"
    ),
    "strict_joint": (
        "target has the predeclared wrong binary HANS label and both controls "
        "have their independently validated exact MNLI labels"
    ),
}


def _metric_outcomes(
    outcomes: list[dict[str, Any]], metric: str
) -> list[int]:
    if metric not in CERTIFICATION_METRICS:
        raise ValueError(
            "verification.certification_metric must be one of "
            f"{sorted(CERTIFICATION_METRICS)}"
        )
    key = "target_failure_observed" if metric == "target_failure" else "z"
    if any(outcome.get(key) is None for outcome in outcomes):
        raise ValueError(
            f"Metric {metric!r} is unavailable for the configured probe design"
        )
    return [int(bool(outcome[key])) for outcome in outcomes]


def _verification_statistic(
    hypothesis_id: str,
    outcomes: list[dict[str, Any]],
    metric: str,
    *,
    delta: float,
    tau_v: float,
    K: int,
    T: int,
) -> dict[str, Any]:
    return verify_binary_outcomes(
        hypothesis_id,
        _metric_outcomes(outcomes, metric),
        delta=delta,
        tau_v=tau_v,
        K=K,
        T=T,
    ).to_dict()


def _diagnostic_statistic(result: dict[str, Any], metric: str) -> dict[str, Any]:
    diagnostic = dict(result)
    diagnostic["metric"] = metric
    diagnostic["outcome_definition"] = METRIC_DEFINITIONS[metric]
    diagnostic["meets_threshold"] = bool(diagnostic.pop("certified"))
    return diagnostic


def _binary_accuracy(
    outcomes: list[dict[str, Any]], key: str
) -> dict[str, int | float | None]:
    available = [outcome[key] for outcome in outcomes if outcome.get(key) is not None]
    n = len(available)
    correct = sum(bool(value) for value in available)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
    }


def _unavailable_diagnostic(metric: str, reason: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "status": "not_available",
        "reason": reason,
        "n": 0,
        "nu_hat": None,
        "confidence_radius": None,
        "lower_bound": None,
        "meets_threshold": None,
        "outcome_definition": METRIC_DEFINITIONS[metric],
    }


def _adaptive_schedule(initial: int, increment: int, maximum: int) -> list[int]:
    if initial <= 0 or increment <= 0 or maximum < initial:
        raise ValueError(
            "Adaptive verification requires 0 < initial <= maximum and increment > 0"
        )
    schedule = list(range(initial, maximum + 1, increment))
    if schedule[-1] != maximum:
        schedule.append(maximum)
    return schedule


def _adaptive_action(
    statistic: dict[str, Any], *, tau_v: float
) -> tuple[str, float | None]:
    estimate = statistic.get("nu_hat")
    radius = statistic.get("confidence_radius")
    lower = statistic.get("lower_bound")
    if estimate is None or radius is None or lower is None:
        return "continue", None
    upper = min(1.0, float(estimate) + float(radius))
    if float(lower) >= tau_v:
        return "certify", upper
    if upper < tau_v:
        return "reject", upper
    return "continue", upper


def _run_adaptive_verification(
    *,
    run_dir: Path,
    output_dir: Path,
    hypotheses: list[FailureHypothesis],
    verification_context: list[dict[str, Any]],
    teacher: PersistentJSONTeacherCache,
    teacher_config: dict[str, Any],
    model: dict[str, Any],
    probe_config: ControlledProbeConfig,
    validator: TeacherVerificationProbeValidator,
    certification_metric: str,
    controls_role: str,
    delta: float,
    tau_v: float,
    K: int,
    configured_T: int,
    schedule: list[int],
    minimum_valid_probe_sets: int,
) -> None:
    # Every scheduled look receives a term in the same Hoeffding union bound.
    # This keeps early stopping valid without replacing the original bound.
    effective_T = configured_T * len(schedule)
    all_generated = []
    all_predictions: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    set_deterministic_seed(probe_config.seed)
    student = _build_student(model) if hypotheses else None
    batch_size = int(model.get("batch_size", 4))

    for hypothesis in hypotheses:
        generated_for_hypothesis = []
        accepted_for_hypothesis = []
        outcomes_for_hypothesis: list[dict[str, Any]] = []
        predictions_for_hypothesis: list[dict[str, Any]] = []
        attempted = 0
        failed_generation = 0
        failed_validation = 0
        rejected_set_ids: set[str] = set()
        trace: list[dict[str, Any]] = []
        stop_reason = "maximum_probe_sets_reached"

        for look_index, cumulative_attempt_target in enumerate(schedule, start=1):
            stage_requested = cumulative_attempt_target - attempted
            if stage_requested <= 0:
                raise RuntimeError("Adaptive verification schedule did not advance")
            stage_probe_config = replace(
                probe_config,
                n_probe_sets=stage_requested,
                min_valid_probe_sets=1,
            )
            generator = ControlledVerificationProbeGenerator(
                teacher, stage_probe_config
            )
            generated, generation_log = generator.generate(
                hypothesis,
                verification_context,
                ordinal_start=attempted,
                avoid_probes=generated_for_hypothesis,
            )
            attempted += int(generation_log["attempted_probe_sets"])
            failed_generation += int(
                generation_log.get("failed_generation_probe_sets", 0)
            )
            decisions, validation_log = validator.validate(hypothesis, generated)
            failed_validation += int(
                validation_log.get("failed_validation_probe_sets", 0)
            )
            accepted, rejected = finalize_valid_probe_sets(
                generated,
                decisions,
                verification_context,
                near_duplicate_threshold=probe_config.near_duplicate_threshold,
                previously_accepted=accepted_for_hypothesis,
            )
            generated_for_hypothesis.extend(generated)
            accepted_for_hypothesis.extend(accepted)
            rejected_set_ids.update(str(row["probe_set_id"]) for row in rejected)
            all_rejected.extend(rejected)
            all_logs.extend(
                [
                    {
                        "stage": "generation",
                        "adaptive_look": look_index,
                        **generation_log,
                    },
                    {
                        "stage": "independent_validation",
                        "adaptive_look": look_index,
                        **validation_log,
                    },
                ]
            )

            if accepted:
                assert student is not None
                prediction_rows, stage_outcomes = predict_probe_sets(
                    hypothesis, accepted, student, batch_size=batch_size
                )
            else:
                prediction_rows, stage_outcomes = [], []
            predictions_for_hypothesis.extend(prediction_rows)
            for outcome in stage_outcomes:
                outcome["adaptive_look"] = look_index
                outcome["certification_metric"] = certification_metric
                outcome["certification_z"] = _metric_outcomes(
                    [outcome], certification_metric
                )[0]
                outcome["certification_outcome_definition"] = METRIC_DEFINITIONS[
                    certification_metric
                ]
            outcomes_for_hypothesis.extend(stage_outcomes)

            selected_statistic = _verification_statistic(
                hypothesis.id,
                outcomes_for_hypothesis,
                certification_metric,
                delta=delta,
                tau_v=tau_v,
                K=K,
                T=effective_T,
            )
            if len(outcomes_for_hypothesis) < minimum_valid_probe_sets:
                action, upper_bound = "continue", None
            else:
                action, upper_bound = _adaptive_action(
                    selected_statistic, tau_v=tau_v
                )
            trace.append(
                {
                    "look": look_index,
                    "attempted_probe_sets": attempted,
                    "valid_probe_sets": len(outcomes_for_hypothesis),
                    "nu_hat": selected_statistic.get("nu_hat"),
                    "confidence_radius": selected_statistic.get(
                        "confidence_radius"
                    ),
                    "lower_bound": selected_statistic.get("lower_bound"),
                    "upper_bound": upper_bound,
                    "action": action,
                }
            )
            if action == "certify":
                stop_reason = "hoeffding_lcb_certified"
                break
            if action == "reject":
                stop_reason = "hoeffding_ucb_below_threshold"
                break

        all_generated.extend(generated_for_hypothesis)
        all_predictions.extend(predictions_for_hypothesis)
        all_outcomes.extend(outcomes_for_hypothesis)
        target_statistic = _verification_statistic(
            hypothesis.id,
            outcomes_for_hypothesis,
            "target_failure",
            delta=delta,
            tau_v=tau_v,
            K=K,
            T=effective_T,
        )
        strict_statistic = (
            None
            if probe_config.probe_design == "target_only"
            else _verification_statistic(
                hypothesis.id,
                outcomes_for_hypothesis,
                "strict_joint",
                delta=delta,
                tau_v=tau_v,
                K=K,
                T=effective_T,
            )
        )
        result = dict(
            target_statistic
            if certification_metric == "target_failure"
            else strict_statistic or {}
        )
        reliable = len(outcomes_for_hypothesis) >= minimum_valid_probe_sets
        if not reliable:
            result["certified"] = False
            result["status"] = "unreliable_probe_design"
            stop_reason = "insufficient_valid_probe_sets"
        result["hypothesis_name"] = hypothesis.name
        result["certification_metric"] = certification_metric
        result["certification_outcome_definition"] = METRIC_DEFINITIONS[
            certification_metric
        ]
        result["probe_design"] = {
            "attempted_probe_sets": attempted,
            "generated_probe_sets": (
                len(generated_for_hypothesis) // probe_config.probes_per_set
            ),
            "probe_design": probe_config.probe_design,
            "probes_per_set": probe_config.probes_per_set,
            "failed_generation_probe_sets": failed_generation,
            "valid_probe_sets": len(outcomes_for_hypothesis),
            "rejected_probe_sets": len(rejected_set_ids),
            "failed_validation_probe_sets": failed_validation,
            "minimum_valid_probe_sets": minimum_valid_probe_sets,
            "near_duplicate_threshold": probe_config.near_duplicate_threshold,
            "reliable_for_lcb": reliable,
            "adaptive": True,
            "look_schedule": schedule,
            "looks_used": len(trace),
            "stop_reason": stop_reason,
            "trace": trace,
        }
        result["diagnostics"] = {
            "target_failure": _diagnostic_statistic(
                target_statistic, "target_failure"
            ),
            "strict_joint": (
                _unavailable_diagnostic(
                    "strict_joint", "controls_not_generated_in_target_only_design"
                )
                if strict_statistic is None
                else _diagnostic_statistic(strict_statistic, "strict_joint")
            ),
            "controls": {
                "role": controls_role,
                "counterfactual": _binary_accuracy(
                    outcomes_for_hypothesis, "counterfactual_control_correct"
                ),
                "invariant": _binary_accuracy(
                    outcomes_for_hypothesis, "invariant_control_correct"
                ),
            },
        }
        result["z_definition"] = METRIC_DEFINITIONS[certification_metric]
        results.append(result)

    write_jsonl(
        output_dir / "generated_probes.jsonl",
        [probe.to_dict() for probe in all_generated],
    )
    write_jsonl(output_dir / "probes.jsonl", all_predictions)
    write_jsonl(output_dir / "rejected_probes.jsonl", all_rejected)
    write_jsonl(output_dir / "probe_set_outcomes.jsonl", all_outcomes)
    write_jsonl(run_dir / "teacher_calls" / "verification.jsonl", all_logs)
    results_payload = {
        "mode": "llm_main",
        "oracle_debug": False,
        "verification_mode": "new_controlled_generated_probes",
        "verification_protocol_version": (
            "target_failure_target_only_adaptive_hoeffding_v1"
            if probe_config.probe_design == "target_only"
            else "target_failure_adaptive_hoeffding_v2"
        ),
        "probe_design": probe_config.probe_design,
        "probes_per_set": probe_config.probes_per_set,
        "certification_metric": certification_metric,
        "controls_role": controls_role,
        "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
        "verification_split_usage": "seed_context_only",
        "student_predictions_hidden_until_all_probe_designs_finalized": False,
        "teacher_never_receives_student_outcomes": True,
        "verification_probes_training_eligible": False,
        "adaptive_verification": True,
        "adaptive_look_schedule": schedule,
        "delta": delta,
        "tau_v": tau_v,
        "K": K,
        "T": effective_T,
        "configured_max_rounds": configured_T,
        "requested_probe_sets": probe_config.n_probe_sets,
        "minimum_valid_probe_sets": minimum_valid_probe_sets,
        "near_duplicate_threshold": probe_config.near_duplicate_threshold,
        "probe_prompt_version": probe_config.prompt_version,
        "label_validation_prompt_version": validator.prompt_version,
        "teacher": {
            "provider": teacher_config.get("provider", "huggingface"),
            "name_or_path": teacher_config.get("name_or_path"),
        },
        "student": {
            "name_or_path": model.get("name_or_path"),
            "adapter_path": model.get("adapter_path"),
        },
        "bound_assumptions": (
            "Hoeffding remains the certification bound. K covers candidate "
            "hypotheses and effective T covers every predeclared adaptive look. "
            "Later teacher calls never receive student outcomes; continuation is "
            "the only outcome-dependent action. Probe sets must remain sufficiently "
            "independent for the union-adjusted guarantee."
        ),
        "results": results,
    }
    write_json(output_dir / "results.json", results_payload)
    _write_verification_report(
        output_dir / "REPORT.md",
        certification_metric=certification_metric,
        tau_v=tau_v,
        delta=delta,
        results=results,
    )
    print(f"Wrote adaptive Hoeffding verification artifacts to {output_dir}")


def _write_verification_report(
    path: Path,
    *,
    certification_metric: str,
    tau_v: float,
    delta: float,
    results: list[dict[str, Any]],
) -> None:
    def percentage(value: Any) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.2f}%"

    lines = [
        "# Controlled verification report",
        "",
        f"Certification metric: `{certification_metric}`",
        f"Certification threshold: `LCB >= {tau_v}` with `delta = {delta}`.",
        "Counterfactual/invariant controls and strict-joint statistics are "
        "diagnostic only when target-failure certification is selected.",
        "Verification probes are never eligible for training; newly generated "
        "synthetic pairs are used for training only after certification.",
        "",
        "| Hypothesis | Valid sets | Target rate | Target LCB | Strict-joint rate | Strict-joint LCB | Counterfactual acc. | Invariant acc. | Stop | Certified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        diagnostics = result["diagnostics"]
        target = diagnostics["target_failure"]
        strict = diagnostics["strict_joint"]
        controls = diagnostics["controls"]
        lines.append(
            "| {name} | {n} | {target_rate} | {target_lcb} | "
            "{strict_rate} | {strict_lcb} | {counterfactual} | {invariant} | "
            "{stop} | {certified} |".format(
                name=result.get("hypothesis_name", result["hypothesis_id"]),
                n=result["n"],
                target_rate=percentage(target["nu_hat"]),
                target_lcb=percentage(target["lower_bound"]),
                strict_rate=percentage(strict["nu_hat"]),
                strict_lcb=percentage(strict["lower_bound"]),
                counterfactual=percentage(
                    controls["counterfactual"]["accuracy"]
                ),
                invariant=percentage(controls["invariant"]["accuracy"]),
                stop=result.get("probe_design", {}).get("stop_reason", "one_shot"),
                certified="yes" if result["certified"] else "no",
            )
        )
    _atomic_text_write(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_hypotheses(payload: dict[str, object]) -> list[FailureHypothesis]:
    if payload.get("mode") != "llm_main" or payload.get("oracle_debug") is not False:
        raise ValueError("Expected a primary LLM hypothesis artifact")
    if payload.get("split_role") != "hypothesis_discovery":
        raise ValueError("Candidate hypotheses must originate from discovery data")
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        raise ValueError("hypotheses must be a list")
    hypotheses = [
        FailureHypothesis.from_mapping(item, require_llm_schema=True)
        for item in raw
        if isinstance(item, dict)
    ]
    if len(hypotheses) != len(raw):
        raise ValueError("Every hypothesis must be a JSON object")
    if len({item.id for item in hypotheses}) != len(hypotheses):
        raise ValueError("Duplicate candidate hypothesis IDs")
    return hypotheses


def _build_student(model: dict[str, Any]) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model.get("name_or_path", "Qwen/Qwen3-1.7B")),
        adapter_path=model.get("adapter_path"),
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Main config must set mode: llm_main and oracle_debug: false")
    hans = require_mapping(config, "hans")
    verification = require_mapping(config, "verification")
    teacher_config = require_mapping(config, "teacher")
    model = require_mapping(config, "model")
    run_dir = Path(args.run_dir)
    teacher = PersistentJSONTeacherCache(
        build_teacher(teacher_config),
        run_dir / "teacher_calls" / "response_cache",
    )
    requested_sets = int(
        verification.get("n_probe_sets", verification.get("n_probes", 50))
    )
    probe_config = ControlledProbeConfig(
        n_probe_sets=requested_sets,
        min_valid_probe_sets=int(
            verification.get("min_valid_probe_sets", requested_sets)
        ),
        max_context_records=int(verification.get("max_context_records", 24)),
        max_probe_sets_per_call=int(
            verification.get("max_probe_sets_per_call", 8)
        ),
        max_attempts_per_batch=int(
            verification.get("max_attempts_per_batch", 3)
        ),
        near_duplicate_threshold=float(
            verification.get("near_duplicate_threshold", 0.92)
        ),
        seed=int(verification.get("seed", 42)),
        prompt_version=str(
            verification.get(
                "probe_prompt_version", "controlled_verification_probes_v3"
            )
        ),
        probe_design=str(
            verification.get("probe_design", "controlled_triplet")
        ),
    )
    validator = TeacherVerificationProbeValidator(
        teacher,
        seed=int(verification.get("label_validation_seed", probe_config.seed)),
        prompt_version=str(
            verification.get(
                "label_validation_prompt_version",
                "verification_probe_validation_v1",
            )
        ),
        max_probe_sets_per_call=int(
            verification.get("validation_max_probe_sets_per_call", 8)
        ),
        max_attempts_per_batch=int(
            verification.get("validation_max_attempts_per_batch", 3)
        ),
    )
    delta = float(verification["delta"])
    tau_v = float(verification["tau_v"])
    K = int(verification["max_hypotheses_per_round"])
    T = int(verification["max_rounds"])
    certification_metric = str(
        verification.get("certification_metric", "strict_joint")
    )
    if certification_metric not in CERTIFICATION_METRICS:
        raise ValueError(
            "verification.certification_metric must be one of "
            f"{sorted(CERTIFICATION_METRICS)}"
        )
    if probe_config.probe_design == "target_only" and certification_metric != "target_failure":
        raise ValueError(
            "target_only probe design requires certification_metric: target_failure"
        )
    controls_role = (
        "not_generated_target_only"
        if probe_config.probe_design == "target_only"
        else "diagnostic_only_for_verification"
        if certification_metric == "target_failure"
        else "certification_gate"
    )
    adaptive_value = verification.get("adaptive_verification", {})
    if not isinstance(adaptive_value, dict):
        raise ValueError("verification.adaptive_verification must be a mapping")
    adaptive_enabled = bool(adaptive_value.get("enabled", False))

    required = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_split_v2.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    output_dir = run_dir / "verification"
    outputs = [
        output_dir / "generated_probes.jsonl",
        output_dir / "probes.jsonl",
        output_dir / "rejected_probes.jsonl",
        output_dir / "probe_set_outcomes.jsonl",
        output_dir / "results.json",
        output_dir / "REPORT.md",
        run_dir / "teacher_calls" / "verification.jsonl",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite verification artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    source = read_jsonl(required["predictions"])
    split = reconstruct_five_way_split(source, hans)
    validate_manifest(read_json(required["manifest"]), split)
    indexed = records_by_id(source)
    verification_context = materialize_role(
        indexed, split.verification_ids, "verification"
    )
    hypotheses = _load_hypotheses(read_json(required["hypotheses"]))
    if len(hypotheses) > K:
        raise ValueError(f"Candidate count exceeds configured K={K}")

    if adaptive_enabled:
        schedule = _adaptive_schedule(
            int(adaptive_value.get("initial_probe_sets", 20)),
            int(adaptive_value.get("increment_probe_sets", 10)),
            requested_sets,
        )
        _run_adaptive_verification(
            run_dir=run_dir,
            output_dir=output_dir,
            hypotheses=hypotheses,
            verification_context=verification_context,
            teacher=teacher,
            teacher_config=teacher_config,
            model=model,
            probe_config=probe_config,
            validator=validator,
            certification_metric=certification_metric,
            controls_role=controls_role,
            delta=delta,
            tau_v=tau_v,
            K=K,
            configured_T=T,
            schedule=schedule,
            minimum_valid_probe_sets=probe_config.min_valid_probe_sets,
        )
        return

    generator = ControlledVerificationProbeGenerator(teacher, probe_config)
    all_generated = []
    all_rejected: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []
    finalized: dict[str, list[Any]] = {}
    design_metadata: dict[str, dict[str, Any]] = {}

    # Finalize every candidate before a student is instantiated. Rejections
    # depend only on labels and constraints, never on observed student failures.
    for hypothesis in hypotheses:
        generated, generation_log = generator.generate(
            hypothesis, verification_context
        )
        decisions, validation_log = validator.validate(hypothesis, generated)
        accepted, rejected = finalize_valid_probe_sets(
            generated,
            decisions,
            verification_context,
            near_duplicate_threshold=probe_config.near_duplicate_threshold,
        )
        all_generated.extend(generated)
        all_rejected.extend(rejected)
        all_logs.extend(
            [
                {"stage": "generation", **generation_log},
                {"stage": "independent_validation", **validation_log},
            ]
        )
        finalized[hypothesis.id] = accepted
        design_metadata[hypothesis.id] = {
            "attempted_probe_sets": int(
                generation_log.get("attempted_probe_sets", probe_config.n_probe_sets)
            ),
            "generated_probe_sets": len(generated) // probe_config.probes_per_set,
            "probe_design": probe_config.probe_design,
            "probes_per_set": probe_config.probes_per_set,
            "failed_generation_probe_sets": int(
                generation_log.get("failed_generation_probe_sets", 0)
            ),
            "valid_probe_sets": len(accepted) // probe_config.probes_per_set,
            "rejected_probe_sets": len({row["probe_set_id"] for row in rejected}),
            "failed_validation_probe_sets": int(
                validation_log.get("failed_validation_probe_sets", 0)
            ),
            "minimum_valid_probe_sets": probe_config.min_valid_probe_sets,
            "near_duplicate_threshold": probe_config.near_duplicate_threshold,
            "reliable_for_lcb": len(accepted) // probe_config.probes_per_set
            >= probe_config.min_valid_probe_sets,
        }

    all_predictions: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if any(finalized.values()):
        set_deterministic_seed(probe_config.seed)
        student = _build_student(model)
    else:
        student = None
    batch_size = int(model.get("batch_size", 4))

    # Query the student once on the frozen accepted probe collection. No teacher
    # call occurs after an outcome is visible, preventing failure-conditioned retry.
    for hypothesis in hypotheses:
        probes = finalized[hypothesis.id]
        if probes:
            assert student is not None
            prediction_rows, outcomes = predict_probe_sets(
                hypothesis, probes, student, batch_size=batch_size
            )
        else:
            prediction_rows, outcomes = [], []
        all_predictions.extend(prediction_rows)
        for outcome in outcomes:
            outcome["certification_metric"] = certification_metric
            outcome["certification_z"] = _metric_outcomes(
                [outcome], certification_metric
            )[0]
            outcome["certification_outcome_definition"] = METRIC_DEFINITIONS[
                certification_metric
            ]
        all_outcomes.extend(outcomes)
        target_statistic = _verification_statistic(
            hypothesis.id,
            outcomes,
            "target_failure",
            delta=delta,
            tau_v=tau_v,
            K=K,
            T=T,
        )
        strict_statistic = (
            None
            if probe_config.probe_design == "target_only"
            else _verification_statistic(
                hypothesis.id,
                outcomes,
                "strict_joint",
                delta=delta,
                tau_v=tau_v,
                K=K,
                T=T,
            )
        )
        result = dict(
            target_statistic
            if certification_metric == "target_failure"
            else strict_statistic or {}
        )
        if not design_metadata[hypothesis.id]["reliable_for_lcb"]:
            result["certified"] = False
            result["status"] = "unreliable_probe_design"
        result["hypothesis_name"] = hypothesis.name
        result["certification_metric"] = certification_metric
        result["certification_outcome_definition"] = METRIC_DEFINITIONS[
            certification_metric
        ]
        result["probe_design"] = design_metadata[hypothesis.id]
        result["diagnostics"] = {
            "target_failure": _diagnostic_statistic(
                target_statistic, "target_failure"
            ),
            "strict_joint": (
                _unavailable_diagnostic(
                    "strict_joint", "controls_not_generated_in_target_only_design"
                )
                if strict_statistic is None
                else _diagnostic_statistic(strict_statistic, "strict_joint")
            ),
            "controls": {
                "role": controls_role,
                "counterfactual": _binary_accuracy(
                    outcomes, "counterfactual_control_correct"
                ),
                "invariant": _binary_accuracy(
                    outcomes, "invariant_control_correct"
                ),
            },
        }
        # Retain the old field for readers of historical artifacts. New code
        # should use certification_outcome_definition and diagnostics.
        result["z_definition"] = METRIC_DEFINITIONS[certification_metric]
        results.append(result)

    write_jsonl(
        output_dir / "generated_probes.jsonl",
        [probe.to_dict() for probe in all_generated],
    )
    write_jsonl(output_dir / "probes.jsonl", all_predictions)
    write_jsonl(output_dir / "rejected_probes.jsonl", all_rejected)
    write_jsonl(output_dir / "probe_set_outcomes.jsonl", all_outcomes)
    write_jsonl(run_dir / "teacher_calls" / "verification.jsonl", all_logs)
    results_payload = {
            "mode": "llm_main",
            "oracle_debug": False,
            "verification_mode": "new_controlled_generated_probes",
            "verification_protocol_version": (
                "target_failure_target_only_verified_v1"
                if probe_config.probe_design == "target_only"
                else "target_failure_verified_v1"
                if certification_metric == "target_failure"
                else "strict_joint_verified_v1"
            ),
            "probe_design": probe_config.probe_design,
            "probes_per_set": probe_config.probes_per_set,
            "certification_metric": certification_metric,
            "controls_role": controls_role,
            "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
            "verification_split_usage": "seed_context_only",
            "student_predictions_hidden_until_all_probe_designs_finalized": True,
            "verification_probes_training_eligible": False,
            "delta": delta,
            "tau_v": tau_v,
            "K": K,
            "T": T,
            "requested_probe_sets": probe_config.n_probe_sets,
            "minimum_valid_probe_sets": probe_config.min_valid_probe_sets,
            "near_duplicate_threshold": probe_config.near_duplicate_threshold,
            "probe_prompt_version": probe_config.prompt_version,
            "label_validation_prompt_version": validator.prompt_version,
            "teacher": {
                "provider": teacher_config.get("provider", "huggingface"),
                "name_or_path": teacher_config.get("name_or_path"),
            },
            "student": {
                "name_or_path": model.get("name_or_path"),
                "adapter_path": model.get("adapter_path"),
            },
            "bound_assumptions": (
                "The union-adjusted Hoeffding LCB is conditional on bounded binary "
                "outcomes, configured K/T accounting, and sufficiently independent "
                "probe sets. One-shot generation, scenario separation, and pre-outcome "
                "validation reduce adaptivity but do not prove iid sampling. Controls "
                "are reported diagnostically and do not enter target-failure "
                "certification when that metric is configured."
            ),
            "results": results,
        }
    write_json(output_dir / "results.json", results_payload)
    _write_verification_report(
        output_dir / "REPORT.md",
        certification_metric=certification_metric,
        tau_v=tau_v,
        delta=delta,
        results=results,
    )
    print(f"Wrote controlled verification artifacts to {output_dir}")


if __name__ == "__main__":
    main()
