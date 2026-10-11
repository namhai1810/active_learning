#!/usr/bin/env python3
"""Offline empirical validation of verification, gain UCB, and safe stopping."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import HANSExample
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION, split_hans_five_way
from src.evaluation.hans import evaluate_hans
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.hypotheses.types import FailureHypothesis
from src.intervention.pilot import PilotRunConfig, run_targeted_pilots
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.theory_validation.gain_coverage import evaluate_gain_coverage
from src.theory_validation.reference_data import validate_reference_disjointness
from src.theory_validation.reference_gain import run_reference_gain_repetitions
from src.theory_validation.stopping import audit_safe_stopping
from src.theory_validation.verification import evaluate_verification_confidence
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Theory-validation YAML")
    parser.add_argument("--run-dir", required=True, help="Completed online run")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only theory_validation/ artifacts",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    if not records and not allow_empty:
        raise ValueError(f"No records found in {path}")
    return records


def _resolve_optional_path(config_path: Path, value: Any, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a path string or null")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_hypotheses(payload: dict[str, Any]) -> dict[str, FailureHypothesis]:
    hypotheses = {}
    for value in payload.get("hypotheses", []):
        hypothesis = FailureHypothesis(**value)
        if HANS_ORACLE_HYPOTHESES.get(hypothesis.heuristic) != hypothesis:
            raise ValueError(f"Unsupported oracle hypothesis: {value}")
        hypotheses[hypothesis.id] = hypothesis
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


def _as_hans(records: Any) -> list[HANSExample]:
    return [
        HANSExample(
            id=str(record["id"]),
            premise=str(record["premise"]),
            hypothesis=str(record["hypothesis"]),
            gold=str(record["gold"]),
            heuristic=str(record["heuristic"]),
            subcase=str(record.get("subcase", "unknown")),
        )
        for record in records
    ]


def _group_gains(
    records: list[dict[str, Any]], order_field: str
) -> dict[str, list[float]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        hypothesis_id = str(record.get("hypothesis_id", ""))
        gain = record.get("gain")
        order = record.get(order_field)
        if (
            not hypothesis_id
            or isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or isinstance(order, bool)
            or not isinstance(order, int)
        ):
            raise ValueError("Malformed gain observation")
        grouped.setdefault(hypothesis_id, []).append((order, float(gain)))
    return {
        hypothesis_id: [gain for _, gain in sorted(values)]
        for hypothesis_id, values in grouped.items()
    }


def _print_tables(
    verification_summary: list[dict[str, Any]],
    gain_summary: dict[str, dict[str, Any]],
    stopping: dict[str, Any],
) -> None:
    print("verification: n | cert_rate | precision | recall | false_cert_rate")
    for row in verification_summary:
        print(
            row["requested_n"],
            row["certification_rate"],
            row["certification_precision"],
            row["robust_failure_recall"],
            row["false_certification_rate"],
            sep=" | ",
        )
    print("gain UCB: m | cases | coverage | mean_b | mean_abs_error")
    for key, row in gain_summary.items():
        print(
            key,
            row["num_cases"],
            row["coverage_rate"],
            row["mean_confidence_radius"],
            row["mean_absolute_error"],
            sep=" | ",
        )
    print(
        "stopping:",
        stopping["online_action"],
        stopping["audit_performed"],
        stopping["best_remaining_reference_net_value"],
        stopping["empirical_safe_stop_pass"],
        sep=" | ",
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_with_extends(config_path)
    if config.get("mode") != "oracle_hans" or config.get("oracle_debug") is not True:
        raise ValueError("Config must declare oracle_hans oracle-debug mode")
    hans = require_mapping(config, "hans")
    reference_config = require_mapping(config, "reference_data")
    verification_config = require_mapping(config, "verification_validation")
    gain_config = require_mapping(config, "gain_validation")
    stopping_config = require_mapping(config, "stopping_validation")
    decision_config = require_mapping(config, "decision")
    intervention_config = require_mapping(config, "intervention")
    pilot_mapping = require_mapping(config, "pilot")
    model = require_mapping(config, "model")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)

    reference_path = _resolve_optional_path(
        config_path,
        reference_config.get("file"),
        "reference_data.file",
    )
    if reference_path is None or not reference_path.is_file():
        raise FileNotFoundError(
            "A separate HANS-style reference_data.file is required"
        )
    precomputed_gain_path = _resolve_optional_path(
        config_path,
        gain_config.get("precomputed_reference_observations_file"),
        "gain_validation.precomputed_reference_observations_file",
    )

    run_dir = Path(args.run_dir)
    paths = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_oracle_split_v2.json",
        "verification": run_dir / "verification" / "results.json",
        "gain_estimates": run_dir / "gain" / "gain_estimates.json",
        "pilot_observations": run_dir / "gain" / "pilot_observations.jsonl",
        "decision": run_dir / "decision" / "decision.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required online artifact not found: {path}")

    # The online decision is read and frozen before any offline oracle input.
    online_decision = _read_json(paths["decision"])
    frozen_decision = paths["decision"].read_bytes()
    online_action = str(online_decision.get("action", ""))
    if online_action not in {"stop", "continue"}:
        raise ValueError("Online decision action must be stop or continue")
    lambda_cost = float(decision_config["lambda_cost"])
    if online_decision.get("lambda_cost") != lambda_cost:
        raise ValueError("Theory config lambda_cost differs from online decision")

    source_records = _read_jsonl(paths["predictions"])
    split = split_hans_five_way(
        [{"id": record["id"]} for record in source_records],
        float(hans["discovery_fraction"]),
        float(hans["verification_fraction"]),
        float(hans["intervention_fraction"]),
        float(hans["gain_eval_fraction"]),
        int(hans["split_seed"]),
    )
    manifest = _read_json(paths["manifest"])
    if manifest.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Theory validation requires the five-way split")
    expected_manifest_ids = {
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    for field, ids in expected_manifest_ids.items():
        if manifest.get(field) != ids:
            raise ValueError(f"Online manifest mismatch for {field}")
    records_by_id = {str(record["id"]): record for record in source_records}

    def materialize(ids: tuple[str, ...], role: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {**records_by_id[example_id], "split_role": role}
            for example_id in ids
        )

    online_verification = materialize(split.verification_ids, "verification")
    intervention_pool = materialize(
        split.intervention_pool_ids, "intervention_pool"
    )
    gain_eval = materialize(split.gain_eval_ids, "gain_eval")
    reference_groups = validate_reference_disjointness(
        _read_jsonl(reference_path),
        list(records_by_id),
    )
    verification_reference = list(reference_groups["verification_reference"])
    gain_reference = list(reference_groups["gain_reference"])
    hypotheses = _load_hypotheses(_read_json(paths["hypotheses"]))

    online_verification_artifact = _read_json(paths["verification"])
    delta = float(online_verification_artifact["delta"])
    tau_v = float(online_verification_artifact["tau_v"])
    K = int(online_verification_artifact["K"])
    T = int(online_verification_artifact["T"])
    configured_constants = require_mapping(config, "online_constants")
    expected_constants = {
        "delta": delta,
        "tau_v": tau_v,
        "K": K,
        "T": T,
    }
    for field, value in expected_constants.items():
        if configured_constants.get(field) != value:
            raise ValueError(f"Online theoretical constant mismatch: {field}")

    current_checkpoint = checkpoint_descriptor(model)
    gain_artifact = _read_json(paths["gain_estimates"])
    if gain_artifact.get("starting_checkpoint") != current_checkpoint:
        raise ValueError("Theory validation checkpoint differs from online pilots")
    online_gain_bound = gain_artifact.get("gain_bound")
    if not isinstance(online_gain_bound, dict):
        raise ValueError("Online gain artifact is missing gain_bound")
    for field, value in {"delta": delta, "K": K, "T": T}.items():
        if online_gain_bound.get(field) != value:
            raise ValueError(f"Online gain-bound constant mismatch: {field}")

    output_dir = run_dir / "theory_validation"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_precomputed_predictions = bool(
        reference_config.get("use_precomputed_predictions", False)
    )
    if not use_precomputed_predictions:
        set_deterministic_seed(int(verification_config["seed"]))
        reference_student = _build_student(model)
        prediction_rows, _ = evaluate_hans(
            reference_student,
            _as_hans(verification_reference),
            batch_size=int(model.get("batch_size", 4)),
        )
        predictions_by_id = {row["id"]: row["prediction"] for row in prediction_rows}
        verification_reference = [
            {**record, "prediction": predictions_by_id[str(record["id"])]}
            for record in verification_reference
        ]
        del reference_student
    elif any("prediction" not in record for record in verification_reference):
        raise ValueError("Precomputed verification references need prediction fields")

    n_values = [int(value) for value in verification_config["n_values"]]
    verification_rows, verification_summary = evaluate_verification_confidence(
        hypotheses=list(hypotheses.values()),
        online_verification_records=online_verification,
        reference_records=verification_reference,
        n_values=n_values,
        reference_probe_count=int(verification_config["reference_probe_count"]),
        delta=delta,
        tau_v=tau_v,
        gamma=float(verification_config["gamma"]),
        K=K,
        T=T,
        seed=int(verification_config["seed"]),
    )
    write_jsonl(output_dir / "verification" / "results.jsonl", verification_rows)
    write_json(
        output_dir / "verification" / "summary.json",
        {
            "n_values": n_values,
            "gamma": float(verification_config["gamma"]),
            "results": verification_summary,
        },
    )

    certified_ids = [
        str(result["hypothesis_id"])
        for result in online_verification_artifact.get("results", [])
        if result.get("certified") is True
    ]
    existing_online_observations = _read_jsonl(
        paths["pilot_observations"], allow_empty=True
    )
    online_observations = existing_online_observations
    m_values = [int(value) for value in gain_config["m_values"]]
    M = float(gain_config["M"])
    tolerance = float(gain_config.get("tolerance", 1e-12))
    if online_gain_bound.get("M") != M:
        raise ValueError("Online and theory-validation gain bounds use different M")
    if (
        not bool(gain_config.get("allow_small_reference_for_smoke", False))
        and int(gain_config["reference_repetitions"]) <= max(m_values)
    ):
        raise ValueError(
            "Full validation requires more reference repetitions than max(m_values)"
        )
    pilot_config = PilotRunConfig(
        m=max(m_values),
        pilot_train_size=int(pilot_mapping["pilot_train_size"]),
        gain_eval_size=int(pilot_mapping["gain_eval_size"]),
        seed=int(pilot_mapping["seed"]),
        M=M,
        tolerance=tolerance,
        non_entailment_label=str(pilot_mapping["non_entailment_label"]),
    )

    def student_factory(seed: int) -> QwenNLIStudent:
        set_deterministic_seed(seed)
        return _build_student(model)

    def train_callback(
        student: QwenNLIStudent,
        examples: list[Any],
        adapter_dir: Path,
        seed: int,
    ) -> float:
        return train_lora(student, examples, training, adapter_dir, seed)

    def risk_callback(student: QwenNLIStudent, records: Any) -> float:
        _, metrics = evaluate_hans(
            student,
            _as_hans(records),
            batch_size=int(model.get("batch_size", 4)),
        )
        return 1.0 - float(metrics["overall_accuracy"])

    if bool(gain_config.get("generate_online_pilots", False)):
        online_observations = []
        for hypothesis_id in certified_ids:
            online_observations.extend(
                item.to_dict()
                for item in run_targeted_pilots(
                    hypothesis=hypotheses[hypothesis_id],
                    intervention_records=intervention_pool,
                    gain_eval_records=gain_eval,
                    config=pilot_config,
                    starting_checkpoint=current_checkpoint,
                    training_config=asdict(training),
                    pilot_root=output_dir / "gain_ucb" / "online_pilots",
                    student_factory=student_factory,
                    train_callback=train_callback,
                    risk_callback=risk_callback,
                )
            )
    write_jsonl(
        output_dir / "gain_ucb" / "online_pilot_observations.jsonl",
        online_observations,
    )

    reference_statuses = {
        row["hypothesis_id"]: row["reference_status"]
        for row in verification_rows
        if row["requested_n"] == n_values[0]
    }
    robust_ids = [
        hypothesis_id
        for hypothesis_id, status in reference_statuses.items()
        if status == "robust_valid"
    ]
    reference_needed = sorted(
        set(certified_ids)
        | (set(robust_ids) if online_action == "stop" else set())
    )
    if bool(gain_config.get("generate_reference_repetitions", False)):
        reference_observations = []
        for hypothesis_id in reference_needed:
            reference_observations.extend(
                run_reference_gain_repetitions(
                    hypothesis=hypotheses[hypothesis_id],
                    intervention_records=intervention_pool,
                    gain_reference_records=gain_reference,
                    repetitions=int(gain_config["reference_repetitions"]),
                    pilot_train_size=int(pilot_mapping["pilot_train_size"]),
                    seed=int(gain_config["seed"]),
                    non_entailment_label=str(pilot_mapping["non_entailment_label"]),
                    M=M,
                    tolerance=tolerance,
                    starting_checkpoint=current_checkpoint,
                    training_config=asdict(training),
                    output_dir=output_dir / "gain_ucb" / "reference_repetitions",
                    student_factory=student_factory,
                    train_callback=train_callback,
                    risk_callback=risk_callback,
                )
            )
    else:
        if precomputed_gain_path is None or not precomputed_gain_path.is_file():
            reference_observations = []
        else:
            reference_observations = _read_jsonl(precomputed_gain_path, allow_empty=True)
    write_jsonl(
        output_dir / "gain_ucb" / "reference_observations.jsonl",
        reference_observations,
    )

    online_gains = _group_gains(online_observations, "pilot_index")
    reference_gains = _group_gains(reference_observations, "repetition")
    gain_rows, gain_summary = evaluate_gain_coverage(
        certified_hypothesis_ids=certified_ids,
        online_gains=online_gains,
        reference_gains=reference_gains,
        m_values=m_values,
        reference_repetitions=int(gain_config["reference_repetitions"]),
        M=M,
        delta=delta,
        K=K,
        T=T,
        tolerance=tolerance,
    )
    write_jsonl(output_dir / "gain_ucb" / "results.jsonl", gain_rows)
    write_json(
        output_dir / "gain_ucb" / "summary.json",
        {
            "m_values": m_values,
            "coverage_by_m": gain_summary,
            "theoretical_target": 1.0 - delta,
            "interpretation": (
                "Observed coverage is descriptive empirical evidence and does not "
                "establish exact theoretical coverage."
            ),
        },
    )

    full_size = int(intervention_config["full_intervention_size"])
    costs = {
        hypothesis_id: float(
            min(
                full_size,
                sum(
                    record.get("heuristic") == hypothesis.heuristic
                    for record in intervention_pool
                ),
            )
        )
        for hypothesis_id, hypothesis in hypotheses.items()
    }
    delta_references = {
        hypothesis_id: sum(values[: int(gain_config["reference_repetitions"])])
        / len(values[: int(gain_config["reference_repetitions"])])
        for hypothesis_id, values in reference_gains.items()
        if values[: int(gain_config["reference_repetitions"])]
    }
    if bool(stopping_config.get("enabled", True)):
        stopping_audit = audit_safe_stopping(
            online_action=online_action,
            reference_statuses=reference_statuses,
            delta_references=delta_references,
            costs=costs,
            lambda_cost=lambda_cost,
        )
    else:
        stopping_audit = {
            "online_action": online_action,
            "audit_performed": False,
            "reason": "disabled_by_config",
            "best_remaining_reference_net_value": None,
            "empirical_safe_stop_pass": None,
        }
    write_json(output_dir / "stopping" / "audit.json", stopping_audit)

    summary = {
        "oracle_debug": True,
        "evaluation_only": True,
        "verification": {"n_values": n_values, "results": verification_summary},
        "gain_ucb": {"m_values": m_values, "coverage_by_m": gain_summary},
        "safe_stopping": {
            "online_action": online_action,
            "audit_performed": stopping_audit["audit_performed"],
            "best_remaining_reference_net_value": stopping_audit[
                "best_remaining_reference_net_value"
            ],
            "empirical_safe_stop_pass": stopping_audit[
                "empirical_safe_stop_pass"
            ],
        },
        "interpretation": (
            "These experiments provide descriptive empirical support for theorem "
            "assumptions; they do not prove those assumptions mathematically."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    if paths["decision"].read_bytes() != frozen_decision:
        raise RuntimeError("Offline theory validation modified online decision.json")
    if checkpoint_descriptor(model) != current_checkpoint:
        raise RuntimeError("Offline theory validation modified the starting checkpoint")
    _print_tables(verification_summary, gain_summary, stopping_audit)
    print(f"Wrote theory-validation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
