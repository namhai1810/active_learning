#!/usr/bin/env python3
"""Run independent synthetic-data pilots for verified main-method hypotheses."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import HANSExample
from src.data.labels import MNLI_LABELS
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION
from src.active_learning.multiround import partition_ids_by_round
from src.benchmark.hans_manifest import FrozenHansManifest
from src.benchmark.stopping_subset import validate_stopping_subset_manifest
from src.data.pipeline_split import (
    materialize_role,
    records_by_id,
    reconstruct_five_way_split,
    validate_manifest,
)
from src.evaluation.hans import evaluate_hans
from src.gain.estimation import estimate_gain
from src.hypotheses.types import FailureHypothesis
from src.intervention.synthetic import SyntheticPilotConfig, run_synthetic_pilots
from src.models.lora_training import (
    EarlyStoppingConfig,
    LoraTrainingConfig,
    train_lora,
    train_lora_stateful_early_stopping,
)
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.teacher import build_teacher
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor, hash_local_path
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed
from src.verification.llm_operationalizer import (
    LLMHypothesisOperationalizer,
    OperationalizationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _binary_validation(
    probabilities: list[list[float]], examples: list[HANSExample]
) -> dict[str, Any]:
    if len(probabilities) != len(examples) or not examples:
        raise ValueError("Pilot validation probabilities and examples must align")
    losses: list[float] = []
    correct = 0
    for row, example in zip(probabilities, examples, strict=True):
        if len(row) != len(MNLI_LABELS):
            raise ValueError("Pilot validation probabilities must use MNLI order")
        entailment = float(row[0])
        non_entailment = float(row[1]) + float(row[2])
        target = entailment if example.gold == "entailment" else non_entailment
        losses.append(-math.log(max(target, 1e-12)))
        prediction = (
            "entailment" if entailment >= non_entailment else "non-entailment"
        )
        correct += int(prediction == example.gold)
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": correct / len(examples),
        "num_examples": len(examples),
        "source": "predeclared_gain_eval_stopping_subset",
        "sealed_final_test_used": False,
    }


def _load_hypotheses(payload: dict[str, Any]) -> dict[str, FailureHypothesis]:
    if payload.get("mode") != "llm_main" or payload.get("oracle_debug") is not False:
        raise ValueError("Expected main-method hypotheses")
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        raise ValueError("hypotheses must be a list")
    result: dict[str, FailureHypothesis] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("Every hypothesis must be an object")
        item = FailureHypothesis.from_mapping(value, require_llm_schema=True)
        if item.id in result:
            raise ValueError(f"Duplicate hypothesis: {item.id}")
        result[item.id] = item
    return result


def _certified(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("results")
    if not isinstance(raw, list):
        raise ValueError("Verification results must be a list")
    ids = [
        str(item["hypothesis_id"])
        for item in raw
        if isinstance(item, dict) and item.get("certified") is True
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate certified hypotheses")
    return ids


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Main config must set mode: llm_main and oracle_debug: false")
    hans = require_mapping(config, "hans")
    pilot = require_mapping(config, "pilot")
    gain = require_mapping(config, "gain")
    model = require_mapping(config, "model")
    training = LoraTrainingConfig.from_mapping(require_mapping(config, "training"))
    teacher_config = require_mapping(config, "teacher")
    teacher = build_teacher(teacher_config)
    pilot_config = SyntheticPilotConfig(
        m=int(pilot.get("m", 4)),
        pilot_train_size=int(pilot.get("pilot_train_size", 32)),
        gain_eval_size=int(pilot.get("gain_eval_size", 64)),
        seed=int(pilot.get("seed", 42)),
        M=float(gain.get("M", 1.0)),
        tolerance=float(gain.get("tolerance", 1e-12)),
    )
    op_config = OperationalizationConfig(
        max_records_per_call=int(pilot.get("max_records_per_call", 50)),
        seed=int(pilot.get("applicability_seed", pilot_config.seed)),
        prompt_version=str(
            pilot.get("prompt_version", "hypothesis_operationalization_v1")
        ),
    )
    delta = float(gain.get("delta", 0.05))
    K = int(gain.get("K", 5))
    T = int(gain.get("T", 1))
    active_learning = config.get("active_learning", {})
    if not isinstance(active_learning, dict):
        raise ValueError("active_learning must be a mapping")
    round_index = int(active_learning.get("current_round", 1))
    declared_rounds = int(active_learning.get("max_rounds", 1))
    if not 1 <= round_index <= declared_rounds:
        raise ValueError("active_learning.current_round is outside max_rounds")
    if T < declared_rounds:
        raise ValueError("gain.T must cover active_learning.max_rounds")
    preserve_adapters = bool(pilot.get("preserve_adapters", True))
    pilot_training_policy = str(pilot.get("training_policy", "independent_lora"))
    if pilot_training_policy not in {
        "independent_lora",
        "stateful_early_stopping_clone",
    }:
        raise ValueError("Unsupported pilot.training_policy")
    early_stopping = (
        EarlyStoppingConfig.from_mapping(require_mapping(config, "training"))
        if pilot_training_policy == "stateful_early_stopping_clone"
        else None
    )
    optimizer_state_input = pilot.get("optimizer_state_path")

    run_dir = Path(args.run_dir)
    required = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_split_v2.json",
        "verification": run_dir / "verification" / "results.json",
        "synthetic": run_dir / "synthetic" / "accepted.jsonl",
        "synthetic_manifest": run_dir / "synthetic" / "manifest.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    output_dir = run_dir / "gain"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(output_dir)

    source = read_jsonl(required["predictions"])
    split = reconstruct_five_way_split(source, hans)
    validate_manifest(read_json(required["manifest"]), split)
    indexed = records_by_id(source)
    gain_eval_shards = partition_ids_by_round(
        split.gain_eval_ids,
        declared_rounds,
        seed=int(hans["split_seed"]),
        namespace="adaptive_gain_eval",
    )
    round_gain_ids = gain_eval_shards[round_index - 1]
    if not round_gain_ids:
        raise ValueError("The configured gain_eval split is too small for max_rounds")
    gain_eval = materialize_role(indexed, round_gain_ids, "gain_eval")
    stopping_examples: list[HANSExample] = []
    if pilot_training_policy == "stateful_early_stopping_clone":
        closed = require_mapping(config, "closed_loop")
        benchmark = require_mapping(config, "stage2_benchmark")
        frozen = FrozenHansManifest.from_mapping(
            read_json(Path(str(benchmark["frozen_manifest_path"])))
        )
        subset = read_json(Path(str(closed["stopping_subset_manifest"])))
        all_gain_rows = [indexed[item] for item in frozen.gain_eval_ids]
        stopping_ids, _ = validate_stopping_subset_manifest(
            subset,
            all_gain_rows,
            frozen.gain_eval_ids,
            frozen_manifest_hash=frozen.manifest_hash,
        )
        stopping_examples = _as_hans([indexed[item] for item in stopping_ids])
    hypotheses = _load_hypotheses(read_json(required["hypotheses"]))
    verification = read_json(required["verification"])
    if (
        verification.get("mode") != "llm_main"
        or verification.get("oracle_debug") is not False
        or verification.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION
        or verification.get("verification_mode")
        != "new_controlled_generated_probes"
        or verification.get("verification_probes_training_eligible") is not False
    ):
        raise ValueError("Verification used a different split version")
    certified_ids = _certified(verification)
    if len(certified_ids) > K:
        raise ValueError(f"Certified hypothesis count exceeds configured K={K}")
    unknown = sorted(set(certified_ids) - set(hypotheses))
    if unknown:
        raise ValueError(f"Unknown certified hypotheses: {unknown}")
    synthetic_manifest = read_json(required["synthetic_manifest"])
    if (
        synthetic_manifest.get("mode") != "llm_main"
        or synthetic_manifest.get("oracle_debug") is not False
        or synthetic_manifest.get("source_role") != "intervention_pool"
        or synthetic_manifest.get("training_label_space")
        != ["entailment", "neutral", "contradiction"]
        or synthetic_manifest.get("pair_schema", {}).get("selection_unit")
        != "complete_pair"
    ):
        raise ValueError("Synthetic manifest is not a valid main-method artifact")
    if synthetic_manifest.get("certified_hypothesis_ids") != certified_ids:
        raise ValueError("Synthetic manifest does not match certified hypotheses")
    synthetic = [
        SyntheticNLIExample.from_mapping(row)
        for row in read_jsonl(required["synthetic"], allow_empty=True)
    ]

    operationalizer = LLMHypothesisOperationalizer(teacher, op_config)
    checkpoint_before = checkpoint_descriptor(model)
    observations = []
    estimates = []
    applicability: dict[str, list[str]] = {}
    applicability_logs: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    batch_size = int(model.get("batch_size", 4))
    for hypothesis_id in certified_ids:
        hypothesis = hypotheses[hypothesis_id]
        operationalization = operationalizer.operationalize(
            hypothesis, gain_eval, expected_role="gain_eval"
        )
        applicability_logs.extend(operationalization.logs)
        applicable_ids = list(operationalization.applicable_ids)
        applicability[hypothesis_id] = applicable_ids
        eligible_eval = [
            record for record in gain_eval if str(record["id"]) in set(applicable_ids)
        ]
        eligible_train = [
            item for item in synthetic if item.hypothesis_id == hypothesis_id
        ]
        feasible = min(
            pilot_config.m,
            len(eligible_train) // pilot_config.pilot_train_size,
            len(eligible_eval) // pilot_config.gain_eval_size,
        )
        if not operationalization.testable or feasible < pilot_config.m:
            shortages.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "operationalization_status": operationalization.reason,
                    "accepted_synthetic": len(eligible_train),
                    "applicable_gain_eval": len(eligible_eval),
                    "requested_pilots": pilot_config.m,
                    "feasible_disjoint_pilots": feasible,
                }
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
            if pilot_training_policy == "independent_lora":
                return train_lora(student, examples, training, adapter_dir, seed)

            def validation_callback() -> dict[str, Any]:
                probabilities = student.predict_proba(
                    [item.premise for item in stopping_examples],
                    [item.hypothesis for item in stopping_examples],
                    batch_size=batch_size,
                )
                return _binary_validation(probabilities, stopping_examples)

            assert early_stopping is not None
            result = train_lora_stateful_early_stopping(
                student,
                examples,
                training,
                early_stopping,
                validation_callback,
                adapter_dir,
                seed,
                optimizer_state_input=optimizer_state_input,
                optimizer_state_output=adapter_dir.parent / "optimizer_state.pt",
            )
            write_json(
                adapter_dir.parent / "training_history.json",
                {
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "epochs_completed": result.epochs_completed,
                    "stopped_early": result.stopped_early,
                    "optimizer_step_start": result.optimizer_step_start,
                    "optimizer_step_end": result.optimizer_step_end,
                    "optimizer_step_last_attempted": (
                        result.optimizer_step_last_attempted
                    ),
                    "restore_to_best_verified": (
                        result.optimizer_step_end
                        <= result.optimizer_step_last_attempted
                    ),
                    "epochs": list(result.epoch_history),
                },
            )
            return result.total_loss

        def risk_callback(student: QwenNLIStudent, selected: Any) -> float:
            if pilot_training_policy == "stateful_early_stopping_clone":
                examples = _as_hans(selected)
                probabilities = student.predict_proba(
                    [item.premise for item in examples],
                    [item.hypothesis for item in examples],
                    batch_size=batch_size,
                )
                accuracy = _binary_validation(probabilities, examples)["accuracy"]
                return 1.0 - float(accuracy)
            _, metrics = evaluate_hans(
                student, _as_hans(selected), batch_size=batch_size
            )
            accuracy = metrics["overall_accuracy"]
            if accuracy is None:
                raise RuntimeError("Gain evaluation returned no accuracy")
            return 1.0 - float(accuracy)

        current = run_synthetic_pilots(
            hypothesis_id=hypothesis_id,
            synthetic_examples=eligible_train,
            applicable_gain_eval=eligible_eval,
            config=pilot_config,
            starting_checkpoint=checkpoint_before,
            training_config=asdict(training),
            pilot_root=output_dir / "pilots",
            student_factory=student_factory,
            train_callback=train_callback,
            risk_callback=risk_callback,
        )
        observations.extend(current)
        estimates.append(
            estimate_gain(
                hypothesis_id,
                [item.gain for item in current],
                M=pilot_config.M,
                delta=delta,
                K=K,
                T=T,
                tolerance=pilot_config.tolerance,
            )
        )
        for item in current:
            metadata_path = (
                output_dir
                / "pilots"
                / hypothesis_id
                / f"pilot_{item.pilot_index:03d}"
                / "metadata.json"
            )
            write_json(metadata_path, item.to_dict())
            if not preserve_adapters and item.adapter_path:
                shutil.rmtree(item.adapter_path)
        if checkpoint_descriptor(model) != checkpoint_before:
            raise RuntimeError("Starting checkpoint changed during pilot execution")

    write_jsonl(
        output_dir / "pilot_observations.jsonl",
        [item.to_dict() for item in observations],
    )
    write_json(
        output_dir / "gain_eval_applicability.json",
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "selection_is_outcome_independent": True,
            "prompt_version": op_config.prompt_version,
            "teacher": {
                "provider": teacher_config.get("provider", "huggingface"),
                "name_or_path": teacher_config.get("name_or_path"),
            },
            "applicable_ids": applicability,
        },
    )
    write_jsonl(
        run_dir / "teacher_calls" / "gain_eval_applicability.jsonl",
        applicability_logs,
    )
    write_json(
        output_dir / "gain_estimates.json",
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "status": (
                "no_certified_hypotheses"
                if not certified_ids
                else "estimated"
                if all(item.status == "estimated" for item in estimates)
                else "insufficient_data"
            ),
            "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
            "certified_hypothesis_ids": certified_ids,
            "pilot_config": asdict(pilot_config),
            "gain_bound": {"M": pilot_config.M, "delta": delta, "K": K, "T": T},
            "adaptive_round": round_index,
            "declared_max_rounds": declared_rounds,
            "gain_eval_round_ids": list(round_gain_ids),
            "gain_eval_reused_across_rounds": False,
            "starting_checkpoint": checkpoint_before,
            "starting_optimizer_state": {
                "path": optimizer_state_input,
                "local_sha256": hash_local_path(optimizer_state_input),
            },
            "pilot_training_policy": pilot_training_policy,
            "main_training_policy_matched": (
                pilot_training_policy == "stateful_early_stopping_clone"
            ),
            "starting_checkpoint_unchanged": checkpoint_descriptor(model) == checkpoint_before,
            "adapters_preserved": preserve_adapters,
            "data_shortages": shortages,
            "independence_limitation": (
                "The K*T union adjustment supports adaptive round selection only if "
                "each round's bounded pilot gains are conditionally independent with "
                "a stable conditional mean given prior history. Gain-eval shards and "
                "pilot batches are disjoint, but related LoRA runs and generated data "
                "may still violate that assumption; these are not causal intervals."
            ),
            "estimates": [item.to_dict() for item in estimates],
        },
    )
    print(json.dumps([item.to_dict() for item in estimates], indent=2))


if __name__ == "__main__":
    main()
