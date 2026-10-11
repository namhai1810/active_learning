#!/usr/bin/env python3
"""Run one controlled oracle-HANS ablation variant."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.runner import ABLATION_VARIANTS, evaluate_ablation
from src.data.hans import HANSExample
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION, split_hans_five_way
from src.evaluation.hans import evaluate_hans
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.hypotheses.types import FailureHypothesis
from src.intervention.full import run_full_intervention
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml, require_mapping
from src.utils.io import _atomic_text_write, write_json
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    parser.add_argument("--config", required=True, help="Shared ablation YAML config")
    parser.add_argument("--run-dir", required=True, help="Milestone 6 run directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected ablation variant directory",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
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


def _load_hypotheses(payload: dict[str, Any]) -> dict[str, FailureHypothesis]:
    if payload.get("oracle_debug") is not True:
        raise ValueError("Hypothesis artifact must declare oracle_debug: true")
    hypotheses = {}
    for value in payload.get("hypotheses", []):
        hypothesis = FailureHypothesis(**value)
        if HANS_ORACLE_HYPOTHESES.get(hypothesis.heuristic) != hypothesis:
            raise ValueError(f"Unsupported or modified oracle hypothesis: {value}")
        if hypothesis.id in hypotheses:
            raise ValueError(f"Duplicate hypothesis: {hypothesis.id}")
        hypotheses[hypothesis.id] = hypothesis
    return hypotheses


def _validate_manifest(manifest: dict[str, Any], split: Any) -> None:
    if manifest.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Ablations require the five-way HANS split manifest")
    expected = {
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    for field, ids in expected.items():
        if manifest.get(field) != ids:
            raise ValueError(f"Manifest field {field!r} does not match reconstructed split")


def _validate_pilot_artifacts(
    estimates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    starting_checkpoint: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    counts: dict[str, int] = {}
    for observation in observations:
        hypothesis_id = str(observation.get("hypothesis_id", ""))
        if not hypothesis_id:
            raise ValueError("Pilot observation is missing hypothesis_id")
        if observation.get("starting_checkpoint") != starting_checkpoint:
            raise ValueError("Pilot observation used a different starting checkpoint")
        if observation.get("training_config") != training_config:
            raise ValueError("Pilot observation used a different LoRA training recipe")
        counts[hypothesis_id] = counts.get(hypothesis_id, 0) + 1
    for estimate in estimates:
        if estimate.get("status") == "estimated":
            hypothesis_id = str(estimate.get("hypothesis_id", ""))
            if estimate.get("m") != counts.get(hypothesis_id, 0):
                raise ValueError(
                    f"Gain estimate m does not match observations for {hypothesis_id}"
                )


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
    examples = []
    for record in records:
        required = ("id", "premise", "hypothesis", "gold", "heuristic")
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"Gain-eval record is missing fields: {missing}")
        examples.append(
            HANSExample(
                id=str(record["id"]),
                premise=str(record["premise"]),
                hypothesis=str(record["hypothesis"]),
                gold=str(record["gold"]),
                heuristic=str(record["heuristic"]),
                subcase=str(record.get("subcase", "unknown")),
            )
        )
    return examples


def _update_summary(root: Path) -> list[dict[str, Any]]:
    records = []
    for variant in ABLATION_VARIANTS:
        path = root / variant / "metrics.json"
        if path.is_file():
            records.append(_read_json(path))
    write_json(root / "summary.json", {"oracle_debug": True, "results": records})
    headers = (
        "variant",
        "certified",
        "selected_h",
        "action",
        "acquired",
        "gain_before",
        "gain_after",
        "observed_gain",
    )
    rows = [headers]
    for record in records:
        def fmt(value: Any) -> str:
            if value is None:
                return "-"
            return f"{value:.4f}" if isinstance(value, float) else str(value)

        rows.append(
            (
                record["variant"],
                str(record["num_certified_hypotheses"]),
                record["selected_hypothesis_id"] or "-",
                record["action"],
                str(record["num_acquired"]),
                fmt(record["gain_eval_before"]),
                fmt(record["gain_eval_after"]),
                fmt(record["observed_gain"]),
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    for row_index, row in enumerate(rows):
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("-+-".join("-" * width for width in widths))
    return records


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("mode") != "oracle_hans" or config.get("oracle_debug") is not True:
        raise ValueError("Config must declare mode: oracle_hans and oracle_debug: true")
    hans = require_mapping(config, "hans")
    ablation = require_mapping(config, "ablation")
    decision_config = require_mapping(config, "decision")
    cost_config = require_mapping(config, "cost")
    intervention_config = require_mapping(config, "intervention")
    pilot_config = require_mapping(config, "pilot")
    gain_config = require_mapping(config, "gain")
    model = require_mapping(config, "model")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)
    if cost_config.get("type") != "acquired_examples":
        raise ValueError("Milestone 8 supports only cost.type: acquired_examples")

    seed = int(ablation["seed"])
    lambda_cost = float(decision_config["lambda_cost"])
    full_size = int(intervention_config["full_intervention_size"])
    intervention_seed = int(intervention_config["seed"])
    non_entailment_label = str(
        intervention_config.get("non_entailment_label", "contradiction")
    )
    M = float(gain_config.get("M", 1.0))
    tolerance = float(gain_config.get("tolerance", 1e-12))
    if full_size <= 0:
        raise ValueError("full_intervention_size must be positive")

    run_dir = Path(args.run_dir)
    paths = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_oracle_split_v2.json",
        "verification": run_dir / "verification" / "results.json",
        "gains": run_dir / "gain" / "gain_estimates.json",
        "pilots": run_dir / "gain" / "pilot_observations.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    records = _read_jsonl(paths["predictions"])
    if {record.get("split_role") for record in records} != {None}:
        raise ValueError("Source predictions must be an unsplit Milestone 3 artifact")
    split = split_hans_five_way(
        [{"id": record["id"]} for record in records],
        float(hans["discovery_fraction"]),
        float(hans["verification_fraction"]),
        float(hans["intervention_fraction"]),
        float(hans["gain_eval_fraction"]),
        int(hans["split_seed"]),
    )
    _validate_manifest(_read_json(paths["manifest"]), split)
    records_by_id = {str(record["id"]): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Source predictions contain duplicate IDs")

    def materialize(ids: tuple[str, ...], role: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {**records_by_id[example_id], "split_role": role}
            for example_id in ids
        )

    intervention_pool = materialize(split.intervention_pool_ids, "intervention_pool")
    gain_eval = materialize(split.gain_eval_ids, "gain_eval")
    hypotheses = _load_hypotheses(_read_json(paths["hypotheses"]))
    verification = _read_json(paths["verification"])
    gains = _read_json(paths["gains"])
    if verification.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Verification artifact uses the wrong split version")
    if gains.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Gain artifact uses the wrong split version")
    current_checkpoint = checkpoint_descriptor(model)
    if gains.get("starting_checkpoint") != current_checkpoint:
        raise ValueError("Ablation checkpoint differs from the Milestone 6 checkpoint")
    raw_verification = verification.get("results")
    raw_estimates = gains.get("estimates")
    if not isinstance(raw_verification, list) or not isinstance(raw_estimates, list):
        raise ValueError("Verification and gain artifacts must contain result lists")
    observations = _read_jsonl(paths["pilots"], allow_empty=True)
    serialized_training = json.loads(json.dumps(asdict(training)))
    _validate_pilot_artifacts(
        raw_estimates,
        observations,
        current_checkpoint,
        serialized_training,
    )

    recorded_pilot = gains.get("pilot_config")
    if not isinstance(recorded_pilot, dict):
        raise ValueError("Gain artifact is missing pilot_config")
    fairness_fields = (
        "m",
        "pilot_train_size",
        "gain_eval_size",
        "seed",
        "non_entailment_label",
    )
    for field in fairness_fields:
        if recorded_pilot.get(field) != pilot_config.get(field):
            raise ValueError(f"Shared pilot config mismatch for {field}")
    if recorded_pilot.get("M") != M or recorded_pilot.get("tolerance") != tolerance:
        raise ValueError("Shared pilot gain bound configuration does not match")

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
    outcome = evaluate_ablation(
        variant=args.variant,
        hypothesis_ids=list(hypotheses),
        verification_results=raw_verification,
        gain_estimates=raw_estimates,
        costs=costs,
        lambda_cost=lambda_cost,
        seed=seed,
    )

    ablation_root = run_dir / "ablations"
    variant_dir = ablation_root / args.variant
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {variant_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {**config, "variant": args.variant}
    _atomic_text_write(
        variant_dir / "config.yaml",
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
    )
    write_json(
        variant_dir / "candidates.json",
        {
            "oracle_debug": True,
            "variant": args.variant,
            "num_candidate_hypotheses": len(hypotheses),
            "accepted_hypothesis_ids": list(outcome.accepted_hypothesis_ids),
            "lambda_cost": lambda_cost,
            "cost_type": "acquired_examples",
            "candidates": [item.to_dict() for item in outcome.candidates],
        },
    )
    decision_payload = {
        "oracle_debug": True,
        "variant": args.variant,
        **outcome.decision.to_dict(),
    }
    write_json(variant_dir / "decision.json", decision_payload)
    frozen_decision = (variant_dir / "decision.json").read_bytes()

    result = None
    selected_ids: list[str] = []
    if outcome.decision.action == "continue":
        hypothesis_id = outcome.decision.selected_hypothesis_id
        if hypothesis_id is None or hypothesis_id not in hypotheses:
            raise RuntimeError("Continue decision did not resolve a hypothesis")
        selected = hypotheses[hypothesis_id]
        batch_size = int(model.get("batch_size", 4))

        def student_factory(full_seed: int) -> QwenNLIStudent:
            set_deterministic_seed(full_seed)
            return _build_student(model)

        def train_callback(
            student: QwenNLIStudent,
            examples: list[Any],
            adapter_dir: Path,
            full_seed: int,
        ) -> float:
            return train_lora(student, examples, training, adapter_dir, full_seed)

        def risk_callback(student: QwenNLIStudent, selected_records: Any) -> float:
            _, metrics = evaluate_hans(
                student, _as_hans(selected_records), batch_size=batch_size
            )
            accuracy = metrics["overall_accuracy"]
            if accuracy is None:
                raise RuntimeError("Gain evaluation returned no accuracy")
            return 1.0 - float(accuracy)

        result = run_full_intervention(
            decision=outcome.decision,
            hypothesis=selected,
            intervention_records=intervention_pool,
            gain_eval_records=gain_eval,
            full_intervention_size=full_size,
            seed=intervention_seed,
            non_entailment_label=non_entailment_label,
            M=M,
            tolerance=tolerance,
            starting_checkpoint=current_checkpoint,
            training_config=asdict(training),
            output_dir=variant_dir / "full_intervention",
            student_factory=student_factory,
            train_callback=train_callback,
            risk_callback=risk_callback,
        )
        selected_ids = result.train_ids
        if checkpoint_descriptor(model) != current_checkpoint:
            raise RuntimeError("Starting checkpoint changed during ablation")
        write_json(
            variant_dir
            / "full_intervention"
            / hypothesis_id
            / "training_metadata.json",
            {
                **result.to_dict(),
                "variant": args.variant,
                "starting_checkpoint_unchanged": True,
                "pilot_adapter_reused": False,
            },
        )
        if (variant_dir / "decision.json").read_bytes() != frozen_decision:
            raise RuntimeError("Post-intervention evaluation changed the decision")

    write_json(
        variant_dir / "selected_example_ids.json",
        {
            "variant": args.variant,
            "hypothesis_id": outcome.decision.selected_hypothesis_id,
            "budget": full_size,
            "ids": selected_ids,
        },
    )
    metrics = {
        "variant": args.variant,
        "seed": seed,
        "num_candidate_hypotheses": len(hypotheses),
        "num_certified_hypotheses": len(outcome.accepted_hypothesis_ids),
        "action": outcome.decision.action,
        "selected_hypothesis_id": outcome.decision.selected_hypothesis_id,
        "num_acquired": 0 if result is None else result.cost,
        "gain_eval_before": None if result is None else 1.0 - result.risk_before,
        "gain_eval_after": None if result is None else 1.0 - result.risk_after,
        "observed_gain": None if result is None else result.observed_full_gain,
        "estimated_gain": outcome.decision.delta_hat,
        "gain_confidence_radius": outcome.decision.confidence_radius,
        "cost": outcome.decision.cost,
        "optimistic_net_value": outcome.decision.optimistic_net_value,
    }
    write_json(variant_dir / "metrics.json", metrics)
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    _update_summary(ablation_root)
    print(f"Wrote ablation artifacts to {variant_dir}")


if __name__ == "__main__":
    main()
