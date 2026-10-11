#!/usr/bin/env python3
"""Make one cost-aware STOP/CONTINUE decision and at most one intervention."""

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
from src.decision.cost_aware import build_decision_candidates, select_intervention
from src.evaluation.hans import evaluate_hans
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.hypotheses.types import FailureHypothesis
from src.intervention.full import run_full_intervention
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml, require_mapping
from src.utils.io import write_json
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Cost-aware YAML config")
    parser.add_argument("--run-dir", required=True, help="Milestone 6 run directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this run's decision/ directory",
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
        raise ValueError("Cost-aware decision requires the five-way split manifest")
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


def _validate_pilot_integrity(
    estimates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    starting_checkpoint: dict[str, Any],
) -> None:
    observation_counts: dict[str, int] = {}
    for observation in observations:
        hypothesis_id = str(observation.get("hypothesis_id", ""))
        if not hypothesis_id:
            raise ValueError("Pilot observation is missing hypothesis_id")
        if observation.get("starting_checkpoint") != starting_checkpoint:
            raise ValueError("Pilot observation used a different starting checkpoint")
        observation_counts[hypothesis_id] = observation_counts.get(hypothesis_id, 0) + 1
    for estimate in estimates:
        if estimate.get("status") != "estimated":
            continue
        hypothesis_id = str(estimate.get("hypothesis_id", ""))
        if estimate.get("m") != observation_counts.get(hypothesis_id, 0):
            raise ValueError(
                f"Gain estimate m does not match pilot observations for {hypothesis_id}"
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


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("mode") != "oracle_hans" or config.get("oracle_debug") is not True:
        raise ValueError("Config must declare mode: oracle_hans and oracle_debug: true")
    hans = require_mapping(config, "hans")
    decision_config = require_mapping(config, "decision")
    cost_config = require_mapping(config, "cost")
    intervention_config = require_mapping(config, "intervention")
    gain_config = require_mapping(config, "gain")
    model = require_mapping(config, "model")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)

    if cost_config.get("type") != "acquired_examples":
        raise ValueError("Milestone 7 supports only cost.type: acquired_examples")
    lambda_cost = float(decision_config["lambda_cost"])
    full_size = int(intervention_config["full_intervention_size"])
    seed = int(intervention_config["seed"])
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
        raise ValueError("Source predictions must be the unsplit Milestone 3 artifact")
    # Partition ID-only views. Final-test prediction/correctness fields are
    # never materialized into the decision or intervention inputs.
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

    intervention_pool = materialize(
        split.intervention_pool_ids, "intervention_pool"
    )
    gain_eval = materialize(split.gain_eval_ids, "gain_eval")
    hypotheses = _load_hypotheses(_read_json(paths["hypotheses"]))
    verification = _read_json(paths["verification"])
    gains = _read_json(paths["gains"])
    if verification.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Verification results use the wrong split version")
    if gains.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Gain estimates use the wrong split version")

    current_checkpoint = checkpoint_descriptor(model)
    if gains.get("starting_checkpoint") != current_checkpoint:
        raise ValueError(
            "Configured starting checkpoint differs from the Milestone 6 checkpoint"
        )
    raw_estimates = gains.get("estimates")
    raw_verification = verification.get("results")
    if not isinstance(raw_estimates, list) or not isinstance(raw_verification, list):
        raise ValueError("Verification and gain artifacts must contain result lists")
    observations = _read_jsonl(paths["pilots"], allow_empty=True)
    _validate_pilot_integrity(raw_estimates, observations, current_checkpoint)

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
    candidates = build_decision_candidates(
        raw_verification,
        raw_estimates,
        costs,
        lambda_cost,
    )
    decision = select_intervention(candidates, lambda_cost)

    output_dir = run_dir / "decision"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "candidates.json",
        {
            "oracle_debug": True,
            "cost_type": "acquired_examples",
            "requested_full_intervention_size": full_size,
            "lambda_cost": lambda_cost,
            "candidates": [candidate.to_dict() for candidate in candidates],
        },
    )
    decision_payload = {
        "oracle_debug": True,
        **decision.to_dict(),
        "decision_rule": "continue iff max_h U(h) > 0",
        "interpretation": (
            "CONTINUE means the intervention cannot yet be ruled out as having "
            "positive net value; it is not guaranteed to be profitable."
        ),
    }
    write_json(output_dir / "decision.json", decision_payload)
    frozen_decision = (output_dir / "decision.json").read_bytes()

    if decision.action == "stop":
        print(json.dumps(decision_payload, indent=2, sort_keys=True))
        print(f"Wrote STOP decision to {output_dir}")
        return

    hypothesis_id = decision.selected_hypothesis_id
    if hypothesis_id is None or hypothesis_id not in hypotheses:
        raise RuntimeError("Continue decision did not resolve a known hypothesis")
    selected_hypothesis = hypotheses[hypothesis_id]
    batch_size = int(model.get("batch_size", 4))

    def student_factory(intervention_seed: int) -> QwenNLIStudent:
        set_deterministic_seed(intervention_seed)
        return _build_student(model)

    def train_callback(
        student: QwenNLIStudent,
        examples: list[Any],
        adapter_dir: Path,
        intervention_seed: int,
    ) -> float:
        return train_lora(
            student,
            examples,
            training,
            adapter_output_dir=adapter_dir,
            seed=intervention_seed,
        )

    def risk_callback(student: QwenNLIStudent, selected: Any) -> float:
        _, metrics = evaluate_hans(
            student, _as_hans(selected), batch_size=batch_size
        )
        accuracy = metrics["overall_accuracy"]
        if accuracy is None:
            raise RuntimeError("Post-intervention evaluation returned no accuracy")
        return 1.0 - float(accuracy)

    result = run_full_intervention(
        decision=decision,
        hypothesis=selected_hypothesis,
        intervention_records=intervention_pool,
        gain_eval_records=gain_eval,
        full_intervention_size=full_size,
        seed=seed,
        non_entailment_label=non_entailment_label,
        M=M,
        tolerance=tolerance,
        starting_checkpoint=current_checkpoint,
        training_config=asdict(training),
        output_dir=output_dir / "full_intervention",
        student_factory=student_factory,
        train_callback=train_callback,
        risk_callback=risk_callback,
    )
    checkpoint_after = checkpoint_descriptor(model)
    if checkpoint_after != current_checkpoint:
        raise RuntimeError("Starting checkpoint changed during full intervention")
    metadata = {
        **result.to_dict(),
        "starting_checkpoint_unchanged": True,
        "source_model_kind": "current_student_checkpoint",
        "pilot_adapter_reused": False,
    }
    write_json(
        output_dir
        / "full_intervention"
        / hypothesis_id
        / "training_metadata.json",
        metadata,
    )
    write_json(
        output_dir / "post_intervention_metrics.json",
        {
            "oracle_debug": True,
            "hypothesis_id": hypothesis_id,
            "evaluation_role": "gain_eval",
            "risk_before": result.risk_before,
            "risk_after": result.risk_after,
            "observed_full_gain": result.observed_full_gain,
            "num_eval_examples": len(result.eval_ids),
            "post_decision_only": True,
            "decision_recomputed": False,
        },
    )
    if (output_dir / "decision.json").read_bytes() != frozen_decision:
        raise RuntimeError("Post-intervention evaluation modified the frozen decision")
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote CONTINUE decision and one full intervention to {output_dir}")


if __name__ == "__main__":
    main()
