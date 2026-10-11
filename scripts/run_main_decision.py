#!/usr/bin/env python3
"""Freeze a cost-aware decision, then optionally run one full synthetic intervention."""

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
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION
from src.data.pipeline_split import (
    materialize_role,
    records_by_id,
    reconstruct_five_way_split,
    validate_manifest,
)
from src.decision.cost_aware import build_decision_candidates, select_intervention
from src.decision.artifact_cost import build_artifact_costs
from src.evaluation.hans import evaluate_hans
from src.hypotheses.types import FailureHypothesis
from src.intervention.synthetic import (
    run_synthetic_full_intervention,
    select_full_synthetic_examples,
)
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--defer-final-test",
        action="store_true",
        help="Do not expose final_test; used by the multi-round controller",
    )
    return parser.parse_args()


def _build_student(
    model: dict[str, Any], *, adapter_path: str | None = None
) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model.get("name_or_path", "Qwen/Qwen3-1.7B")),
        adapter_path=(model.get("adapter_path") if adapter_path is None else adapter_path),
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


def _validate_pilots(
    observations: list[dict[str, Any]],
    estimates: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, list[str]]:
    counts: dict[str, int] = {}
    used_train_ids: dict[str, list[str]] = {}
    all_train_ids: set[str] = set()
    for observation in observations:
        hypothesis_id = str(observation.get("hypothesis_id", ""))
        if not hypothesis_id or observation.get("starting_checkpoint") != checkpoint:
            raise ValueError("Pilot metadata has an invalid checkpoint or hypothesis")
        if observation.get("training_config") != training_config:
            raise ValueError("Pilot and full intervention LoRA recipes differ")
        raw_ids = observation.get("train_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("Pilot train_ids must be a list")
        train_ids = [str(item) for item in raw_ids]
        if all_train_ids & set(train_ids):
            raise ValueError("Synthetic pilot training sets overlap")
        all_train_ids.update(train_ids)
        used_train_ids.setdefault(hypothesis_id, []).extend(train_ids)
        counts[hypothesis_id] = counts.get(hypothesis_id, 0) + 1
    for estimate in estimates:
        if estimate.get("status") == "estimated":
            hypothesis_id = str(estimate.get("hypothesis_id", ""))
            if estimate.get("m") != counts.get(hypothesis_id, 0):
                raise ValueError("Gain estimate count does not match pilot records")
    return used_train_ids


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Main config must set mode: llm_main and oracle_debug: false")
    hans = require_mapping(config, "hans")
    decision_config = require_mapping(config, "decision")
    cost_config = require_mapping(config, "cost")
    intervention = require_mapping(config, "intervention")
    gain_config = require_mapping(config, "gain")
    model = require_mapping(config, "model")
    training = LoraTrainingConfig.from_mapping(require_mapping(config, "training"))
    cost_type = cost_config.get("type")
    if cost_type not in {"validated_synthetic_examples", "full_pipeline"}:
        raise ValueError(
            "Main method requires cost.type: full_pipeline or "
            "validated_synthetic_examples"
        )
    lambda_cost = float(decision_config["lambda_cost"])
    full_size = int(intervention["full_intervention_size"])
    seed = int(intervention.get("seed", 42))
    if full_size <= 0:
        raise ValueError("full_intervention_size must be positive")

    run_dir = Path(args.run_dir)
    required = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_split_v2.json",
        "verification": run_dir / "verification" / "results.json",
        "synthetic": run_dir / "synthetic" / "accepted.jsonl",
        "gains": run_dir / "gain" / "gain_estimates.json",
        "pilots": run_dir / "gain" / "pilot_observations.jsonl",
        "applicability": run_dir / "gain" / "gain_eval_applicability.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    source = read_jsonl(required["predictions"])
    split = reconstruct_five_way_split(source, hans)
    validate_manifest(read_json(required["manifest"]), split)
    indexed = records_by_id(source)
    # Only gain_eval is materialized before the frozen decision. final_test is
    # deliberately materialized after a full intervention completes.
    gain_eval = materialize_role(indexed, split.gain_eval_ids, "gain_eval")
    hypotheses = _load_hypotheses(read_json(required["hypotheses"]))
    verification = read_json(required["verification"])
    gains = read_json(required["gains"])
    applicability = read_json(required["applicability"])
    if (
        verification.get("mode") != "llm_main"
        or verification.get("oracle_debug") is not False
        or verification.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION
        or verification.get("verification_mode")
        != "new_controlled_generated_probes"
        or verification.get("verification_probes_training_eligible") is not False
    ):
        raise ValueError("Verification uses an incompatible split")
    if (
        gains.get("mode") != "llm_main"
        or gains.get("oracle_debug") is not False
        or gains.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION
    ):
        raise ValueError("Pilot gains use an incompatible split")
    if (
        applicability.get("mode") != "llm_main"
        or applicability.get("oracle_debug") is not False
        or applicability.get("selection_is_outcome_independent") is not True
    ):
        raise ValueError("Gain-eval applicability was not outcome-independent")
    raw_verification = verification.get("results")
    raw_estimates = gains.get("estimates")
    raw_applicable_ids = applicability.get("applicable_ids")
    if not isinstance(raw_verification, list) or not isinstance(raw_estimates, list):
        raise ValueError("Verification and gain artifacts require result lists")
    if not isinstance(raw_applicable_ids, dict):
        raise ValueError("Applicability artifact requires applicable_ids mapping")

    checkpoint = checkpoint_descriptor(model)
    if gains.get("starting_checkpoint") != checkpoint:
        raise ValueError("Pilot and full intervention checkpoints differ")
    observations = read_jsonl(required["pilots"], allow_empty=True)
    used_pilot_ids = _validate_pilots(
        observations, raw_estimates, checkpoint, asdict(training)
    )
    synthetic = [
        SyntheticNLIExample.from_mapping(row)
        for row in read_jsonl(required["synthetic"], allow_empty=True)
    ]
    full_sets = {
        hypothesis_id: select_full_synthetic_examples(
            synthetic,
            hypothesis_id=hypothesis_id,
            excluded_ids=used_pilot_ids.get(hypothesis_id, []),
            full_size=full_size,
            seed=seed,
        )
        for hypothesis_id in hypotheses
    }
    example_count_costs = {
        hypothesis_id: float(len(examples))
        for hypothesis_id, examples in full_sets.items()
        if examples
    }
    candidate_ids = list(example_count_costs)
    if cost_type == "full_pipeline":
        evaluation_counts = {
            hypothesis_id: len(raw_applicable_ids.get(hypothesis_id, []))
            if isinstance(raw_applicable_ids.get(hypothesis_id), list)
            else 0
            for hypothesis_id in candidate_ids
        }
        costs, cost_breakdowns = build_artifact_costs(
            run_dir=run_dir,
            hypothesis_ids=candidate_ids,
            full_training_counts={
                key: int(value) for key, value in example_count_costs.items()
            },
            evaluation_counts=evaluation_counts,
            pilot_observations=observations,
            cost_config=cost_config,
        )
    else:
        costs = example_count_costs
        cost_breakdowns = None
    # A certified hypothesis with no validated, non-pilot full data is not an
    # eligible intervention candidate and therefore cannot receive zero cost.
    eligible_verification = [
        item
        for item in raw_verification
        if isinstance(item, dict)
        and str(item.get("hypothesis_id", "")) in costs
    ]
    candidates = build_decision_candidates(
        eligible_verification,
        raw_estimates,
        costs,
        lambda_cost,
        cost_breakdowns=cost_breakdowns,
    )
    remaining_budget_value = decision_config.get("remaining_budget")
    remaining_budget = (
        None if remaining_budget_value is None else float(remaining_budget_value)
    )
    decision = select_intervention(
        candidates, lambda_cost, max_cost=remaining_budget
    )

    output_dir = run_dir / "decision"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(output_dir)
    write_json(
        output_dir / "candidates.json",
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "cost_type": cost_type,
            "cost_currency": cost_config.get("currency", "examples"),
            "requested_full_intervention_size": full_size,
            "lambda_cost": lambda_cost,
            "remaining_budget": remaining_budget,
            "candidates": [item.to_dict() for item in candidates],
        },
    )
    decision_payload = {
        "mode": "llm_main",
        "oracle_debug": False,
        **decision.to_dict(),
        "decision_rule": "continue iff max_h U(h) > 0",
        "full_intervention_committed_before_decision": False,
        "interpretation": (
            "A positive U is an optimistic eligibility rule under the configured "
            "bound assumptions, not a guarantee of causal or positive net gain."
        ),
    }
    write_json(output_dir / "decision.json", decision_payload)
    frozen_decision = (output_dir / "decision.json").read_bytes()
    if decision.action == "stop":
        print(json.dumps(decision_payload, indent=2, sort_keys=True))
        return

    hypothesis_id = decision.selected_hypothesis_id
    if hypothesis_id is None or hypothesis_id not in hypotheses:
        raise RuntimeError("CONTINUE did not select a known hypothesis")
    selected_train = full_sets[hypothesis_id]
    applicable_ids_value = raw_applicable_ids.get(hypothesis_id)
    if not isinstance(applicable_ids_value, list):
        raise ValueError("Selected hypothesis lacks gain-eval applicability IDs")
    applicable_id_set = {str(item) for item in applicable_ids_value}
    eligible_eval = [
        record for record in gain_eval if str(record["id"]) in applicable_id_set
    ]
    if len(eligible_eval) != len(applicable_id_set):
        raise ValueError("Applicability artifact contains a non-gain_eval ID")
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
        return train_lora(student, examples, training, adapter_dir, intervention_seed)

    def risk_callback(student: QwenNLIStudent, selected: Any) -> float:
        _, metrics = evaluate_hans(student, _as_hans(selected), batch_size=batch_size)
        accuracy = metrics["overall_accuracy"]
        if accuracy is None:
            raise RuntimeError("Post-intervention gain evaluation returned no accuracy")
        return 1.0 - float(accuracy)

    result = run_synthetic_full_intervention(
        decision=decision,
        training_examples=selected_train,
        applicable_gain_eval=eligible_eval,
        seed=seed,
        M=float(gain_config.get("M", 1.0)),
        tolerance=float(gain_config.get("tolerance", 1e-12)),
        starting_checkpoint=checkpoint,
        training_config=asdict(training),
        output_dir=output_dir / "full_intervention",
        student_factory=student_factory,
        train_callback=train_callback,
        risk_callback=risk_callback,
        expected_training_examples=len(selected_train),
    )
    if checkpoint_descriptor(model) != checkpoint:
        raise RuntimeError("Starting checkpoint changed during full intervention")
    write_json(
        output_dir / "full_intervention" / hypothesis_id / "training_metadata.json",
        {
            **result.to_dict(),
            "source_data_kind": "new_validated_synthetic_nli",
            "pilot_train_ids_excluded": used_pilot_ids.get(hypothesis_id, []),
            "pilot_adapter_reused": False,
            "starting_checkpoint_unchanged": True,
        },
    )
    write_json(
        output_dir / "post_intervention_metrics.json",
        {
            "mode": "llm_main",
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

    next_model = dict(model)
    next_model["adapter_path"] = result.adapter_path
    next_checkpoint = checkpoint_descriptor(next_model)
    write_json(output_dir / "next_checkpoint.json", next_checkpoint)
    if args.defer_final_test:
        if (output_dir / "decision.json").read_bytes() != frozen_decision:
            raise RuntimeError("Post-decision work modified the frozen decision")
        print(json.dumps(decision_payload, indent=2, sort_keys=True))
        return

    # final_test crosses the data firewall only now, after decision and training.
    final_records = materialize_role(indexed, split.final_test_ids, "final_test")
    final_student = _build_student(model, adapter_path=result.adapter_path)
    final_predictions, final_metrics = evaluate_hans(
        final_student, _as_hans(final_records), batch_size=batch_size
    )
    write_jsonl(output_dir / "final_test_predictions.jsonl", final_predictions)
    write_json(
        output_dir / "final_test_metrics.json",
        {
            "mode": "llm_main",
            "evaluation_role": "final_test",
            "evaluated_after_frozen_continue_and_full_training": True,
            **final_metrics,
        },
    )
    if (output_dir / "decision.json").read_bytes() != frozen_decision:
        raise RuntimeError("Post-decision work modified the frozen decision")
    print(json.dumps(decision_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
