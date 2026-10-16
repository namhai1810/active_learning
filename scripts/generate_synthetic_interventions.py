#!/usr/bin/env python3
"""Generate and quality-filter new NLI data for certified hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline_split import (
    materialize_role,
    records_by_id,
    reconstruct_five_way_split,
    validate_manifest,
)
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION
from src.hypotheses.types import FailureHypothesis
from src.synthetic.generator import SyntheticGenerationConfig, TargetedSyntheticGenerator
from src.synthetic.validator import (
    LeakageIndex,
    SyntheticValidationConfig,
    TeacherSyntheticQualityJudge,
    validate_synthetic_examples,
)
from src.teacher import PersistentJSONTeacherCache, build_teacher
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl


NO_CERTIFIED_STATUS = "abstained_no_certified_hypotheses"
NO_CERTIFIED_REASON = "no_certified_hypotheses"
NO_CERTIFIED_DETAIL = (
    "Controlled verification certified zero hypotheses; synthetic generation "
    "abstained fail-closed rather than training on unverified targets."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_candidates(payload: dict[str, Any]) -> dict[str, FailureHypothesis]:
    if payload.get("mode") != "llm_main" or payload.get("oracle_debug") is not False:
        raise ValueError("Expected main-method LLM hypotheses")
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        raise ValueError("hypotheses must be a list")
    candidates: dict[str, FailureHypothesis] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("Every hypothesis must be an object")
        hypothesis = FailureHypothesis.from_mapping(
            value, require_llm_schema=True
        )
        if hypothesis.id in candidates:
            raise ValueError(f"Duplicate hypothesis: {hypothesis.id}")
        candidates[hypothesis.id] = hypothesis
    return candidates


def _certified_ids(payload: dict[str, Any]) -> list[str]:
    if payload.get("mode") != "llm_main" or payload.get("oracle_debug") is not False:
        raise ValueError("Expected main-method verification results")
    if (
        payload.get("verification_mode") != "new_controlled_generated_probes"
        or payload.get("verification_probes_training_eligible") is not False
    ):
        raise ValueError("Synthetic generation requires controlled, training-ineligible probes")
    raw = payload.get("results")
    if not isinstance(raw, list):
        raise ValueError("Verification results must be a list")
    result = [
        str(item["hypothesis_id"])
        for item in raw
        if isinstance(item, dict) and item.get("certified") is True
    ]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate certified hypothesis IDs")
    return result


def _synthetic_manifest(
    *,
    stage2_policy: str,
    stage2_global_limit: int | None,
    minimum_accepted: int,
    maximum_generated: int,
    teacher_config: dict[str, Any],
    certified: list[str],
    generation_config: SyntheticGenerationConfig,
    validation_config: SyntheticValidationConfig,
    quality_prompt_version: str,
    quality_max_examples_per_call: int,
    quality_max_attempts_per_batch: int,
    generation_rounds: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
    abstained: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "llm_main",
        "oracle_debug": False,
        "stage2_policy": stage2_policy,
        "stage2_generation_limit_examples": stage2_global_limit,
        "stage2_min_accepted_examples": minimum_accepted,
        "stage2_max_generated_examples": maximum_generated,
        "verification_required": stage2_policy == "failure_targeted_verified",
        "binary_hans_labels_converted": False,
        "pair_selection_unit": "complete_pair",
        "teacher": {
            "provider": teacher_config.get("provider", "huggingface"),
            "name_or_path": teacher_config.get("name_or_path"),
        },
        "certified_hypothesis_ids": certified,
        "generation_config": {
            "counterfactual_pairs_per_hypothesis": generation_config.counterfactual_pairs_per_hypothesis,
            "invariant_pairs_per_hypothesis": generation_config.invariant_pairs_per_hypothesis,
            "reasoning_contrast_pairs_per_hypothesis": generation_config.reasoning_contrast_pairs_per_hypothesis,
            "total_pairs_per_hypothesis": generation_config.total_pairs,
            "total_examples_per_hypothesis": generation_config.total_examples,
            "max_seed_examples": generation_config.max_seed_examples,
            "seed_examples_per_call": generation_config.seed_examples_per_call,
            "max_pairs_per_call": generation_config.max_pairs_per_call,
            "max_attempts_per_batch": generation_config.max_attempts_per_batch,
            "seed": generation_config.seed,
            "prompt_version": generation_config.prompt_version,
        },
        "validation_config": {
            "near_duplicate_threshold": validation_config.near_duplicate_threshold,
            "min_sentence_tokens": validation_config.min_sentence_tokens,
            "max_sentence_tokens": validation_config.max_sentence_tokens,
            "quality_prompt_version": quality_prompt_version,
            "quality_max_examples_per_call": quality_max_examples_per_call,
            "quality_max_attempts_per_batch": quality_max_attempts_per_batch,
        },
        "source_role": "intervention_pool",
        "training_label_space": ["entailment", "neutral", "contradiction"],
        "pair_schema": {
            "pair_types": ["counterfactual", "invariant", "reasoning_contrast"],
            "members_per_pair": 2,
            "selection_unit": "complete_pair",
        },
        "heldout_text_exposed_to_teacher": False,
        "batch_checkpointing": {
            "enabled": True,
            "directory": "synthetic/checkpoints",
            "atomic_writes": True,
            "generation_and_quality_control": True,
        },
        "generation_rounds": generation_rounds,
        "final_test_materialized": False,
        "leakage_reference_roles": [
            "intervention_pool",
            "hypothesis_discovery",
            "verification",
            "verification_probe",
            "gain_eval",
        ],
        "counts": counts,
    }
    if abstained:
        payload.update(
            {
                "status": NO_CERTIFIED_STATUS,
                "abstained": True,
                "abstention_reason": NO_CERTIFIED_REASON,
                "abstention_detail": NO_CERTIFIED_DETAIL,
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Main config must set mode: llm_main and oracle_debug: false")
    hans = require_mapping(config, "hans")
    generation = require_mapping(config, "synthetic_generation")
    validation = require_mapping(config, "synthetic_validation")
    teacher_config = require_mapping(config, "teacher")
    stage2_policy = str(
        config.get("stage2_policy", "failure_targeted_verified")
    )
    allowed_policies = {
        "failure_targeted_verified",
        "failure_targeted_verification_bypassed",
        "random_non_failure_targeted",
    }
    if stage2_policy not in allowed_policies:
        raise ValueError(f"Unsupported stage2_policy: {stage2_policy}")
    run_dir = Path(args.run_dir)
    generation_config = SyntheticGenerationConfig(
        counterfactual_pairs_per_hypothesis=int(
            generation["counterfactual_pairs_per_hypothesis"]
        ),
        invariant_pairs_per_hypothesis=int(
            generation["invariant_pairs_per_hypothesis"]
        ),
        reasoning_contrast_pairs_per_hypothesis=int(
            generation.get("reasoning_contrast_pairs_per_hypothesis", 0)
        ),
        max_seed_examples=int(generation.get("max_seed_examples", 24)),
        seed_examples_per_call=int(generation.get("seed_examples_per_call", 4)),
        max_pairs_per_call=int(generation.get("max_pairs_per_call", 8)),
        max_attempts_per_batch=int(generation.get("max_attempts_per_batch", 3)),
        seed=int(generation.get("seed", 42)),
        prompt_version=str(
            generation.get("prompt_version", "targeted_synthetic_nli_pairs_v3")
        ),
    )
    validation_config = SyntheticValidationConfig(
        near_duplicate_threshold=float(
            validation.get("near_duplicate_threshold", 0.92)
        ),
        min_sentence_tokens=int(validation.get("min_sentence_tokens", 3)),
        max_sentence_tokens=int(validation.get("max_sentence_tokens", 80)),
    )
    quality_prompt_version = str(
        validation.get("prompt_version", "synthetic_pair_quality_control_v2")
    )
    quality_max_examples_per_call = int(
        validation.get("max_examples_per_call", 24)
    )
    quality_max_attempts_per_batch = int(
        validation.get("max_attempts_per_batch", 3)
    )

    required = {
        "predictions": run_dir / "predictions.jsonl",
        "manifest": run_dir / "hans_split_v2.json",
    }
    if stage2_policy != "random_non_failure_targeted":
        required["hypotheses"] = run_dir / "hypotheses.json"
    if stage2_policy == "failure_targeted_verified":
        required["verification"] = run_dir / "verification" / "results.json"
        required["verification_probes"] = (
            run_dir / "verification" / "generated_probes.jsonl"
        )
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
    output_dir = run_dir / "synthetic"
    output_paths = [
        output_dir / "generated.jsonl",
        output_dir / "accepted.jsonl",
        output_dir / "rejected.jsonl",
        output_dir / "manifest.json",
        run_dir / "teacher_calls" / "synthetic.jsonl",
    ]
    checkpoint_dir = output_dir / "checkpoints"
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        interrupted_final_write = (
            checkpoint_dir.is_dir() and len(existing) < len(output_paths)
        )
        if interrupted_final_write:
            print(
                "Resuming an interrupted final artifact write from batch checkpoints",
                flush=True,
            )
        else:
            raise FileExistsError(
                "Refusing to overwrite synthetic artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    source = read_jsonl(required["predictions"])
    split = reconstruct_five_way_split(source, hans)
    validate_manifest(read_json(required["manifest"]), split)
    indexed = records_by_id(source)
    intervention = materialize_role(
        indexed, split.intervention_pool_ids, "intervention_pool"
    )
    if stage2_policy == "random_non_failure_targeted":
        support_id = str(intervention[0]["id"])
        random_hypothesis = FailureHypothesis(
            id="stage2_random_synthetic",
            name="Random synthetic NLI coverage",
            description=(
                "Generate diverse controlled NLI pairs without targeting an "
                "observed student failure."
            ),
            predicted_failure_behavior="No failure behavior is targeted.",
            predicted_failure_labels=("entailment", "non-entailment"),
            verification_criteria=("Pair has valid explicit three-class NLI labels.",),
            intervention_target="Broad random NLI coverage with controlled pairs.",
            support_example_ids=(support_id,),
            source="stage2_random_baseline",
            prompt_version=generation_config.prompt_version,
        )
        hypotheses = {random_hypothesis.id: random_hypothesis}
        certified = [random_hypothesis.id]
        verification_payload = None
    else:
        hypotheses = _load_candidates(read_json(required["hypotheses"]))
        if stage2_policy == "failure_targeted_verified":
            verification_payload = read_json(required["verification"])
            if verification_payload.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
                raise ValueError("Verification used an incompatible split manifest")
            certified = _certified_ids(verification_payload)
        else:
            verification_payload = None
            certified = list(hypotheses)
        unknown = sorted(set(certified) - set(hypotheses))
        if unknown:
            raise ValueError(f"Selected hypotheses are missing from candidates: {unknown}")

    stage2_limit_value = config.get("stage2_generation_limit_examples")
    stage2_global_limit = (
        None if stage2_limit_value is None else int(stage2_limit_value)
    )
    if stage2_global_limit is not None:
        if stage2_global_limit <= 0 or stage2_global_limit % 2:
            raise ValueError("stage2_generation_limit_examples must be positive and even")
        if stage2_global_limit != generation_config.total_examples:
            raise ValueError(
                "Stage 2 global generation limit must equal the configured base "
                "pair schedule; it is divided across selected hypotheses"
            )
    minimum_accepted = int(config.get("stage2_min_accepted_examples", 0))
    maximum_generated = int(
        config.get(
            "stage2_max_generated_examples",
            stage2_global_limit or generation_config.total_examples,
        )
    )
    if minimum_accepted < 0 or minimum_accepted % 2:
        raise ValueError("stage2_min_accepted_examples must be non-negative and even")
    if maximum_generated <= 0 or maximum_generated % 2:
        raise ValueError("stage2_max_generated_examples must be positive and even")
    if maximum_generated < generation_config.total_examples:
        raise ValueError(
            "stage2_max_generated_examples cannot be smaller than the base schedule"
        )
    if minimum_accepted > maximum_generated:
        raise ValueError(
            "stage2_min_accepted_examples cannot exceed the generation safety cap"
        )
    if not certified and stage2_policy == "failure_targeted_verified":
        write_jsonl(output_dir / "generated.jsonl", [])
        write_jsonl(output_dir / "accepted.jsonl", [])
        write_jsonl(output_dir / "rejected.jsonl", [])
        write_jsonl(run_dir / "teacher_calls" / "synthetic.jsonl", [])
        write_json(
            output_dir / "manifest.json",
            _synthetic_manifest(
                stage2_policy=stage2_policy,
                stage2_global_limit=stage2_global_limit,
                minimum_accepted=minimum_accepted,
                maximum_generated=maximum_generated,
                teacher_config=teacher_config,
                certified=[],
                generation_config=generation_config,
                validation_config=validation_config,
                quality_prompt_version=quality_prompt_version,
                quality_max_examples_per_call=quality_max_examples_per_call,
                quality_max_attempts_per_batch=quality_max_attempts_per_batch,
                generation_rounds=[],
                counts={},
                abstained=True,
            ),
        )
        print(
            "Wrote fail-closed empty synthetic artifacts: controlled "
            "verification certified zero hypotheses",
            flush=True,
        )
        return
    if minimum_accepted and not certified:
        raise ValueError("No certified hypotheses are available for the required pool")

    # Do not construct or load the teacher until after the verified-policy
    # abstention above. A zero-certified run must make no synthetic teacher call.
    teacher = PersistentJSONTeacherCache(
        build_teacher(teacher_config),
        run_dir / "teacher_calls" / "response_cache",
    )
    quality_judge = TeacherSyntheticQualityJudge(
        teacher,
        seed=int(validation.get("seed", 42)),
        prompt_version=quality_prompt_version,
        max_examples_per_call=quality_max_examples_per_call,
        max_attempts_per_batch=quality_max_attempts_per_batch,
    )

    # Leakage references remain local comparison signatures. No held-out text or
    # signature is ever included in a teacher generation or QC request.
    leakage_records = [*intervention]
    leakage_records.extend(
        materialize_role(indexed, split.discovery_ids, "hypothesis_discovery")
    )
    leakage_records.extend(
        materialize_role(indexed, split.verification_ids, "verification")
    )
    leakage_records.extend(materialize_role(indexed, split.gain_eval_ids, "gain_eval"))
    verification_probes = (
        read_jsonl(required["verification_probes"], allow_empty=True)
        if "verification_probes" in required
        else []
    )
    leakage_records.extend(
        {
            "id": str(record["id"]),
            "premise": record["premise"],
            "hypothesis": record["hypothesis"],
            "split_role": "verification_probe",
        }
        for record in verification_probes
    )
    leakage_index = LeakageIndex.from_records(leakage_records)

    generated_all = []
    accepted_all = []
    rejected_all: list[dict[str, Any]] = []
    call_logs: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    quality_all = {}
    generation_rounds: list[dict[str, Any]] = []
    round_index = 0
    round_counterfactual_pairs = generation_config.counterfactual_pairs_per_hypothesis
    round_invariant_pairs = generation_config.invariant_pairs_per_hypothesis
    round_reasoning_pairs = generation_config.reasoning_contrast_pairs_per_hypothesis
    while True:
        round_start = len(generated_all)
        hypothesis_count = len(certified)
        for hypothesis_index, hypothesis_id in enumerate(certified):
            hypothesis = hypotheses[hypothesis_id]

            def allocated(total: int) -> int:
                return total // hypothesis_count + (hypothesis_index < total % hypothesis_count)

            counterfactual_pairs = allocated(round_counterfactual_pairs)
            invariant_pairs = allocated(round_invariant_pairs)
            reasoning_pairs = allocated(round_reasoning_pairs)
            if counterfactual_pairs <= 0 or invariant_pairs <= 0:
                raise ValueError(
                    "Every selected hypothesis must receive counterfactual and invariant pairs"
                )
            current_generation_config = replace(
                generation_config,
                counterfactual_pairs_per_hypothesis=counterfactual_pairs,
                invariant_pairs_per_hypothesis=invariant_pairs,
                reasoning_contrast_pairs_per_hypothesis=reasoning_pairs,
            )
            previous_for_hypothesis = [
                item for item in generated_all if item.hypothesis_id == hypothesis_id
            ]
            generator = TargetedSyntheticGenerator(
                teacher,
                current_generation_config,
                failure_targeted=stage2_policy != "random_non_failure_targeted",
            )
            checkpoint_key = hashlib.sha256(
                hypothesis.id.encode("utf-8")
            ).hexdigest()[:16]
            generation_checkpoint = (
                checkpoint_dir
                / f"{checkpoint_key}.round_{round_index:03d}.generation.json"
            )
            quality_checkpoint = (
                checkpoint_dir / f"{checkpoint_key}.round_{round_index:03d}.quality.json"
            )
            generation_resume = (
                read_json(generation_checkpoint)
                if generation_checkpoint.is_file()
                else None
            )
            generated, generation_log = generator.generate(
                hypothesis,
                intervention,
                generation_round=round_index,
                pair_ordinal_start=len(previous_for_hypothesis) // 2,
                avoid_examples=generated_all,
                resume_state=generation_resume,
                checkpoint_callback=lambda state, path=generation_checkpoint: write_json(
                    path, state
                ),
            )
            quality_resume = (
                read_json(quality_checkpoint) if quality_checkpoint.is_file() else None
            )
            quality, quality_log = quality_judge.judge(
                hypothesis,
                generated,
                resume_state=quality_resume,
                checkpoint_callback=lambda state, path=quality_checkpoint: write_json(
                    path, state
                ),
            )
            duplicate_quality_ids = set(quality_all) & set(quality)
            if duplicate_quality_ids:
                raise ValueError(
                    f"Duplicate synthetic quality IDs: {sorted(duplicate_quality_ids)}"
                )
            quality_all.update(quality)
            generated_all.extend(generated)
            call_logs.extend(generation_log["calls"])
            call_logs.extend(quality_log["calls"])

        accepted_all, rejected_all = validate_synthetic_examples(
            generated_all, quality_all, leakage_index, validation_config
        )
        generation_rounds.append(
            {
                "round": round_index,
                "generated_examples": len(generated_all) - round_start,
                "cumulative_generated_examples": len(generated_all),
                "cumulative_accepted_examples": len(accepted_all),
                "cumulative_rejected_examples": len(rejected_all),
            }
        )
        print(
            "Synthetic acceptance after round "
            f"{round_index}: {len(accepted_all)}/{len(generated_all)} examples",
            flush=True,
        )
        if minimum_accepted == 0 or len(accepted_all) >= minimum_accepted:
            break
        remaining_capacity = maximum_generated - len(generated_all)
        if remaining_capacity < 4 * hypothesis_count:
            raise ValueError(
                "Synthetic pool did not reach the required accepted count before "
                f"the generation cap: accepted={len(accepted_all)}, "
                f"required={minimum_accepted}, generated={len(generated_all)}, "
                f"cap={maximum_generated}"
            )
        deficit = minimum_accepted - len(accepted_all)
        observed_rate = len(accepted_all) / max(1, len(generated_all))
        planning_rate = max(0.25, observed_rate)
        planned_examples = max(
            64,
            2 * deficit,
            2 * math.ceil(deficit / planning_rate / 2),
            4 * hypothesis_count,
        )
        planned_examples = min(remaining_capacity, planned_examples)
        planned_examples -= planned_examples % 2
        planned_pairs = planned_examples // 2
        round_counterfactual_pairs = max(
            hypothesis_count, (planned_pairs + 1) // 2
        )
        round_invariant_pairs = planned_pairs - round_counterfactual_pairs
        if round_invariant_pairs < hypothesis_count:
            round_invariant_pairs = hypothesis_count
            round_counterfactual_pairs = planned_pairs - round_invariant_pairs
        round_reasoning_pairs = 0
        round_index += 1
    for hypothesis_id in certified:
        generated = [
            item for item in generated_all if item.hypothesis_id == hypothesis_id
        ]
        accepted = [
            item for item in accepted_all if item.hypothesis_id == hypothesis_id
        ]
        rejected = [
            item for item in rejected_all if item["hypothesis_id"] == hypothesis_id
        ]
        counts[hypothesis_id] = {
            "generated": len(generated),
            "generated_pairs": len({item.pair_id for item in generated}),
            "accepted": len(accepted),
            "accepted_pairs": len({item.pair_id for item in accepted}),
            "rejected": len(rejected),
            "rejected_pairs": len({str(item["pair_id"]) for item in rejected}),
        }

    write_jsonl(
        output_dir / "generated.jsonl", [item.to_dict() for item in generated_all]
    )
    write_jsonl(
        output_dir / "accepted.jsonl", [item.to_dict() for item in accepted_all]
    )
    write_jsonl(output_dir / "rejected.jsonl", rejected_all)
    write_jsonl(run_dir / "teacher_calls" / "synthetic.jsonl", call_logs)
    write_json(
        output_dir / "manifest.json",
        _synthetic_manifest(
            stage2_policy=stage2_policy,
            stage2_global_limit=stage2_global_limit,
            minimum_accepted=minimum_accepted,
            maximum_generated=maximum_generated,
            teacher_config=teacher_config,
            certified=certified,
            generation_config=generation_config,
            validation_config=validation_config,
            quality_prompt_version=quality_prompt_version,
            quality_max_examples_per_call=quality_max_examples_per_call,
            quality_max_attempts_per_batch=quality_max_attempts_per_batch,
            generation_rounds=generation_rounds,
            counts=counts,
        ),
    )
    print(f"Wrote validated synthetic artifacts to {output_dir}")


if __name__ == "__main__":
    main()
