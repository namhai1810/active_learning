#!/usr/bin/env python3
"""Generate and quality-filter new NLI data for certified hypotheses."""

from __future__ import annotations

import argparse
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
from src.teacher import build_teacher
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl


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
    teacher = build_teacher(teacher_config)
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
        max_pairs_per_call=int(generation.get("max_pairs_per_call", 8)),
        seed=int(generation.get("seed", 42)),
        prompt_version=str(
            generation.get("prompt_version", "targeted_synthetic_nli_pairs_v2")
        ),
    )
    validation_config = SyntheticValidationConfig(
        near_duplicate_threshold=float(
            validation.get("near_duplicate_threshold", 0.92)
        ),
        min_sentence_tokens=int(validation.get("min_sentence_tokens", 3)),
        max_sentence_tokens=int(validation.get("max_sentence_tokens", 80)),
    )
    quality_judge = TeacherSyntheticQualityJudge(
        teacher,
        seed=int(validation.get("seed", 42)),
        prompt_version=str(
            validation.get("prompt_version", "synthetic_pair_quality_control_v2")
        ),
        max_examples_per_call=int(validation.get("max_examples_per_call", 24)),
    )

    run_dir = Path(args.run_dir)
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
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
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
    generated_all = []
    accepted_all = []
    rejected_all: list[dict[str, Any]] = []
    call_logs: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    quality_all = {}
    for hypothesis_index, hypothesis_id in enumerate(certified):
        hypothesis = hypotheses[hypothesis_id]
        current_generation_config = generation_config
        if stage2_global_limit is not None and certified:
            count = len(certified)
            current_generation_config = replace(
                generation_config,
                counterfactual_pairs_per_hypothesis=(
                    generation_config.counterfactual_pairs_per_hypothesis // count
                    + (hypothesis_index < generation_config.counterfactual_pairs_per_hypothesis % count)
                ),
                invariant_pairs_per_hypothesis=(
                    generation_config.invariant_pairs_per_hypothesis // count
                    + (hypothesis_index < generation_config.invariant_pairs_per_hypothesis % count)
                ),
                reasoning_contrast_pairs_per_hypothesis=(
                    generation_config.reasoning_contrast_pairs_per_hypothesis // count
                    + (hypothesis_index < generation_config.reasoning_contrast_pairs_per_hypothesis % count)
                ),
            )
        generator = TargetedSyntheticGenerator(
            teacher,
            current_generation_config,
            failure_targeted=stage2_policy != "random_non_failure_targeted",
        )
        generated, generation_log = generator.generate(hypothesis, intervention)
        quality, quality_log = quality_judge.judge(hypothesis, generated)
        duplicate_quality_ids = set(quality_all) & set(quality)
        if duplicate_quality_ids:
            raise ValueError(
                f"Duplicate synthetic quality IDs: {sorted(duplicate_quality_ids)}"
            )
        quality_all.update(quality)
        generated_all.extend(generated)
        call_logs.extend([generation_log, quality_log])

    # Held-out records are accessed only after teacher generation and converted
    # to comparison signatures for a leakage veto. Their text is never placed
    # in a generation or quality-control prompt.
    leakage_records = [*intervention]
    leakage_records.extend(
        materialize_role(indexed, split.discovery_ids, "hypothesis_discovery")
    )
    leakage_records.extend(
        materialize_role(indexed, split.verification_ids, "verification")
    )
    leakage_records.extend(materialize_role(indexed, split.gain_eval_ids, "gain_eval"))
    # Generated verification probes are local veto references only. Their text
    # is loaded after generation/QC and is never a training seed or teacher input.
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
    accepted_all, rejected_all = validate_synthetic_examples(
        generated_all, quality_all, leakage_index, validation_config
    )
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
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "stage2_policy": stage2_policy,
            "stage2_generation_limit_examples": stage2_global_limit,
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
                "max_pairs_per_call": generation_config.max_pairs_per_call,
                "seed": generation_config.seed,
                "prompt_version": generation_config.prompt_version,
            },
            "validation_config": {
                "near_duplicate_threshold": validation_config.near_duplicate_threshold,
                "min_sentence_tokens": validation_config.min_sentence_tokens,
                "max_sentence_tokens": validation_config.max_sentence_tokens,
                "quality_prompt_version": quality_judge.prompt_version,
                "quality_max_examples_per_call": quality_judge.max_examples_per_call,
            },
            "source_role": "intervention_pool",
            "training_label_space": ["entailment", "neutral", "contradiction"],
            "pair_schema": {
                "pair_types": ["counterfactual", "invariant", "reasoning_contrast"],
                "members_per_pair": 2,
                "selection_unit": "complete_pair",
            },
            "heldout_text_exposed_to_teacher": False,
            "final_test_materialized": False,
            "leakage_reference_roles": [
                "intervention_pool",
                "hypothesis_discovery",
                "verification",
                "verification_probe",
                "gain_eval",
            ],
            "counts": counts,
        },
    )
    print(f"Wrote validated synthetic artifacts to {output_dir}")


if __name__ == "__main__":
    main()
