#!/usr/bin/env python3
"""Build primary-method failure hypotheses with a configured teacher LLM."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline_split import (
    manifest_payload,
    materialize_role,
    records_by_id,
    reconstruct_five_way_split,
)
from src.hypotheses.llm_generator import LLMDiscoveryConfig, LLMFailureHypothesisGenerator
from src.hypotheses.error_scorer import ErrorScoringConfig, TeacherErrorProbabilityScorer
from src.teacher import PersistentJSONTeacherCache, build_teacher
from src.utils.artifacts import read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl


OUTPUT_FILES = (
    "hypotheses.json",
    "hypothesis_evidence.json",
    "hans_split_v2.json",
    "teacher_calls/discovery.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Main config must set mode: llm_main and oracle_debug: false")
    hans = require_mapping(config, "hans")
    generation = require_mapping(config, "hypothesis_generation")
    teacher_config = require_mapping(config, "teacher")
    discovery_config = LLMDiscoveryConfig(
        max_hypotheses=int(generation.get("max_hypotheses", 5)),
        min_hypotheses=int(generation.get("min_hypotheses", 1)),
        max_error_examples=int(generation.get("max_error_examples", 80)),
        max_contrast_examples=int(generation.get("max_contrast_examples", 20)),
        sampling_strategy=str(generation.get("sampling_strategy", "exponential")),
        sampling_alpha=float(generation.get("sampling_alpha", 0.5)),
        sampling_beta=float(generation.get("sampling_beta", 10.0)),
        sampling_threshold=float(generation.get("sampling_threshold", 0.7)),
        contrast_sampling_strategy=str(
            generation.get("contrast_sampling_strategy", "highest_error_score")
        ),
        diversity_sampling=bool(generation.get("diversity_sampling", False)),
        max_candidate_token_jaccard=float(
            generation.get("max_candidate_token_jaccard", 1.0)
        ),
        max_attempts_per_call=int(generation.get("max_attempts_per_call", 3)),
        seed=int(generation.get("seed", 42)),
        prompt_version=str(generation.get("prompt_version", "failure_discovery_v1")),
    )

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    output_paths = [run_dir / name for name in OUTPUT_FILES]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite main-method artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    source = read_jsonl(predictions_path)
    split = reconstruct_five_way_split(source, hans)
    indexed = records_by_id(source)
    discovery = materialize_role(
        indexed, split.discovery_ids, "hypothesis_discovery"
    )
    teacher = PersistentJSONTeacherCache(
        build_teacher(teacher_config),
        run_dir / "teacher_calls" / "response_cache",
    )
    scorer_config_value = require_mapping(generation, "error_scorer")
    if scorer_config_value.get("type", "teacher_llm") != "teacher_llm":
        raise ValueError("Only the explicit teacher_llm discovery error scorer is supported")
    scorer = TeacherErrorProbabilityScorer(
        teacher,
        ErrorScoringConfig(
            max_records_per_call=int(
                scorer_config_value.get("max_records_per_call", 40)
            ),
            max_attempts_per_batch=int(
                scorer_config_value.get("max_attempts_per_batch", 3)
            ),
            seed=int(scorer_config_value.get("seed", discovery_config.seed)),
            prompt_version=str(
                scorer_config_value.get(
                    "prompt_version", "discovery_error_probability_v1"
                )
            ),
            source=str(scorer_config_value.get("source", "teacher_llm_critic")),
        ),
    )
    generator = LLMFailureHypothesisGenerator(teacher, discovery_config, scorer)
    hypotheses, evidence, call_log = generator.generate(discovery)

    write_json(
        run_dir / "hypotheses.json",
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "split_role": "hypothesis_discovery",
            "generator": "llm_teacher",
            "teacher": {
                "provider": teacher_config.get("provider", "huggingface"),
                "name_or_path": teacher_config.get("name_or_path"),
            },
            "prompt_version": discovery_config.prompt_version,
            "generation_config": asdict(discovery_config),
            "hypotheses": [item.to_dict() for item in hypotheses],
        },
    )
    write_json(run_dir / "hypothesis_evidence.json", evidence)
    write_json(
        run_dir / "hans_split_v2.json",
        manifest_payload(split, hans, mode="llm_main"),
    )
    write_jsonl(run_dir / "teacher_calls" / "discovery.jsonl", [call_log])
    print(f"Wrote {len(hypotheses)} LLM hypothesis candidates to {run_dir}")


if __name__ == "__main__":
    main()
