#!/usr/bin/env python3
"""Generate new controlled probes, query the student, and compute an LCB."""

from __future__ import annotations

import argparse
import sys
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
from src.teacher import build_teacher
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed
from src.verification.bounds import verify_binary_outcomes
from src.verification.generated_probes import (
    ControlledProbeConfig,
    ControlledVerificationProbeGenerator,
    TeacherVerificationProbeValidator,
    finalize_valid_probe_sets,
    predict_probe_sets,
)


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
    teacher = build_teacher(teacher_config)
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
        near_duplicate_threshold=float(
            verification.get("near_duplicate_threshold", 0.92)
        ),
        seed=int(verification.get("seed", 42)),
        prompt_version=str(
            verification.get(
                "probe_prompt_version", "controlled_verification_probes_v1"
            )
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
    )
    delta = float(verification["delta"])
    tau_v = float(verification["tau_v"])
    K = int(verification["max_hypotheses_per_round"])
    T = int(verification["max_rounds"])

    run_dir = Path(args.run_dir)
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
            "generated_probe_sets": len(generated) // 3,
            "valid_probe_sets": len(accepted) // 3,
            "rejected_probe_sets": len({row["probe_set_id"] for row in rejected}),
            "minimum_valid_probe_sets": probe_config.min_valid_probe_sets,
            "near_duplicate_threshold": probe_config.near_duplicate_threshold,
            "reliable_for_lcb": len(accepted) // 3
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
        all_outcomes.extend(outcomes)
        result = verify_binary_outcomes(
            hypothesis.id,
            [int(outcome["z"]) for outcome in outcomes],
            delta=delta,
            tau_v=tau_v,
            K=K,
            T=T,
        ).to_dict()
        if not design_metadata[hypothesis.id]["reliable_for_lcb"]:
            result["certified"] = False
            result["status"] = "unreliable_probe_design"
        result["probe_design"] = design_metadata[hypothesis.id]
        result["z_definition"] = (
            "target has the predeclared wrong binary label and both controls "
            "have their independently validated exact MNLI labels"
        )
        results.append(result)

    write_jsonl(
        output_dir / "generated_probes.jsonl",
        [probe.to_dict() for probe in all_generated],
    )
    write_jsonl(output_dir / "probes.jsonl", all_predictions)
    write_jsonl(output_dir / "rejected_probes.jsonl", all_rejected)
    write_jsonl(output_dir / "probe_set_outcomes.jsonl", all_outcomes)
    write_jsonl(run_dir / "teacher_calls" / "verification.jsonl", all_logs)
    write_json(
        output_dir / "results.json",
        {
            "mode": "llm_main",
            "oracle_debug": False,
            "verification_mode": "new_controlled_generated_probes",
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
                "probe sets. One-shot generation, scenario separation, controls, and "
                "pre-outcome validation reduce adaptivity but do not prove iid sampling."
            ),
            "results": results,
        },
    )
    print(f"Wrote controlled verification artifacts to {output_dir}")


if __name__ == "__main__":
    main()
