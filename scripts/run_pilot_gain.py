#!/usr/bin/env python3
"""Run independent debug-only targeted pilots for certified HANS hypotheses."""

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
from src.gain.estimation import estimate_gain
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.hypotheses.types import FailureHypothesis
from src.intervention.pilot import PilotRunConfig, run_targeted_pilots
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.utils.config import load_yaml, require_mapping
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pilot-gain YAML config")
    parser.add_argument("--run-dir", required=True, help="Verified oracle run directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this run's gain/ directory",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


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


def _load_hypotheses(payload: dict[str, Any]) -> dict[str, FailureHypothesis]:
    if payload.get("oracle_debug") is not True:
        raise ValueError("Hypotheses artifact must declare oracle_debug: true")
    loaded = {}
    for value in payload.get("hypotheses", []):
        hypothesis = FailureHypothesis(**value)
        if HANS_ORACLE_HYPOTHESES.get(hypothesis.heuristic) != hypothesis:
            raise ValueError(f"Unsupported or modified oracle hypothesis: {value}")
        loaded[hypothesis.id] = hypothesis
    return loaded


def _validate_manifest(manifest: dict[str, Any], split: Any) -> None:
    if manifest.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Pilot gain requires the versioned five-way HANS manifest")
    expected = {
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    for name, ids in expected.items():
        if manifest.get(name) != ids:
            raise ValueError(f"Manifest field {name!r} does not match reconstructed split")
    sets = [set(ids) for ids in expected.values()]
    if any(sets[left] & sets[right] for left in range(5) for right in range(left + 1, 5)):
        raise ValueError("Five-way HANS roles overlap")


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
    pilot = require_mapping(config, "pilot")
    gain = require_mapping(config, "gain")
    model = require_mapping(config, "model")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)
    pilot_config = PilotRunConfig(
        m=int(pilot.get("m", 4)),
        pilot_train_size=int(pilot.get("pilot_train_size", 32)),
        gain_eval_size=int(pilot.get("gain_eval_size", 64)),
        seed=int(pilot.get("seed", 42)),
        M=float(gain.get("M", 1.0)),
        tolerance=float(gain.get("tolerance", 1e-12)),
        non_entailment_label=str(pilot.get("non_entailment_label", "contradiction")),
    )
    delta = float(gain.get("delta", 0.05))
    K = int(gain.get("K", 3))
    T = int(gain.get("T", 1))
    if not 0.0 < delta < 1.0 or K <= 0 or T <= 0:
        raise ValueError("gain.delta must be in (0, 1), and K/T must be positive")
    preserve_adapters = bool(pilot.get("preserve_adapters", True))

    run_dir = Path(args.run_dir)
    paths = {
        "predictions": run_dir / "predictions.jsonl",
        "hypotheses": run_dir / "hypotheses.json",
        "manifest": run_dir / "hans_oracle_split_v2.json",
        "verification": run_dir / "verification" / "results.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    output_dir = run_dir / "gain"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}; pass --overwrite explicitly"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(paths["predictions"])
    if {record.get("split_role") for record in records} != {None}:
        raise ValueError("Source predictions must be the unsplit Milestone 3 artifact")
    split = split_hans_five_way(
        records,
        float(hans["discovery_fraction"]),
        float(hans["verification_fraction"]),
        float(hans["intervention_fraction"]),
        float(hans["gain_eval_fraction"]),
        int(hans["split_seed"]),
    )
    _validate_manifest(_read_json(paths["manifest"]), split)
    hypotheses = _load_hypotheses(_read_json(paths["hypotheses"]))
    verification = _read_json(paths["verification"])
    if verification.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Verification results were not produced from the five-way split")
    certified_ids = [
        str(result["hypothesis_id"])
        for result in verification.get("results", [])
        if result.get("certified") is True
    ]
    if len(certified_ids) != len(set(certified_ids)):
        raise ValueError("Verification results contain duplicate certified hypotheses")
    unknown = [item for item in certified_ids if item not in hypotheses]
    if unknown:
        raise ValueError(f"Certified hypotheses are absent from candidates: {unknown}")
    if len(certified_ids) > K:
        raise ValueError(f"Certified hypothesis count exceeds configured K={K}")

    checkpoint_before = checkpoint_descriptor(model)
    observations = []
    estimates = []
    shortages: list[dict[str, Any]] = []
    overlap_report: dict[str, Any] = {}
    batch_size = int(model.get("batch_size", 4))

    for hypothesis_id in certified_ids:
        hypothesis = hypotheses[hypothesis_id]
        eligible_train = sum(
            record["heuristic"] == hypothesis.heuristic
            for record in split.intervention_pool
        )
        eligible_eval = sum(
            record["heuristic"] == hypothesis.heuristic for record in split.gain_eval
        )
        if eligible_train < pilot_config.pilot_train_size or eligible_eval < pilot_config.gain_eval_size:
            shortages.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "eligible_intervention": eligible_train,
                    "requested_pilot_train_size": pilot_config.pilot_train_size,
                    "eligible_gain_eval": eligible_eval,
                    "requested_gain_eval_size": pilot_config.gain_eval_size,
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
            return train_lora(student, examples, training, adapter_dir, seed)

        def risk_callback(student: QwenNLIStudent, selected: Any) -> float:
            _, metrics = evaluate_hans(
                student, _as_hans(selected), batch_size=batch_size
            )
            accuracy = metrics["overall_accuracy"]
            if accuracy is None:
                raise RuntimeError("Gain evaluation returned no accuracy")
            return 1.0 - float(accuracy)

        current = run_targeted_pilots(
            hypothesis=hypothesis,
            intervention_records=split.intervention_pool,
            gain_eval_records=split.gain_eval,
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
        eval_sets = [set(item.eval_ids) for item in current]
        overlap_report[hypothesis_id] = {
            "pairwise_disjoint": not any(
                eval_sets[left] & eval_sets[right]
                for left in range(len(eval_sets))
                for right in range(left + 1, len(eval_sets))
            ),
            "num_available_matching_gain_eval": eligible_eval,
        }
        for item in current:
            write_json(
                output_dir
                / "pilots"
                / hypothesis_id
                / f"pilot_{item.pilot_index:03d}"
                / "metadata.json",
                item.to_dict(),
            )
            if not preserve_adapters:
                shutil.rmtree(item.adapter_path)
        if checkpoint_descriptor(model) != checkpoint_before:
            raise RuntimeError("Starting checkpoint changed during pilot execution")

    write_jsonl(
        output_dir / "pilot_observations.jsonl",
        [item.to_dict() for item in observations],
    )
    if not certified_ids:
        status = "no_certified_hypotheses"
    elif all(item.status == "estimated" for item in estimates):
        status = "estimated"
    else:
        status = "insufficient_data"
    write_json(
        output_dir / "gain_estimates.json",
        {
            "mode": "oracle_hans",
            "oracle_debug": True,
            "status": status,
            "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
            "certified_hypothesis_ids": certified_ids,
            "pilot_config": asdict(pilot_config),
            "gain_bound": {"M": pilot_config.M, "delta": delta, "K": K, "T": T},
            "starting_checkpoint": checkpoint_before,
            "starting_checkpoint_unchanged": checkpoint_descriptor(model) == checkpoint_before,
            "local_checkpoint_hash_available": any(
                checkpoint_before[field] is not None
                for field in ("model_local_sha256", "adapter_local_sha256")
            ),
            "adapters_preserved": preserve_adapters,
            "gain_eval_overlap": overlap_report,
            "data_shortages": shortages,
            "independence_limitation": (
                "The confidence radius treats pilot gains as independent bounded "
                "observations. Reusing a fixed HANS benchmark, overlapping eval "
                "sets when data are scarce, and related LoRA runs can violate this "
                "assumption; intervals are therefore debug estimates, not causal guarantees."
            ),
            "estimates": [item.to_dict() for item in estimates],
        },
    )
    print(json.dumps([item.to_dict() for item in estimates], indent=2, sort_keys=True))
    print(f"Wrote pilot-gain artifacts to {output_dir}")


if __name__ == "__main__":
    main()
