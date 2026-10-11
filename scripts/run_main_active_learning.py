#!/usr/bin/env python3
"""Run the complete primary method as an adaptive multi-round AL loop."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import HANSExample
from src.benchmark.hans_manifest import FrozenHansManifest
from src.data.pipeline_split import reconstruct_five_way_split
from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config
from src.evaluation.hans import evaluate_hans
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer
from src.models.qwen_nli import QwenNLIStudent
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


STAGES = (
    "build_hypotheses.py",
    "verify_hypotheses.py",
    "generate_synthetic_interventions.py",
    "run_main_pilot_gain.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing evaluation directory containing predictions.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _student(model: dict[str, Any]) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model.get("name_or_path", "Qwen/Qwen3-1.7B")),
        adapter_path=model.get("adapter_path"),
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def _as_hans(rows: list[dict[str, Any]]) -> list[HANSExample]:
    return [
        HANSExample(
            id=str(row["id"]),
            premise=str(row["premise"]),
            hypothesis=str(row["hypothesis"]),
            gold=str(row["gold"]),
            heuristic=str(row["heuristic"]),
            subcase=str(row.get("subcase", "unknown")),
        )
        for row in rows
    ]


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _round_config(
    base: dict[str, Any], round_index: int, max_rounds: int, remaining: float | None
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["active_learning"]["current_round"] = round_index
    config["active_learning"]["max_rounds"] = max_rounds
    config["gain"]["T"] = max_rounds
    config["verification"]["max_rounds"] = max_rounds
    config["decision"]["remaining_budget"] = remaining
    # Fresh deterministic randomness is part of the conditional-on-history
    # round design. It also avoids regenerating identical probe/data sets.
    for section, keys in {
        "hypothesis_generation": ("seed",),
        "verification": ("seed", "label_validation_seed"),
        "synthetic_generation": ("seed",),
        "synthetic_validation": ("seed",),
        "pilot": ("seed", "applicability_seed"),
        "intervention": ("seed",),
    }.items():
        mapping = config.get(section)
        if isinstance(mapping, dict):
            for key in keys:
                if key in mapping:
                    mapping[key] = int(mapping[key]) + round_index - 1
    scorer = config.get("hypothesis_generation", {}).get("error_scorer")
    if isinstance(scorer, dict) and "seed" in scorer:
        scorer["seed"] = int(scorer["seed"]) + round_index - 1
    return config


def _prepare_predictions(
    source: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    round_index: int,
) -> list[dict[str, Any]]:
    hans = require_mapping(config, "hans")
    split = reconstruct_five_way_split(source, hans)
    final_ids = set(split.final_test_ids)
    non_final = [row for row in source if str(row["id"]) not in final_ids]
    if round_index > 1:
        set_deterministic_seed(int(hans["split_seed"]) + round_index - 1)
        student = _student(require_mapping(config, "model"))
        predictions, _ = evaluate_hans(
            student,
            _as_hans(non_final),
            batch_size=int(config["model"].get("batch_size", 4)),
        )
        indexed = {str(row["id"]): row for row in predictions}
        non_final = [indexed[str(row["id"])] for row in non_final]
    indexed_non_final = {str(row["id"]): row for row in non_final}
    # The final partition is represented by ID only. No final-test text, label,
    # prediction, or correctness is available to any adaptive round stage.
    return [
        {"id": str(row["id"])}
        if str(row["id"]) in final_ids
        else indexed_non_final[str(row["id"])]
        for row in source
    ]


def _run_stage(script: str, config_path: Path, round_dir: Path, *extra: str) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--config",
        str(config_path),
        "--run-dir",
        str(round_dir),
        *extra,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if config.get("mode") != "llm_main" or config.get("oracle_debug") is not False:
        raise ValueError("Config must declare mode: llm_main and oracle_debug: false")
    active = require_mapping(config, "active_learning")
    max_rounds = int(active["max_rounds"])
    if max_rounds <= 0:
        raise ValueError("active_learning.max_rounds must be positive")
    budget_value = active.get("max_budget")
    max_budget = None if budget_value is None else float(budget_value)
    if max_budget is not None and max_budget < 0:
        raise ValueError("active_learning.max_budget must be non-negative or null")
    if require_mapping(config, "cost").get("type") != "full_pipeline":
        raise ValueError("Multi-round execution requires cost.type: full_pipeline")

    run_dir = Path(args.run_dir).resolve()
    source_path = run_dir / "predictions.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {source_path}")
    source = read_jsonl(source_path)
    manifest_path_value = active.get("frozen_manifest_path")
    frozen_manifest = None
    if manifest_path_value is not None:
        frozen_manifest = FrozenHansManifest.from_mapping(
            read_json(Path(str(manifest_path_value)))
        )
        source_ids = {str(row["id"]) for row in source}
        manifest_ids = set().union(
            set(frozen_manifest.hypothesis_discovery_ids),
            set(frozen_manifest.verification_ids),
            set(frozen_manifest.intervention_pool_ids),
            set(frozen_manifest.gain_eval_ids),
            set(frozen_manifest.final_test_ids),
        )
        if source_ids != manifest_ids:
            raise ValueError("Source IDs differ from the frozen Stage 2 manifest")
    output = run_dir / "active_learning"
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output}; pass --overwrite explicitly"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    spent = 0.0
    current_model = copy.deepcopy(require_mapping(config, "model"))
    rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds_reached"
    for round_index in range(1, max_rounds + 1):
        remaining = None if max_budget is None else max(0.0, max_budget - spent)
        round_config = _round_config(config, round_index, max_rounds, remaining)
        round_config["model"] = copy.deepcopy(current_model)
        round_dir = output / "rounds" / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=False)
        config_path = round_dir / "config.yaml"
        _write_yaml(config_path, round_config)
        write_jsonl(
            round_dir / "predictions.jsonl",
            _prepare_predictions(source, round_config, round_index=round_index),
        )
        starting_checkpoint = checkpoint_descriptor(current_model)
        for stage in STAGES:
            _run_stage(stage, config_path, round_dir)
        _run_stage(
            "run_main_decision.py",
            config_path,
            round_dir,
            "--defer-final-test",
        )
        decision = read_json(round_dir / "decision" / "decision.json")
        round_cost = read_json(round_dir / "decision" / "round_cost.json")
        record: dict[str, Any] = {
            "round": round_index,
            "starting_checkpoint": starting_checkpoint,
            "decision": decision,
            "spent_before": spent,
            "cost": round_cost,
            "stages_completed": [
                "discover_hypotheses",
                "verify_failures",
                "generate_targeted_data",
                "estimate_pilot_gains",
                "stop_continue_decision",
            ],
        }
        if decision.get("action") == "stop":
            spent += float(round_cost["incurred_exploration_cost"])
            record["ending_checkpoint"] = starting_checkpoint
            record["spent_after"] = spent
            rounds.append(record)
            stop_reason = str(decision.get("reason", "stopped"))
            break
        selected_cost = decision.get("cost")
        if not isinstance(selected_cost, (int, float)) or isinstance(selected_cost, bool):
            raise ValueError("CONTINUE decision must contain a numeric cost")
        spent += float(round_cost["realized_round_cost_after_continue"])
        next_checkpoint = read_json(round_dir / "decision" / "next_checkpoint.json")
        current_model["adapter_path"] = next_checkpoint["adapter_path"]
        ending_checkpoint = checkpoint_descriptor(current_model)
        record["stages_completed"].append("update_student")
        record["ending_checkpoint"] = ending_checkpoint
        record["spent_after"] = spent
        rounds.append(record)
        if max_budget is not None and spent >= max_budget:
            stop_reason = "max_budget_reached"
            break

    write_jsonl(output / "rounds.jsonl", rounds)
    # The final test crosses the firewall exactly once, after every adaptive
    # decision and checkpoint update has been frozen.
    split = reconstruct_five_way_split(source, require_mapping(config, "hans"))
    indexed = {str(row["id"]): row for row in source}
    final_rows = [indexed[item] for item in split.final_test_ids]
    if any("premise" not in row or "gold" not in row for row in final_rows):
        sealed_value = active.get("sealed_final_test_path")
        if sealed_value is None:
            raise ValueError(
                "ID-only final rows require active_learning.sealed_final_test_path"
            )
        sealed = read_jsonl(Path(str(sealed_value)))
        sealed_index = {str(row["id"]): row for row in sealed}
        if set(sealed_index) != set(split.final_test_ids):
            raise ValueError("Sealed final-test IDs differ from the frozen split")
        if frozen_manifest is not None and set(sealed_index) != set(
            frozen_manifest.final_test_ids
        ):
            raise ValueError("Sealed final-test IDs differ from the shared manifest")
        final_rows = [sealed_index[item] for item in split.final_test_ids]
    final_checkpoint = checkpoint_descriptor(current_model)
    set_deterministic_seed(int(config["hans"]["split_seed"]))
    final_student = _student(current_model)
    final_timer = _start_gpu_timer(final_student)
    final_predictions, final_metrics = evaluate_hans(
        final_student,
        _as_hans(final_rows),
        batch_size=int(current_model.get("batch_size", 4)),
    )
    final_gpu_seconds = float(
        _stop_gpu_timer(final_student, final_timer) or 0.0
    )
    write_jsonl(output / "final_test_predictions.jsonl", final_predictions)
    write_json(
        output / "final_test_metrics.json",
        {
            "mode": "llm_main",
            "evaluation_role": "final_test",
            "evaluated_once_after_adaptive_loop": True,
            "final_checkpoint": final_checkpoint,
            **final_metrics,
        },
    )
    adaptive_cost = spent
    rates, currency = cost_rates_from_config(require_mapping(config, "cost"))
    final_test_cost = compute_cost(
        CostUsage(
            evaluation_runs=1,
            evaluation_items=len(final_rows),
            evaluation_gpu_seconds=final_gpu_seconds,
        ),
        rates,
        currency=currency,
    ).to_dict()
    spent += float(final_test_cost["total"])
    write_json(output / "final_test_cost.json", final_test_cost)
    summary = {
        "mode": "llm_main",
        "max_rounds": max_rounds,
        "rounds_attempted": len(rounds),
        "rounds_updated": sum(row["decision"]["action"] == "continue" for row in rounds),
        "max_budget": max_budget,
        "adaptive_modeled_cost": adaptive_cost,
        "final_test_evaluation_cost": float(final_test_cost["total"]),
        "total_modeled_cost": spent,
        "stop_reason": stop_reason,
        "final_checkpoint": final_checkpoint,
        "final_test_exposed_during_adaptation": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
