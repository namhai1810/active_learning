#!/usr/bin/env python3
"""Run fixed-budget Stage-2 training with checkpoint-conditioned verification."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.closed_loop import (
    merge_hypothesis_candidates,
    verification_states,
)
from src.benchmark.hans_manifest import FrozenHansManifest, audit_final_test_overlap
from src.benchmark.selection import select_closed_loop_soft_pairs
from src.benchmark.stopping_subset import validate_stopping_subset_manifest
from src.data.hans import HANSExample
from src.data.pipeline_split import manifest_payload, reconstruct_five_way_split
from src.data.labels import MNLI_LABELS
from src.data.mnli import load_mnli
from src.evaluation.hans import evaluate_hans
from src.evaluation.mnli import evaluate_mnli
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _student(model: dict[str, Any], adapter: str | None) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=adapter,
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def _clear_cuda_cache() -> None:
    """Return unreferenced model storage to CUDA after the caller deletes it."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _hans(rows: list[dict[str, Any]]) -> list[HANSExample]:
    return [
        HANSExample(
            id=str(row["id"]),
            premise=str(row["premise"]),
            hypothesis=str(row["hypothesis"]),
            gold=str(row["gold"]),
            heuristic=str(row["heuristic"]),
            subcase=str(row["subcase"]),
        )
        for row in rows
    ]


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    _atomic_text_write(path, yaml.safe_dump(value, sort_keys=False))


def _run_stage(script: str, config: Path, run_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / script),
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _pair_tokens(group: list[SyntheticNLIExample]) -> frozenset[str]:
    import re

    return frozenset(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                f"{item.premise} {item.hypothesis}" for item in group
            ).lower(),
        )
    )


def _merge_pool(
    current: list[SyntheticNLIExample],
    additions: list[SyntheticNLIExample],
    *,
    threshold: float,
) -> tuple[list[SyntheticNLIExample], dict[str, int]]:
    def groups(rows: list[SyntheticNLIExample]) -> list[list[SyntheticNLIExample]]:
        indexed: dict[str, list[SyntheticNLIExample]] = {}
        for item in rows:
            indexed.setdefault(item.pair_id, []).append(item)
        if any(len(value) != 2 for value in indexed.values()):
            raise ValueError("Closed-loop pool contains an incomplete pair")
        return list(indexed.values())

    merged = list(current)
    known_ids = {item.id for item in current}
    known_pair_ids = {item.pair_id for item in current}
    signatures = [_pair_tokens(group) for group in groups(current)]
    rejected = 0
    accepted_pairs = 0
    for group in groups(additions):
        signature = _pair_tokens(group)
        duplicate = bool(
            known_pair_ids & {item.pair_id for item in group}
            or known_ids & {item.id for item in group}
        )
        if not duplicate:
            for prior in signatures:
                union = signature | prior
                similarity = len(signature & prior) / len(union) if union else 1.0
                if similarity >= threshold:
                    duplicate = True
                    break
        if duplicate:
            rejected += 1
            continue
        merged.extend(group)
        known_ids.update(item.id for item in group)
        known_pair_ids.add(group[0].pair_id)
        signatures.append(signature)
        accepted_pairs += 1
    return merged, {
        "accepted_new_pairs": accepted_pairs,
        "rejected_cross_round_duplicate_pairs": rejected,
        "cumulative_pairs": len(merged) // 2,
    }


def _round_config(
    base: dict[str, Any], *, round_index: int, adapter: str | None
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["model"]["adapter_path"] = adapter
    config["model"]["device_map"] = os.environ.get(
        "CLOSED_LOOP_STUDENT_DEVICE", "cuda:1"
    )
    config["teacher"]["device_map"] = os.environ.get(
        "CLOSED_LOOP_TEACHER_DEVICE", "cuda:0"
    )
    for section, keys in {
        "hypothesis_generation": ("seed",),
        "verification": ("seed", "label_validation_seed"),
        "synthetic_generation": ("seed",),
        "synthetic_validation": ("seed",),
    }.items():
        value = config.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    value[key] = int(value[key]) + round_index - 1
    scorer = config.get("hypothesis_generation", {}).get("error_scorer")
    if isinstance(scorer, dict) and "seed" in scorer:
        scorer["seed"] = int(scorer["seed"]) + round_index - 1
    return config


def _prepare_current_predictions(
    shared_rows: list[dict[str, Any]],
    manifest: FrozenHansManifest,
    model: dict[str, Any],
    adapter: str | None,
) -> tuple[list[dict[str, Any]], int]:
    indexed = {str(row["id"]): dict(row) for row in shared_rows}
    discovery_rows = [indexed[item] for item in manifest.hypothesis_discovery_ids]
    student = _student(model, adapter)
    student.progress_label = "closed-loop discovery re-evaluation"
    predictions, _ = evaluate_hans(
        student, _hans(discovery_rows), batch_size=int(model.get("batch_size", 4))
    )
    del student
    _clear_cuda_cache()
    for row in predictions:
        indexed[str(row["id"])] = row
    final_ids = set(manifest.final_test_ids)
    output = [
        {"id": str(row["id"])}
        if str(row["id"]) in final_ids
        else indexed[str(row["id"])]
        for row in shared_rows
    ]
    errors = sum(
        not bool(indexed[item]["correct"])
        for item in manifest.hypothesis_discovery_ids
    )
    return output, errors


def _write_round_split_manifest(
    path: Path,
    predictions: list[dict[str, Any]],
    hans_config: dict[str, Any],
) -> None:
    """Write the online five-way manifest expected by Stage-2 subprocesses.

    The benchmark's frozen manifest and the online pipeline manifest describe
    the same partition but deliberately use different schemas.  Refresh rounds
    must materialize the latter even when hypothesis discovery is skipped.
    """
    split = reconstruct_five_way_split(predictions, hans_config)
    write_json(path, manifest_payload(split, hans_config, mode="llm_main"))


def _skipped_discovery_payload(reason: str) -> dict[str, Any]:
    """Preserve discovery provenance when a refresh reuses archived candidates."""
    return {
        "mode": "llm_main",
        "oracle_debug": False,
        "split_role": "hypothesis_discovery",
        "generator": reason,
        "hypotheses": [],
    }


def _refresh_hypotheses(
    *,
    output: Path,
    base_config: dict[str, Any],
    round_index: int,
    adapter: str | None,
    shared_rows: list[dict[str, Any]],
    manifest: FrozenHansManifest,
    archive: list[dict[str, Any]],
    active_ids: list[str],
    generate_pool: bool,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, dict[str, Any]],
    list[SyntheticNLIExample],
    dict[str, Any],
]:
    closed = require_mapping(base_config, "closed_loop")
    round_dir = output / "refresh" / f"round_{round_index:03d}"
    round_dir.mkdir(parents=True)
    config = _round_config(base_config, round_index=round_index, adapter=adapter)
    config_path = round_dir / "config.yaml"
    _write_yaml(config_path, config)
    predictions, discovery_errors = _prepare_current_predictions(
        shared_rows, manifest, config["model"], adapter
    )
    write_jsonl(round_dir / "predictions.jsonl", predictions)

    maximum_candidates = int(closed.get("max_hypotheses_per_round", 5))
    proposal_capacity = max(0, maximum_candidates - len(set(active_ids)))
    novel_search_performed = discovery_errors == 0 or proposal_capacity > 0
    if discovery_errors and proposal_capacity > 0:
        _run_stage("build_hypotheses.py", config_path, round_dir)
        proposal_payload = read_json(round_dir / "hypotheses.json")
        proposals = [dict(item) for item in proposal_payload["hypotheses"]]
    else:
        proposal_payload = _skipped_discovery_payload(
            (
                "skipped_no_discovery_errors"
                if discovery_errors == 0
                else "skipped_active_candidate_capacity_full"
            )
        )
        proposals = []
        _write_round_split_manifest(
            round_dir / "hans_split_v2.json",
            predictions,
            require_mapping(config, "hans"),
        )

    candidates, merge_log = merge_hypothesis_candidates(
        archive,
        proposals,
        active_ids=active_ids,
        maximum=maximum_candidates,
        near_duplicate_threshold=float(
            closed.get("hypothesis_near_duplicate_threshold", 0.78)
        ),
    )
    proposal_payload["hypotheses"] = candidates
    proposal_payload["closed_loop_merge"] = merge_log
    write_json(round_dir / "hypotheses.json", proposal_payload)
    write_json(round_dir / "hypothesis_merge.json", merge_log)
    if not (round_dir / "hans_split_v2.json").is_file():
        raise RuntimeError("Discovery stage did not write its split manifest")

    if candidates:
        _run_stage("verify_hypotheses.py", config_path, round_dir)
        verification = read_json(round_dir / "verification" / "results.json")
        states = verification_states(verification)
    else:
        states = {}
    next_active = sorted(
        hypothesis_id
        for hypothesis_id, value in states.items()
        if value["state"] in {"certified", "unresolved"}
    )
    archive_by_id = {str(item["id"]): dict(item) for item in archive}
    for item in candidates:
        archive_by_id[str(item["id"])] = dict(item)
    next_archive = list(archive_by_id.values())

    additions: list[SyntheticNLIExample] = []
    if generate_pool and next_active:
        _run_stage("generate_synthetic_interventions.py", config_path, round_dir)
        additions = [
            SyntheticNLIExample.from_mapping(row)
            for row in read_jsonl(round_dir / "synthetic" / "accepted.jsonl")
        ]
    lifecycle = {
        "refresh_round": round_index,
        "checkpoint_adapter": adapter,
        "discovery_errors": discovery_errors,
        "proposal_capacity_before_verification": proposal_capacity,
        "novel_search_performed": novel_search_performed,
        "previous_active_ids": active_ids,
        "candidate_ids": [str(item["id"]) for item in candidates],
        "novel_hypothesis_ids": merge_log["novel_accepted"],
        "reactivated_hypothesis_ids": merge_log["historical_reactivated"],
        "states": states,
        "next_active_ids": next_active,
        "generated_training_examples": len(additions),
        "verification_probes_training_eligible": False,
    }
    write_json(round_dir / "lifecycle.json", lifecycle)
    return next_archive, next_active, states, additions, lifecycle


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from src.evaluation.metrics import compute_hans_metrics

    return compute_hans_metrics(
        [str(row["gold"]) for row in rows],
        [str(row["prediction"]) for row in rows],
        [str(row["heuristic"]) for row in rows],
        [str(row["subcase"]) for row in rows],
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    closed = require_mapping(config, "closed_loop")
    benchmark = require_mapping(config, "stage2_benchmark")
    require_mapping(config, "teacher")
    require_mapping(config, "model")
    require_mapping(config, "training")
    require_mapping(config, "hypothesis_generation")
    require_mapping(config, "verification")
    require_mapping(config, "synthetic_generation")
    require_mapping(config, "synthetic_validation")
    seed = int(closed.get("seed", 42))
    total_budget = int(closed.get("total_budget", 128))
    per_round = int(closed.get("per_round_examples", 32))
    max_rounds = int(closed.get("max_rounds", 4))
    if total_budget != per_round * max_rounds or per_round % 2:
        raise ValueError("Closed-loop budget must equal max_rounds * even per-round size")
    set_deterministic_seed(seed)

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _write_yaml(output / "config.yaml", config)

    manifest = FrozenHansManifest.from_mapping(
        read_json(Path(str(benchmark["frozen_manifest_path"])))
    )
    shared_path = Path(str(closed["shared_predictions_path"]))
    shared_rows = read_jsonl(shared_path)
    shared_index = {str(row["id"]): row for row in shared_rows}
    source_pool = Path(str(closed["initial_pool_dir"]))
    pool = [
        SyntheticNLIExample.from_mapping(row)
        for row in read_jsonl(source_pool / "synthetic" / "accepted.jsonl")
    ]
    initial_hypotheses = read_json(source_pool / "hypotheses.json")["hypotheses"]
    archive = [dict(item) for item in initial_hypotheses]
    initial_manifest = read_json(source_pool / "synthetic" / "manifest.json")
    states = {
        str(key): dict(value)
        for key, value in initial_manifest["hypothesis_verification_states"].items()
    }
    # The v5 initial family was not predeclared as one of v6's repeated looks.
    # Treat all five initial candidates as provisional until their first fresh
    # v6 re-verification; this prevents v6 from inheriting a premature drop.
    active_ids = sorted(str(item["id"]) for item in archive)
    for hypothesis_id in active_ids:
        if states[hypothesis_id]["state"] == "refuted":
            states[hypothesis_id] = {
                **states[hypothesis_id],
                "state": "unresolved",
                "certified": False,
                "v6_initial_status": "provisional_recheck_required",
            }

    subset = read_json(Path(str(closed["stopping_subset_manifest"])))
    gain_rows = [shared_index[item] for item in manifest.gain_eval_ids]
    stopping_ids, report_ids = validate_stopping_subset_manifest(
        subset,
        gain_rows,
        manifest.gain_eval_ids,
        frozen_manifest_hash=manifest.manifest_hash,
    )
    stopping_examples = _hans([shared_index[item] for item in stopping_ids])
    base_stopping_rows = [shared_index[item] for item in stopping_ids]
    learning_curve = [{"round": 0, "metrics": _metrics_from_rows(base_stopping_rows)}]

    model = require_mapping(config, "model")
    model["device_map"] = os.environ.get("CLOSED_LOOP_STUDENT_DEVICE", "cuda:1")
    training = LoraTrainingConfig.from_mapping(require_mapping(config, "training"))
    selection = require_mapping(closed, "selection")
    current_adapter = model.get("adapter_path")
    selected_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    cumulative_hypothesis_counts: Counter[str] = Counter()
    round_records: list[dict[str, Any]] = []
    stop_reason = "maximum_budget_reached"

    for round_index in range(1, max_rounds + 1):
        refresh_record: dict[str, Any] | None = None
        if round_index > 1:
            archive, active_ids, states, additions, refresh_record = _refresh_hypotheses(
                output=output,
                base_config=config,
                round_index=round_index,
                adapter=current_adapter,
                shared_rows=shared_rows,
                manifest=manifest,
                archive=archive,
                active_ids=active_ids,
                generate_pool=True,
            )
            pool, pool_merge = _merge_pool(
                pool,
                additions,
                threshold=float(closed.get("pool_near_duplicate_threshold", 0.92)),
            )
            refresh_record["pool_merge"] = pool_merge
        if not active_ids:
            stop_reason = (
                "all_tracked_hypotheses_refuted_and_no_novel_hypothesis"
                if refresh_record is None
                or refresh_record.get("novel_search_performed") is True
                else "all_tracked_hypotheses_refuted_novel_search_deferred_by_candidate_cap"
            )
            break

        selected_set = set(selected_ids)
        eligible = [
            item
            for item in pool
            if item.hypothesis_id in set(active_ids) or item.id in selected_set
        ]
        student = _student(model, current_adapter)
        student.progress_label = f"closed-loop round {round_index} probability scoring"
        probabilities = student.predict_proba(
            [item.premise for item in eligible],
            [item.hypothesis for item in eligible],
            batch_size=int(model.get("batch_size", 4)),
        )
        label_index = {label: index for index, label in enumerate(MNLI_LABELS)}
        nll = {
            item.id: -math.log(
                max(float(probabilities[index][label_index[item.label]]), 1e-12)
            )
            for index, item in enumerate(eligible)
        }
        entropy = {
            item.id: -sum(
                float(value) * math.log(max(float(value), 1e-12))
                for value in probabilities[index]
            )
            / math.log(len(MNLI_LABELS))
            for index, item in enumerate(eligible)
        }
        student.progress_label = f"closed-loop round {round_index} embedding extraction"
        vectors = student.encode_pair_embeddings(
            [item.premise for item in eligible],
            [item.hypothesis for item in eligible],
            batch_size=int(selection.get("embedding_batch_size", 8)),
        )
        evidence = {key: states[key] for key in active_ids}
        for item in eligible:
            if item.id in selected_set and item.hypothesis_id not in evidence:
                evidence[item.hypothesis_id] = {"state": "unresolved"}
        acquired, diagnostics = select_closed_loop_soft_pairs(
            eligible,
            budget=per_round,
            seed=seed + round_index - 1,
            nll_scores=nll,
            entropy_scores=entropy,
            embeddings={item.id: vectors[index] for index, item in enumerate(eligible)},
            verification_evidence=evidence,
            excluded_ids=selected_ids,
            prior_hypothesis_pair_counts=cumulative_hypothesis_counts,
            nll_weight=float(selection.get("nll_weight", 0.65)),
            entropy_weight=float(selection.get("entropy_weight", 0.35)),
            novelty_weight=float(selection.get("novelty_weight", 0.50)),
            hypothesis_bonus_weight=float(
                selection.get("hypothesis_bonus_weight", 0.35)
            ),
            new_hypothesis_bonus_weight=float(
                selection.get("new_hypothesis_bonus_weight", 0.50)
            ),
        )
        if len(acquired) != per_round:
            del student
            _clear_cuda_cache()
            stop_reason = "insufficient_verified_synthetic_pairs"
            break
        round_dir = output / "training" / f"round_{round_index:03d}"
        start = time.perf_counter()
        loss = train_lora(
            student,
            [item.to_mnli() for item in acquired],
            training,
            round_dir / "adapter",
            seed + round_index - 1,
        )
        current_adapter = str(round_dir / "adapter")
        stopping_predictions, stopping_metrics = evaluate_hans(
            student,
            stopping_examples,
            batch_size=int(model.get("batch_size", 4)),
        )
        elapsed = time.perf_counter() - start
        del student
        _clear_cuda_cache()
        write_jsonl(round_dir / "stopping_predictions.jsonl", stopping_predictions)
        ids = [item.id for item in acquired]
        selected_ids.extend(ids)
        selected_rows.extend(item.to_dict() for item in acquired)
        cumulative_hypothesis_counts.update(
            item.hypothesis_id for item in acquired if item.pair_member == "base"
        )
        learning_curve.append({"round": round_index, "metrics": stopping_metrics})
        record = {
            "round": round_index,
            "requested_examples": per_round,
            "actual_examples": len(acquired),
            "cumulative_examples": len(selected_ids),
            "starting_adapter": model.get("adapter_path") if round_index == 1 else round_records[-1]["ending_adapter"],
            "ending_adapter": current_adapter,
            "train_loss": loss,
            "wall_clock_seconds_training_and_stopping_eval": elapsed,
            "active_hypothesis_ids": active_ids,
            "verification_refresh": refresh_record,
            "selection": diagnostics,
            "stopping_metrics": stopping_metrics,
        }
        round_records.append(record)
        write_jsonl(output / "rounds.jsonl", round_records)

    final_audit = None
    if bool(closed.get("final_hypothesis_audit", True)) and current_adapter:
        archive, final_active, final_states, _, final_audit = _refresh_hypotheses(
            output=output,
            base_config=config,
            round_index=max_rounds + 1,
            adapter=current_adapter,
            shared_rows=shared_rows,
            manifest=manifest,
            archive=archive,
            active_ids=active_ids,
            generate_pool=False,
        )
        if not final_active:
            stop_reason = (
                "all_tracked_hypotheses_refuted_and_no_novel_hypothesis"
                if final_audit.get("novel_search_performed") is True
                else "all_tracked_hypotheses_refuted_novel_search_deferred_by_candidate_cap"
            )
        else:
            stop_reason = "maximum_budget_reached_with_remaining_hypotheses"

    final_student = _student(model, current_adapter)
    gain_report_rows = [shared_index[item] for item in report_ids]
    gain_predictions, gain_metrics = evaluate_hans(
        final_student,
        _hans(gain_report_rows),
        batch_size=int(model.get("batch_size", 4)),
    )
    write_jsonl(output / "gain_report_predictions.jsonl", gain_predictions)
    final_rows = read_jsonl(Path(str(benchmark["sealed_final_test_path"])))
    final_predictions, hans_metrics = evaluate_hans(
        final_student,
        _hans(final_rows),
        batch_size=int(model.get("batch_size", 4)),
    )
    write_jsonl(output / "hans_final_predictions.jsonl", final_predictions)
    mnli_config = require_mapping(config, "mnli_evaluation")
    mnli_metrics: dict[str, Any] = {}
    for name, split in (
        ("matched", str(mnli_config.get("matched_split", "validation_matched"))),
        ("mismatched", str(mnli_config.get("mismatched_split", "validation_mismatched"))),
    ):
        examples = load_mnli(
            split=split,
            dataset_name=str(mnli_config.get("name", "nyu-mll/glue")),
            dataset_config=str(mnli_config.get("config", "mnli")),
            cache_dir=mnli_config.get("cache_dir"),
            max_examples=mnli_config.get("max_examples"),
            drop_unlabeled=True,
        )
        _, mnli_metrics[name] = evaluate_mnli(
            final_student, examples, batch_size=int(model.get("batch_size", 4))
        )
    del final_student
    _clear_cuda_cache()

    leakage = audit_final_test_overlap(
        selected_rows,
        final_rows,
        near_duplicate_threshold=float(benchmark.get("near_duplicate_threshold", 0.92)),
    )
    write_json(output / "leakage_audit.json", leakage)
    if leakage["id_overlap"]:
        raise ValueError("Final-test ID overlap detected")
    result = {
        "method": "ours_verified_closed_loop_v1",
        "seed": seed,
        "budget": total_budget,
        "actual_training_examples": len(selected_ids),
        "training_objective": "per_example_causal_lm_cross_entropy",
        "protocol": "checkpoint_conditioned_discover_verify_generate_select_train",
        "stop_reason": stop_reason,
        "final_checkpoint": checkpoint_descriptor({**model, "adapter_path": current_adapter}),
        "hans_final": hans_metrics,
        "gain_report": gain_metrics,
        "mnli": mnli_metrics,
        "learning_curve": learning_curve,
        "final_hypothesis_audit": final_audit,
        "selected_hypothesis_pair_counts": dict(sorted(cumulative_hypothesis_counts.items())),
        "verification_probes_used_for_training": False,
    }
    write_json(output / "result.json", result)
    write_json(output / "hypothesis_archive.json", {"hypotheses": archive})
    lines = [
        "# Stage 2 closed-loop verification report",
        "",
        f"- Seed: `{seed}`",
        f"- Fixed maximum budget: `{total_budget}` examples",
        f"- Actual trained examples: `{len(selected_ids)}`",
        f"- Completed rounds: `{len(round_records)}/{max_rounds}`",
        f"- Stop reason: `{stop_reason}`",
        f"- HANS final accuracy: `{float(hans_metrics['overall_accuracy']):.6f}`",
        f"- Gain-report accuracy: `{float(gain_metrics['overall_accuracy']):.6f}`",
        f"- MNLI matched accuracy: `{float(mnli_metrics['matched']['accuracy']):.6f}`",
        f"- MNLI mismatched accuracy: `{float(mnli_metrics['mismatched']['accuracy']):.6f}`",
        "",
        "Verification probes were used only for certification/diagnosis and never for training.",
        "A hypothesis is resolved only when its fresh checkpoint-conditioned Hoeffding UCB is below tau_v.",
    ]
    _atomic_text_write(output / "CLOSED_LOOP_REPORT.md", "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
