#!/usr/bin/env python3
"""Compare three-way-to-HANS and direct-binary HANS prediction rules.

This script is diagnostic-only. It loads existing V7 verification ON or OFF
round-3 and round-4 adapters, runs probability inference on the frozen HANS
final set, and writes all outputs outside the original run directory. An
existing diagnostic from the other verification branch can optionally be
included for a same-round, same-budget comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import HANSExample
from src.data.labels import MNLI_LABELS
from src.evaluation.metrics import compute_hans_metrics
from src.models.adapter_continuity import audit_loaded_adapter
from src.models.qwen_nli import QwenNLIStudent
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


DEFAULT_RUN_ROOT = Path("outputs/stage2_runs_v7_training_verification_ablation")
DEFAULT_DIAGNOSTIC_ROOT = Path(
    "outputs/stage2_diagnostics/v7_hans_prediction_rules"
)
DEFAULT_REPORT_ROOT = Path("reports/stage2_v7_training_verification_ablation")

RULE_THREE_WAY = "argmax(P(E), P(N), P(C)) -> HANS binary"
RULE_BINARY = "argmax(P(E), P(N)+P(C))"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification-mode",
        choices=("on", "off"),
        default="on",
        help="Verification branch whose round-3/round-4 adapters are scored.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Defaults to the selected verification branch and seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the branch-specific diagnostic output directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to a branch-specific Markdown report.",
    )
    parser.add_argument(
        "--reference-diagnostic",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic.json from the other verification branch. "
            "Adds same-round, same-budget ON/OFF comparison tables."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the model batch size from the run config snapshot.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a validated completed probability cache for either round.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite diagnostic artifacts only.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def make_student(model: dict[str, Any], adapter: Path) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=str(adapter),
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def hans_examples(rows: list[dict[str, Any]]) -> list[HANSExample]:
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


def validate_probability_rows(
    rows: list[dict[str, Any]], examples: list[HANSExample], adapter: Path
) -> list[list[float]]:
    if len(rows) != len(examples):
        raise ValueError(
            f"Probability cache has {len(rows)} rows; expected {len(examples)}"
        )
    probabilities: list[list[float]] = []
    for row, example in zip(rows, examples, strict=True):
        if str(row.get("id")) != example.id:
            raise ValueError(
                f"Probability cache ID mismatch: {row.get('id')!r} != {example.id!r}"
            )
        if str(row.get("adapter")) != str(adapter):
            raise ValueError("Probability cache was produced by a different adapter")
        observed_order = tuple(row.get("label_order", []))
        if observed_order != MNLI_LABELS:
            raise ValueError(
                f"Probability cache label order is {observed_order!r}, expected {MNLI_LABELS!r}"
            )
        values = [float(value) for value in row["probabilities"]]
        if len(values) != len(MNLI_LABELS):
            raise ValueError("Probability cache row has the wrong class width")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Probability cache contains a value outside [0, 1]")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Probability cache row does not sum to one")
        probabilities.append(values)
    return probabilities


def probability_cache_rows(
    probabilities: list[list[float]], examples: list[HANSExample], adapter: Path
) -> list[dict[str, Any]]:
    return [
        {
            "id": example.id,
            "adapter": str(adapter),
            "label_order": list(MNLI_LABELS),
            "probabilities": [float(value) for value in row],
        }
        for example, row in zip(examples, probabilities, strict=True)
    ]


def inference_for_round(
    *,
    round_index: int,
    adapter: Path,
    model: dict[str, Any],
    examples: list[HANSExample],
    batch_size: int,
    output_dir: Path,
    resume: bool,
    overwrite: bool,
    seed: int,
) -> tuple[list[list[float]], dict[str, Any], float, str]:
    cache_path = output_dir / f"round_{round_index:03d}_probabilities.jsonl"
    audit_path = output_dir / f"round_{round_index:03d}_adapter_load_audit.json"
    if cache_path.is_file() and resume:
        rows = read_jsonl(cache_path)
        probabilities = validate_probability_rows(rows, examples, adapter)
        audit = read_json(audit_path) if audit_path.is_file() else {}
        print(f"[RESUME] Reusing {cache_path} ({len(rows):,} examples)", flush=True)
        return probabilities, audit, 0.0, "reused_cache"
    if cache_path.exists() and not overwrite:
        raise FileExistsError(
            f"Diagnostic cache already exists: {cache_path}. "
            "Use --resume to reuse it or --overwrite to recompute it."
        )

    print(
        f"[ROUND {round_index}] Loading adapter and scoring {len(examples):,} HANS examples",
        flush=True,
    )
    set_deterministic_seed(seed)
    student = make_student(model, adapter)
    audit = audit_loaded_adapter(student.model, adapter)
    if not bool(audit.get("matched")):
        raise RuntimeError(f"Round {round_index} adapter load audit failed")
    student.progress_label = f"HANS diagnostic round {round_index} probabilities"
    started = time.perf_counter()
    probabilities = student.predict_proba(
        [example.premise for example in examples],
        [example.hypothesis for example in examples],
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started
    if len(probabilities) != len(examples):
        raise RuntimeError("Model returned a different number of probability rows")
    write_jsonl(
        cache_path,
        probability_cache_rows(probabilities, examples, adapter),
    )
    write_json(audit_path, audit)
    del student
    clear_cuda_cache()
    print(
        f"[ROUND {round_index}] Wrote probability cache in {elapsed:.1f} seconds",
        flush=True,
    )
    return probabilities, audit, elapsed, "inference"


def predictions_from_probabilities(
    probabilities: list[list[float]],
) -> tuple[list[str], list[str], list[str]]:
    three_way_predictions: list[str] = []
    binary_predictions: list[str] = []
    mnli_predictions: list[str] = []
    for row in probabilities:
        best_index = max(range(len(MNLI_LABELS)), key=row.__getitem__)
        mnli_label = MNLI_LABELS[best_index]
        mnli_predictions.append(mnli_label)
        three_way_predictions.append(
            "entailment" if mnli_label == "entailment" else "non-entailment"
        )
        entailment = float(row[0])
        non_entailment = float(row[1]) + float(row[2])
        binary_predictions.append(
            "entailment" if entailment >= non_entailment else "non-entailment"
        )
    return three_way_predictions, binary_predictions, mnli_predictions


def metrics_for_predictions(
    predictions: list[str], examples: list[HANSExample]
) -> dict[str, Any]:
    return compute_hans_metrics(
        [example.gold for example in examples],
        predictions,
        [example.heuristic for example in examples],
        [example.subcase for example in examples],
    )


def compare_rules(
    probabilities: list[list[float]], examples: list[HANSExample]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    three_way, binary, mnli_predictions = predictions_from_probabilities(probabilities)
    three_way_metrics = metrics_for_predictions(three_way, examples)
    binary_metrics = metrics_for_predictions(binary, examples)
    rows: list[dict[str, Any]] = []
    direction_counts: Counter[str] = Counter()
    three_way_only_correct = 0
    binary_only_correct = 0
    for example, probability, mnli, prediction_a, prediction_b in zip(
        examples,
        probabilities,
        mnli_predictions,
        three_way,
        binary,
        strict=True,
    ):
        different = prediction_a != prediction_b
        correct_a = prediction_a == example.gold
        correct_b = prediction_b == example.gold
        if different:
            direction_counts[f"{prediction_a} -> {prediction_b}"] += 1
            three_way_only_correct += int(correct_a and not correct_b)
            binary_only_correct += int(correct_b and not correct_a)
        rows.append(
            {
                "id": example.id,
                "gold": example.gold,
                "heuristic": example.heuristic,
                "subcase": example.subcase,
                "p_entailment": float(probability[0]),
                "p_neutral": float(probability[1]),
                "p_contradiction": float(probability[2]),
                "mnli_argmax_prediction": mnli,
                "three_way_to_binary_prediction": prediction_a,
                "direct_binary_prediction": prediction_b,
                "predictions_differ": different,
                "three_way_correct": correct_a,
                "direct_binary_correct": correct_b,
            }
        )
    disagreement_count = sum(row["predictions_differ"] for row in rows)
    return (
        {
            "rules": {
                "three_way_to_binary": RULE_THREE_WAY,
                "direct_binary": RULE_BINARY,
            },
            "metrics": {
                "three_way_to_binary": three_way_metrics,
                "direct_binary": binary_metrics,
            },
            "disagreement": {
                "count": disagreement_count,
                "rate": disagreement_count / len(rows),
                "direction_counts": dict(sorted(direction_counts.items())),
                "three_way_only_correct": three_way_only_correct,
                "direct_binary_only_correct": binary_only_correct,
            },
        },
        rows,
    )


def stored_prediction_consistency(
    stored_path: Path | None,
    diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if stored_path is None or not stored_path.is_file():
        return {
            "available": False,
            "path": str(stored_path) if stored_path is not None else None,
            "reason": "no_stored_full_hans_predictions_for_checkpoint",
        }
    stored = read_jsonl(stored_path)
    if len(stored) != len(diagnostic_rows):
        raise ValueError(
            f"Stored prediction count differs for {stored_path}: "
            f"{len(stored)} != {len(diagnostic_rows)}"
        )
    mismatch_ids: list[str] = []
    for expected, observed in zip(stored, diagnostic_rows, strict=True):
        if str(expected["id"]) != str(observed["id"]):
            raise ValueError(f"Stored prediction IDs differ in {stored_path}")
        if str(expected["prediction"]) != str(
            observed["three_way_to_binary_prediction"]
        ):
            mismatch_ids.append(str(expected["id"]))
    return {
        "available": True,
        "path": str(stored_path),
        "num_rows": len(stored),
        "mismatch_count": len(mismatch_ids),
        "mismatch_ids_first_20": mismatch_ids[:20],
    }


def validation_history(run_dir: Path, round_index: int) -> dict[str, Any]:
    path = run_dir / f"training/round_{round_index:03d}/training_history.json"
    payload = read_json(path)
    epochs = []
    for item in payload["epochs"]:
        validation = item["validation"]
        epochs.append(
            {
                "epoch": int(item["epoch"]),
                "optimizer_step_end": int(item["optimizer_step_end"]),
                "validation_nll": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_entailment_accuracy": float(
                    validation["entailment_accuracy"]
                ),
                "validation_non_entailment_accuracy": float(
                    validation["non_entailment_accuracy"]
                ),
                "improved": bool(item["improved"]),
            }
        )
    return {
        "path": str(path),
        "prediction_rule": RULE_BINARY,
        "validation_source": "predeclared_gain_eval_stopping_subset",
        "num_validation_examples": 900,
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_loss": float(payload["best_validation_loss"]),
        "stopped_early": bool(payload["stopped_early"]),
        "epochs": epochs,
    }


def metric_delta(round3: dict[str, Any], round4: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for rule in ("three_way_to_binary", "direct_binary"):
        output[rule] = {}
        for key in (
            "overall_accuracy",
            "entailment_accuracy",
            "non_entailment_accuracy",
        ):
            output[rule][key] = (
                float(round4["metrics"][rule][key])
                - float(round3["metrics"][rule][key])
            )
    return output


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.4f}%"


def signed_points(value: Any) -> str:
    return f"{100.0 * float(value):+.4f} pp"


def mode_label(mode: str) -> str:
    return str(mode).upper()


def metric_triplet_delta(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    """Return left-minus-right deltas for the three headline metrics."""
    return {
        key: float(left[key]) - float(right[key])
        for key in (
            "overall_accuracy",
            "entailment_accuracy",
            "non_entailment_accuracy",
        )
    }


def validate_reference_diagnostic(
    result: dict[str, Any], reference: dict[str, Any]
) -> None:
    if int(reference.get("seed", -1)) != int(result["seed"]):
        raise ValueError("Reference diagnostic uses a different seed")
    if int(reference.get("num_examples", -1)) != int(result["num_examples"]):
        raise ValueError("Reference diagnostic uses a different HANS example count")
    if str(reference.get("verification_mode")) == str(result["verification_mode"]):
        raise ValueError("Reference diagnostic must be from the other verification mode")
    if set(reference.get("rounds", {})) != {"3", "4"}:
        raise ValueError("Reference diagnostic must contain rounds 3 and 4")


def cross_verification_comparison(
    result: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    validate_reference_diagnostic(result, reference)
    def comparison_view(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "seed": payload["seed"],
            "verification_mode": payload["verification_mode"],
            "num_examples": payload["num_examples"],
            "rounds": payload["rounds"],
            "round3_to_round4_delta": payload["round3_to_round4_delta"],
            "diagnostic_path": payload.get("diagnostic_path"),
        }

    by_mode = {
        str(result["verification_mode"]): comparison_view(result),
        str(reference["verification_mode"]): comparison_view(reference),
    }
    if set(by_mode) != {"on", "off"}:
        raise ValueError("Cross-verification comparison requires one ON and one OFF run")
    differences: dict[str, Any] = {}
    for round_index in (3, 4):
        differences[str(round_index)] = {}
        for rule in ("three_way_to_binary", "direct_binary"):
            on_metrics = by_mode["on"]["rounds"][str(round_index)]["metrics"][rule]
            off_metrics = by_mode["off"]["rounds"][str(round_index)]["metrics"][rule]
            differences[str(round_index)][rule] = metric_triplet_delta(
                on_metrics, off_metrics
            )
    return {
        "modes": by_mode,
        "on_minus_off": differences,
        "examples_per_round": 32,
    }


def report_text(result: dict[str, Any]) -> str:
    rounds = result["rounds"]
    verification_mode = str(result["verification_mode"])
    branch = (
        "ON/OFF"
        if result.get("cross_verification_comparison")
        else mode_label(verification_mode)
    )
    lines = [
        f"# HANS prediction-rule diagnostic: V7 Verification {branch} "
        f"seed {result['seed']}",
        "",
        "> **Diagnostic only.** This analysis re-scores the frozen HANS final set "
        "with existing round-3 and round-4 checkpoints. It does not modify the "
        "model, training, evaluator, checkpoints, or primary results.",
        "",
        "## Prediction rules",
        "",
        f"1. `{RULE_THREE_WAY}` (current evaluator rule).",
        f"2. `{RULE_BINARY}` (direct binary aggregation).",
        "",
        "`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are "
        "resolved in favor of entailment for both rules, consistent with the "
        "current first-index argmax and binary-validation implementation.",
        "",
        "## Full HANS metrics",
        "",
        "| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for round_index in (3, 4):
        for rule_key, rule_label in (
            ("three_way_to_binary", "Three-way argmax -> binary"),
            ("direct_binary", "Direct binary"),
        ):
            metrics = rounds[str(round_index)]["metrics"][rule_key]
            num_correct = int(metrics["num_correct"])
            num_wrong = int(metrics["num_examples"]) - num_correct
            lines.append(
                f"| Round {round_index} | {32 * round_index} examples | "
                f"{rule_label} | {num_correct:,} | {num_wrong:,} | "
                f"{percent(metrics['overall_accuracy'])} | "
                f"{percent(metrics['entailment_accuracy'])} | "
                f"{percent(metrics['non_entailment_accuracy'])} |"
            )

    lines.extend(
        [
            "",
            "## Prediction disagreement between rules",
            "",
            "| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for round_index in (3, 4):
        disagreement = rounds[str(round_index)]["disagreement"]
        directions = ", ".join(
            f"{key}: {value}"
            for key, value in disagreement["direction_counts"].items()
        ) or "none"
        lines.append(
            f"| Round {round_index} | {disagreement['count']:,} / "
            f"{rounds[str(round_index)]['metrics']['three_way_to_binary']['num_examples']:,} | "
            f"{percent(disagreement['rate'])} | "
            f"{disagreement['three_way_only_correct']:,} | "
            f"{disagreement['direct_binary_only_correct']:,} | {directions} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy change from round 3 to round 4",
            "",
            "Positive values mean round 4 is higher than round 3.",
            "",
            "| Rule | Overall delta | Entailment delta | Non-entailment delta |",
            "|---|---:|---:|---:|",
        ]
    )
    deltas = result["round3_to_round4_delta"]
    for rule_key, label in (
        ("three_way_to_binary", "Three-way argmax -> binary"),
        ("direct_binary", "Direct binary"),
    ):
        values = deltas[rule_key]
        lines.append(
            f"| {label} | {signed_points(values['overall_accuracy'])} | "
            f"{signed_points(values['entailment_accuracy'])} | "
            f"{signed_points(values['non_entailment_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation history for rounds 3-4",
            "",
            "The logged epoch accuracy and NLL use the direct-binary rule "
            "`argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping "
            "subset. The sealed final HANS set was not used for these epoch logs.",
            "",
        ]
    )
    for round_index in (3, 4):
        history = result["validation_history"][str(round_index)]
        lines.extend(
            [
                f"### Round {round_index}",
                "",
                f"Best epoch by validation NLL: **{history['best_epoch']}**; "
                f"best NLL: **{history['best_validation_loss']:.6f}**.",
                "",
                "| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for epoch in history["epochs"]:
            lines.append(
                f"| {epoch['epoch']} | {epoch['optimizer_step_end']} | "
                f"{epoch['validation_nll']:.6f} | "
                f"{percent(epoch['validation_accuracy'])} | "
                f"{percent(epoch['validation_entailment_accuracy'])} | "
                f"{percent(epoch['validation_non_entailment_accuracy'])} | "
                f"{'yes' if epoch['improved'] else 'no'} |"
            )
        lines.append("")

    cross = result.get("cross_verification_comparison")
    if cross:
        by_mode = cross["modes"]
        lines.extend(
            [
                "## Verification ON/OFF at the same round and budget",
                "",
                "All entries re-score the same frozen 15,000-example HANS set. "
                "The final test is used only for this post-hoc diagnostic, not "
                "for checkpoint selection or training.",
                "",
                "| Round | Budget | Branch | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |",
                "|---:|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for round_index in (3, 4):
            for mode in ("on", "off"):
                for rule_key, rule_label in (
                    ("three_way_to_binary", "Three-way argmax -> binary"),
                    ("direct_binary", "Direct binary"),
                ):
                    metrics = by_mode[mode]["rounds"][str(round_index)]["metrics"][rule_key]
                    correct = int(metrics["num_correct"])
                    wrong = int(metrics["num_examples"]) - correct
                    lines.append(
                        f"| {round_index} | {32 * round_index} examples | "
                        f"{mode_label(mode)} | {rule_label} | {correct:,} | "
                        f"{wrong:,} | {percent(metrics['overall_accuracy'])} | "
                        f"{percent(metrics['entailment_accuracy'])} | "
                        f"{percent(metrics['non_entailment_accuracy'])} |"
                    )

        lines.extend(
            [
                "",
                "### ON minus OFF accuracy at matched budget",
                "",
                "Positive values mean ON is numerically higher; negative values "
                "mean OFF is numerically higher. These are descriptive final-test "
                "differences, not a rule-selection criterion.",
                "",
                "| Round | Budget | Rule | Overall ON-OFF | Entailment ON-OFF | Non-entailment ON-OFF |",
                "|---:|---:|---|---:|---:|---:|",
            ]
        )
        for round_index in (3, 4):
            for rule_key, rule_label in (
                ("three_way_to_binary", "Three-way argmax -> binary"),
                ("direct_binary", "Direct binary"),
            ):
                values = cross["on_minus_off"][str(round_index)][rule_key]
                lines.append(
                    f"| {round_index} | {32 * round_index} examples | "
                    f"{rule_label} | {signed_points(values['overall_accuracy'])} | "
                    f"{signed_points(values['entailment_accuracy'])} | "
                    f"{signed_points(values['non_entailment_accuracy'])} |"
                )

        lines.extend(
            [
                "",
                "### Round 3 to round 4 within each branch",
                "",
                "Positive values mean round 4 is numerically higher than round 3.",
                "",
                "| Branch | Rule | Overall delta | Entailment delta | Non-entailment delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for mode in ("on", "off"):
            deltas = by_mode[mode]["round3_to_round4_delta"]
            for rule_key, rule_label in (
                ("three_way_to_binary", "Three-way argmax -> binary"),
                ("direct_binary", "Direct binary"),
            ):
                values = deltas[rule_key]
                lines.append(
                    f"| {mode_label(mode)} | {rule_label} | "
                    f"{signed_points(values['overall_accuracy'])} | "
                    f"{signed_points(values['entailment_accuracy'])} | "
                    f"{signed_points(values['non_entailment_accuracy'])} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Provenance and consistency checks",
            "",
            f"- Frozen HANS examples: `{result['num_examples']:,}`.",
            f"- Round-3 adapter: `{result['checkpoints']['3']}`.",
            f"- Round-4 adapter: `{result['checkpoints']['4']}`.",
            f"- Round-3 current-rule consistency check: "
            f"`{consistency_summary(rounds['3']['stored_prediction_consistency'])}`.",
            f"- Round-4 current-rule consistency check: "
            f"`{consistency_summary(rounds['4']['stored_prediction_consistency'])}`.",
            "- Raw probability and per-example comparison artifacts are stored "
            f"under `{result['output_dir']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def consistency_summary(payload: dict[str, Any]) -> str:
    if not bool(payload.get("available", True)):
        return "not available (no stored full-HANS predictions for this checkpoint)"
    return f"{int(payload['mismatch_count'])} mismatches"


def main() -> None:
    args = parse_args()
    default_run_dir = (
        DEFAULT_RUN_ROOT / f"verification_{args.verification_mode}" / f"seed_{args.seed}"
    )
    default_output_dir = (
        DEFAULT_DIAGNOSTIC_ROOT
        / f"verification_{args.verification_mode}"
        / f"seed_{args.seed}"
    )
    if args.reference_diagnostic is not None:
        default_report = DEFAULT_REPORT_ROOT / "HANS_PREDICTION_RULE_DIAGNOSTIC_ON_OFF.md"
    elif args.verification_mode == "on" and args.seed == 42:
        default_report = DEFAULT_REPORT_ROOT / "HANS_PREDICTION_RULE_DIAGNOSTIC.md"
    else:
        default_report = (
            DEFAULT_REPORT_ROOT
            / f"HANS_PREDICTION_RULE_DIAGNOSTIC_{args.verification_mode.upper()}_SEED_{args.seed}.md"
        )
    run_dir = resolve_project_path(args.run_dir or default_run_dir)
    output_dir = resolve_project_path(args.output_dir or default_output_dir)
    report_path = resolve_project_path(args.report or default_report)
    reference_path: Path | None = None
    reference: dict[str, Any] | None = None
    if args.reference_diagnostic is not None:
        reference_path = resolve_project_path(args.reference_diagnostic)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference diagnostic is missing: {reference_path}"
            )
        reference = read_json(reference_path)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config snapshot is missing: {config_path}")
    config = load_yaml_with_extends(config_path)
    model = require_mapping(config, "model")
    benchmark = require_mapping(config, "stage2_benchmark")
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(model.get("batch_size", 4))
    )
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")

    final_path = resolve_project_path(str(benchmark["sealed_final_test_path"]))
    examples = hans_examples(read_jsonl(final_path))
    if not examples:
        raise ValueError("Frozen HANS final set is empty")
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen HANS final set contains duplicate IDs")

    checkpoints = {
        3: run_dir / "training/round_003/adapter",
        4: run_dir / "training/round_004/adapter",
    }
    round3_stored = (
        run_dir / "posthoc/round_003_stopping_accuracy_71/hans_final_predictions.jsonl"
    )
    stored_predictions: dict[int, Path | None] = {
        3: round3_stored if round3_stored.is_file() else None,
        4: run_dir / "hans_final_predictions.jsonl",
    }
    for round_index, adapter in checkpoints.items():
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Round {round_index} adapter is missing: {adapter}")

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / "diagnostic.json"
    if diagnostic_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Diagnostic result already exists: {diagnostic_path}. "
            "Use --resume or --overwrite."
        )

    round_results: dict[str, Any] = {}
    total_started = time.perf_counter()
    for round_index in (3, 4):
        probabilities, audit, inference_seconds, source = inference_for_round(
            round_index=round_index,
            adapter=checkpoints[round_index],
            model=model,
            examples=examples,
            batch_size=batch_size,
            output_dir=output_dir,
            resume=args.resume,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        comparison, prediction_rows = compare_rules(probabilities, examples)
        comparison_path = output_dir / f"round_{round_index:03d}_comparison.jsonl"
        write_jsonl(comparison_path, prediction_rows)
        comparison["checkpoint"] = str(checkpoints[round_index])
        comparison["probability_source"] = source
        comparison["inference_seconds"] = inference_seconds
        comparison["adapter_load_audit"] = audit
        comparison["stored_prediction_consistency"] = stored_prediction_consistency(
            stored_predictions[round_index], prediction_rows
        )
        round_results[str(round_index)] = comparison
        print(
            f"[ROUND {round_index}] rule disagreements: "
            f"{comparison['disagreement']['count']:,} "
            f"({100.0 * comparison['disagreement']['rate']:.4f}%)",
            flush=True,
        )

    histories = {
        str(round_index): validation_history(run_dir, round_index)
        for round_index in (3, 4)
    }
    result = {
        "status": "diagnostic_only",
        "mutates_original_run": False,
        "seed": args.seed,
        "verification_mode": args.verification_mode,
        "num_examples": len(examples),
        "sealed_final_test_path": str(final_path),
        "mnli_label_order": list(MNLI_LABELS),
        "tie_policy": "entailment",
        "checkpoints": {
            str(key): str(value) for key, value in checkpoints.items()
        },
        "rounds": round_results,
        "round3_to_round4_delta": metric_delta(
            round_results["3"], round_results["4"]
        ),
        "validation_history": histories,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "diagnostic_path": str(diagnostic_path),
        "total_wall_seconds": time.perf_counter() - total_started,
    }
    if reference is not None and reference_path is not None:
        result["reference_diagnostic_path"] = str(reference_path)
        result["cross_verification_comparison"] = cross_verification_comparison(
            result, reference
        )
    write_json(diagnostic_path, result)
    _atomic_text_write(report_path, report_text(result) + "\n")
    _atomic_text_write(output_dir / ".complete", "complete\n")
    print(f"[DONE] Diagnostic JSON: {diagnostic_path}", flush=True)
    print(f"[DONE] Diagnostic report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
