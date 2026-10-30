#!/usr/bin/env python3
"""Report the matched v7 training/verification ON-OFF ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_V7_RESULT = (
    PROJECT_ROOT
    / "outputs"
    / "stage2_runs_ours_verified_v7_dual"
    / "experiment_b"
    / "ours_verified"
    / "budget_128"
    / "seed_42"
    / "branches"
    / "continuation"
    / "result.json"
)
MODES = ("on", "off")
SEEDS = (42, 43, 44)
GPU_TIME_FIELDS = (
    "acquisition_gpu_wall_seconds",
    "training_gpu_wall_seconds",
    "validation_gpu_wall_seconds",
    "stopping_evaluation_gpu_wall_seconds",
    "final_evaluation_gpu_wall_seconds",
)
LOOP_GPU_TIME_FIELDS = (
    "acquisition_gpu_wall_seconds",
    "training_gpu_wall_seconds",
    "validation_gpu_wall_seconds",
    "stopping_evaluation_gpu_wall_seconds",
)
DIAGNOSTIC_ROOT = (
    PROJECT_ROOT / "outputs" / "stage2_diagnostics" / "v7_hans_prediction_rules"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in: {path}")
                rows.append(value)
    return rows


def _nested(result: dict[str, Any], *keys: str) -> float:
    value: Any = result
    for key in keys:
        value = value[key]
    return float(value)


def _summary(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _display(values: list[float], *, percent: bool = True) -> str:
    mean, std = _summary(values)
    scale = 100.0 if percent else 1.0
    return f"{mean * scale:.4f} ± {std * scale:.4f}"


def _display_scaled(values: list[float], scale: float, decimals: int = 2) -> str:
    mean, std = _summary(values)
    return f"{mean * scale:.{decimals}f} ± {std * scale:.{decimals}f}"


def _display_delta(
    values: list[float], *, scale: float = 1.0, decimals: int = 2, suffix: str = ""
) -> str:
    mean, std = _summary(values)
    return (
        f"{mean * scale:+.{decimals}f} ± {std * scale:.{decimals}f}"
        f"{suffix}"
    )


def _load_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            path = root / f"verification_{mode}" / f"seed_{seed}" / "result.json"
            if not path.is_file():
                continue
            result = _read_json(path)
            if result.get("verification_mode") != mode:
                raise ValueError(f"Verification mode mismatch: {path}")
            if int(result.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch: {path}")
            if result.get("acquisition_stopping") != "disabled_fixed_4x32":
                raise ValueError(f"Adaptive acquisition unexpectedly enabled: {path}")
            if int(result.get("requested_budget", -1)) != 128:
                raise ValueError(f"Unexpected acquisition budget: {path}")
            if (
                result.get("fixed_budget_comparable") is not True
                or int(result.get("actual_training_examples", -1)) != 128
                or int(result.get("rounds", -1)) != 4
            ):
                raise ValueError(f"Incomplete fixed-budget run cannot be reported: {path}")
            runs[mode].append(result)
    policies = {
        json.dumps(result["training_policy"], sort_keys=True)
        for mode_runs in runs.values()
        for result in mode_runs
    }
    if len(policies) > 1:
        raise ValueError("ON/OFF runs do not share one training policy")
    return runs


def _load_round_cost_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for seed in SEEDS:
            run_dir = root / f"verification_{mode}" / f"seed_{seed}"
            result_path = run_dir / "result.json"
            rounds_path = run_dir / "rounds.jsonl"
            if not result_path.is_file():
                continue
            if not rounds_path.is_file():
                raise FileNotFoundError(f"Missing per-round resource log: {rounds_path}")
            result = _read_json(result_path)
            accuracy_by_round = {
                int(point["round"]): float(point["metrics"]["overall_accuracy"])
                for point in result["learning_curve"]
            }
            round_records = _read_jsonl(rounds_path)
            if len(round_records) != 4:
                raise ValueError(f"Expected four round records: {rounds_path}")
            cumulative_tokens = 0.0
            cumulative_gpu_seconds = 0.0
            cumulative_total_wall_seconds = 0.0
            for record in round_records:
                round_index = int(record["round"])
                usage = record["resource_usage"]
                teacher = usage["teacher"]
                teacher_tokens = float(teacher["input_tokens"]) + float(
                    teacher["output_tokens"]
                )
                gpu_seconds = sum(
                    float(usage.get(key, 0.0)) for key in LOOP_GPU_TIME_FIELDS
                )
                # Refresh and loop GPU stages are sequential. Teacher wall time is
                # already contained in refresh_wall_seconds and must not be added again.
                total_wall_seconds = float(usage["refresh_wall_seconds"]) + gpu_seconds
                cumulative_tokens += teacher_tokens
                cumulative_gpu_seconds += gpu_seconds
                cumulative_total_wall_seconds += total_wall_seconds
                rows.append(
                    {
                        "verification": mode,
                        "seed": seed,
                        "round": round_index,
                        "budget": round_index * 32,
                        "stopping_accuracy": accuracy_by_round[round_index],
                        "teacher_tokens_incremental": teacher_tokens,
                        "teacher_tokens_cumulative": cumulative_tokens,
                        "gpu_seconds_incremental": gpu_seconds,
                        "gpu_seconds_cumulative": cumulative_gpu_seconds,
                        "total_wall_seconds_incremental": total_wall_seconds,
                        "total_wall_seconds_cumulative": (
                            cumulative_total_wall_seconds
                        ),
                    }
                )
    return rows


def _load_diagnostics() -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            path = DIAGNOSTIC_ROOT / f"verification_{mode}" / f"seed_{seed}" / "diagnostic.json"
            if path.is_file():
                diagnostics[mode].append(_read_json(path))
    return diagnostics


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    completed = sum(len(items) for items in runs.values())
    if not completed:
        raise FileNotFoundError(f"No completed ablation results under {root}")

    metric_rows: list[dict[str, str]] = []
    for mode in MODES:
        for result in runs[mode]:
            metric_rows.append(
                {
                    "verification": mode,
                    "seed": str(result["seed"]),
                    "hans_accuracy": str(
                        _nested(result, "hans_final", "overall_accuracy")
                    ),
                    "hans_non_entailment_accuracy": str(
                        _nested(result, "hans_final", "non_entailment_accuracy")
                    ),
                    "mnli_matched_accuracy": str(
                        _nested(result, "mnli", "matched", "accuracy")
                    ),
                    "mnli_mismatched_accuracy": str(
                        _nested(result, "mnli", "mismatched", "accuracy")
                    ),
                    "optimizer_steps": str(result["optimizer_steps"]),
                }
            )
    csv_path = output / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    round_cost_rows = _load_round_cost_rows(root)
    round_cost_csv_path = output / "per_round_performance_cost.csv"
    with round_cost_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(round_cost_rows[0]))
        writer.writeheader()
        writer.writerows(round_cost_rows)

    lines = [
        "# V7 Training Early Stopping + Verification ON/OFF",
        "",
        "The two arms share the same initial pool, 4x32 acquisition budget, "
        "selector, stateful LoRA+AdamW training policy, seed schedule, and "
        "synthetic QC. Only hypothesis verification differs.",
        "",
        "Training early stopping uses binary NLL on the predeclared gain-eval "
        "stopping subset. The sealed final test is evaluated only after training.",
        "",
        "## Final metrics",
        "",
        "| Verification | Seeds complete | HANS | HANS non-ent. | MNLI-m | MNLI-mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        items = runs[mode]
        if not items:
            lines.append(f"| {mode.upper()} | 0/3 | pending | pending | pending | pending |")
            continue
        lines.append(
            f"| {mode.upper()} | {len(items)}/3 | "
            f"{_display([_nested(x, 'hans_final', 'overall_accuracy') for x in items])} | "
            f"{_display([_nested(x, 'hans_final', 'non_entailment_accuracy') for x in items])} | "
            f"{_display([_nested(x, 'mnli', 'matched', 'accuracy') for x in items])} | "
            f"{_display([_nested(x, 'mnli', 'mismatched', 'accuracy') for x in items])} |"
        )

    if OLD_V7_RESULT.is_file():
        old_v7 = _read_json(OLD_V7_RESULT)
        lines.extend(
            [
                "",
                "Historical old-v7 reference (different training policy): "
                f"HANS {_nested(old_v7, 'hans_final', 'overall_accuracy') * 100:.4f}%, "
                f"MNLI-m {_nested(old_v7, 'mnli', 'matched', 'accuracy') * 100:.4f}%, "
                f"MNLI-mm {_nested(old_v7, 'mnli', 'mismatched', 'accuracy') * 100:.4f}%. "
                "It is context only, not the controlled ON/OFF contrast.",
            ]
        )

    diagnostics = _load_diagnostics()
    if all(len(diagnostics[mode]) == len(SEEDS) for mode in MODES):
        lines.extend(
            [
                "",
                "## Final-test HANS metrics: round 4, budget 128",
                "",
                "These values use the saved final-test artifacts from the predeclared "
                "round-4 endpoint. Three-way → binary is the current primary "
                "evaluator; Direct Binary is a diagnostic re-aggregation of the same "
                "saved probabilities. The sealed final HANS set was not used for "
                "training, early stopping, or checkpoint selection.",
                "",
                "| Round | Budget | Rule | ON | OFF | ON − OFF mean |",
                "|---:|---:|---|---:|---:|---:|",
            ]
        )

        def append_diagnostic_rows(round_index: int) -> None:
            for metric_key, label in (
                ("three_way_to_binary", "Three-way → binary"),
                ("direct_binary", "Direct Binary"),
            ):
                on_by_seed = {
                    int(item["seed"]): float(
                        item["rounds"][str(round_index)]["metrics"][metric_key]["overall_accuracy"]
                    )
                    for item in diagnostics["on"]
                }
                off_by_seed = {
                    int(item["seed"]): float(
                        item["rounds"][str(round_index)]["metrics"][metric_key]["overall_accuracy"]
                    )
                    for item in diagnostics["off"]
                }
                on_values = [on_by_seed[seed] for seed in SEEDS]
                off_values = [off_by_seed[seed] for seed in SEEDS]
                paired_deltas = [
                    on_by_seed[seed] - off_by_seed[seed] for seed in SEEDS
                ]
                lines.append(
                    f"| {round_index} | {round_index * 32} | {label} | "
                    f"{_display(on_values)} | {_display(off_values)} | "
                    f"{_display_delta(paired_deltas, scale=100.0, decimals=4, suffix=' pp')} |"
                )

        append_diagnostic_rows(4)
        lines.extend(
            [
                "",
                "At round 4, the descriptive ON/OFF ordering changes with the "
                "mapping rule: OFF is 0.4422 pp higher under the current Three-way "
                "rule, while ON is 0.6622 pp higher under Direct Binary. This "
                "sensitivity result must not be used to choose a rule from final-test "
                "performance.",
                "",
                "### Post-hoc checkpoint diagnostic: round 3, budget 96",
                "",
                "Round 3 is shown only as a post-hoc checkpoint diagnostic and is "
                "not part of the four-point validation curve or a new stopping rule.",
                "",
                "| Round | Budget | Rule | ON | OFF | ON − OFF mean |",
                "|---:|---:|---|---:|---:|---:|",
            ]
        )
        append_diagnostic_rows(3)
        lines.extend(
            [
                "",
                "Full per-seed diagnostic: "
                "[HANS_PREDICTION_RULE_DIAGNOSTIC_ALL_SEEDS.md]"
                "(HANS_PREDICTION_RULE_DIAGNOSTIC_ALL_SEEDS.md).",
                "",
                "### Where Verification ON is numerically better",
                "",
                "- **Matched budget 96 (round 3), current Three-way rule:** ON "
                "reaches 69.6867 ± 1.4063%, compared with OFF at 67.7711 ± "
                "2.6777% (**ON +1.9156 pp**).",
                "- **Matched budget 96, Direct Binary:** ON reaches 69.6956 ± "
                "1.3954%, compared with OFF at 67.7889 ± 2.6951% (**ON "
                "+1.9067 pp**).",
                "- **Matched budget 128 (round 4), Direct Binary diagnostic:** "
                "ON reaches 73.4667 ± 1.5154%, compared with OFF at 72.8044 ± "
                "1.4932% (**ON +0.6622 pp**).",
                "- On the leakage-safe validation curve at budget 96, ON is "
                "also +1.5555 pp above OFF (69.1111% versus 67.5556%).",
                "",
                "The defensible positive claim is therefore: **verification ON "
                "shows an accuracy advantage at the 96-example checkpoint and "
                "under Direct Binary aggregation at 128 examples.** It is not "
                "accurate to claim that ON universally beats OFF: at the "
                "predeclared 128-example endpoint under the current primary "
                "Three-way evaluator, OFF is 0.4422 pp higher, and ON incurs "
                "substantially more teacher-token and wall-clock cost.",
            ]
        )

    lines.extend(
        [
            "",
            "Mean ± sample standard deviation is reported when at least two seeds "
            "are complete; a single completed seed has standard deviation 0.",
            "",
            "## Learning curves",
            "",
            "| Verification | Round | Seeds | Stopping accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        by_round: dict[int, list[float]] = {}
        for result in runs[mode]:
            for point in result["learning_curve"]:
                by_round.setdefault(int(point["round"]), []).append(
                    float(point["metrics"]["overall_accuracy"])
                )
        for round_index in sorted(by_round):
            values = by_round[round_index]
            lines.append(
                f"| {mode.upper()} | {round_index} | {len(values)} | "
                f"{_display(values)} |"
            )

    lines.extend(
        [
            "",
            "## Per-round performance and cost",
            "",
            "Accuracy is measured on the predeclared 900-example stopping subset "
            "using Direct Binary, which is the only leakage-safe metric available at "
            "all four budgets. Teacher tokens are input plus output tokens. GPU time "
            "is the sum of acquisition, training, validation, and stopping-evaluation "
            "wall-clock proxies.",
            "",
            "`Total wall cost` is a time proxy, not a monetary USD cost: "
            "`refresh_wall_seconds + GPU proxy seconds`. Teacher time is already "
            "inside refresh time and is not added again. Values exclude the shared "
            "initial-pool cost, post-round-4 final hypothesis audit, sealed-final "
            "evaluation, and abandoned/retry overhead.",
            "",
            "### Incremental cost incurred in each round",
            "",
            "| Verification | Round | Budget | Accuracy | Teacher tokens | GPU proxy (min) | Total wall cost (min) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for round_index in range(1, 5):
            items = [
                row
                for row in round_cost_rows
                if row["verification"] == mode and row["round"] == round_index
            ]
            lines.append(
                f"| {mode.upper()} | {round_index} | {round_index * 32} | "
                f"{_display([float(x['stopping_accuracy']) for x in items])} | "
                f"{_display_scaled([float(x['teacher_tokens_incremental']) for x in items], 1.0, 0)} | "
                f"{_display_scaled([float(x['gpu_seconds_incremental']) for x in items], 1.0 / 60.0)} | "
                f"{_display_scaled([float(x['total_wall_seconds_incremental']) for x in items], 1.0 / 60.0)} |"
            )

    lines.extend(
        [
            "",
            "### Cumulative cost to reach each checkpoint",
            "",
            "| Verification | Round | Budget | Accuracy | Cumulative teacher tokens | Cumulative GPU proxy (min) | Cumulative total wall cost (h) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for round_index in range(1, 5):
            items = [
                row
                for row in round_cost_rows
                if row["verification"] == mode and row["round"] == round_index
            ]
            lines.append(
                f"| {mode.upper()} | {round_index} | {round_index * 32} | "
                f"{_display([float(x['stopping_accuracy']) for x in items])} | "
                f"{_display_scaled([float(x['teacher_tokens_cumulative']) for x in items], 1.0, 0)} | "
                f"{_display_scaled([float(x['gpu_seconds_cumulative']) for x in items], 1.0 / 60.0)} | "
                f"{_display_scaled([float(x['total_wall_seconds_cumulative']) for x in items], 1.0 / 3600.0)} |"
            )

    lines.extend(
        [
            "",
            "### Paired ON − OFF deltas at matched seeds",
            "",
            "Positive accuracy means ON is higher. Positive cost means ON uses "
            "more resources. Deltas are paired by seed before reporting mean ± "
            "sample standard deviation.",
            "",
            "| Budget | Accuracy delta | Teacher-token delta | GPU-proxy delta | Total-wall delta |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for round_index in range(1, 5):
        on_by_seed = {
            int(row["seed"]): row
            for row in round_cost_rows
            if row["verification"] == "on" and row["round"] == round_index
        }
        off_by_seed = {
            int(row["seed"]): row
            for row in round_cost_rows
            if row["verification"] == "off" and row["round"] == round_index
        }

        def paired_delta(key: str) -> list[float]:
            return [
                float(on_by_seed[seed][key]) - float(off_by_seed[seed][key])
                for seed in SEEDS
            ]

        lines.append(
            f"| {round_index * 32} | "
            f"{_display_delta(paired_delta('stopping_accuracy'), scale=100.0, decimals=4, suffix=' pp')} | "
            f"{_display_delta(paired_delta('teacher_tokens_cumulative'), decimals=0)} | "
            f"{_display_delta(paired_delta('gpu_seconds_cumulative'), scale=1.0 / 60.0, suffix=' min')} | "
            f"{_display_delta(paired_delta('total_wall_seconds_cumulative'), scale=1.0 / 3600.0, suffix=' h')} |"
        )

    lines.extend(
        [
            "",
            "The budget-32 point has zero in-run teacher tokens because both arms "
            "reuse the same prebuilt initial pool; that shared construction cost is "
            "outside the ablation run scope. ON begins paying the verification cost "
            "during the refresh used to construct the budget-64 checkpoint.",
            "",
            "## Performance–cost curves",
            "",
            "![Validation performance against budget, cumulative teacher tokens, GPU time, and total wall cost](figures/05_performance_cost_curves.svg)",
            "",
            "Each marker is one fixed budget (32, 64, 96, or 128 examples); "
            "lines show three-seed means and vertical bars show sample standard "
            "deviation of validation accuracy. These curves use validation accuracy, "
            "not repeated final-HANS evaluation.",
        ]
    )

    lines.extend(
        [
            "",
            "## Cost-accounting reconciliation",
            "",
            "The performance–cost curve stops accounting at each trained "
            "checkpoint. The run-level result additionally records the post-round-4 "
            "hypothesis audit and sealed-final evaluation. These later operations "
            "are shown separately and are not charged to an earlier budget point.",
            "",
            "| Run | Curve tokens to B128 | Post-R4 audit tokens | Result tokens | Curve GPU (min) | Logged loop GPU (min) | Final eval (min) | Check |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for mode in MODES:
        for result in runs[mode]:
            seed = int(result["seed"])
            curve_row = next(
                row
                for row in round_cost_rows
                if row["verification"] == mode
                and int(row["seed"]) == seed
                and int(row["round"]) == 4
            )
            resources = result["resource_usage"]
            teacher = resources["teacher"]
            result_tokens = float(teacher["input_tokens"]) + float(
                teacher["output_tokens"]
            )
            curve_tokens = float(curve_row["teacher_tokens_cumulative"])
            post_audit_tokens = result_tokens - curve_tokens
            curve_gpu_seconds = float(curve_row["gpu_seconds_cumulative"])
            logged_loop_gpu_seconds = sum(
                float(resources.get(key, 0.0)) for key in LOOP_GPU_TIME_FIELDS
            )
            final_eval_seconds = float(
                resources.get("final_evaluation_gpu_wall_seconds", 0.0)
            )
            check = (
                "exact"
                if abs(curve_gpu_seconds - logged_loop_gpu_seconds) < 1e-6
                and abs(curve_tokens + post_audit_tokens - result_tokens) < 1e-6
                else "review"
            )
            lines.append(
                f"| {mode.upper()} seed {seed} | {curve_tokens:.0f} | "
                f"{post_audit_tokens:.0f} | {result_tokens:.0f} | "
                f"{curve_gpu_seconds / 60.0:.2f} | "
                f"{logged_loop_gpu_seconds / 60.0:.2f} | "
                f"{final_eval_seconds / 60.0:.2f} | {check} |"
            )
    lines.extend(
        [
            "",
            "Accounting rules:",
            "",
            "- `teacher_wall_clock_seconds` is contained in "
            "`refresh_wall_seconds`; the two are never summed.",
            "- GPU fields are measured wall-clock proxies, not hardware-utilization "
            "integrals or billable GPU-hours.",
            "- Shared initial-pool construction is excluded from both arms, as "
            "declared by `resource_usage.scope`; therefore round 1 has zero in-run "
            "teacher tokens.",
            "- Post-round-4 final hypothesis audit and sealed-final evaluation are "
            "reported separately from the cost-to-checkpoint curve.",
            "- Failed attempts, Ctrl-C/retry overhead, and archived incomplete runs "
            "are not represented by successful-run artifacts.",
        ]
    )

    lines.extend(
        [
            "",
            "## Training and resource summary",
            "",
            "| Run | Committed AdamW steps | Best epochs per round | "
            "GPU proxy seconds incl. final eval | Teacher seconds | Teacher input/output tokens incl. final audit |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for result in runs[mode]:
            epochs = ",".join(
                str(item["best_epoch"]) for item in result["training_history"]
            )
            resources = result["resource_usage"]
            teacher = resources["teacher"]
            gpu_seconds = sum(float(resources.get(key, 0.0)) for key in GPU_TIME_FIELDS)
            lines.append(
                f"| {mode.upper()} seed {result['seed']} | "
                f"{result['optimizer_steps']} | {epochs} | "
                f"{gpu_seconds:.2f} | "
                f"{float(teacher.get('teacher_wall_clock_seconds', 0.0)):.2f} | "
                f"{int(teacher['input_tokens'])}/{int(teacher['output_tokens'])} |"
            )

    missing = [
        f"{mode}:{seed}"
        for mode in MODES
        for seed in SEEDS
        if not (
            root / f"verification_{mode}" / f"seed_{seed}" / "result.json"
        ).is_file()
    ]
    lines.extend(
        [
            "",
            "## Completion",
            "",
            f"- Completed runs: {completed}/6",
            f"- Missing runs: {', '.join(missing) if missing else 'none'}",
            "",
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
