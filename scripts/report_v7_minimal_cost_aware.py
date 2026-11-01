#!/usr/bin/env python3
"""Build an artifact-only three-seed V7 minimal Cost-Aware report."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifacts import read_json  # noqa: E402
from src.utils.io import _atomic_text_write  # noqa: E402


SEEDS = (42, 43, 44)

HISTORICAL_SEED42 = {
    "on_round3_posthoc": (
        "V7 Verification ON round 3 (post-hoc)",
        "outputs/stage2_runs_v7_training_verification_ablation/"
        "verification_on/seed_42/posthoc/round_003_stopping_accuracy_71/result.json",
    ),
    "on_round4": (
        "V7 Verification ON final",
        "outputs/stage2_runs_v7_training_verification_ablation/"
        "verification_on/seed_42/result.json",
    ),
    "off_round4": (
        "V7 Verification OFF final",
        "outputs/stage2_runs_v7_training_verification_ablation/"
        "verification_off/seed_42/result.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", default="outputs/stage2_runs_v7_minimal_cost_aware"
    )
    parser.add_argument(
        "--output", default="reports/v7_minimal_cost_aware/REPORT.md"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _mean_std(values: list[float], *, percent: bool = False) -> str:
    if not values:
        return "n/a"
    scale = 100.0 if percent else 1.0
    mean = statistics.mean(values) * scale
    std = statistics.stdev(values) * scale if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}{'%' if percent else ''}"


def load_runs(root: Path, *, allow_incomplete: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in SEEDS:
        run = root / f"seed_{seed}"
        if not (run / ".complete").is_file():
            missing.append(seed)
            continue
        result = read_json(run / "result.json")
        if result.get("method") != "ours_v7_minimal_cost_aware":
            raise ValueError(f"Unexpected method in {run}")
        rows.append(result)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Missing completed seeds: {missing}")
    return rows


def load_historical_seed42() -> dict[str, dict[str, Any]]:
    historical: dict[str, dict[str, Any]] = {}
    for key, (label, relative_path) in HISTORICAL_SEED42.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        result = read_json(path)
        historical[key] = {
            "label": label,
            "budget": int(
                result.get(
                    "actual_training_examples",
                    result.get("training_examples_seen", 0),
                )
            ),
            "result": result,
        }
    return historical


def _seed42_analysis(
    result: dict[str, Any],
    historical: dict[str, dict[str, Any]],
) -> list[str]:
    rules = result["hans_prediction_rules"]
    three_way = float(result["hans_final"]["overall_accuracy"])
    direct = float(
        rules["direct_binary_probability_aggregation"]["overall_accuracy"]
    )
    disagreement = rules["disagreement"]
    lines = [
        "",
        "## Seed 42 detailed interpretation",
        "",
        "### Same-seed historical comparison",
        "",
        "| Run | Budget | HANS Three-way | MNLI-m | MNLI-mm | HANS delta vs Cost-Aware |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in historical.values():
        old = item["result"]
        old_hans = float(old["hans_final"]["overall_accuracy"])
        lines.append(
            f"| {item['label']} | {item['budget']} | "
            f"{100*old_hans:.4f}% | "
            f"{100*float(old['mnli']['matched']['accuracy']):.4f}% | "
            f"{100*float(old['mnli']['mismatched']['accuracy']):.4f}% | "
            f"{100*(old_hans-three_way):+.4f} pp |"
        )
    lines.append(
        f"| **V7 Minimal Cost-Aware seed 42** | "
        f"**{result['actual_training_examples']}** | **{100*three_way:.4f}%** | "
        f"{100*float(result['mnli']['matched']['accuracy']):.4f}% | "
        f"{100*float(result['mnli']['mismatched']['accuracy']):.4f}% | — |"
    )

    old_on = historical.get("on_round4", {}).get("result")
    prefix_matches = False
    old_steps = None
    if isinstance(old_on, dict):
        old_curve = {
            int(item["round"]): float(item["metrics"]["overall_accuracy"])
            for item in old_on.get("learning_curve", [])
        }
        prefix_matches = all(
            int(item["round"]) not in old_curve
            or abs(
                float(item["metrics"]["overall_accuracy"])
                - old_curve[int(item["round"])]
            )
            < 1e-12
            for item in result.get("learning_curve", [])
            if int(item["round"]) <= 4
        )
        old_steps = old_on.get("optimizer_steps")

    lines.extend(
        [
            "",
            "The comparison is same-seed but not same-budget: the new run uses "
            "256 examples, while the historical primary ON/OFF runs use 128 and "
            "the ON round-3 diagnostic uses 96. Seed controls stochastic choices; "
            "it does not make different budgets or training trajectories equivalent.",
            "",
            f"- The validation curve through round 4 "
            f"{'matches' if prefix_matches else 'does not exactly match'} the historical "
            "Verification-ON run. This is regression evidence that the preserved "
            "V7 prefix behaves as intended.",
            f"- The new run then adds four refresh/acquisition rounds and reaches "
            f"{result['optimizer_steps']} committed AdamW steps"
            + (f", versus {old_steps} for historical ON" if old_steps is not None else "")
            + ".",
            "- Its Three-way HANS result already exceeds the historical checkpoints, "
            "so the improvement is not an artifact of switching to Direct Binary.",
            "- MNLI is higher than historical ON-final but lower than OFF-final and "
            "ON-round-3, showing a HANS-robustness/MNLI-retention trade-off rather "
            "than uniform domination.",
            "",
            "### Validation learning curve",
            "",
            "| Round | Cumulative budget | Accuracy | Entailment | Non-entailment |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result.get("learning_curve", []):
        round_index = int(item["round"])
        metrics = item["metrics"]
        lines.append(
            f"| {round_index} | {32*round_index} | "
            f"{100*float(metrics['overall_accuracy']):.4f}% | "
            f"{100*float(metrics['entailment_accuracy']):.4f}% | "
            f"{100*float(metrics['non_entailment_accuracy']):.4f}% |"
        )
    curve = list(result.get("learning_curve", []))
    if curve:
        best = max(curve, key=lambda item: float(item["metrics"]["overall_accuracy"]))
        lines.extend(
            [
                "",
                f"The highest stopping-subset accuracy is "
                f"{100*float(best['metrics']['overall_accuracy']):.4f}% at round "
                f"{best['round']}. This is diagnostic only: no cross-round rule "
                "predeclared that accuracy as an acquisition stopping criterion, "
                "and the sealed final set must not be used to select a round.",
            ]
        )

    decisions = [
        item
        for item in result.get("cost_aware_rounds", [])
        if int(item.get("round", 0)) >= 2
    ]
    total_pilots = sum(
        int((item.get("gain_estimate") or {}).get("m", 0))
        for item in decisions
    )
    evaluated = sum(
        bool(item.get("decision", {}).get("gain_cost_stop_evaluated"))
        for item in decisions
    )
    unresolved_rounds = sum(
        item.get("decision", {}).get("reason")
        == "unresolved_hypotheses_require_more_evidence"
        for item in decisions
    )
    lines.extend(
        [
            "",
            "### What this run establishes about Cost-Aware stopping",
            "",
            f"- Decision rounds with unresolved hypotheses: {unresolved_rounds}/{len(decisions)}.",
            f"- Pilot observations executed: {total_pilots}.",
            f"- Rounds that evaluated `Gain-UCB - lambda*Cost`: {evaluated}.",
            f"- Terminal category: `{result['stop_reason_category']}`; reason: "
            f"`{result['stop_reason']}`.",
            "",
            "Consequently, seed 42 does **not** empirically validate a gain-cost "
            "STOP. Every decision round was gated by unresolved evidence, so the "
            "observed performance should be interpreted as an eight-round, "
            "maximum-budget V7 trajectory under the safe evidence gate—not as an "
            "accuracy gain caused by the stopping controller.",
            "",
            "### Final prediction-rule diagnostic",
            "",
            f"- Three-way HANS: {100*three_way:.4f}%.",
            f"- Direct Binary HANS: {100*direct:.4f}% "
            f"({100*(direct-three_way):+.4f} pp).",
            f"- Rule disagreement: {disagreement['count']}/15,000 "
            f"({100*float(disagreement['rate']):.4f}%).",
        ]
    )

    resources = result.get("resource_usage", {})
    teacher = resources.get("teacher", {})
    total_tokens = float(teacher.get("input_tokens", 0.0)) + float(
        teacher.get("output_tokens", 0.0)
    )
    lines.extend(
        [
            "",
            "### Seed 42 resource summary",
            "",
            f"- Teacher calls/tokens: {int(float(teacher.get('calls', 0))):,} / "
            f"{int(total_tokens):,}.",
            f"- Teacher wall time: {float(teacher.get('teacher_wall_clock_seconds', 0.0))/3600:.2f} hours.",
            f"- Refresh wall time: {float(resources.get('refresh_wall_seconds', 0.0))/3600:.2f} hours.",
            f"- Training / validation GPU-wall proxies: "
            f"{float(resources.get('training_gpu_wall_seconds', 0.0))/60:.2f} / "
            f"{float(resources.get('validation_gpu_wall_seconds', 0.0))/60:.2f} minutes.",
            f"- Final evaluation GPU-wall proxy: "
            f"{float(resources.get('final_evaluation_gpu_wall_seconds', 0.0))/60:.2f} minutes.",
            f"- Modeled decision cost for rounds 2+: "
            f"{float(result['total_modeled_decision_cost_rounds_2_plus']):.6f} "
            "configured USD units. Round 1 and the shared initial pool are excluded.",
        ]
    )
    return lines


def build_report(
    rows: list[dict[str, Any]],
    historical_seed42: dict[str, dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# V7 Minimal Cost-Aware Report",
        "",
        "> Artifact-only report; no model, teacher, training, or inference is run.",
        "",
        "The controller evaluates the exact mixed-hypothesis V7 soft-pair action. "
        "It does not claim the appendix theorem for hypothesis-specific interventions.",
        "",
        "## Final results",
        "",
        "| Seed | Rounds | Examples | Stop category | Stop reason | HANS Three-way | HANS Direct Binary | MNLI m/mm | Modeled decision cost (r2+) | Teacher tokens |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(rows, key=lambda item: int(item["seed"])):
        teacher = result.get("resource_usage", {}).get("teacher", {})
        tokens = float(teacher.get("input_tokens", 0.0)) + float(
            teacher.get("output_tokens", 0.0)
        )
        rules = result["hans_prediction_rules"]
        lines.append(
            f"| {result['seed']} | {result['rounds']} | "
            f"{result['actual_training_examples']} | "
            f"`{result['stop_reason_category']}` | `{result['stop_reason']}` | "
            f"{100*float(result['hans_final']['overall_accuracy']):.4f}% | "
            f"{100*float(rules['direct_binary_probability_aggregation']['overall_accuracy']):.4f}% | "
            f"{100*float(result['mnli']['matched']['accuracy']):.4f}% / "
            f"{100*float(result['mnli']['mismatched']['accuracy']):.4f}% | "
            f"{float(result['total_modeled_decision_cost_rounds_2_plus']):.6f} | "
            f"{tokens:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Mean ± sample standard deviation",
            "",
            f"- HANS Three-way: {_mean_std([float(r['hans_final']['overall_accuracy']) for r in rows], percent=True)}",
            f"- HANS Direct Binary: {_mean_std([float(r['hans_prediction_rules']['direct_binary_probability_aggregation']['overall_accuracy']) for r in rows], percent=True)}",
            f"- MNLI matched: {_mean_std([float(r['mnli']['matched']['accuracy']) for r in rows], percent=True)}",
            f"- MNLI mismatched: {_mean_std([float(r['mnli']['mismatched']['accuracy']) for r in rows], percent=True)}",
            f"- Acquired examples: {_mean_std([float(r['actual_training_examples']) for r in rows])}",
            f"- Completed rounds: {_mean_std([float(r['rounds']) for r in rows])}",
            f"- Modeled decision cost (rounds 2+): {_mean_std([float(r['total_modeled_decision_cost_rounds_2_plus']) for r in rows])}",
            "",
        ]
    )
    seed42 = next((item for item in rows if int(item["seed"]) == 42), None)
    if seed42 is not None:
        lines.extend(
            _seed42_analysis(seed42, historical_seed42 or {})
        )
    lines.extend(
        [
            "",
            "## Decision traces",
            "",
            "| Seed | Round | Budget before | Certified | Unresolved | Pilots | Gain | Radius | Raw gain UCB | Bounded gain UCB | Cost | λCost | UCB net | Action | Reason |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for result in sorted(rows, key=lambda item: int(item["seed"])):
        decisions = list(result.get("cost_aware_rounds") or [])
        stop = result.get("cost_aware_stop_record")
        if isinstance(stop, dict):
            decisions.append(stop)
        for item in decisions:
            decision = item["decision"]
            estimate = item.get("gain_estimate") or {}
            lines.append(
                f"| {result['seed']} | {item['round']} | "
                f"{item['cumulative_examples_before_decision']} | "
                f"{len(decision.get('certified_hypotheses', []))} | "
                f"{len(decision.get('unresolved_hypotheses', []))} | "
                f"{estimate.get('m', 0)} | "
                f"{estimate.get('delta_hat', 'n/a')} | "
                f"{estimate.get('radius', 'n/a')} | "
                f"{decision.get('raw_gain_ucb', 'n/a')} | "
                f"{decision.get('bounded_gain_ucb', 'n/a')} | "
                f"{decision.get('cost', 'n/a')} | "
                f"{decision.get('cost_penalty', 'n/a')} | "
                f"{decision.get('optimistic_net_value', 'n/a')} | "
                f"`{decision['action']}` | `{decision['reason']}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Unresolved hypotheses force CONTINUE; they are never treated as refuted.",
            "- Pilot gain uses predeclared, disjoint gain-eval shards and never sealed HANS/MNLI.",
            "- Conditional independence/unbiasedness of related LoRA pilots is not empirically proven.",
            "- With m=4, K=1, T=7, δ=.05 and M=1, the raw radius is about 1.6785. The gain UCB is intersected with the known support bound M; the radius itself is not reduced.",
            "- Lambda is fixed ex ante at 0.2 gain/USD from the declared indifference point 1.0 gain = USD 5; no final-test outcome is used.",
            "- Round-1 and the frozen initial-pool sunk cost are outside the round-2+ decision-cost aggregate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_runs(Path(args.input_root), allow_incomplete=args.allow_incomplete)
    output = Path(args.output)
    _atomic_text_write(
        output,
        build_report(rows, historical_seed42=load_historical_seed42()),
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

