#!/usr/bin/env python3
"""Run the predeclared Stage 2 Experiment A or B grid without overwriting runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_A_METHODS = (
    "zero_shot", "random_mnli", "uncertainty_mnli",
    "random_synthetic", "failure_driven_no_verification", "ours_verified",
)
LITERATURE_METHODS = (
    "margin_mnli", "coreset_mnli", "badge_mnli", "cal_mnli",
    "margin_synthetic", "coreset_synthetic", "badge_synthetic", "cal_synthetic",
)
EXPERIMENT_B_METHODS = (
    "random_mnli", "uncertainty_mnli", "ours_verified", *LITERATURE_METHODS,
)
ALL_METHODS = tuple(dict.fromkeys((*EXPERIMENT_A_METHODS, *EXPERIMENT_B_METHODS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=("A", "B"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        help="Run only these methods (default: the complete experiment grid).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run only these configured seeds (default: every configured seed).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        help="Run only these configured budgets (default: every configured budget).",
    )
    parser.add_argument(
        "--ours-objective",
        choices=("pairwise", "ce"),
        default="pairwise",
        help=(
            "Training objective used by ours_verified. Acquisition and selected "
            "IDs are unchanged, enabling a controlled objective ablation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.config import load_yaml_with_extends, require_mapping

    config = load_yaml_with_extends(args.config)
    benchmark = require_mapping(config, "stage2_benchmark")
    configured_seeds = [int(item) for item in benchmark["seeds"]]
    configured_budgets = [int(item) for item in benchmark["budgets"]]
    if args.experiment == "A":
        default_methods = EXPERIMENT_A_METHODS
        default_budgets = configured_budgets
        rounds, per_round = 1, None
    else:
        default_methods = ("random_mnli", "uncertainty_mnli", "ours_verified")
        allowed_methods = EXPERIMENT_B_METHODS
        default_budgets = [128]
        rounds, per_round = 4, 32
    methods = tuple(args.methods) if args.methods else default_methods
    seeds = list(args.seeds) if args.seeds else configured_seeds
    budgets = list(args.budgets) if args.budgets else default_budgets
    if args.experiment == "A":
        allowed_methods = EXPERIMENT_A_METHODS
    invalid_methods = sorted(set(methods) - set(allowed_methods))
    invalid_seeds = sorted(set(seeds) - set(configured_seeds))
    invalid_budgets = sorted(set(budgets) - set(default_budgets))
    if invalid_methods:
        raise ValueError(
            f"Methods are outside Experiment {args.experiment}: {invalid_methods}"
        )
    if invalid_seeds:
        raise ValueError(f"Seeds are outside the configured grid: {invalid_seeds}")
    if invalid_budgets:
        raise ValueError(
            f"Budgets are outside Experiment {args.experiment}: {invalid_budgets}"
        )
    root = Path(args.output_root).resolve() / f"experiment_{args.experiment.lower()}"
    jobs = [
        (method, budget, seed)
        for method in methods
        for budget in budgets
        for seed in seeds
    ]
    suite_started = time.monotonic()
    for index, (method, budget, seed) in enumerate(jobs, start=1):
        output = root / method / f"budget_{budget}" / f"seed_{seed}"
        label = (
            f"method={method} budget={budget} seed={seed}"
        )
        if (output / "result.json").is_file():
            print(f"[RUN {index}/{len(jobs)}][SKIP] {label}", flush=True)
            continue
        print(f"[RUN {index}/{len(jobs)}][START] {label}", flush=True)
        started = time.monotonic()
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_stage2_benchmark.py"),
            "--config", args.config,
            "--experiment", args.experiment,
            "--method", method,
            "--seed", str(seed),
            "--budget", str(budget),
            "--rounds", str(rounds),
            "--output-dir", str(output),
        ]
        if method == "ours_verified":
            command.extend(["--ours-objective", args.ours_objective])
        if per_round is not None:
            command.extend(["--per-round-budget", str(per_round)])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        elapsed = time.monotonic() - started
        print(
            f"[RUN {index}/{len(jobs)}][DONE] {label} elapsed={elapsed / 60:.1f}m",
            flush=True,
        )
    suite_elapsed = time.monotonic() - suite_started
    print(
        f"[SUITE DONE] experiment={args.experiment} runs={len(jobs)} "
        f"elapsed={suite_elapsed / 3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
