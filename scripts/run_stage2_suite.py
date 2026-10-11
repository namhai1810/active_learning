#!/usr/bin/env python3
"""Run the predeclared Stage 2 Experiment A or B grid without overwriting runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=("A", "B"))
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.config import load_yaml_with_extends, require_mapping

    config = load_yaml_with_extends(args.config)
    benchmark = require_mapping(config, "stage2_benchmark")
    seeds = [int(item) for item in benchmark["seeds"]]
    if args.experiment == "A":
        methods = (
            "zero_shot", "random_mnli", "uncertainty_mnli",
            "random_synthetic", "failure_driven_no_verification", "ours_verified",
        )
        budgets = [int(item) for item in benchmark["budgets"]]
        rounds, per_round = 1, None
    else:
        methods = ("random_mnli", "uncertainty_mnli", "ours_verified")
        budgets = [128]
        rounds, per_round = 4, 32
    root = Path(args.output_root).resolve() / f"experiment_{args.experiment.lower()}"
    for method in methods:
        for budget in budgets:
            for seed in seeds:
                output = root / method / f"budget_{budget}" / f"seed_{seed}"
                if (output / "result.json").is_file():
                    print(f"skip completed: {output}")
                    continue
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
                if per_round is not None:
                    command.extend(["--per-round-budget", str(per_round)])
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
