#!/usr/bin/env python3
"""Build debug-only HANS oracle hypotheses from existing predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import HANS_HEURISTICS
from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION, split_hans_five_way
from src.hypotheses.oracle_hans import HansOracleHypothesisGenerator
from src.utils.config import load_yaml, require_mapping
from src.utils.io import write_json

ORACLE_OUTPUT_FILES = (
    "hypotheses.json",
    "hypothesis_evidence.json",
    "hans_oracle_split_v2.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Oracle YAML config")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing Milestone 3 run directory containing predictions.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only existing oracle hypothesis artifacts",
    )
    return parser.parse_args()


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
        raise ValueError(f"No predictions found in {path}")
    return records


def _validate_config(
    config: dict[str, Any],
) -> tuple[float, float, float, float, int, int]:
    if config.get("mode") != "oracle_hans":
        raise ValueError("Oracle config must set mode: oracle_hans")
    if config.get("oracle_debug") is not True:
        raise ValueError("Oracle config must explicitly set oracle_debug: true")
    hans = require_mapping(config, "hans")
    generation = require_mapping(config, "hypothesis_generation")
    discovery_fraction = float(hans["discovery_fraction"])
    verification_fraction = float(hans["verification_fraction"])
    intervention_fraction = float(hans["intervention_fraction"])
    gain_eval_fraction = float(hans["gain_eval_fraction"])
    split_seed = int(hans.get("split_seed", 42))
    min_errors = int(generation.get("min_errors", 1))
    return (
        discovery_fraction,
        verification_fraction,
        intervention_fraction,
        gain_eval_fraction,
        split_seed,
        min_errors,
    )


def _print_summary(evidence: dict[str, dict[str, Any]]) -> None:
    header = ("heuristic", "examples", "errors", "error_rate", "candidate")
    rows = [header]
    for heuristic in HANS_HEURISTICS:
        record = evidence[f"h_{heuristic}"]
        rows.append(
            (
                heuristic,
                str(record["num_examples"]),
                str(record["num_errors"]),
                f"{record['error_rate']:.4f}",
                "yes" if record["candidate"] else "no",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(5)]
    for row_index, row in enumerate(rows):
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("-+-".join("-" * width for width in widths))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    (
        discovery_fraction,
        verification_fraction,
        intervention_fraction,
        gain_eval_fraction,
        split_seed,
        min_errors,
    ) = _validate_config(config)
    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Milestone 3 predictions not found: {predictions_path}"
        )

    output_paths = [run_dir / filename for filename in ORACLE_OUTPUT_FILES]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite oracle artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite explicitly."
        )

    records = _read_jsonl(predictions_path)
    role_values = {record.get("split_role") for record in records}
    if role_values != {None}:
        raise ValueError(
            "Input predictions must be an unsplit Milestone 3 artifact. "
            "Refusing to reinterpret existing split_role values."
        )
    split = split_hans_five_way(
        records,
        discovery_fraction,
        verification_fraction,
        intervention_fraction,
        gain_eval_fraction,
        split_seed,
    )

    generator = HansOracleHypothesisGenerator(min_errors=min_errors)
    hypotheses, all_evidence = generator.generate(split.hypothesis_discovery)
    candidate_ids = {hypothesis.id for hypothesis in hypotheses}
    candidate_evidence = {
        hypothesis_id: evidence
        for hypothesis_id, evidence in all_evidence.items()
        if hypothesis_id in candidate_ids
    }

    write_json(
        run_dir / "hypotheses.json",
        {
            "mode": "oracle_hans",
            "oracle_debug": True,
            "split_role": "hypothesis_discovery",
            "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
        },
    )
    write_json(run_dir / "hypothesis_evidence.json", candidate_evidence)
    write_json(
        run_dir / "hans_oracle_split_v2.json",
        {
            "mode": "oracle_hans",
            "oracle_debug": True,
            "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
            "split_seed": split_seed,
            "discovery_fraction": discovery_fraction,
            "verification_fraction": verification_fraction,
            "intervention_fraction": intervention_fraction,
            "gain_eval_fraction": gain_eval_fraction,
            "num_total": len(records),
            "num_hypothesis_discovery": len(split.hypothesis_discovery),
            "num_verification": len(split.verification),
            "num_intervention_pool": len(split.intervention_pool),
            "num_gain_eval": len(split.gain_eval),
            "num_final_test": len(split.final_test),
            "hypothesis_discovery_ids": list(split.discovery_ids),
            "verification_ids": list(split.verification_ids),
            "intervention_pool_ids": list(split.intervention_pool_ids),
            "gain_eval_ids": list(split.gain_eval_ids),
            "final_test_ids": list(split.final_test_ids),
        },
    )

    _print_summary(all_evidence)
    print(
        f"split: discovery={len(split.hypothesis_discovery)}, "
        f"verification={len(split.verification)}, "
        f"intervention={len(split.intervention_pool)}, "
        f"gain_eval={len(split.gain_eval)}, "
        f"final_test={len(split.final_test)}, pairwise_overlap=0"
    )
    print(f"Wrote oracle-debug artifacts to {run_dir}")


if __name__ == "__main__":
    main()
