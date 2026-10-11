#!/usr/bin/env python3
"""Statistically verify debug-only HANS oracle candidate hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION, split_hans_five_way
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.hypotheses.types import FailureHypothesis
from src.utils.config import load_yaml, require_mapping
from src.utils.io import write_json, write_jsonl
from src.verification.bounds import verify_binary_outcomes
from src.verification.hans_oracle_probes import (
    build_probe_outcomes,
    select_hans_oracle_probes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Verification YAML config")
    parser.add_argument("--run-dir", required=True, help="Existing oracle run directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only existing verification artifacts",
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
        raise ValueError(f"No prediction records found in {path}")
    return records


def _validate_split_manifest(
    manifest: dict[str, Any], split: Any
) -> None:
    expected = {
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    if manifest.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Unsupported or missing five-way split version")
    for field, ids in expected.items():
        if manifest.get(field) != ids:
            raise ValueError(
                f"Split manifest field {field!r} does not match the "
                "deterministically reconstructed split"
            )
    sets = [set(ids) for ids in expected.values()]
    if any(sets[left] & sets[right] for left in range(5) for right in range(left + 1, 5)):
        raise ValueError("Five-way HANS split IDs overlap")


def _load_hypotheses(payload: dict[str, Any]) -> list[FailureHypothesis]:
    if payload.get("oracle_debug") is not True:
        raise ValueError("Hypothesis artifact must declare oracle_debug: true")
    if payload.get("split_role") != "hypothesis_discovery":
        raise ValueError(
            "Hypotheses must originate from split_role='hypothesis_discovery'"
        )
    raw_hypotheses = payload.get("hypotheses")
    if not isinstance(raw_hypotheses, list):
        raise ValueError("hypotheses must be a list")
    hypotheses = []
    for value in raw_hypotheses:
        hypothesis = FailureHypothesis(**value)
        expected = HANS_ORACLE_HYPOTHESES.get(hypothesis.heuristic)
        if expected != hypothesis:
            raise ValueError(f"Unsupported or modified oracle hypothesis: {value}")
        hypotheses.append(hypothesis)
    if len({hypothesis.id for hypothesis in hypotheses}) != len(hypotheses):
        raise ValueError("Duplicate candidate hypothesis IDs")
    return hypotheses


def _print_results(results: list[dict[str, Any]]) -> None:
    header = ("hypothesis", "n", "nu_hat", "radius", "LCB", "tau_v", "certified")
    rows = [header]
    for result in results:
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{value:.4f}"

        rows.append(
            (
                result["hypothesis_id"],
                str(result["n"]),
                fmt(result["nu_hat"]),
                fmt(result["confidence_radius"]),
                fmt(result["lower_bound"]),
                fmt(result["threshold"]),
                "yes" if result["certified"] else "no",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(7)]
    for row_index, row in enumerate(rows):
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("-+-".join("-" * width for width in widths))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("mode") != "oracle_hans" or config.get("oracle_debug") is not True:
        raise ValueError("Config must declare mode: oracle_hans and oracle_debug: true")
    hans = require_mapping(config, "hans")
    verification = require_mapping(config, "verification")
    discovery_fraction = float(hans["discovery_fraction"])
    verification_fraction = float(hans["verification_fraction"])
    intervention_fraction = float(hans["intervention_fraction"])
    gain_eval_fraction = float(hans["gain_eval_fraction"])
    split_seed = int(hans["split_seed"])
    delta = float(verification["delta"])
    tau_v = float(verification["tau_v"])
    n_probes = int(verification["n_probes"])
    K = int(verification["max_hypotheses_per_round"])
    T = int(verification["max_rounds"])
    verification_seed = int(verification["seed"])

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    hypotheses_path = run_dir / "hypotheses.json"
    manifest_path = run_dir / "hans_oracle_split_v2.json"
    for path in (predictions_path, hypotheses_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    output_dir = run_dir / "verification"
    output_paths = (output_dir / "probes.jsonl", output_dir / "results.json")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite verification artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite explicitly."
        )

    records = _read_jsonl(predictions_path)
    if {record.get("split_role") for record in records} != {None}:
        raise ValueError("Source predictions must be an unsplit Milestone 3 artifact")
    split = split_hans_five_way(
        records,
        discovery_fraction,
        verification_fraction,
        intervention_fraction,
        gain_eval_fraction,
        split_seed,
    )
    _validate_split_manifest(_read_json(manifest_path), split)
    hypotheses = _load_hypotheses(_read_json(hypotheses_path))
    if len(hypotheses) > K:
        raise ValueError(
            f"Found {len(hypotheses)} candidates, exceeding configured K={K}"
        )

    all_probe_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    probe_ids: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        selected = select_hans_oracle_probes(
            hypothesis,
            split.verification,
            n_probes=n_probes,
            seed=verification_seed,
        )
        probes = build_probe_outcomes(hypothesis, selected)
        result = verify_binary_outcomes(
            hypothesis.id,
            [probe["z"] for probe in probes],
            delta=delta,
            tau_v=tau_v,
            K=K,
            T=T,
        )
        all_probe_records.extend(probes)
        result_records.append(result.to_dict())
        probe_ids[hypothesis.id] = [probe["example_id"] for probe in probes]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "probes.jsonl", all_probe_records)
    write_json(
        output_dir / "results.json",
        {
            "oracle_debug": True,
            "verification_mode": "hans_oracle_debug",
            "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
            "independence_assumption": (
                "The concentration bound assumes conditionally independent "
                "Bernoulli probe outcomes; a fixed HANS benchmark does not by "
                "itself establish this assumption."
            ),
            "delta": delta,
            "tau_v": tau_v,
            "K": K,
            "T": T,
            "requested_n_probes": n_probes,
            "probe_ids": probe_ids,
            "results": result_records,
        },
    )
    _print_results(result_records)
    print(
        f"split: discovery={len(split.hypothesis_discovery)}, "
        f"verification={len(split.verification)}, "
        f"intervention={len(split.intervention_pool)}, "
        f"gain_eval={len(split.gain_eval)}, "
        f"final_test={len(split.final_test)}, pairwise_overlap=0"
    )
    print(f"Wrote verification artifacts to {output_dir}")


if __name__ == "__main__":
    main()
