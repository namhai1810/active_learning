#!/usr/bin/env python3
"""CPU-only replay of legacy and range-aware V7 Cost-Aware decisions.

This script reads cached observations and cost ledgers only.  It never loads a
model, invokes a teacher, trains, performs inference, or mutates source runs.
Rounds after the first counterfactual Range-Aware STOP are reported as
off-trajectory diagnostics because their artifacts could not exist on the
counterfactual stopped trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision.v7_minimal_cost_aware import MIXED_POLICY_ID
from src.gain.range_aware import compare_legacy_and_range_aware_bounds
EXPECTED = {
    "required_pilots": 32,
    "M": 1.0,
    "delta": 0.05,
    "K": 1,
    "T": 7,
    "lambda_cost": 0.2,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _protocol_config_path(run_dir: Path) -> Path:
    """Resolve the immutable config artifact for native and replay runs."""
    native = run_dir / "config.yaml"
    replay_snapshot = run_dir / "config_snapshot.json"
    if native.is_file():
        return native
    if replay_snapshot.is_file():
        return replay_snapshot
    raise FileNotFoundError(
        "Missing protocol config artifact; expected config.yaml or "
        f"config_snapshot.json under {run_dir}"
    )


def _validate_protocol(run_dir: Path) -> dict[str, Any]:
    config_path = _protocol_config_path(run_dir)
    if config_path.suffix == ".json":
        config = _read_json(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("cost_aware"), dict):
        raise ValueError(f"Missing resolved cost_aware config: {config_path}")
    cost_aware = config["cost_aware"]
    for key, expected in EXPECTED.items():
        actual = cost_aware.get(key)
        if actual != expected:
            raise ValueError(
                f"Protocol mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    semantics = cost_aware.get("decision_cost_semantics")
    if semantics != "marginal_incremental_cost_if_continue_after_pilots":
        raise ValueError(f"Unexpected decision cost semantics: {semantics!r}")
    return cost_aware


def _round_payloads(run_dir: Path) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "cost_aware").glob("round_*/decision.json")):
        payload = _read_json(path)
        payloads[int(payload["round"])] = payload
    result_path = run_dir / "result.json"
    if result_path.is_file():
        result = _read_json(result_path)
        for payload in result.get("cost_aware_rounds", []):
            if isinstance(payload, dict) and payload.get("round") is not None:
                payloads.setdefault(int(payload["round"]), payload)
    return payloads


def _cached_observations(
    run_dir: Path,
    round_index: int,
    payload: dict[str, Any],
    result: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    round_root = run_dir / "cost_aware" / f"round_{round_index:03d}"
    jsonl_path = round_root / "pilot_observations.jsonl"
    if jsonl_path.is_file():
        return _read_jsonl(jsonl_path), [
            {"path": str(jsonl_path), "sha256": _sha256(jsonl_path)}
        ]
    paths = sorted((round_root / "pilots").glob("pilot_*/observation.json"))
    if paths:
        return [_read_json(path) for path in paths], [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ]
    embedded = payload.get("pilot_observations")
    if isinstance(embedded, list) and embedded:
        return [dict(item) for item in embedded], [
            {
                "path": f"{run_dir}/result-or-decision:round-{round_index}",
                "sha256": _sha256_payload(embedded),
            }
        ]
    if result is not None:
        manifest = (
            result.get("replay", {})
            .get("external_pilot_cache", {})
            .get("observations", [])
        )
        selected = [
            item for item in manifest
            if isinstance(item, dict) and int(item.get("round", -1)) == round_index
        ]
        if selected:
            source_paths = [Path(str(item["path"])) for item in selected]
            return [_read_json(path) for path in source_paths], [
                {"path": str(path), "sha256": _sha256(path)} for path in source_paths
            ]
    return [], []


def replay_seed(run_dir: Path, seed: int) -> dict[str, Any]:
    _validate_protocol(run_dir)
    protocol_config_path = _protocol_config_path(run_dir)
    result_path = run_dir / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    payloads = _round_payloads(run_dir)
    rows: list[dict[str, Any]] = []
    first_range_stop: int | None = None

    for round_index in sorted(payloads):
        if round_index < 2:
            continue
        payload = payloads[round_index]
        old_decision = payload.get("decision") or {}
        cost_payload = payload.get("decision_cost") or {}
        observations, provenance = _cached_observations(
            run_dir, round_index, payload, result
        )
        old_evaluated = bool(old_decision.get("gain_cost_stop_evaluated"))
        if not old_evaluated or len(observations) != EXPECTED["required_pilots"]:
            row = {
                "seed": seed,
                "round": round_index,
                "m": len(observations),
                "evidence_gate_preserved": True,
                "old": {
                    "action": old_decision.get("action"),
                    "reason": old_decision.get("reason"),
                },
                "range_aware": {
                    "action": old_decision.get("action"),
                    "reason": old_decision.get("reason"),
                    "comparison_status": "not_evaluated_due_to_evidence_gate",
                },
                "provenance": provenance,
            }
        else:
            comparison = compare_legacy_and_range_aware_bounds(
                observations,
                hypothesis_id=str(
                    (payload.get("gain_estimate") or {}).get(
                        "hypothesis_id", MIXED_POLICY_ID
                    )
                ),
                required_pilots=EXPECTED["required_pilots"],
                M=EXPECTED["M"],
                delta=EXPECTED["delta"],
                K=EXPECTED["K"],
                T=EXPECTED["T"],
                decision_cost=float(cost_payload["decision_cost"]),
                lambda_cost=EXPECTED["lambda_cost"],
            )
            row = {
                "seed": seed,
                "round": round_index,
                "m": comparison["m"],
                "evidence_gate_preserved": True,
                "delta_hat": comparison["delta_hat"],
                "baseline_accuracy": comparison["range_aware_audit"][
                    "baseline_accuracy"
                ],
                "support": [
                    comparison["range_aware_audit"]["support_lower"],
                    comparison["range_aware_audit"]["support_upper"],
                ],
                "decision_cost": comparison["decision_cost"],
                "old": comparison["legacy_symmetric_support"],
                "range_aware": comparison["range_aware_fixed_baseline"],
                "assumption_audit": comparison["range_aware_audit"],
                "provenance": provenance,
            }
            if comparison["range_aware_fixed_baseline"]["action"] == "stop":
                if first_range_stop is None:
                    first_range_stop = round_index
        row["trajectory_status"] = (
            "on_counterfactual_trajectory"
            if first_range_stop is None or round_index <= first_range_stop
            else "post_stop_diagnostic_only"
        )
        rows.append(row)

    complete = (run_dir / ".complete").is_file()
    return {
        "schema_version": "v7_range_aware_cpu_replay_v1",
        "seed": seed,
        "source_run": str(run_dir.resolve()),
        "source_complete": complete,
        "source_artifacts": {
            "config": {
                "path": str(protocol_config_path),
                "sha256": _sha256(protocol_config_path),
            },
            "result": (
                {"path": str(result_path), "sha256": _sha256(result_path)}
                if result_path.is_file()
                else None
            ),
        },
        "protocol": {
            **EXPECTED,
            "gain": "accuracy_after - accuracy_before",
            "range_aware_radius": "sqrt(log(2*K*T/delta)/(2*m))",
            "support": "[-A0, 1-A0] with width 1",
            "decision_cost_semantics": (
                "marginal_incremental_cost_if_continue_after_pilots"
            ),
            "population_level_guarantee_claimed": False,
        },
        "first_range_aware_stop_round": first_range_stop,
        "range_aware_stop_reason": (
            "nonpositive_mixed_policy_optimistic_net_gain"
            if first_range_stop is not None
            else ("no_gain_cost_stop_in_available_rounds" if rows else "no_rounds_available")
        ),
        "rounds": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# V7 Minimal Cost-Aware: Old vs Range-Aware Hoeffding",
        "",
        "> CPU replay from cached artifacts only. No teacher, training, pilot, or inference is run.",
        "> Guarantees are conditional on the fixed evaluation shard and configured training randomization; no population-level guarantee is claimed.",
        "",
        "| Seed | Round | m | Delta-hat | A0 | Old b | New b | Old Net-UCB | New Net-UCB | Old action | New action | Old reason | New reason | Net gap to STOP | b reduction to STOP | Trajectory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        for row in result["rounds"]:
            old = row["old"]
            new = row["range_aware"]
            lines.append(
                "| {seed} | {round} | {m} | {delta} | {baseline} | {old_b} | "
                "{new_b} | {old_net} | {new_net} | {old_action} | {new_action} | "
                "{old_reason} | {new_reason} | {gap} | {radius_gap} | {trajectory} |".format(
                    seed=row["seed"],
                    round=row["round"],
                    m=row["m"],
                    delta=_fmt(row.get("delta_hat")),
                    baseline=_fmt(row.get("baseline_accuracy")),
                    old_b=_fmt(old.get("radius")),
                    new_b=_fmt(new.get("radius")),
                    old_net=_fmt(old.get("net_ucb")),
                    new_net=_fmt(new.get("net_ucb")),
                    old_action=old.get("action", "—"),
                    new_action=new.get("action", "—"),
                    old_reason=old.get("reason", "—"),
                    new_reason=new.get("reason", "—"),
                    gap=_fmt(new.get("net_ucb_gap_to_stop")),
                    radius_gap=_fmt(new.get("radius_reduction_needed_to_stop")),
                    trajectory=row["trajectory_status"],
                )
            )
    lines.extend(["", "## Counterfactual stopping summary", ""])
    for result in results:
        lines.append(
            f"- Seed {result['seed']}: first Range-Aware STOP round = "
            f"{_fmt(result['first_range_aware_stop_round'])}; "
            f"reason = `{result['range_aware_stop_reason']}`."
        )
    lines.extend(
        [
            "",
            "Rounds after a first counterfactual STOP are diagnostic only: cached later-round artifacts belong to the historical continued trajectory.",
            "Evidence gates are preserved; missing certified hypotheses or incomplete pilots are not converted into Gain-Cost STOP decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_resume_extension(
    existing: dict[str, Any], computed: dict[str, Any], destination: Path
) -> None:
    """Allow newly completed source rounds, but never rewrite old comparisons."""
    for key in ("schema_version", "seed", "source_run", "protocol"):
        if existing.get(key) != computed.get(key):
            raise ValueError(f"CPU replay resume metadata changed at {key}: {destination}")
    computed_by_round = {int(row["round"]): row for row in computed.get("rounds", [])}
    for old_row in existing.get("rounds", []):
        round_index = int(old_row["round"])
        if computed_by_round.get(round_index) != old_row:
            raise ValueError(
                f"Cached comparison for round {round_index} changed: {destination}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument(
        "--input-root",
        default="outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage2_diagnostics/v7_range_aware_replay",
    )
    parser.add_argument(
        "--report",
        default="reports/v7_minimal_cost_aware/RANGE_AWARE_REPLAY.md",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        source = input_root / f"seed_{seed}"
        destination = output_root / f"seed_{seed}" / "comparison.json"
        computed = replay_seed(source, seed)
        if destination.exists() and args.resume:
            existing = _read_json(destination)
            _validate_resume_extension(existing, computed, destination)
            _atomic_json(destination, computed)
        elif destination.exists():
            raise FileExistsError(
                f"Diagnostic output exists; use --resume to validate and reuse: {destination}"
            )
        else:
            _atomic_json(destination, computed)
        results.append(computed)
    report_path = Path(args.report)
    report_text = render_report(results)
    if report_path.exists() and args.resume:
        _atomic_text(report_path, report_text)
    elif report_path.exists():
        raise FileExistsError(f"Report exists; use --resume: {report_path}")
    else:
        _atomic_text(report_path, report_text)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
