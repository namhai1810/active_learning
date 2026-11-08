#!/usr/bin/env python3
"""ARCHIVED CPU simulation of v4 adaptive pilot allocation.

Only cached pilot observations are revealed, in pilot-index order. Hidden
observations may be used after a decision solely to measure realized GPU-time
savings; they never influence allocation or statistical certificates.

Retained for historical report reproducibility; not part of the V7 runtime.
See ``archive/v7_cost_aware_experiments/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decision.adaptive_pilot_allocation import (  # noqa: E402
    AdaptivePilotConfig,
    FrozenPilotCostProjection,
    PilotObservation,
    freeze_pilot_cost_projection,
    simulate_adaptive_allocation,
    simulate_fixed_32,
    simulate_no_pilots,
)
from src.utils.config import load_yaml_with_extends  # noqa: E402


SCHEMA = "v7_adaptive_pilot_allocation_simulation_v4"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _first(mapping: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _number(mapping.get(key))
        if value is not None:
            return value
    return None


def _decision_records(run_dir: Path, result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for row in result.get("cost_aware_rounds", []):
        round_index = row.get("round", row.get("round_index"))
        if round_index is not None:
            records[int(round_index)] = row
    for path in sorted((run_dir / "cost_aware").glob("round_*/decision.json")):
        payload = read_json(path)
        round_index = payload.get("round", payload.get("round_index"))
        if round_index is None:
            round_index = int(path.parent.name.rsplit("_", 1)[-1])
        records[int(round_index)] = payload
    return records


def _decision_fields(record: Mapping[str, Any]) -> tuple[Optional[float], str, str]:
    decision = record.get("decision", record)
    cost = record.get("decision_cost", {})
    decision_cost = _first(cost, "decision_cost", "projected_decision_cost") if isinstance(cost, Mapping) else None
    if decision_cost is None:
        decision_cost = _first(decision, "cost", "decision_cost")
    action = str(decision.get("action", "continue")).lower()
    reason = str(decision.get("reason", "unavailable"))
    return decision_cost, action, reason


def _evidence_gate(reason: str) -> Optional[str]:
    lowered = reason.lower()
    if any(token in lowered for token in ("no_certified", "no_eligible", "insufficient_evidence", "evidence_not_exhausted")):
        return reason
    return None


def _hash_ids(ids: Any) -> Optional[str]:
    if not isinstance(ids, list):
        return None
    encoded = json.dumps([str(value) for value in ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pilot_observations(run_dir: Path, round_index: int) -> list[PilotObservation]:
    path = run_dir / "cost_aware" / f"round_{round_index:03d}" / "pilot_observations.jsonl"
    if not path.exists():
        return []
    observations: list[PilotObservation] = []
    for row_number, row in enumerate(read_jsonl(path)):
        baseline = None
        risk_before = _number(row.get("risk_before"))
        if risk_before is not None:
            baseline = 1.0 - risk_before
        if baseline is None:
            baseline = _number(row.get("accuracy_before"))
        gain = _number(row.get("gain"))
        if baseline is None or gain is None:
            raise ValueError(f"round {round_index} pilot {row_number} lacks baseline/gain")
        pilot_index = int(row.get("pilot_index", row_number))
        observations.append(
            PilotObservation(
                pilot_index=pilot_index,
                gain=gain,
                baseline_accuracy=baseline,
                gpu_seconds=_number(row.get("gpu_seconds")),
                eval_ids_hash=_hash_ids(row.get("eval_ids")),
            )
        )
    return observations


def _frozen_projection(
    resolved_config: Mapping[str, Any],
    analysis_spec: Mapping[str, Any],
) -> FrozenPilotCostProjection:
    """Freeze pilot cost before any cached observation is revealed."""

    cost_aware = resolved_config.get("cost_aware")
    training = resolved_config.get("training")
    closed_loop = resolved_config.get("closed_loop")
    if not isinstance(cost_aware, Mapping):
        raise ValueError("resolved config lacks cost_aware")
    if not isinstance(training, Mapping):
        raise ValueError("resolved config lacks training")
    if not isinstance(closed_loop, Mapping):
        raise ValueError("resolved config lacks closed_loop")
    cost = cost_aware.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError("resolved config lacks cost_aware.cost")

    training_examples = int(closed_loop["per_round_examples"])
    gain_examples = int(cost_aware["pilot_eval_size"])
    max_epochs = int(training["max_epochs"])
    expected = {
        "training_examples_per_pilot": training_examples,
        "gain_evaluation_examples_per_pass": gain_examples,
        "max_epochs": max_epochs,
    }
    for key, actual in expected.items():
        declared = int(analysis_spec[key])
        if declared != actual:
            raise ValueError(
                f"frozen pilot projection mismatch for {key}: "
                f"analysis={declared}, run_config={actual}"
            )
    return freeze_pilot_cost_projection(
        dict(cost),
        training_examples=training_examples,
        gain_evaluation_examples=gain_examples,
        gain_evaluation_passes=int(analysis_spec["gain_evaluation_passes"]),
        stopping_validation_examples=int(
            analysis_spec["stopping_validation_examples"]
        ),
        max_epochs=max_epochs,
        expected_currency=str(cost_aware["calibration_currency"]),
    )


def _row(
    seed: int,
    round_index: int,
    result: Any,
    *,
    post_stop: bool,
    pilot_projection: Optional[FrozenPilotCostProjection],
) -> dict[str, Any]:
    payload = result.to_dict()
    looks = payload.pop("looks")
    final_look = looks[-1] if looks else {}
    payload.update(
        {
            "seed": seed,
            "round_index": round_index,
            "diagnostic_only_after_counterfactual_stop": post_stop,
            "frozen_pilot_gpu_second_cost": (
                pilot_projection.frozen_pilot_gpu_second_cost
                if pilot_projection is not None else None
            ),
            "pilot_cost_currency": (
                pilot_projection.currency if pilot_projection is not None else None
            ),
            "projected_pilot_cost_per_pilot": (
                pilot_projection.projected_total_cost_per_pilot
                if pilot_projection is not None else None
            ),
            "projected_pilot_utility_penalty_per_pilot": (
                result.projected_pilot_utility_penalty_per_pilot
            ),
            "pilot_cost_projection": (
                pilot_projection.to_dict() if pilot_projection is not None else None
            ),
            "final_bootstrap_stop_probability": final_look.get(
                "bootstrap_stop_probability"
            ),
            "final_expected_information_benefit": final_look.get(
                "expected_information_benefit"
            ),
            "final_projected_information_cost_usd": final_look.get(
                "projected_information_cost"
            ),
            "final_projected_information_utility_penalty": final_look.get(
                "projected_information_utility_penalty"
            ),
            "final_value_of_information": final_look.get(
                "value_of_information"
            ),
            "final_voi_target_m": final_look.get("voi_target_m"),
            "looks": looks,
        }
    )
    return payload


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["looks_json"] = json.dumps(item.pop("looks", []), sort_keys=True)
        converted.append(item)
    return converted


def _strategy_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(int(row["seed"]), str(row["strategy"]), int(row["pilot_cap"])) for row in rows})
    summaries: list[dict[str, Any]] = []
    for seed, strategy, cap in groups:
        selected = [
            row for row in rows
            if int(row["seed"]) == seed
            and str(row["strategy"]) == strategy
            and int(row["pilot_cap"]) == cap
        ]
        summaries.append(
            {
                "seed": seed,
                "strategy": strategy,
                "pilot_cap": cap,
                "rounds": len(selected),
                "pilots_used": sum(int(row["pilots_used"]) for row in selected),
                "certified_stops": sum(row["certification"] == "certified_stop" for row in selected),
                "certified_continues": sum(row["certification"] == "certified_continue" for row in selected),
                "uncertified_continues": sum(row["certification"] == "uncertified_continue" for row in selected),
                "pilot_gpu_hours_used": sum(float(row["pilot_gpu_seconds_used"]) for row in selected) / 3600,
                "pilot_gpu_hours_saved": sum(float(row["pilot_gpu_seconds_saved"]) for row in selected) / 3600,
                "projected_pilot_cost": sum(
                    int(row["pilots_used"])
                    * float(row.get("projected_pilot_cost_per_pilot") or 0.0)
                    for row in selected
                ),
                "confidence_valid": all(bool(row["confidence_valid"]) for row in selected),
            }
        )
    return summaries


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> str:
    primary = [row for row in rows if row["strategy"] in {"adaptive", "fixed_32", "no_pilots_continue"}]
    lines = [
        "# V7 Decision-Aware Adaptive Pilot Allocation v4",
        "",
        "> CPU simulation over cached pilots. No future pilot observation, future round metric, or sealed test is visible to allocation decisions.",
        "",
        "## Validation",
        "",
        "| Seed | Status | Frozen GPU USD/s | Projected USD/pilot | Currency | Missing |",
        "|---:|---|---:|---:|---|---|",
    ]
    for item in validation:
        lines.append(
            f"| {item['seed']} | {item['status']} | {_fmt(item.get('frozen_pilot_gpu_second_cost'))} | "
            f"{_fmt(item.get('projected_pilot_cost_per_pilot'))} | "
            f"{item.get('pilot_cost_currency') or 'N/A'} | "
            f"{', '.join(item.get('missing', [])) or 'none'} |"
        )
    ready_projection = next(
        (
            item.get("pilot_cost_projection")
            for item in validation
            if item.get("pilot_cost_projection") is not None
        ),
        None,
    )
    if ready_projection is not None:
        lines += [
            "",
            "### Frozen per-pilot cost projection",
            "",
            "| Component | Projected GPU s | Projected USD |",
            "|---|---:|---:|",
            f"| LoRA training | {_fmt(ready_projection['projected_training_gpu_seconds'])} | {_fmt(ready_projection['projected_training_cost'])} |",
            f"| Evaluation + validation | {_fmt(ready_projection['projected_evaluation_gpu_seconds'])} | {_fmt(ready_projection['projected_evaluation_cost'])} |",
            f"| **Total / pilot** | {_fmt(ready_projection['projected_training_gpu_seconds'] + ready_projection['projected_evaluation_gpu_seconds'])} | **{_fmt(ready_projection['projected_total_cost_per_pilot'])}** |",
            "",
            "The USD total includes the unchanged per-run, per-item, and GPU-hour rates. VOI converts it to gain utility exactly once through `lambda`; measured pilot runtimes are reporting-only.",
        ]
    lines += [
        "",
        "## Primary comparison",
        "",
        "| Seed | R | Strategy | Pilots | Action | Certification | Net-UCB | P(STOP) | VOI utility | GPU h saved | Reason | Diagnostic only |",
        "|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in primary:
        lines.append(
            f"| {row['seed']} | {row['round_index']} | {row['strategy']} | {row['pilots_used']} | "
            f"{row['action']} | {row['certification']} | {_fmt(row.get('final_net_ucb'))} | "
            f"{_fmt(row.get('final_bootstrap_stop_probability'))} | "
            f"{_fmt(row.get('final_value_of_information'))} | "
            f"{_fmt(row.get('pilot_gpu_hours_saved'))} | {row['reason']} | "
            f"{row['diagnostic_only_after_counterfactual_stop']} |"
        )
    lines += [
        "",
        "## Aggregate pilot savings",
        "",
        "| Seed | Strategy | Cap | Pilots used | Projected pilot cost | Certified STOP | Certified CONTINUE | Uncertified CONTINUE | GPU h used | GPU h saved | Valid bounds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['seed']} | {row['strategy']} | {row['pilot_cap']} | {row['pilots_used']} | "
            f"{_fmt(row['projected_pilot_cost'])} | "
            f"{row['certified_stops']} | {row['certified_continues']} | {row['uncertified_continues']} | "
            f"{_fmt(row['pilot_gpu_hours_used'])} | {_fmt(row['pilot_gpu_hours_saved'])} | {row['confidence_valid']} |"
        )
    lines += [
        "",
        "## Pilot-cap sensitivity",
        "",
        "| Seed | Cap | Pilots used | Certified STOP | Uncertified CONTINUE | GPU h saved |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['seed']} | {row['pilot_cap']} | {row['pilots_used']} | "
            f"{row['certified_stops']} | {row['uncertified_continues']} | {_fmt(row['pilot_gpu_hours_saved'])} |"
        )
    lines += [
        "",
        "## Statistical interpretation",
        "",
        "- Adaptive looks use equal Bonferroni alpha spending over the declared schedule; the fixed-32 comparator remains a single-look v2.1 bound.",
        "- STOP is emitted only when a valid Range-Aware Net-UCB is non-positive.",
        "- A positive valid Net-LCB may certify CONTINUE; otherwise ending sampling is explicitly `uncertified_continue`.",
        "- Bootstrap stop probability is a decision-aware allocation heuristic using revealed pilots only. It never creates a statistical certificate.",
        "- VOI is reported in normalized gain utility: expected avoided USD and projected pilot USD are both multiplied by the frozen lambda (gain/USD).",
        "- Pilot cost is projected from the frozen recipe and cost config before outcomes are revealed; observed runtimes never enter VOI.",
        "- Realized savings use hidden pilot timings only after the simulated decision and never affect allocation.",
        "- The shared fixed 64-example evaluation shard limits population-level interpretation and may induce dependence not covered by an iid population claim.",
        "- Rounds after a counterfactual STOP remain diagnostic only, not a valid online trajectory.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v7_adaptive_pilot_allocation_v4.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--pilot-caps", nargs="+", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    sources = {int(item["seed"]): item for item in config["sources"]}
    seeds = args.seeds or sorted(sources)
    caps = args.pilot_caps or [int(value) for value in config["protocol"]["pilot_caps"]]
    primary_schedule = tuple(int(value) for value in config["protocol"]["schedule"])
    output_root = Path(config["outputs"]["root"])
    report_path = Path(config["outputs"]["report"])
    output_root.mkdir(parents=True, exist_ok=True)

    validation: list[dict[str, Any]] = []
    contexts: dict[
        int,
        tuple[
            Path,
            Mapping[str, Any],
            dict[int, Mapping[str, Any]],
            Optional[FrozenPilotCostProjection],
        ],
    ] = {}
    projection_spec = config["frozen_pilot_cost_projection"]
    for seed in seeds:
        source = sources.get(seed)
        missing: list[str] = []
        if source is None:
            validation.append({"seed": seed, "status": "missing", "missing": ["source_spec"]})
            continue
        run_dir = Path(source["run_dir"])
        result_path = run_dir / "result.json"
        if not result_path.exists():
            missing.append(str(result_path))
            validation.append({"seed": seed, "status": "missing", "missing": missing})
            continue
        result = read_json(result_path)
        decisions = _decision_records(run_dir, result)
        source_dir = Path(source["source_run_dir"]) if source.get("source_run_dir") else run_dir
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            config_path = source_dir / "config.yaml"
        projection: Optional[FrozenPilotCostProjection] = None
        if config_path.exists():
            try:
                resolved_config = load_yaml_with_extends(config_path)
                projection = _frozen_projection(resolved_config, projection_spec)
                if not math.isclose(
                    float(resolved_config["cost_aware"]["lambda_cost"]),
                    float(config["protocol"]["lambda_cost"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("analysis lambda differs from resolved run config")
            except (KeyError, TypeError, ValueError) as error:
                missing.append(f"frozen_pilot_cost_projection:{error}")
        else:
            missing.append("resolved_run_config")
        if projection is None:
            missing.append("frozen_pilot_cost_projection")
        for round_index in range(2, 9):
            record = decisions.get(round_index)
            if record is None:
                missing.append(f"decision_round_{round_index}")
                continue
            _, _, reason = _decision_fields(record)
            gate = _evidence_gate(reason)
            observations = _pilot_observations(run_dir, round_index)
            if gate is None and len(observations) < 32:
                missing.append(f"round_{round_index}_pilots:{len(observations)}/32")
        validation.append(
            {
                "seed": seed,
                "status": "ready" if not missing else "partial",
                "missing": sorted(set(missing)),
                "frozen_pilot_gpu_second_cost": (
                    projection.frozen_pilot_gpu_second_cost
                    if projection is not None else None
                ),
                "pilot_cost_currency": (
                    projection.currency if projection is not None else None
                ),
                "projected_pilot_cost_per_pilot": (
                    projection.projected_total_cost_per_pilot
                    if projection is not None else None
                ),
                "projected_pilot_utility_penalty_per_pilot": (
                    float(config["protocol"]["lambda_cost"])
                    * projection.projected_total_cost_per_pilot
                    if projection is not None else None
                ),
                "pilot_cost_projection": (
                    projection.to_dict() if projection is not None else None
                ),
            }
        )
        contexts[seed] = (run_dir, result, decisions, projection)

    write_json(output_root / "validation.json", {"schema": SCHEMA, "seeds": validation})
    if args.validate_only:
        print(f"Wrote {output_root / 'validation.json'}")
        return

    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for seed, (run_dir, _result, decisions, projection) in contexts.items():
        stopped: dict[str, bool] = {"adaptive": False, "fixed_32": False, "no_pilots_continue": False}
        primary_config = AdaptivePilotConfig(
            schedule=primary_schedule,
            delta=float(config["protocol"]["delta"]),
            K=int(config["protocol"]["K"]),
            T=int(config["protocol"]["T"]),
            lambda_cost=float(config["protocol"]["lambda_cost"]),
            bootstrap_draws=int(config["protocol"]["bootstrap_draws"]),
        )
        for round_index in range(2, 9):
            record = decisions.get(round_index)
            if record is None:
                continue
            decision_cost, _recorded_action, reason = _decision_fields(record)
            gate = _evidence_gate(reason)
            observations = _pilot_observations(run_dir, round_index)
            simulations = (
                simulate_fixed_32(observations, decision_cost=decision_cost, config=primary_config, evidence_gate=gate),
                simulate_adaptive_allocation(
                    observations,
                    decision_cost=decision_cost,
                    pilot_cost_projection=projection,
                    config=primary_config,
                    random_seed=seed * 1000 + round_index,
                    evidence_gate=gate,
                ),
                simulate_no_pilots(observations, decision_cost=decision_cost, evidence_gate=gate),
            )
            for simulation in simulations:
                key = simulation.strategy
                post_stop = stopped.get(key, False)
                rows.append(
                    _row(
                        seed,
                        round_index,
                        simulation,
                        post_stop=post_stop,
                        pilot_projection=projection,
                    )
                )
                if simulation.action == "stop" and simulation.certification == "certified_stop":
                    stopped[key] = True

        for cap in caps:
            schedule = tuple(value for value in primary_schedule if value <= cap)
            if not schedule or schedule[-1] != cap:
                raise ValueError(f"pilot cap {cap} is not in declared schedule")
            cap_config = AdaptivePilotConfig(
                schedule=schedule,
                delta=float(config["protocol"]["delta"]),
                K=int(config["protocol"]["K"]),
                T=int(config["protocol"]["T"]),
                lambda_cost=float(config["protocol"]["lambda_cost"]),
                bootstrap_draws=int(config["protocol"]["bootstrap_draws"]),
            )
            counterfactual_stop = False
            for round_index in range(2, 9):
                record = decisions.get(round_index)
                if record is None:
                    continue
                decision_cost, _action, reason = _decision_fields(record)
                simulation = simulate_adaptive_allocation(
                    _pilot_observations(run_dir, round_index),
                    decision_cost=decision_cost,
                    pilot_cost_projection=projection,
                    config=cap_config,
                    random_seed=seed * 100000 + cap * 100 + round_index,
                    evidence_gate=_evidence_gate(reason),
                )
                sensitivity_rows.append(
                    _row(
                        seed,
                        round_index,
                        simulation,
                        post_stop=counterfactual_stop,
                        pilot_projection=projection,
                    )
                )
                if simulation.action == "stop" and simulation.certification == "certified_stop":
                    counterfactual_stop = True

    summaries = _strategy_summary(rows)
    sensitivity_summaries = _strategy_summary(sensitivity_rows)
    payload = {
        "schema": SCHEMA,
        "protocol": config["protocol"],
        "frozen_pilot_cost_projection_spec": projection_spec,
        "voi_unit_semantics": {
            "monetary_cost": "USD",
            "decision_utility": "normalized_gain",
            "conversion": "normalized_gain = lambda_cost * USD",
            "measured_runtime_used_for_allocation": False,
        },
        "validation": validation,
        "rows": rows,
        "summaries": summaries,
        "future_observations_used_for_allocation": False,
        "sealed_test_used": False,
    }
    write_json(output_root / "simulation.json", payload)
    write_csv(output_root / "simulation.csv", _csv_rows(rows))
    write_json(output_root / "sensitivity.json", {"schema": SCHEMA, "rows": sensitivity_rows, "summaries": sensitivity_summaries})
    write_csv(output_root / "sensitivity.csv", _csv_rows(sensitivity_rows))
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        write_json(output_root / f"seed_{seed}" / "simulation.json", {"schema": SCHEMA, "seed": seed, "rows": seed_rows})
        write_csv(output_root / f"seed_{seed}" / "simulation.csv", _csv_rows(seed_rows))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(rows, summaries, sensitivity_summaries, validation),
        encoding="utf-8",
    )
    print(f"Wrote {output_root / 'simulation.json'}")
    print(f"Wrote {output_root / 'simulation.csv'}")
    print(f"Wrote {output_root / 'sensitivity.json'}")
    print(f"Wrote {output_root / 'sensitivity.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
