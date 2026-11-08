#!/usr/bin/env python3
"""Static, CPU-only validation of the Range-Aware protocol configuration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gain.estimation import gain_confidence_radius


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Config inheritance cycle: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    parent_name = payload.get("extends")
    if parent_name is None:
        return _merge({}, payload)
    parent = _load(resolved.parent / str(parent_name), (*seen, resolved))
    return _merge(parent, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage2_v7_minimal_cost_aware_range_aware.yaml",
    )
    args = parser.parse_args()
    config = _load(Path(args.config))
    policy = config["cost_aware"]
    expected = {
        "gain_confidence_method": "fixed_baseline_range_aware",
        "required_pilots": 32,
        "M": 1.0,
        "delta": 0.05,
        "K": 1,
        "T": 7,
        "lambda_cost": 0.2,
        "decision_cost_semantics": (
            "marginal_incremental_cost_if_continue_after_pilots"
        ),
    }
    mismatches = {
        key: {"expected": value, "actual": policy.get(key)}
        for key, value in expected.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Range-Aware protocol mismatch: {mismatches}")
    m = int(policy["required_pilots"])
    K = int(policy["K"])
    T = int(policy["T"])
    delta = float(policy["delta"])
    radius = gain_confidence_radius(
        m, float(policy["M"]), K, T, delta, support_width=1.0
    )
    direct = math.sqrt(math.log(2.0 * K * T / delta) / (2.0 * m))
    if not math.isclose(radius, direct, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Range-Aware radius implementation differs from formula")
    print(
        json.dumps(
            {
                "status": "valid",
                "config": str(Path(args.config)),
                "m": m,
                "delta": delta,
                "K": K,
                "T": T,
                "support_width": 1.0,
                "radius": radius,
                "formula": "sqrt(log(2*K*T/delta)/(2*m))",
                "population_level_guarantee_claimed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
