#!/usr/bin/env python3
"""Build one Stage 2 synthetic acquisition pool under a declared policy."""

from __future__ import annotations

import argparse
import copy
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.hans_manifest import FrozenHansManifest
from src.data.pipeline_split import manifest_payload, reconstruct_five_way_split
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json


POLICY_CONFIG = {
    "random_synthetic": "stage2_generation_random_synthetic.yaml",
    "failure_driven_no_verification": "stage2_generation_no_verification.yaml",
    "ours_verified": "stage2_generation_ours.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--method", required=True, choices=sorted(POLICY_CONFIG))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete pool and skip stages with complete artifacts.",
    )
    return parser.parse_args()


def _seeded(config: dict[str, Any], seed: int, hans: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["hans"] = copy.deepcopy(hans)
    result["model"] = copy.deepcopy(model)
    for section, keys in {
        "hypothesis_generation": ("seed",),
        "verification": ("seed", "label_validation_seed"),
        "synthetic_generation": ("seed",),
        "synthetic_validation": ("seed",),
    }.items():
        value = result.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    value[key] = seed if key == "seed" else seed + 100_000
    scorer = result.get("hypothesis_generation", {}).get("error_scorer")
    if isinstance(scorer, dict):
        scorer["seed"] = seed
    return result


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _run(
    script: str,
    config: Path,
    run_dir: Path,
    *,
    stage: str,
    overwrite: bool = False,
) -> None:
    print(f"  [STAGE] {stage}", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def _complete(run_dir: Path, relative_paths: tuple[str, ...]) -> bool:
    present = [(run_dir / path).is_file() for path in relative_paths]
    return bool(present) and all(present)


def _has_any(run_dir: Path, relative_paths: tuple[str, ...]) -> bool:
    return any((run_dir / path).exists() for path in relative_paths)


def main() -> None:
    args = parse_args()
    benchmark_config = load_yaml_with_extends(args.benchmark_config)
    benchmark = require_mapping(benchmark_config, "stage2_benchmark")
    if args.seed not in [int(item) for item in benchmark["seeds"]]:
        raise ValueError("Seed is outside the predeclared Stage 2 grid")
    hans = require_mapping(benchmark_config, "hans")
    model = require_mapping(benchmark_config, "model")
    shared_predictions = Path(str(benchmark["frozen_manifest_path"])).parent / "predictions.jsonl"
    if not shared_predictions.is_file():
        raise FileNotFoundError("Run prepare_stage2_benchmark.py first")
    run_dir = Path(args.output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.is_file():
        if not filecmp.cmp(shared_predictions, predictions_path, shallow=False):
            raise ValueError("Existing pool predictions differ from frozen predictions")
    else:
        shutil.copy2(shared_predictions, predictions_path)
    frozen = FrozenHansManifest.from_mapping(
        read_json(Path(str(benchmark["frozen_manifest_path"])))
    )
    source = read_jsonl(predictions_path)
    split = reconstruct_five_way_split(source, hans)
    frozen_roles = (
        frozen.hypothesis_discovery_ids,
        frozen.verification_ids,
        frozen.intervention_pool_ids,
        frozen.gain_eval_ids,
        frozen.final_test_ids,
    )
    reconstructed_roles = (
        split.discovery_ids,
        split.verification_ids,
        split.intervention_pool_ids,
        split.gain_eval_ids,
        split.final_test_ids,
    )
    if frozen_roles != reconstructed_roles:
        raise ValueError("Synthetic pool split differs from the frozen Stage 2 manifest")

    discovery = _seeded(
        load_yaml_with_extends(PROJECT_ROOT / "configs" / "main_hypothesis_discovery.yaml"),
        args.seed, hans, model,
    )
    verification = _seeded(
        load_yaml_with_extends(PROJECT_ROOT / "configs" / "main_verification.yaml"),
        args.seed, hans, model,
    )
    generation = _seeded(
        load_yaml_with_extends(PROJECT_ROOT / "configs" / POLICY_CONFIG[args.method]),
        args.seed, hans, model,
    )
    configs = run_dir / "stage2_configs"
    discovery_path = configs / "discovery.yaml"
    verification_path = configs / "verification.yaml"
    generation_path = configs / "generation.yaml"
    for path, value in (
        (discovery_path, discovery),
        (verification_path, verification),
        (generation_path, generation),
    ):
        if not path.is_file():
            _write_yaml(path, value)

    if args.method == "random_synthetic":
        split_path = run_dir / "hans_split_v2.json"
        if not split_path.is_file():
            write_json(
                split_path,
                manifest_payload(split, hans, mode="llm_main"),
            )
    else:
        discovery_outputs = (
            "hypotheses.json",
            "hypothesis_evidence.json",
            "hans_split_v2.json",
            "teacher_calls/discovery.jsonl",
        )
        if _complete(run_dir, discovery_outputs):
            print("  [SKIP] failure hypotheses already complete", flush=True)
        else:
            _run(
                "build_hypotheses.py",
                discovery_path,
                run_dir,
                stage="discovering failure hypotheses",
                overwrite=_has_any(run_dir, discovery_outputs),
            )
        if args.method == "ours_verified":
            verification_outputs = (
                "verification/generated_probes.jsonl",
                "verification/probes.jsonl",
                "verification/rejected_probes.jsonl",
                "verification/probe_set_outcomes.jsonl",
                "verification/results.json",
                "teacher_calls/verification.jsonl",
            )
            if _complete(run_dir, verification_outputs):
                print("  [SKIP] controlled verification already complete", flush=True)
            else:
                _run(
                    "verify_hypotheses.py",
                    verification_path,
                    run_dir,
                    stage="generating and verifying controlled probes",
                    overwrite=_has_any(run_dir, verification_outputs),
                )
    synthetic_outputs = (
        "synthetic/generated.jsonl",
        "synthetic/accepted.jsonl",
        "synthetic/rejected.jsonl",
        "synthetic/manifest.json",
        "teacher_calls/synthetic.jsonl",
    )
    if _complete(run_dir, synthetic_outputs):
        print("  [SKIP] synthetic generation and QC already complete", flush=True)
    else:
        _run(
            "generate_synthetic_interventions.py",
            generation_path,
            run_dir,
            stage="generating and quality-checking synthetic training data",
        )
    (run_dir / ".complete").touch()
    print(f"Wrote Stage 2 acquisition pool to {run_dir}")


if __name__ == "__main__":
    main()
