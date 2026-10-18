#!/usr/bin/env python3
"""Build one Stage 2 synthetic acquisition pool under a declared policy."""

from __future__ import annotations

import argparse
import copy
import filecmp
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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

NO_CERTIFIED_STATUS = "abstained_no_certified_hypotheses"
NO_CERTIFIED_REASON = "no_certified_hypotheses"
SYNTHETIC_OUTPUTS = (
    "synthetic/generated.jsonl",
    "synthetic/accepted.jsonl",
    "synthetic/rejected.jsonl",
    "synthetic/manifest.json",
    "teacher_calls/synthetic.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--method", required=True, choices=sorted(POLICY_CONFIG))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--discovery-config",
        help="Optional hypothesis-discovery config (default: main config).",
    )
    parser.add_argument(
        "--verification-config",
        help="Optional controlled-verification config (default: main config).",
    )
    parser.add_argument(
        "--generation-config",
        help="Optional synthetic-generation config (default: method config).",
    )
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


def _is_no_certified_abstention(run_dir: Path) -> bool:
    """Recognize only a complete, explicit, fail-closed verified-pool result."""
    if not _complete(run_dir, SYNTHETIC_OUTPUTS):
        return False
    try:
        manifest = read_json(run_dir / "synthetic" / "manifest.json")
    except (OSError, ValueError):
        return False
    if not (
        manifest.get("status") == NO_CERTIFIED_STATUS
        and manifest.get("abstained") is True
        and manifest.get("stage2_policy") == "failure_targeted_verified"
        and manifest.get("certified_hypothesis_ids") == []
        and manifest.get("counts") == {}
        and manifest.get("abstention_reason") == NO_CERTIFIED_REASON
    ):
        return False
    empty_jsonl_paths = (
        run_dir / "synthetic" / "generated.jsonl",
        run_dir / "synthetic" / "accepted.jsonl",
        run_dir / "synthetic" / "rejected.jsonl",
        run_dir / "teacher_calls" / "synthetic.jsonl",
    )
    try:
        return all(
            not read_jsonl(path, allow_empty=True) for path in empty_jsonl_paths
        )
    except (OSError, ValueError):
        return False


def _archive_synthetic_attempt(run_dir: Path, reason: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir / "synthetic_attempts" / f"{timestamp}_{reason}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = (
            run_dir / "synthetic_attempts" / f"{timestamp}_{reason}_{suffix}"
        )
    destination.mkdir(parents=True)
    synthetic_dir = run_dir / "synthetic"
    if synthetic_dir.exists():
        shutil.move(str(synthetic_dir), str(destination / "synthetic"))
    teacher_log = run_dir / "teacher_calls" / "synthetic.jsonl"
    if teacher_log.exists():
        shutil.move(str(teacher_log), str(destination / "synthetic_teacher_calls.jsonl"))
    completion_marker = run_dir / ".complete"
    if completion_marker.exists():
        shutil.move(str(completion_marker), str(destination / ".complete"))
    print(
        f"  [RECOVER] Archived synthetic attempt ({reason}) to {destination}",
        flush=True,
    )
    return destination


def _archive_verification_attempt(run_dir: Path, reason: str) -> Path:
    """Preserve stale verification outputs before a prompt/config revision."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir / "verification_attempts" / f"{timestamp}_{reason}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = (
            run_dir / "verification_attempts" / f"{timestamp}_{reason}_{suffix}"
        )
    destination.mkdir(parents=True)
    verification_dir = run_dir / "verification"
    if verification_dir.exists():
        shutil.move(str(verification_dir), str(destination / "verification"))
    teacher_log = run_dir / "teacher_calls" / "verification.jsonl"
    if teacher_log.exists():
        shutil.move(
            str(teacher_log), str(destination / "verification_teacher_calls.jsonl")
        )
    verification_config = run_dir / "stage2_configs" / "verification.yaml"
    if verification_config.exists():
        shutil.move(
            str(verification_config), str(destination / "verification.yaml")
        )
    print(
        f"  [RECOVER] Archived verification attempt ({reason}) to {destination}",
        flush=True,
    )
    return destination


def _archive_discovery_attempt(run_dir: Path, reason: str) -> Path:
    """Preserve stale discovery outputs before changing its search profile."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir / "discovery_attempts" / f"{timestamp}_{reason}"
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = run_dir / "discovery_attempts" / f"{timestamp}_{reason}_{suffix}"
    destination.mkdir(parents=True)
    for relative_path in (
        "hypotheses.json",
        "hypothesis_evidence.json",
        "hans_split_v2.json",
    ):
        source = run_dir / relative_path
        if source.exists():
            shutil.move(str(source), str(destination / source.name))
    teacher_log = run_dir / "teacher_calls" / "discovery.jsonl"
    if teacher_log.exists():
        shutil.move(str(teacher_log), str(destination / "discovery_teacher_calls.jsonl"))
    discovery_config = run_dir / "stage2_configs" / "discovery.yaml"
    if discovery_config.exists():
        shutil.move(str(discovery_config), str(destination / "discovery.yaml"))
    print(
        f"  [RECOVER] Archived discovery attempt ({reason}) to {destination}",
        flush=True,
    )
    return destination


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

    discovery_source = Path(args.discovery_config).resolve() if args.discovery_config else (
        PROJECT_ROOT / "configs" / "main_hypothesis_discovery.yaml"
    )
    verification_source = Path(args.verification_config).resolve() if args.verification_config else (
        PROJECT_ROOT / "configs" / "main_verification.yaml"
    )
    generation_source = Path(args.generation_config).resolve() if args.generation_config else (
        PROJECT_ROOT / "configs" / POLICY_CONFIG[args.method]
    )
    discovery = _seeded(
        load_yaml_with_extends(discovery_source), args.seed, hans, model
    )
    verification = _seeded(
        load_yaml_with_extends(verification_source), args.seed, hans, model
    )
    generation = _seeded(
        load_yaml_with_extends(generation_source), args.seed, hans, model
    )
    configs = run_dir / "stage2_configs"
    discovery_path = configs / "discovery.yaml"
    verification_path = configs / "verification.yaml"
    generation_path = configs / "generation.yaml"
    discovery_config_changed = (
        discovery_path.is_file()
        and load_yaml_with_extends(discovery_path) != discovery
    )
    if args.method != "random_synthetic" and discovery_config_changed:
        if (
            (run_dir / "synthetic").exists()
            or (run_dir / "teacher_calls" / "synthetic.jsonl").exists()
            or (run_dir / ".complete").exists()
        ):
            _archive_synthetic_attempt(run_dir, "discovery_config_changed")
        if args.method == "ours_verified" and (
            (run_dir / "verification").exists()
            or (run_dir / "teacher_calls" / "verification.jsonl").exists()
        ):
            _archive_verification_attempt(run_dir, "discovery_config_changed")
        _archive_discovery_attempt(run_dir, "discovery_config_changed")
    if not discovery_path.is_file() or discovery_config_changed:
        _write_yaml(discovery_path, discovery)
    verification_config_changed = (
        verification_path.is_file()
        and load_yaml_with_extends(verification_path) != verification
    )
    if args.method == "ours_verified" and verification_config_changed:
        has_verification_attempt = (
            (run_dir / "verification").exists()
            or (run_dir / "teacher_calls" / "verification.jsonl").exists()
        )
        if has_verification_attempt:
            _archive_verification_attempt(run_dir, "verification_config_changed")
        if (
            (run_dir / "synthetic").exists()
            or (run_dir / "teacher_calls" / "synthetic.jsonl").exists()
            or (run_dir / ".complete").exists()
        ):
            _archive_synthetic_attempt(run_dir, "verification_config_changed")
    if not verification_path.is_file() or verification_config_changed:
        _write_yaml(verification_path, verification)

    required_accepted = max(int(item) for item in benchmark["budgets"])
    configured_minimum = int(generation.get("stage2_min_accepted_examples", 0))
    if configured_minimum < required_accepted:
        raise ValueError(
            "Synthetic generation must require at least the largest benchmark "
            f"budget: configured={configured_minimum}, required={required_accepted}"
        )
    generation_config_changed = False
    if generation_path.is_file():
        generation_config_changed = (
            load_yaml_with_extends(generation_path) != generation
        )
    accepted_path = run_dir / "synthetic" / "accepted.jsonl"
    accepted_count = (
        len(read_jsonl(accepted_path, allow_empty=True))
        if accepted_path.is_file()
        else None
    )
    if generation_config_changed and (run_dir / "synthetic").exists():
        _archive_synthetic_attempt(run_dir, "generation_config_changed")
    elif (
        accepted_count is not None
        and accepted_count < required_accepted
        and not (
            args.method == "ours_verified"
            and _is_no_certified_abstention(run_dir)
        )
    ):
        _archive_synthetic_attempt(
            run_dir, f"insufficient_{accepted_count}_of_{required_accepted}"
        )
    if not generation_path.is_file() or generation_config_changed:
        _write_yaml(generation_path, generation)

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
                "verification/REPORT.md",
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
    if _complete(run_dir, SYNTHETIC_OUTPUTS):
        print("  [SKIP] synthetic generation and QC already complete", flush=True)
    else:
        _run(
            "generate_synthetic_interventions.py",
            generation_path,
            run_dir,
            stage="generating and quality-checking synthetic training data",
        )
    final_accepted = len(
        read_jsonl(run_dir / "synthetic" / "accepted.jsonl", allow_empty=True)
    )
    no_certified_abstention = (
        args.method == "ours_verified" and _is_no_certified_abstention(run_dir)
    )
    if final_accepted < required_accepted and not no_certified_abstention:
        raise ValueError(
            "Synthetic pool is too small for the benchmark: "
            f"accepted={final_accepted}, required={required_accepted}"
        )
    (run_dir / ".complete").touch()
    if no_certified_abstention:
        print(
            "  [ABSTAIN] No hypotheses were certified; recorded an empty "
            "fail-closed acquisition pool",
            flush=True,
        )
    print(f"Wrote Stage 2 acquisition pool to {run_dir}")


if __name__ == "__main__":
    main()
