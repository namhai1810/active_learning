#!/usr/bin/env python3
"""Supplement missing one-step checkpoint metrics without training anything.

The script accepts only inference plans authorized by the CPU validation step.
It evaluates every requested checkpoint on the exact same per-seed held-out
manifest and writes new diagnostic artifacts outside historical run folders.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_hypotheses import _build_student
from src.utils.config import load_yaml_with_extends


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def hash_local_path(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("example_id"))
    if value is None:
        raise ValueError("Evaluation source row has no id/example_id")
    return str(value)


def _gold_binary(row: Mapping[str, Any]) -> int:
    value = row.get("gold", row.get("gold_label", row.get("label")))
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"entailment", "entails", "e"}:
            return 0
        if normalized in {"neutral", "contradiction", "non-entailment", "nonentailment", "n", "c"}:
            return 1
    if value in (0, False):
        return 0
    if value in (1, 2, True):
        return 1
    raise ValueError(f"Unsupported NLI gold label: {value!r}")


def _load_adapter(student: Any, adapter_path: Path) -> None:
    if hasattr(student, "load_adapter"):
        student.load_adapter(str(adapter_path))
        return
    model = getattr(student, "model", None)
    if model is not None and hasattr(model, "load_adapter"):
        model.load_adapter(str(adapter_path))
        if hasattr(model, "set_adapter"):
            adapters = getattr(model, "peft_config", {})
            if isinstance(adapters, Mapping) and adapters:
                model.set_adapter(next(reversed(adapters)))
        return
    if model is not None:
        from peft import PeftModel

        student.model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        return
    raise TypeError("QwenNLIStudent exposes no supported adapter-loading interface")


def _invoke_prediction_method(method: Any, rows: Sequence[Mapping[str, Any]], batch_size: int) -> Any:
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    parameters = inspect.signature(method).parameters
    kwargs: dict[str, Any] = {}
    if "batch_size" in parameters:
        kwargs["batch_size"] = batch_size
    if "examples" in parameters or "rows" in parameters:
        key = "examples" if "examples" in parameters else "rows"
        return method(**{key: rows}, **kwargs)
    return method(premises, hypotheses, **kwargs)


def _normalize_probabilities(raw: Any, expected: int) -> np.ndarray:
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, Mapping):
        raw = raw.get("probabilities", raw.get("probs"))
    normalized: list[list[float]] = []
    for item in raw:
        if isinstance(item, Mapping):
            normalized.append(
                [
                    float(item["entailment"]),
                    float(item["neutral"]),
                    float(item["contradiction"]),
                ]
            )
        else:
            vector = list(item)
            if len(vector) != 3:
                raise ValueError("Student inference must return E/N/C probabilities")
            normalized.append([float(value) for value in vector])
    array = np.asarray(normalized, dtype=np.float64)
    if array.shape != (expected, 3):
        raise ValueError(f"Unexpected probability shape: {array.shape}; expected {(expected, 3)}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Invalid probabilities returned by student")
    return array


def _predict_probability_batch(student: Any, rows: Sequence[Mapping[str, Any]], batch_size: int) -> np.ndarray:
    for name in ("predict_proba", "predict_probabilities"):
        method = getattr(student, name, None)
        if method is not None:
            return _normalize_probabilities(
                _invoke_prediction_method(method, rows, batch_size), len(rows)
            )
    raise TypeError(
        "QwenNLIStudent has no probability inference method. Refusing to derive "
        "accuracy from hard labels because both HANS rules must remain auditable."
    )


def _predict_probabilities(
    student: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    description: str,
    show_progress: bool,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    starts = range(0, len(rows), batch_size)
    progress = tqdm(
        starts,
        total=(len(rows) + batch_size - 1) // batch_size,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for start in progress:
        batch = rows[start : start + batch_size]
        chunks.append(_predict_probability_batch(student, batch, batch_size))
    return np.concatenate(chunks, axis=0)


def _accuracy(values: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean(values == gold))


def _evaluate(
    *,
    seed: int,
    round_index: int,
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    run_config: Mapping[str, Any],
    output_dir: Path,
    batch_size: int,
    resume: bool,
    show_progress: bool,
) -> None:
    metrics_path = output_dir / "metrics.json"
    checkpoint_hash = hash_local_path(checkpoint)
    if resume and metrics_path.exists():
        saved = read_json(metrics_path)
        if (
            saved.get("checkpoint_hash") == checkpoint_hash
            and saved.get("evaluation_manifest_hash") == manifest["manifest_hash"]
        ):
            print(f"[REUSE] seed={seed} round={round_index}")
            return
        raise RuntimeError(f"Existing metrics provenance mismatch: {metrics_path}")

    model_config = run_config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Resolved run config has no model mapping")
    student = _build_student(dict(model_config))
    _load_adapter(student, checkpoint)
    started = time.perf_counter()
    probabilities = _predict_probabilities(
        student,
        rows,
        batch_size,
        f"seed {seed} R{round_index}",
        show_progress,
    )
    inference_seconds = time.perf_counter() - started
    gold = np.asarray([_gold_binary(row) for row in rows], dtype=np.int64)
    three_way = (np.argmax(probabilities, axis=1) != 0).astype(np.int64)
    direct_binary = (probabilities[:, 1] + probabilities[:, 2] > probabilities[:, 0]).astype(np.int64)
    predictions = []
    for row, probs, pred_three, pred_direct, target in zip(
        rows, probabilities, three_way, direct_binary, gold
    ):
        predictions.append(
            {
                "id": _row_id(row),
                "gold_binary": int(target),
                "p_entailment": float(probs[0]),
                "p_neutral": float(probs[1]),
                "p_contradiction": float(probs[2]),
                "three_way_binary_prediction": int(pred_three),
                "direct_binary_prediction": int(pred_direct),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics = {
        "schema": "v7_one_step_checkpoint_metric_v1",
        "seed": seed,
        "round": round_index,
        "num_examples": len(rows),
        "primary_rule": "direct_binary",
        "overall_accuracy": _accuracy(direct_binary, gold),
        "direct_binary_accuracy": _accuracy(direct_binary, gold),
        "three_way_binary_accuracy": _accuracy(three_way, gold),
        "prediction_disagreement_count": int(np.sum(three_way != direct_binary)),
        "inference_seconds": inference_seconds,
        "examples_per_second": len(rows) / inference_seconds if inference_seconds > 0 else None,
        "evaluation_manifest_hash": manifest["manifest_hash"],
        "evaluation_ids_hash": manifest["evaluation_ids_hash"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_hash": checkpoint_hash,
        "observed_gain_role": "single_training_realization_not_expected_gain",
    }
    write_json(metrics_path, metrics)
    print(f"[DONE] seed={seed} round={round_index}: {metrics_path}")
    del student
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", type=Path, default=Path("configs/v7_one_step_cost_aware_benchmark.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rounds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = read_json(args.benchmark_config)
    source_specs = {int(item["seed"]): item for item in spec.get("sources", [])}
    seeds = args.seeds or sorted(source_specs)
    metrics_root = Path(spec["evaluation"]["checkpoint_metrics_root"])
    output_root = metrics_root.parent
    requested = set(args.rounds) if args.rounds else None

    for seed in seeds:
        manifest_path = output_root / "evaluation_manifests" / f"seed_{seed}.json"
        plan_path = output_root / f"inference_plan_seed_{seed}.json"
        if not manifest_path.exists() or not plan_path.exists():
            raise FileNotFoundError(f"Run validation first; seed {seed} has no manifest/plan")
        manifest, plan = read_json(manifest_path), read_json(plan_path)
        if not manifest.get("authorized_for_inference") or not plan.get("authorized"):
            raise RuntimeError(f"Seed {seed} inference is not authorized: {manifest.get('issues', [])}")
        if plan.get("evaluation_manifest_hash") != manifest.get("manifest_hash"):
            raise RuntimeError(f"Seed {seed} plan/manifest provenance mismatch")
        config_path = Path(str(plan["resolved_run_config_path"]))
        run_config = load_yaml_with_extends(config_path)
        source_rows = {_row_id(row): row for row in read_jsonl(Path(manifest["source_path"]))}
        rows = [source_rows[example_id] for example_id in manifest["evaluation_ids"]]
        if _json_hash([_row_id(row) for row in rows]) != manifest["evaluation_ids_hash"]:
            raise RuntimeError(f"Seed {seed} evaluation order/hash mismatch")
        model_config = dict(run_config.get("model", {}))
        model_config["device"] = args.device
        run_config = dict(run_config)
        run_config["model"] = model_config
        checkpoints = tqdm(
            plan["checkpoints"],
            desc=f"seed {seed} checkpoints",
            unit="checkpoint",
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        for checkpoint in checkpoints:
            round_index = int(checkpoint["round"])
            if requested is not None and round_index not in requested:
                continue
            if checkpoint["status"] != "ready":
                print(f"[MISSING] seed={seed} round={round_index}: checkpoint")
                continue
            path = Path(checkpoint["checkpoint_path"])
            if hash_local_path(path) != checkpoint["checkpoint_hash"]:
                raise RuntimeError(f"Seed {seed} round {round_index} checkpoint hash changed")
            _evaluate(
                seed=seed,
                round_index=round_index,
                checkpoint=path,
                rows=rows,
                manifest=manifest,
                run_config=run_config,
                output_dir=metrics_root / f"seed_{seed}" / f"round_{round_index:03d}",
                batch_size=args.batch_size,
                resume=args.resume,
                show_progress=not args.no_progress,
            )


if __name__ == "__main__":
    main()
