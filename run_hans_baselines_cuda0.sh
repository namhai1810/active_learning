#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr conda environment first:" >&2
  echo "  conda activate active_lr" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A REQUESTED_STAGES=()
if (( $# == 0 )); then
  for stage in zero-shot base random uncertainty; do
    REQUESTED_STAGES["$stage"]=1
  done
else
  for stage in "$@"; do
    case "$stage" in
      zero-shot|base|random|uncertainty)
        REQUESTED_STAGES["$stage"]=1
        ;;
      *)
        echo "ERROR: unknown stage '$stage'." >&2
        echo "Valid stages: zero-shot base random uncertainty" >&2
        exit 2
        ;;
    esac
  done
fi

should_run() {
  [[ -n "${REQUESTED_STAGES[$1]:-}" ]]
}

LOG_DIR="logs/hans_baselines_seed42"
REPORT_DIR="reports/hans_baselines_seed42"
PARTIAL_DIR="$LOG_DIR/partial_runs"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$PARTIAL_DIR"

echo "Execution plan:"
should_run zero-shot && echo "  - HANS zero-shot: 30,000 examples"
should_run base && echo "  - Base: 30,000 examples"
should_run random && echo "  - Random: acquire/train 32 MNLI examples, then 30,000 HANS examples"
should_run uncertainty && echo "  - Uncertainty: score 512 MNLI examples, train 32, then 30,000 HANS examples"
echo "Each NLI example is scored against 3 candidate labels. tqdm will show batch rate and ETA."
echo

run_stage() {
  local label="$1"
  local run_dir="$2"
  local log_file="$3"
  shift 3

  if [[ -f "$run_dir/metrics.json" && -f "$run_dir/predictions.jsonl" ]]; then
    echo "[SKIP] $label is already complete: $run_dir"
    return 0
  fi
  if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    local archive_name
    archive_name="$(basename "$run_dir")_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "[RECOVER] Archiving interrupted run: $run_dir -> $PARTIAL_DIR/$archive_name"
    mv "$run_dir" "$PARTIAL_DIR/$archive_name"
  fi

  echo "[RUN] $label"
  local stage_start=$SECONDS
  "$@" 2>&1 | tee "$log_file"
  local elapsed=$((SECONDS - stage_start))
  printf '[DONE] %s in %02d:%02d:%02d\n' \
    "$label" "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
}

python - <<'PY'
import json
import os
from pathlib import Path

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active_lr environment")
print(f"Visible CUDA device: {torch.cuda.get_device_name(0)}")

metadata_path = Path("outputs/hans_qwen3_1_7b_base/run.json")
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_seconds = metadata.get("wall_clock_seconds")
    if isinstance(reference_seconds, (int, float)) and reference_seconds > 0:
        print(
            "Historical reference: one full HANS evaluation took "
            f"{reference_seconds / 60:.1f} min. Use this only as an optimistic "
            "per-evaluation reference; the live tqdm ETA accounts for current throughput."
        )
PY

echo "Current physical GPU 0 load (high utilization makes ETA less stable):"
nvidia-smi \
  --id=0 \
  --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
echo

if should_run zero-shot; then
  run_stage \
    "HANS zero-shot" \
    "outputs/hans_qwen3_1_7b_base" \
    "$LOG_DIR/00_hans_zero_shot.log" \
    python scripts/evaluate_hans.py --config configs/hans_qwen17b.yaml
fi

if should_run base; then
  run_stage \
    "Base baseline" \
    "outputs/baseline_base_b32_seed42" \
    "$LOG_DIR/01_base.log" \
    python scripts/run_baseline.py --method base --config configs/baseline_base.yaml
fi

if should_run random; then
  run_stage \
    "Random baseline" \
    "outputs/baseline_random_b32_seed42" \
    "$LOG_DIR/02_random.log" \
    python scripts/run_baseline.py --method random --config configs/baseline_random.yaml
fi

if should_run uncertainty; then
  run_stage \
    "Uncertainty baseline" \
    "outputs/baseline_uncertainty_b32_seed42" \
    "$LOG_DIR/03_uncertainty.log" \
    python scripts/run_baseline.py --method uncertainty --config configs/baseline_uncertainty.yaml
fi

python scripts/report_hans_baselines.py \
  --outputs outputs \
  --report-dir "$REPORT_DIR" \
  --include-standalone-zero-shot \
  2>&1 | tee "$LOG_DIR/04_report.log"

echo
echo "Completed all runs. Open:"
echo "  $REPORT_DIR/HANS_BASELINE_REPORT.md"
