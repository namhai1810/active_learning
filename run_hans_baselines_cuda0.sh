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

LOG_DIR="logs/hans_baselines_seed42"
REPORT_DIR="reports/hans_baselines_seed42"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

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
    echo "ERROR: partial run directory detected: $run_dir" >&2
    echo "Move it aside or inspect it before retrying; this script will not overwrite it." >&2
    exit 3
  fi

  echo "[RUN] $label"
  "$@" 2>&1 | tee "$log_file"
}

python - <<'PY'
import os
import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active_lr environment")
print(f"Visible CUDA device: {torch.cuda.get_device_name(0)}")
PY

run_stage \
  "HANS zero-shot" \
  "outputs/hans_qwen3_1_7b_base" \
  "$LOG_DIR/00_hans_zero_shot.log" \
  python scripts/evaluate_hans.py --config configs/hans_qwen17b.yaml

run_stage \
  "Base baseline" \
  "outputs/baseline_base_b32_seed42" \
  "$LOG_DIR/01_base.log" \
  python scripts/run_baseline.py --method base --config configs/baseline_base.yaml

run_stage \
  "Random baseline" \
  "outputs/baseline_random_b32_seed42" \
  "$LOG_DIR/02_random.log" \
  python scripts/run_baseline.py --method random --config configs/baseline_random.yaml

run_stage \
  "Uncertainty baseline" \
  "outputs/baseline_uncertainty_b32_seed42" \
  "$LOG_DIR/03_uncertainty.log" \
  python scripts/run_baseline.py --method uncertainty --config configs/baseline_uncertainty.yaml

python scripts/report_hans_baselines.py \
  --outputs outputs \
  --report-dir "$REPORT_DIR" \
  2>&1 | tee "$LOG_DIR/04_report.log"

echo
echo "Completed all runs. Open:"
echo "  $REPORT_DIR/HANS_BASELINE_REPORT.md"
