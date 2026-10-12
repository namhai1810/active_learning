#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash run_hans_baselines_individual.sh base
  bash run_hans_baselines_individual.sh random
  bash run_hans_baselines_individual.sh uncertainty
  bash run_hans_baselines_individual.sh report
  bash run_hans_baselines_individual.sh all

Actions:
  base         Re-evaluate the unadapted Base checkpoint on HANS.
  random       Run Random MNLI acquisition, LoRA training, and HANS evaluation.
  uncertainty  Run Uncertainty MNLI acquisition, LoRA training, and HANS evaluation.
  report       Aggregate the three completed runs and generate Markdown/CSV/JSON/SVG.
  all          Run base, random, uncertainty, then report in that order.

Environment:
  CUDA_DEVICE  Physical CUDA device to expose (default: 0).

Baseline actions intentionally pass --overwrite to produce a clean re-evaluation
of their named output directories.
EOF
}

ACTION="${1:-}"
case "$ACTION" in
  base|random|uncertainty|report|all)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr environment first:" >&2
  echo "  conda activate active_lr" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="logs/hans_baselines_seed42"
REPORT_DIR="reports/hans_baselines_seed42"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

run_baseline() {
  local method="$1"
  local config="configs/baseline_${method}.yaml"
  local log_file="$LOG_DIR/${method}.log"

  echo "[RUN] ${method}"
  python scripts/run_baseline.py \
    --method "$method" \
    --config "$config" \
    --overwrite \
    2>&1 | tee "$log_file"
  echo "[DONE] ${method}; log: ${log_file}"
}

build_report() {
  echo "[REPORT] Aggregating Base, Random, and Uncertainty"
  python scripts/report_hans_baselines.py \
    --outputs outputs \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$LOG_DIR/report.log"
  echo "[DONE] Open ${REPORT_DIR}/HANS_BASELINE_REPORT.md"
}

case "$ACTION" in
  base|random|uncertainty)
    run_baseline "$ACTION"
    ;;
  report)
    build_report
    ;;
  all)
    run_baseline base
    run_baseline random
    run_baseline uncertainty
    build_report
    ;;
esac
