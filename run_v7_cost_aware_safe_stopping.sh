#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SEED=""
RESUME=false
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    *)
      echo "Usage: $0 --seed 42|43|44 [--resume]" >&2
      exit 2
      ;;
  esac
done

if [[ "$SEED" != "42" && "$SEED" != "43" && "$SEED" != "44" ]]; then
  echo "ERROR: --seed must be 42, 43, or 44" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate active_lr first: conda activate active_lr" >&2
  exit 2
fi

TEACHER_GPU="${TEACHER_GPU:-1}"
STUDENT_GPU="${STUDENT_GPU:-0}"
if [[ "$TEACHER_GPU" == "$STUDENT_GPU" ]]; then
  echo "ERROR: teacher and student must use distinct physical GPUs" >&2
  exit 2
fi

CONFIG="configs/stage2_v7_cost_aware_safe_stopping.yaml"
OUTPUT_ROOT="outputs/stage2_runs_v7_cost_aware_safe_stopping"
OUTPUT="$OUTPUT_ROOT/seed_${SEED}"
LOG_DIR="logs/stage2_v7_cost_aware_safe_stopping/seed_${SEED}"

for path in \
  "outputs/stage2_acquisition_ours_verified_v5/seed_42/.complete" \
  "outputs/stage2_acquisition_ours_verified_v5/seed_42/synthetic/accepted.jsonl" \
  "outputs/stage2_shared/hans_frozen_manifest.json" \
  "outputs/stage2_shared/predictions.jsonl" \
  "outputs/stage2_shared/sealed_final_test.jsonl" \
  "outputs/stage2_shared/gain_eval_stopping_subset_v1.json"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required artifact is missing or empty: $path" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT/.complete" ]]; then
  echo "ERROR: completed output already exists: $OUTPUT" >&2
  exit 1
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
  if [[ "$RESUME" != true ]]; then
    echo "ERROR: incomplete output exists; rerun with --resume: $OUTPUT" >&2
    exit 1
  fi
elif [[ "$RESUME" == true ]]; then
  echo "ERROR: no incomplete output exists to resume: $OUTPUT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$TEACHER_GPU,$STUDENT_GPU"
export CLOSED_LOOP_TEACHER_DEVICE="cuda:0"
export CLOSED_LOOP_STUDENT_DEVICE="cuda:1"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_DIR"

COMMAND=(
  python scripts/run_stage2_closed_loop.py
  --config "$CONFIG"
  --output-dir "$OUTPUT"
  --seed "$SEED"
)
if [[ "$RESUME" == true ]]; then
  COMMAND+=(--resume)
fi

echo "[START] V7 cost-aware safe stopping seed=$SEED"
"${COMMAND[@]}" 2>&1 | tee "$LOG_DIR/run.log"

python scripts/report_v7_cost_aware_stopping.py \
  --input-root "$OUTPUT_ROOT" \
  --output reports/stage2_v7_cost_aware_safe_stopping/REPORT.md
