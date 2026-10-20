#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_ours_verified_v7_dual_closed_loop.yaml"
OUTPUT="outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42"
LOG_DIR="logs/stage2_ours_verified_v7_dual/seed_42_budget_128"
REPORT_DIR="reports/stage2_seed42_comparison_with_v7"

RESUME=false
if [[ "${1:-}" == "--resume" ]] && [[ "$#" -eq 1 ]]; then
  RESUME=true
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--resume]" >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr environment first: conda activate active_lr" >&2
  exit 2
fi

TEACHER_GPU="${TEACHER_GPU:-1}"
STUDENT_GPU="${STUDENT_GPU:-0}"
if [[ "$TEACHER_GPU" == "$STUDENT_GPU" ]]; then
  echo "ERROR: v7 dual requires distinct teacher/student GPUs; use GPU 0 and 1." >&2
  exit 2
fi

if [[ ! -e "outputs/stage2_acquisition_ours_verified_v5/seed_42/.complete" ]]; then
  echo "ERROR: v5 pool completion marker is missing" >&2
  exit 1
fi
for path in \
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
  echo "ERROR: completed v7 output already exists: $OUTPUT" >&2
  exit 1
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
  if [[ "$RESUME" != true ]]; then
    echo "ERROR: incomplete v7 output already exists: $OUTPUT" >&2
    echo "Resume it with: $0 --resume" >&2
    exit 1
  fi
elif [[ "$RESUME" == true ]]; then
  echo "ERROR: no incomplete v7 output exists to resume: $OUTPUT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$TEACHER_GPU,$STUDENT_GPU"
export CLOSED_LOOP_TEACHER_DEVICE="cuda:0"
export CLOSED_LOOP_STUDENT_DEVICE="cuda:1"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_DIR"

echo "[START] v7 shared closed loop: teacher=physical GPU $TEACHER_GPU, student=physical GPU $STUDENT_GPU"
echo "[INFO] One acquisition trace; final branches are continuation and consolidation."
if [[ "$RESUME" == true ]]; then
  echo "[RESUME] Continuing from validated completed-round artifacts."
  python scripts/run_stage2_closed_loop.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --resume \
    2>&1 | tee -a "$LOG_DIR/run.log"
else
  python scripts/run_stage2_closed_loop.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    2>&1 | tee "$LOG_DIR/run.log"
fi

python scripts/report_stage2_v7_comparison.py \
  --v7-output "$OUTPUT" \
  --output-dir "$REPORT_DIR"

echo "[DONE] Continuation: $OUTPUT/branches/continuation/result.json"
echo "[DONE] Consolidation: $OUTPUT/branches/consolidation/result.json"
echo "[DONE] Direct comparison: $OUTPUT/V7_COMPARISON.md"
echo "[DONE] Full seed-42 comparison: $REPORT_DIR/STAGE2_COMPARISON.md"

