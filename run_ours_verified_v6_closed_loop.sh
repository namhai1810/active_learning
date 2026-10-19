#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_ours_verified_v6_closed_loop.yaml"
OUTPUT="outputs/stage2_runs_ours_verified_v6_closed_loop/experiment_b/ours_verified/budget_128/seed_42"
LOG_DIR="logs/stage2_ours_verified_v6_closed_loop/seed_42_budget_128"

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr environment first: conda activate active_lr" >&2
  exit 2
fi

TEACHER_GPU="${TEACHER_GPU:-0}"
STUDENT_GPU="${STUDENT_GPU:-1}"

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
    echo "ERROR: required artifact is missing: $path" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT/result.json" ]]; then
  echo "ERROR: completed v6 result already exists: $OUTPUT/result.json" >&2
  exit 1
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: incomplete output already exists: $OUTPUT" >&2
  echo "Move it aside, or run the Python entrypoint explicitly with --overwrite." >&2
  exit 1
fi

if [[ "$TEACHER_GPU" == "$STUDENT_GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$TEACHER_GPU"
  export CLOSED_LOOP_TEACHER_DEVICE="cuda:0"
  export CLOSED_LOOP_STUDENT_DEVICE="cuda:0"
  echo "[WARN] Single-GPU mode on physical GPU $TEACHER_GPU."
  echo "[WARN] Verification loads teacher and student on the same GPU; sufficient VRAM is required."
else
  export CUDA_VISIBLE_DEVICES="$TEACHER_GPU,$STUDENT_GPU"
  export CLOSED_LOOP_TEACHER_DEVICE="cuda:0"
  export CLOSED_LOOP_STUDENT_DEVICE="cuda:1"
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_DIR"

echo "[START] v6 closed loop: teacher=physical GPU $TEACHER_GPU, student=physical GPU $STUDENT_GPU"
python scripts/run_stage2_closed_loop.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  2>&1 | tee "$LOG_DIR/run.log"

echo "[DONE] Result: $OUTPUT/result.json"
echo "[DONE] Report: $OUTPUT/CLOSED_LOOP_REPORT.md"
