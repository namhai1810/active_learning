#!/usr/bin/env bash

# Ubuntu's /bin/sh is commonly dash. If this file is explicitly invoked with
# `sh run_experiment_A.sh`, restart it with Bash because the workflow uses
# pipefail and Bash arrays.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_benchmark.yaml"
SHARED_DIR="outputs/stage2_shared"
ACQUISITION_ROOT="outputs/stage2_acquisition"
RUN_ROOT="outputs/stage2_runs"
EXPERIMENT_ROOT="$RUN_ROOT/experiment_a"
REPORT_DIR="reports/stage2_experiment_A"
LOG_DIR="logs/stage2_experiment_A"
PARTIAL_DIR="$LOG_DIR/partial_runs"

METHODS=(
  zero_shot
  random_mnli
  uncertainty_mnli
  random_synthetic
  failure_driven_no_verification
  ours_verified
)
SYNTHETIC_METHODS=(
  random_synthetic
  failure_driven_no_verification
  ours_verified
)
BUDGETS=(32 64 128)
SEEDS=(42 43 44)

usage() {
  cat <<'EOF'
Usage:
  ./run_experiment_A.sh
  ./run_experiment_A.sh --report-only
  ./run_experiment_A.sh --help

Default workflow:
  1. Validate the already-prepared frozen Stage 2 HANS artifacts.
  2. Build/validate synthetic pools for three methods and seeds 42, 43, 44.
  3. Run all 54 Experiment A cells sequentially.
  4. Aggregate JSON/CSV/Markdown and generate SVG visualizations.

Completed pools/runs are skipped. Incomplete run directories are moved into
logs/stage2_experiment_A/partial_runs/ before resuming; they are never deleted.

Environment:
  CUDA_DEVICE  Physical GPU to expose (default: 0).
EOF
}

MODE="all"
case "${1:-}" in
  "") ;;
  --report-only) MODE="report" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
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

mkdir -p "$LOG_DIR" "$PARTIAL_DIR"

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required frozen artifact is missing or empty: $path" >&2
    echo "Run this once before Experiment A:" >&2
    echo "  python scripts/prepare_stage2_benchmark.py --config $CONFIG --output-dir $SHARED_DIR" >&2
    exit 2
  fi
}

validate_shared_artifacts() {
  require_file "$SHARED_DIR/hans_frozen_manifest.json"
  require_file "$SHARED_DIR/predictions.jsonl"
  require_file "$SHARED_DIR/sealed_final_test.jsonl"
  require_file "$SHARED_DIR/preparation.json"
  echo "[OK] Frozen Stage 2 artifacts are present."
}

archive_incomplete_directory() {
  local directory="$1"
  local completion_file="$2"
  local label="$3"
  if [[ -d "$directory" && ! -f "$completion_file" ]] && \
     [[ -n "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    local timestamp destination
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    destination="$PARTIAL_DIR/${label}_${timestamp}"
    local suffix=0
    while [[ -e "$destination" ]]; do
      suffix=$((suffix + 1))
      destination="$PARTIAL_DIR/${label}_${timestamp}_${suffix}"
    done
    echo "[RECOVER] Moving incomplete directory: $directory"
    echo "          -> $destination"
    mv "$directory" "$destination"
  fi
}

build_synthetic_pools() {
  echo "[PHASE 1/3] Building synthetic acquisition pools sequentially"
  local pool_index=0
  local pool_total=$((${#SYNTHETIC_METHODS[@]} * ${#SEEDS[@]}))
  for method in "${SYNTHETIC_METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      pool_index=$((pool_index + 1))
      local output="$ACQUISITION_ROOT/$method/seed_$seed"
      local accepted="$output/synthetic/accepted.jsonl"
      local manifest="$output/synthetic/manifest.json"
      if [[ -f "$accepted" && -f "$manifest" ]]; then
        echo "[POOL $pool_index/$pool_total][SKIP] Complete: method=$method seed=$seed"
        continue
      fi
      archive_incomplete_directory \
        "$output" "$output/.complete" "acquisition_${method}_seed_${seed}"
      echo "[POOL $pool_index/$pool_total] method=$method seed=$seed"
      python scripts/build_stage2_synthetic_pool.py \
        --benchmark-config "$CONFIG" \
        --method "$method" \
        --seed "$seed" \
        --output-dir "$output" \
        2>&1 | tee "$LOG_DIR/pool_${method}_seed_${seed}.log"
      if [[ ! -f "$accepted" || ! -f "$manifest" ]]; then
        echo "ERROR: synthetic pool did not produce required artifacts: $output" >&2
        exit 1
      fi
    done
  done
}

recover_incomplete_benchmark_runs() {
  for method in "${METHODS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        local output="$EXPERIMENT_ROOT/$method/budget_$budget/seed_$seed"
        archive_incomplete_directory \
          "$output" "$output/result.json" \
          "run_${method}_budget_${budget}_seed_${seed}"
      done
    done
  done
}

run_benchmark_grid() {
  echo "[PHASE 2/3] Running 6 methods × 3 budgets × 3 seeds = 54 runs"
  recover_incomplete_benchmark_runs
  python scripts/run_stage2_suite.py \
    --config "$CONFIG" \
    --experiment A \
    --output-root "$RUN_ROOT" \
    2>&1 | tee "$LOG_DIR/experiment_A.log"
}

build_report() {
  echo "[PHASE 3/3] Aggregating and visualizing Experiment A"
  local completed
  if [[ -d "$EXPERIMENT_ROOT" ]]; then
    completed="$(find "$EXPERIMENT_ROOT" -name result.json -type f | wc -l)"
  else
    completed=0
  fi
  if [[ "$completed" -ne 54 ]]; then
    echo "ERROR: expected 54 completed Experiment A runs, found $completed." >&2
    echo "Resume with: ./run_experiment_A.sh" >&2
    exit 1
  fi
  python scripts/report_stage2_benchmark.py \
    --runs-root "$EXPERIMENT_ROOT" \
    --output-dir "$REPORT_DIR" \
    2>&1 | tee "$LOG_DIR/report.log"

  local required_reports=(
    "$REPORT_DIR/STAGE2_REPORT.md"
    "$REPORT_DIR/stage2_summary.json"
    "$REPORT_DIR/stage2_summary.csv"
    "$REPORT_DIR/stage2_learning_curves.csv"
    "$REPORT_DIR/figures/experiment_a_final_hans_accuracy.svg"
    "$REPORT_DIR/figures/experiment_a_gain_eval_learning_curve.svg"
    "$REPORT_DIR/figures/experiment_a_accuracy_vs_cost.svg"
  )
  for path in "${required_reports[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "ERROR: report artifact is missing or empty: $path" >&2
      exit 1
    fi
  done
  echo "[DONE] Report: $REPORT_DIR/STAGE2_REPORT.md"
  echo "[DONE] Figures: $REPORT_DIR/figures/"
}

validate_shared_artifacts
if [[ "$MODE" == "report" ]]; then
  build_report
  exit 0
fi

echo "Experiment A executes sequentially and may take a long time."
echo "Completed artifacts are resumable and will be skipped on the next invocation."
build_synthetic_pools
run_benchmark_grid
build_report
