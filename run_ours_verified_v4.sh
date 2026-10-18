#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_benchmark_ours_verified_v4.yaml"
SUBSET_MANIFEST="outputs/stage2_shared/gain_eval_stopping_subset_v1.json"
POOL_DIR="outputs/stage2_acquisition_ours_verified_v2/seed_42"
RUN_ROOT="outputs/stage2_runs_ours_verified_v4_ce"
RUN_DIR="$RUN_ROOT/experiment_b/ours_verified/budget_128/seed_42"
REPORT_DIR="reports/stage2_ours_verified_v4_ce/seed_42_budget_128"
LOG_DIR="logs/stage2_ours_verified_v4_ce/seed_42_budget_128"
PARTIAL_DIR="logs/stage2_ours_verified_v4_ce/partial_runs"
REPORT_ONLY=false

usage() {
  cat <<'EOF'
Usage:
  CUDA_DEVICE=0 ./run_ours_verified_v4.sh
  CUDA_DEVICE=0 ./run_ours_verified_v4.sh --report-only

Protocol v4 is frozen for seed=42 and maximum budget=128. It reuses v3
acquisition/training settings, but reduces adaptive-evaluation cost:
  - 900 stopping examples: 30 per HANS subcase, deterministic metadata-only hash;
  - cached base predictions: no round-0 inference;
  - practical paired-gain stopping at 32/64/96/128;
  - 2,850 held-out gain-report examples evaluated once after checkpoint choice;
  - sealed HANS final and MNLI evaluated once at the selected checkpoint.

Environment:
  CUDA_DEVICE    Physical GPU to expose (default: 0).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-only)
      REPORT_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr environment first:" >&2
  echo "  conda activate active_lr" >&2
  exit 2
fi

if [[ ! -f "$POOL_DIR/.complete" ]]; then
  echo "ERROR: completed v2 pool sentinel is missing: $POOL_DIR/.complete" >&2
  exit 1
fi
for path in \
  "$POOL_DIR/synthetic/accepted.jsonl" \
  "$POOL_DIR/synthetic/manifest.json" \
  "$POOL_DIR/verification/probes.jsonl" \
  "$POOL_DIR/verification/probe_set_outcomes.jsonl" \
  "outputs/stage2_shared/hans_frozen_manifest.json" \
  "outputs/stage2_shared/predictions.jsonl" \
  "outputs/stage2_shared/sealed_final_test.jsonl"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required reusable artifact is missing: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$PARTIAL_DIR"

echo "[1/3] Freezing/validating the 900-example stopping subset"
python scripts/prepare_stage2_stopping_subset.py \
  --config "$CONFIG" \
  --output "$SUBSET_MANIFEST" \
  --examples-per-subcase 30 \
  2>&1 | tee -a "$LOG_DIR/subset.log"

if [[ "$REPORT_ONLY" != true ]]; then
  if [[ -f "$RUN_DIR/result.json" ]]; then
    echo "ERROR: completed v4 result already exists: $RUN_DIR/result.json" >&2
    echo "Use --report-only, or move the old run before starting a new one." >&2
    exit 1
  fi
  if [[ -d "$RUN_DIR" ]] && \
     [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    destination="$PARTIAL_DIR/run_seed_42_$timestamp"
    suffix=0
    while [[ -e "$destination" ]]; do
      suffix=$((suffix + 1))
      destination="${destination}_$suffix"
    done
    echo "[RECOVER] Moving incomplete v4 run to $destination"
    mv "$RUN_DIR" "$destination"
  fi

  echo "[2/3] Running ours_verified v4 CE (seed=42, max budget=128)"
  python scripts/run_stage2_suite.py \
    --config "$CONFIG" \
    --experiment B \
    --output-root "$RUN_ROOT" \
    --methods ours_verified \
    --budgets 128 \
    --seeds 42 \
    --ours-objective ce \
    2>&1 | tee -a "$LOG_DIR/benchmark.log"
fi

if [[ ! -f "$RUN_DIR/result.json" ]]; then
  echo "ERROR: no completed result found at $RUN_DIR/result.json" >&2
  exit 1
fi

echo "[3/3] Writing v4 report"
python scripts/report_stage2_benchmark.py \
  --runs-root "$RUN_ROOT/experiment_b" \
  --output-dir "$REPORT_DIR" \
  --methods ours_verified \
  --budgets 128 \
  --seeds 42 \
  2>&1 | tee -a "$LOG_DIR/report.log"

echo "[DONE] Stopping subset: $SUBSET_MANIFEST"
echo "[DONE] Result: $RUN_DIR/result.json"
echo "[DONE] Stopping trace: $RUN_DIR/rounds.jsonl"
echo "[DONE] Held-out gain report: $RUN_DIR/gain_report_predictions.jsonl"
echo "[DONE] Report: $REPORT_DIR/STAGE2_REPORT.md"
