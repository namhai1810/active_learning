#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_benchmark_ours_verified_v5.yaml"
DISCOVERY_CONFIG="configs/stage2_ours_verified_v2_discovery.yaml"
VERIFICATION_CONFIG="configs/stage2_ours_verified_v2_verification.yaml"
GENERATION_CONFIG="configs/stage2_generation_ours_confidence_safe.yaml"
SOURCE_POOL="outputs/stage2_acquisition_ours_verified_v2/seed_42"
POOL_DIR="outputs/stage2_acquisition_ours_verified_v5/seed_42"
SUBSET_MANIFEST="outputs/stage2_shared/gain_eval_stopping_subset_v1.json"
RUN_ROOT="outputs/stage2_runs_ours_verified_v5_ce"
RUN_DIR="$RUN_ROOT/experiment_b/ours_verified/budget_128/seed_42"
REPORT_DIR="reports/stage2_ours_verified_v5_ce/seed_42_budget_128"
LOG_DIR="logs/stage2_ours_verified_v5_ce/seed_42_budget_128"
PARTIAL_DIR="logs/stage2_ours_verified_v5_ce/partial_runs"

POOL_ONLY=false
SKIP_POOL=false
REPORT_ONLY=false

usage() {
  cat <<'EOF'
Usage:
  CUDA_DEVICE=0 ./run_ours_verified_v5.sh
  POOL_CUDA_DEVICE=0 RUN_CUDA_DEVICE=1 ./run_ours_verified_v5.sh
  ./run_ours_verified_v5.sh --pool-only | --skip-pool | --report-only

Protocol v5 (seed=42, maximum budget=128):
  - reuses completed v2 discovery and controlled verification artifacts;
  - retains certified + unresolved hypotheses and excludes only refuted ones;
  - generates a fresh shared candidate pool for every retained hypothesis;
  - selects complete pairs using target gold-NLL + entropy and contextual novelty;
  - seeds coverage with one pair per retained hypothesis, without equal quotas;
  - uses v4's low-cost stopping/evaluation protocol and CE training.

Environment:
  CUDA_DEVICE         Default physical GPU for both sequential stages.
  POOL_CUDA_DEVICE    GPU for teacher generation (defaults to CUDA_DEVICE).
  RUN_CUDA_DEVICE     GPU for scoring/training/evaluation (defaults to CUDA_DEVICE).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pool-only)
      POOL_ONLY=true
      shift
      ;;
    --skip-pool)
      SKIP_POOL=true
      shift
      ;;
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

mode_count=0
for enabled in "$POOL_ONLY" "$SKIP_POOL" "$REPORT_ONLY"; do
  [[ "$enabled" == true ]] && mode_count=$((mode_count + 1))
done
if (( mode_count > 1 )); then
  echo "ERROR: choose at most one execution mode" >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "active_lr" ]]; then
  echo "ERROR: activate the active_lr environment first:" >&2
  echo "  conda activate active_lr" >&2
  exit 2
fi

for path in \
  "$SOURCE_POOL/hypotheses.json" \
  "$SOURCE_POOL/hypothesis_evidence.json" \
  "$SOURCE_POOL/hans_split_v2.json" \
  "$SOURCE_POOL/verification/results.json" \
  "$SOURCE_POOL/verification/probes.jsonl" \
  "$SOURCE_POOL/verification/probe_set_outcomes.jsonl" \
  "$SOURCE_POOL/teacher_calls/discovery.jsonl" \
  "$SOURCE_POOL/teacher_calls/verification.jsonl" \
  "outputs/stage2_shared/hans_frozen_manifest.json" \
  "outputs/stage2_shared/predictions.jsonl" \
  "outputs/stage2_shared/sealed_final_test.jsonl"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required reusable artifact is missing: $path" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_DIR" "$PARTIAL_DIR"

copy_file_if_missing() {
  local source="$1"
  local destination="$2"
  if [[ ! -e "$destination" ]]; then
    mkdir -p "$(dirname -- "$destination")"
    cp -a "$source" "$destination"
  fi
}

if [[ "$REPORT_ONLY" != true && "$SKIP_POOL" != true ]]; then
  echo "[1/4] Reusing frozen v2 discovery and verification (no repeated teacher calls)"
  copy_file_if_missing "$SOURCE_POOL/hypotheses.json" "$POOL_DIR/hypotheses.json"
  copy_file_if_missing "$SOURCE_POOL/hypothesis_evidence.json" "$POOL_DIR/hypothesis_evidence.json"
  copy_file_if_missing "$SOURCE_POOL/hans_split_v2.json" "$POOL_DIR/hans_split_v2.json"
  copy_file_if_missing "$SOURCE_POOL/teacher_calls/discovery.jsonl" "$POOL_DIR/teacher_calls/discovery.jsonl"
  copy_file_if_missing "$SOURCE_POOL/teacher_calls/verification.jsonl" "$POOL_DIR/teacher_calls/verification.jsonl"
  if [[ ! -d "$POOL_DIR/verification" ]]; then
    mkdir -p "$POOL_DIR"
    cp -a "$SOURCE_POOL/verification" "$POOL_DIR/verification"
  fi
  if [[ -d "$SOURCE_POOL/teacher_calls/response_cache" && \
        ! -e "$POOL_DIR/teacher_calls/response_cache" ]]; then
    mkdir -p "$POOL_DIR/teacher_calls"
    cp -a "$SOURCE_POOL/teacher_calls/response_cache" "$POOL_DIR/teacher_calls/response_cache"
  fi

  export CUDA_VISIBLE_DEVICES="${POOL_CUDA_DEVICE:-${CUDA_DEVICE:-0}}"
  echo "[2/4] Building confidence-safe synthetic pool on physical GPU $CUDA_VISIBLE_DEVICES"
  python scripts/build_stage2_synthetic_pool.py \
    --benchmark-config "$CONFIG" \
    --method ours_verified \
    --seed 42 \
    --output-dir "$POOL_DIR" \
    --discovery-config "$DISCOVERY_CONFIG" \
    --verification-config "$VERIFICATION_CONFIG" \
    --generation-config "$GENERATION_CONFIG" \
    --resume \
    2>&1 | tee -a "$LOG_DIR/pool.log"
fi

if [[ ! -f "$POOL_DIR/.complete" ]]; then
  echo "ERROR: completed v5 pool not found at $POOL_DIR" >&2
  exit 1
fi
if [[ "$POOL_ONLY" == true ]]; then
  echo "[DONE] Confidence-safe pool: $POOL_DIR"
  exit 0
fi

echo "[3/4] Freezing/validating the stopping subset"
python scripts/prepare_stage2_stopping_subset.py \
  --config "$CONFIG" \
  --output "$SUBSET_MANIFEST" \
  --examples-per-subcase 30 \
  2>&1 | tee -a "$LOG_DIR/subset.log"

if [[ "$REPORT_ONLY" != true ]]; then
  if [[ -f "$RUN_DIR/result.json" ]]; then
    echo "ERROR: completed v5 result already exists: $RUN_DIR/result.json" >&2
    echo "Use --report-only, or move the old result before starting a new run." >&2
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
    echo "[RECOVER] Moving incomplete v5 run to $destination"
    mv "$RUN_DIR" "$destination"
  fi

  export CUDA_VISIBLE_DEVICES="${RUN_CUDA_DEVICE:-${CUDA_DEVICE:-0}}"
  echo "[4/4] Scoring, training, and evaluating on physical GPU $CUDA_VISIBLE_DEVICES"
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

echo "[REPORT] Writing v5 report"
python scripts/report_stage2_benchmark.py \
  --runs-root "$RUN_ROOT/experiment_b" \
  --output-dir "$REPORT_DIR" \
  --methods ours_verified \
  --budgets 128 \
  --seeds 42 \
  2>&1 | tee -a "$LOG_DIR/report.log"

echo "[DONE] Pool: $POOL_DIR"
echo "[DONE] Result: $RUN_DIR/result.json"
echo "[DONE] Report: $REPORT_DIR/STAGE2_REPORT.md"
