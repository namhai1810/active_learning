#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_benchmark_ours_verified_target_only.yaml"
DISCOVERY_CONFIG="configs/stage2_ours_verified_efficient_discovery.yaml"
VERIFICATION_CONFIG="configs/stage2_ours_verified_target_only_verification.yaml"
GENERATION_CONFIG="configs/stage2_generation_ours.yaml"
SHARED_DIR="outputs/stage2_shared"
ACQUISITION_ROOT="outputs/stage2_acquisition_ours_verified_target_only"
RUN_ROOT="outputs/stage2_runs_ours_verified_target_only"
LOG_ROOT="logs/stage2_ours_verified_target_only"
STRICT_ACQUISITION_ROOT="outputs/stage2_acquisition_ours_verified"

# Seed 42 informed the profile diagnosis, so default to a fresh configured seed.
SEED=43
BUDGET=128
REPORT_ONLY=false
REUSE_STRICT_PROBES=false

usage() {
  cat <<'EOF'
Usage:
  ./run_ours_verified.sh [--seed 42|43|44] [--budget 32|64|128] [--reuse-strict-probes]
  ./run_ours_verified.sh [--seed N] [--budget N] --report-only

Runs target-failure-certified ours_verified acquisition, training, evaluation,
and reporting. Strict-joint/control metrics remain diagnostics. Artifacts are
isolated from both the original strict-joint run and the six-method grid.

--reuse-strict-probes copies a completed strict-joint seed into the isolated
target-only directory. The builder archives its old decision, replays cached
probe generation/QC, and recomputes certification without regenerating probes.

Environment:
  CUDA_DEVICE  Physical GPU to expose (default: 0).

Defaults: seed=43, budget=128.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --budget)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      BUDGET="$2"
      shift 2
      ;;
    --report-only)
      REPORT_ONLY=true
      shift
      ;;
    --reuse-strict-probes)
      REUSE_STRICT_PROBES=true
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

case "$SEED" in
  42|43|44) ;;
  *) echo "ERROR: seed must be 42, 43, or 44" >&2; exit 2 ;;
esac
case "$BUDGET" in
  32|64|128) ;;
  *) echo "ERROR: budget must be 32, 64, or 128" >&2; exit 2 ;;
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

POOL_DIR="$ACQUISITION_ROOT/seed_$SEED"
STRICT_POOL_DIR="$STRICT_ACQUISITION_ROOT/seed_$SEED"
RUN_DIR="$RUN_ROOT/experiment_a/ours_verified/budget_$BUDGET/seed_$SEED"
REPORT_DIR="reports/stage2_ours_verified_target_only/seed_${SEED}_budget_${BUDGET}"
LOG_DIR="$LOG_ROOT/seed_${SEED}_budget_${BUDGET}"
PARTIAL_DIR="$LOG_ROOT/partial_runs"
mkdir -p "$LOG_DIR" "$PARTIAL_DIR"

for path in \
  "$SHARED_DIR/hans_frozen_manifest.json" \
  "$SHARED_DIR/predictions.jsonl" \
  "$SHARED_DIR/sealed_final_test.jsonl" \
  "$SHARED_DIR/preparation.json"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing frozen artifact: $path" >&2
    echo "Prepare it with:" >&2
    echo "  python scripts/prepare_stage2_benchmark.py --config configs/stage2_benchmark.yaml --output-dir $SHARED_DIR" >&2
    exit 2
  fi
done

if [[ "$REPORT_ONLY" != true ]]; then
  if [[ "$REUSE_STRICT_PROBES" == true && ! -e "$POOL_DIR" ]]; then
    if [[ ! -f "$STRICT_POOL_DIR/verification/results.json" ]]; then
      echo "ERROR: no completed strict-joint verification found at $STRICT_POOL_DIR" >&2
      exit 1
    fi
    mkdir -p "$ACQUISITION_ROOT"
    echo "[REUSE] Copying strict-joint seed artifacts into the isolated target-only pool"
    cp -a "$STRICT_POOL_DIR" "$POOL_DIR"
  elif [[ "$REUSE_STRICT_PROBES" == true && -e "$POOL_DIR" ]]; then
    echo "[REUSE] Target-only pool already exists; resuming it without copying"
  fi

  echo "[1/3] Building the target-failure-certified ours_verified pool (seed=$SEED)"
  python scripts/build_stage2_synthetic_pool.py \
    --benchmark-config "$CONFIG" \
    --method ours_verified \
    --seed "$SEED" \
    --output-dir "$POOL_DIR" \
    --discovery-config "$DISCOVERY_CONFIG" \
    --verification-config "$VERIFICATION_CONFIG" \
    --generation-config "$GENERATION_CONFIG" \
    --resume \
    2>&1 | tee -a "$LOG_DIR/pool.log"

  if [[ -d "$RUN_DIR" && ! -f "$RUN_DIR/result.json" ]] && \
     [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    destination="$PARTIAL_DIR/run_budget_${BUDGET}_seed_${SEED}_$timestamp"
    suffix=0
    while [[ -e "$destination" ]]; do
      suffix=$((suffix + 1))
      destination="${destination}_$suffix"
    done
    echo "[RECOVER] Moving incomplete benchmark run to $destination"
    mv "$RUN_DIR" "$destination"
  fi

  echo "[2/3] Training and evaluating only ours_verified"
  python scripts/run_stage2_suite.py \
    --config "$CONFIG" \
    --experiment A \
    --output-root "$RUN_ROOT" \
    --methods ours_verified \
    --budgets "$BUDGET" \
    --seeds "$SEED" \
    2>&1 | tee -a "$LOG_DIR/benchmark.log"
fi

if [[ ! -f "$RUN_DIR/result.json" ]]; then
  echo "ERROR: no completed result found at $RUN_DIR/result.json" >&2
  exit 1
fi

echo "[3/3] Writing the single-method report"
python scripts/report_stage2_benchmark.py \
  --runs-root "$RUN_ROOT/experiment_a" \
  --output-dir "$REPORT_DIR" \
  --methods ours_verified \
  --budgets "$BUDGET" \
  --seeds "$SEED" \
  2>&1 | tee -a "$LOG_DIR/report.log"

echo "[DONE] Result: $RUN_DIR/result.json"
echo "[DONE] Verification report: $POOL_DIR/verification/REPORT.md"
echo "[DONE] Report: $REPORT_DIR/STAGE2_REPORT.md"
