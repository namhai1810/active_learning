#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/stage2_benchmark_ours_verified_v2.yaml"
DISCOVERY_CONFIG="configs/stage2_ours_verified_v2_discovery.yaml"
VERIFICATION_CONFIG="configs/stage2_ours_verified_v2_verification.yaml"
GENERATION_CONFIG="configs/stage2_generation_ours.yaml"
SHARED_DIR="outputs/stage2_shared"
ACQUISITION_ROOT="outputs/stage2_acquisition_ours_verified_v2"
OLD_CACHE_ROOT="outputs/stage2_acquisition_ours_verified_target_only"

SEED=42
BUDGET=128
OBJECTIVE=pairwise
POOL_ONLY=false
SKIP_POOL=false
REPORT_ONLY=false
COMPARE_ONLY=false

usage() {
  cat <<'EOF'
Usage:
  ./run_ours_verified_v2.sh [--seed 42|43|44] [--budget 32|64|128]
                            [--objective pairwise|ce]
                            [--pool-only | --skip-pool | --report-only]
  ./run_ours_verified_v2.sh [--seed N] [--budget N] --compare-only

Protocol v2 keeps target-failure Hoeffding certification and adds:
  - five diverse discovery hypotheses;
  - predeclared adaptive verification looks at 20/30/40/50/60 sets;
  - hypothesis/pair-type/label stratification;
  - student NLL utility plus lexical diversity;
  - controlled CE versus pairwise objective ablation on the same selected IDs.

Modes:
  --pool-only    Build/resume the shared verified synthetic pool, then stop.
  --skip-pool    Require an already completed pool; run one objective only.
  --report-only  Rebuild the single-objective report from an existing result.
  --compare-only Compare completed pairwise and CE results and verify identical IDs.

Environment:
  CUDA_DEVICE    Physical GPU to expose (default: 0).

The response cache is bootstrapped from an earlier target-only run when
available. Only matching immutable teacher requests are reused; v2 discovery
and verification decisions remain isolated in a new output directory.
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
    --objective)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      OBJECTIVE="$2"
      shift 2
      ;;
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
    --compare-only)
      COMPARE_ONLY=true
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
case "$OBJECTIVE" in
  pairwise|ce) ;;
  *) echo "ERROR: objective must be pairwise or ce" >&2; exit 2 ;;
esac

mode_count=0
for enabled in "$POOL_ONLY" "$SKIP_POOL" "$REPORT_ONLY" "$COMPARE_ONLY"; do
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

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POOL_DIR="$ACQUISITION_ROOT/seed_$SEED"
RUN_ROOT="outputs/stage2_runs_ours_verified_v2_${OBJECTIVE}"
RUN_DIR="$RUN_ROOT/experiment_a/ours_verified/budget_$BUDGET/seed_$SEED"
REPORT_DIR="reports/stage2_ours_verified_v2_${OBJECTIVE}/seed_${SEED}_budget_${BUDGET}"
LOG_ROOT="logs/stage2_ours_verified_v2_${OBJECTIVE}"
LOG_DIR="$LOG_ROOT/seed_${SEED}_budget_${BUDGET}"
PARTIAL_DIR="$LOG_ROOT/partial_runs"
PAIRWISE_RESULT="outputs/stage2_runs_ours_verified_v2_pairwise/experiment_a/ours_verified/budget_${BUDGET}/seed_${SEED}/result.json"
CE_RESULT="outputs/stage2_runs_ours_verified_v2_ce/experiment_a/ours_verified/budget_${BUDGET}/seed_${SEED}/result.json"
ABLATION_DIR="reports/stage2_ours_verified_v2_ablation/seed_${SEED}_budget_${BUDGET}"
mkdir -p "$LOG_DIR" "$PARTIAL_DIR"

if [[ "$COMPARE_ONLY" == true ]]; then
  python scripts/report_ours_objective_ablation.py \
    --pairwise-result "$PAIRWISE_RESULT" \
    --ce-result "$CE_RESULT" \
    --output-dir "$ABLATION_DIR"
  echo "[DONE] Ablation report: $ABLATION_DIR/OBJECTIVE_ABLATION.md"
  exit 0
fi

for path in \
  "$SHARED_DIR/hans_frozen_manifest.json" \
  "$SHARED_DIR/predictions.jsonl" \
  "$SHARED_DIR/sealed_final_test.jsonl" \
  "$SHARED_DIR/preparation.json"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing frozen artifact: $path" >&2
    exit 2
  fi
done

if [[ "$REPORT_ONLY" != true && "$SKIP_POOL" != true ]]; then
  CACHE_SOURCE="$OLD_CACHE_ROOT/seed_$SEED/teacher_calls/response_cache"
  CACHE_DEST="$POOL_DIR/teacher_calls/response_cache"
  if [[ -d "$CACHE_SOURCE" && ! -e "$CACHE_DEST" ]]; then
    echo "[CACHE] Reusing matching immutable teacher responses from $CACHE_SOURCE"
    mkdir -p "$CACHE_DEST"
    cp -a "$CACHE_SOURCE/." "$CACHE_DEST/"
  fi
  echo "[1/3] Building protocol-v2 verified pool (seed=$SEED)"
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
fi

if [[ ! -f "$POOL_DIR/.complete" ]]; then
  echo "ERROR: completed v2 pool not found at $POOL_DIR" >&2
  echo "Build it first with: CUDA_DEVICE=0 ./run_ours_verified_v2.sh --seed $SEED --budget $BUDGET --pool-only" >&2
  exit 1
fi

if [[ "$POOL_ONLY" == true ]]; then
  echo "[DONE] Shared pool: $POOL_DIR"
  exit 0
fi

if [[ "$REPORT_ONLY" != true ]]; then
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

  echo "[2/3] Training/evaluating ours_verified objective=$OBJECTIVE"
  python scripts/run_stage2_suite.py \
    --config "$CONFIG" \
    --experiment A \
    --output-root "$RUN_ROOT" \
    --methods ours_verified \
    --budgets "$BUDGET" \
    --seeds "$SEED" \
    --ours-objective "$OBJECTIVE" \
    2>&1 | tee -a "$LOG_DIR/benchmark.log"
fi

if [[ ! -f "$RUN_DIR/result.json" ]]; then
  echo "ERROR: no completed result found at $RUN_DIR/result.json" >&2
  exit 1
fi

echo "[3/3] Writing objective=$OBJECTIVE report"
python scripts/report_stage2_benchmark.py \
  --runs-root "$RUN_ROOT/experiment_a" \
  --output-dir "$REPORT_DIR" \
  --methods ours_verified \
  --budgets "$BUDGET" \
  --seeds "$SEED" \
  2>&1 | tee -a "$LOG_DIR/report.log"

echo "[DONE] Result: $RUN_DIR/result.json"
echo "[DONE] Verification report: $POOL_DIR/verification/REPORT.md"
echo "[DONE] Objective report: $REPORT_DIR/STAGE2_REPORT.md"
if [[ -f "$PAIRWISE_RESULT" && -f "$CE_RESULT" ]]; then
  if python scripts/report_ours_objective_ablation.py \
      --pairwise-result "$PAIRWISE_RESULT" \
      --ce-result "$CE_RESULT" \
      --output-dir "$ABLATION_DIR"; then
    echo "[DONE] Ablation report: $ABLATION_DIR/OBJECTIVE_ABLATION.md"
  else
    echo "[WARN] Both objective results exist, but they are not a valid controlled ablation."
    echo "       Inspect their run_status and selected_ids.json files."
  fi
fi
