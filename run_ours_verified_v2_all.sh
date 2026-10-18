#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SEED=42
BUDGET=128

usage() {
  cat <<'EOF'
Usage:
  ./run_ours_verified_v2_all.sh [--seed 42|43|44] [--budget 32|64|128]

Execution order:
  1. Build/resume the shared verified pool on physical GPU 0.
  2. Run pairwise training/evaluation on GPU 0 and CE on GPU 1 in parallel.
  3. Verify that both runs used identical selected IDs and write the ablation report.

Defaults: seed=42, budget=128.
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

echo "[ALL 1/3] Building/resuming verified pool on physical GPU 0"
CUDA_DEVICE=0 ./run_ours_verified_v2.sh \
  --seed "$SEED" \
  --budget "$BUDGET" \
  --pool-only

echo "[ALL 2/3] Starting pairwise on GPU 0 and CE on GPU 1"
CUDA_DEVICE=0 ./run_ours_verified_v2.sh \
  --seed "$SEED" \
  --budget "$BUDGET" \
  --objective pairwise \
  --skip-pool &
PAIRWISE_PID=$!

CUDA_DEVICE=1 ./run_ours_verified_v2.sh \
  --seed "$SEED" \
  --budget "$BUDGET" \
  --objective ce \
  --skip-pool &
CE_PID=$!

set +e
wait "$PAIRWISE_PID"
PAIRWISE_STATUS=$?
wait "$CE_PID"
CE_STATUS=$?
set -e

if (( PAIRWISE_STATUS != 0 || CE_STATUS != 0 )); then
  echo "ERROR: objective run failed: pairwise=$PAIRWISE_STATUS ce=$CE_STATUS" >&2
  echo "Inspect logs/stage2_ours_verified_v2_pairwise and logs/stage2_ours_verified_v2_ce." >&2
  exit 1
fi

echo "[ALL 3/3] Writing controlled CE-versus-pairwise ablation report"
CUDA_DEVICE=0 ./run_ours_verified_v2.sh \
  --seed "$SEED" \
  --budget "$BUDGET" \
  --compare-only

echo "[ALL DONE] Protocol v2 completed for seed=$SEED budget=$BUDGET"
echo "Ablation report: reports/stage2_ours_verified_v2_ablation/seed_${SEED}_budget_${BUDGET}/OBJECTIVE_ABLATION.md"
