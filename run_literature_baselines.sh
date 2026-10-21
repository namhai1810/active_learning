#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BUDGET="${BUDGET:-128}"
POOL="${POOL:-all}"
CONFIG="configs/stage2_literature_baselines.yaml"
RUN_ROOT="outputs/stage2_runs_literature"
REPORT_DIR="reports/stage2_literature_baselines/seed_${SEED}_budget_${BUDGET}"

usage() {
  echo "Usage: GPU=0 SEED=42 BUDGET=128 $0 [--pool mnli|synthetic|all] [--report-only|--no-report]"
}

REPORT_ONLY=0
NO_REPORT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pool)
      POOL="$2"
      shift 2
      ;;
    --report-only)
      REPORT_ONLY=1
      shift
      ;;
    --no-report)
      NO_REPORT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${REPORT_ONLY}" -eq 1 && "${NO_REPORT}" -eq 1 ]]; then
  echo "--report-only and --no-report cannot be used together" >&2
  exit 2
fi

if [[ "${POOL}" != "mnli" && "${POOL}" != "synthetic" && "${POOL}" != "all" ]]; then
  echo "POOL must be mnli, synthetic, or all" >&2
  exit 2
fi

if [[ ! -f outputs/stage2_shared/hans_frozen_manifest.json ]]; then
  echo "Missing frozen Stage-2 manifest. Run prepare_stage2_benchmark.py first." >&2
  exit 1
fi

run_methods() {
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run_stage2_suite.py \
    --config "${CONFIG}" \
    --experiment B \
    --output-root "${RUN_ROOT}" \
    --methods "$@" \
    --seeds "${SEED}" \
    --budgets "${BUDGET}"
}

if [[ "${REPORT_ONLY}" -eq 0 ]]; then
  if [[ "${POOL}" == "mnli" || "${POOL}" == "all" ]]; then
    echo "[LITERATURE 1/2] MNLI pool: Margin, Core-set, BADGE, CAL"
    run_methods margin_mnli coreset_mnli badge_mnli cal_mnli
  fi

  if [[ "${POOL}" == "synthetic" || "${POOL}" == "all" ]]; then
    SYNTHETIC_COMPLETE="outputs/stage2_acquisition_ours_verified_v5/seed_${SEED}/.complete"
    if [[ ! -f "${SYNTHETIC_COMPLETE}" ]]; then
      echo "Missing controlled verified synthetic pool: ${SYNTHETIC_COMPLETE}" >&2
      exit 1
    fi
    echo "[LITERATURE 2/2] Controlled synthetic pool: Margin, Core-set, BADGE, CAL"
    run_methods margin_synthetic coreset_synthetic badge_synthetic cal_synthetic
  fi
fi

if [[ "${NO_REPORT}" -eq 0 ]]; then
  echo "[REPORT] Aggregating completed literature runs"
  python scripts/report_stage2_benchmark.py \
    --runs-root "${RUN_ROOT}" \
    --output-dir "${REPORT_DIR}" \
    --seeds "${SEED}" \
    --budgets "${BUDGET}"
fi

echo "[DONE] Runs: ${RUN_ROOT}/experiment_b"
if [[ "${NO_REPORT}" -eq 0 ]]; then
  echo "[DONE] Report: ${REPORT_DIR}/STAGE2_REPORT.md"
fi
