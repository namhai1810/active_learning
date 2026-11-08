# Archived V7 Cost-Aware side experiments

These experiments are retained for historical reproducibility. They are not
part of the V7 runtime and must not be imported by
`scripts/run_stage2_closed_loop.py`.

The files remain at their original paths rather than being physically moved.
Moving them would break historical commands, imports, report links and cached
artifact provenance. This is a logical archive: the source is frozen and the
main research direction is documented in
`docs/v7_next_round_cost_aware_refactor.md`.

## v4 — Decision-Aware Adaptive Pilot Allocation

Status: archived offline simulation.

- `src/decision/adaptive_pilot_allocation.py`
- `scripts/simulate_v7_adaptive_pilot_allocation.py`
- `tests/test_v7_adaptive_pilot_allocation.py`
- `docs/v7_adaptive_pilot_allocation_v4.md`
- `reports/v7_minimal_cost_aware/ADAPTIVE_PILOT_ALLOCATION_V4.md`
- `outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/`

Reason: the VOI allocator and its effective fixed-4 behavior are not the
original next-round expected-net-gain objective. Cached results and code stay
available, but none may influence V7 decisions.

## v5 — Pre-Round Risk-Ceiling Stopping

Status: archived offline replay.

- `scripts/replay_v7_pre_round_stopping_v5.py`
- `tests/test_v7_pre_round_stopping_v5.py`
- `docs/v7_pre_round_stopping_v5.md`
- `reports/v7_minimal_cost_aware/PRE_ROUND_STOPPING_V5.md`
- `outputs/stage2_diagnostics/v7_pre_round_stopping_v5/`

Reason: the risk-ceiling objective is not the expected improvement of the next
active-learning round. It remains useful as a negative diagnostic but is not a
V7 stopping rule.

## Preserved utilities

The following remain active research utilities because they support the
original objective without selecting a different objective:

- `src/gain/estimation.py` — pilot gain estimation and Hoeffding accounting.
- `src/gain/range_aware.py` — range-aware confidence bounds.
- `src/decision/cost_model.py` — frozen rates and component accounting.
- `src/decision/incremental_cost_accounting.py` — cost partition diagnostics.
- `src/decision/next_round_cost_aware.py` — estimator-agnostic future
  integration contract.

Historical v2/v2.1 code, configs, outputs, checkpoints and reports are frozen
and have not been rewritten by this refactor.
