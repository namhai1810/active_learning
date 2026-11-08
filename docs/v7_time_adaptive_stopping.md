# V7 Time-Adaptive Sliding-Window Stopping

This experiment is a read-only CPU replay over frozen V7 pilot trajectories.
It is not connected to the V7 training/runtime controller.

## Estimator family

Before round `t+1`, the experiment evaluates fixed calendar windows
`L=1,...,7` and a cumulative-history estimator. Observations from `t+1` or
later are structurally excluded. Rounds without pilots remain in the calendar
window with zero statistical weight.

The estimator pools individual pilot gains, not round means, so unequal pilot
counts receive the correct observation weights. It reports calendar length,
round-level counts, effective observations, historical mean, and sample
variance.

## Simultaneous radius

The predeclared family contains:

`7 decision rounds × (7 fixed windows + cumulative) = 56 tests`.

For width-one bounded gains, the diagnostic radius is

`sqrt(log(2*56/0.05)/(2*n_eff))`.

Under a suitable conditional-independence or martingale assumption, this
controls deviations from the observation-weighted average of historical
conditional means. It does not, alone, control the next-round expected gain.
Consequently outputs use `assumption_conditional_diagnostic` unless those
assumptions are independently justified.

## Drift and policy mismatch

For an illustrative per-round stability assumption `d`, temporal uncertainty
is age-weighted:

`rho = sum_s m_s*d*(t+1-s) / sum_s m_s`.

Scenarios `d={0, 0.005, 0.01, 0.02}` use hypothetical `eta=0`. Neither is
calibrated from retrospective trajectories. Historical maximum drift is only a
diagnostic and never becomes a theoretical bound.

## Cost and certification

The replay reuses the audited protocol cost lower bound. That LCB is
conditional on an executable 32-example intervention. Since its applicability
is not established before discovery/verification, the replay does not silently
promote it to an unconditional bound.

A certified STOP requires all of the following:

- valid simultaneous sampling concentration;
- valid temporal drift bound `rho`;
- valid historical-policy mismatch bound `eta`;
- predecision applicability of the full-round cost LCB;
- nonpositive Net-UCB.

Missing evidence yields `uncertified_continue`; budget exhaustion yields the
separate state `budget_stop`.

## Commands

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. conda run -n active_lr pytest -q \
  tests/test_v7_time_adaptive_stopping.py \
  tests/test_v7_final_mathematical_feasibility.py \
  tests/test_v7_minimal_cost_aware.py \
  tests/test_cost_aware.py

CUDA_VISIBLE_DEVICES="" conda run -n active_lr \
  python scripts/replay_v7_time_adaptive_stopping.py \
  --seeds 42 43 --validate-only

CUDA_VISIBLE_DEVICES="" conda run -n active_lr \
  python scripts/replay_v7_time_adaptive_stopping.py \
  --seeds 42 43
```

These commands do not load a model, invoke a teacher, run inference, or train.

## Outputs

- `reports/v7_minimal_cost_aware/TIME_ADAPTIVE_STOPPING.md`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/replay.json`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/windows.csv`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/fixed_32_baseline.csv`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/drift_sensitivity.csv`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/historical_drift.csv`
- `outputs/stage2_diagnostics/v7_time_adaptive_stopping/validation.json`

Retrospective accuracy regret and realised cost savings are post-hoc fields and
never influence an online decision.
