# V7 final mathematical-feasibility experiment

This is a read-only, CPU-only audit of the frozen V7 Cost-Aware trajectories
for seeds 42 and 43. It is not wired into V7 training and does not change the
historical v2/v2.1 decisions.

## Decision target

The desired pre-round certificate is

`U_net,t+1 = gain_UCB,t+1 + rho_t+1 - lambda * full_round_cost_LCB,t+1 <= 0`.

The frozen constants are `lambda=0.2`, `delta=0.05`, `K=1`, `T=7`, and
`m=32`. Budget exhaustion is reported separately from a certified gain–cost
STOP.

## A. Gain bounds

For the cached fixed-shard pilot observations,

`X_j = accuracy_after_j - accuracy_before`.

The common baseline gives exact support `[-A0, 1-A0]`, whose width is one.
The audit compares:

- range-aware Hoeffding:
  `sqrt(log(2KT/delta)/(2m))`;
- empirical Bernstein:
  `sqrt(2*s^2*log(2KT/delta)/m) + 7*log(2KT/delta)/(3*(m-1))`.

Both UCBs are intersected with the exact support upper endpoint. The metadata
checks fixed evaluation IDs, fixed baseline/checkpoint/action, the gain
identity, support, and distinct training seeds. It cannot prove stochastic
independence of the deterministic seed schedule or population unbiasedness,
so the output explicitly limits the bound to a conditional fixed-shard target.

## B. Cost lower bound

Only costs with a deterministic positive minimum before the decision enter the
LCB. For an executable 32-example intervention these are:

- two selector scoring passes over at least 32 items;
- one fixed LoRA run charge;
- one epoch validation plus the restored-checkpoint evaluation on the frozen
  900-example stopping subset.

Discovery, teacher tokens, adaptive verification, generation/QC, and GPU time
receive a valid lower bound of zero because the frozen protocol does not prove
a state-independent positive minimum. Historical realised means and projected
GPU seconds are never substituted for a lower confidence bound.

The bound is conditional on an executable intervention. If V7 has no eligible
candidate/data, the terminal reason is a data/evidence gate rather than a
gain–cost certificate.

## C. Cross-round stability

Pilots for round `t` are produced after round `t` discovery, verification,
generation, and selection. They therefore cannot be used for a decision made
before that round. Transferring the prior round's pilot UCB requires a valid
cross-round stability allowance `rho`.

The replay reports consecutive historical drift only as a diagnostic. Two
dependent seed trajectories, missing pilot rounds, and post-hoc maxima do not
provide a valid calibration. If `rho` is unavailable, the pre-round result is
`uncertified_continue`, regardless of whether a counterfactual `rho=0`
calculation looks favorable.

## Outputs

- `reports/v7_minimal_cost_aware/FINAL_MATHEMATICAL_FEASIBILITY.md`
- `outputs/stage2_diagnostics/v7_final_mathematical_feasibility/replay.json`
- `outputs/stage2_diagnostics/v7_final_mathematical_feasibility/per_round.csv`
- `outputs/stage2_diagnostics/v7_final_mathematical_feasibility/drift.csv`
- `outputs/stage2_diagnostics/v7_final_mathematical_feasibility/cost_lower_bound.csv`
- `outputs/stage2_diagnostics/v7_final_mathematical_feasibility/validation.json`

Retrospective checkpoint accuracy is used only to report regret after the
decision analysis. It is never an input to the stopping calculation.

## Commands

```bash
PYTHONPATH=. conda run -n active_lr pytest -q \
  tests/test_v7_final_mathematical_feasibility.py \
  tests/test_v7_next_round_cost_aware.py

CUDA_VISIBLE_DEVICES="" conda run -n active_lr \
  python scripts/replay_v7_final_mathematical_feasibility.py \
  --seeds 42 43 --validate-only

CUDA_VISIBLE_DEVICES="" conda run -n active_lr \
  python scripts/replay_v7_final_mathematical_feasibility.py \
  --seeds 42 43
```

These commands perform no inference or training and do not invoke a teacher.
