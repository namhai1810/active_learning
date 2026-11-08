# V7 Time-Adaptive Sliding-Window Stopping

> CPU-only offline replay over frozen seeds 42/43. No GPU, teacher API, model loading, inference, training, or historical-artifact mutation was performed.

## Executive result

Pooling increases the largest effective sample from 32 to **160** pilots. After simultaneous Bonferroni correction over `7 decisions × (7 windows + cumulative) = 56` tests, the smallest radius is **0.155264**, versus **0.296721** for the existing fixed-32 bound: an absolute reduction of **0.141457** (47.67%).

The optimistic `rho=eta=0` diagnostic produces **0** negative Net-UCB window/decision combinations. None is a certified STOP: pooling concentrates around a historical weighted mean, while conditional sampling, next-round drift `rho`, policy mismatch `eta`, and predecision applicability of the conditional cost-LCB remain unproved.

The valid online conclusion is `uncertified_continue` through R8, followed by `budget_stop`.

## Radius by effective pilot count

| Effective pilots | Simultaneous radius | Change vs fixed-32 |
|---:|---:|---:|
| 32 | 0.347181 | 0.050460 |
| 64 | 0.245494 | -0.051227 |
| 96 | 0.200445 | -0.096276 |
| 128 | 0.173591 | -0.123131 |
| 160 | 0.155264 | -0.141457 |

The L=1 time-adaptive radius is wider than the fixed-32 baseline because the time-adaptive experiment protects the complete family of 56 tests. Larger windows eventually overcome that correction through larger `n_eff`.

## Historical-window estimates

| Seed | Pre-round | Window | Calendar rounds | Observed rounds | n_eff | Mean | Radius | Net-UCB rho=eta=0 | Numeric STOP |
|---:|---:|---|---|---|---:|---:|---:|---:|:---:|
| 42 | 2 | L1 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L2 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L3 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L4 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L5 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L6 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | L7 | [1] | [] | 0 | — | — | — | no |
| 42 | 2 | cumulative | [1] | [] | 0 | — | — | — | no |
| 42 | 3 | L1 | [2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L2 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L3 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L4 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L5 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L6 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | L7 | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 3 | cumulative | [1, 2] | [2] | 32 | 0.016602 | 0.347181 | 0.318503 | no |
| 42 | 4 | L1 | [3] | [3] | 32 | -0.045410 | 0.347181 | 0.256491 | no |
| 42 | 4 | L2 | [2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | L3 | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | L4 | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | L5 | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | L6 | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | L7 | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 4 | cumulative | [1, 2, 3] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | L1 | [4] | [] | 0 | — | — | — | no |
| 42 | 5 | L2 | [3, 4] | [3] | 32 | -0.045410 | 0.347181 | 0.256491 | no |
| 42 | 5 | L3 | [2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | L4 | [1, 2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | L5 | [1, 2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | L6 | [1, 2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | L7 | [1, 2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 5 | cumulative | [1, 2, 3, 4] | [2, 3] | 64 | -0.014404 | 0.245494 | 0.185810 | no |
| 42 | 6 | L1 | [5] | [5] | 32 | -0.026367 | 0.347181 | 0.275534 | no |
| 42 | 6 | L2 | [4, 5] | [5] | 32 | -0.026367 | 0.347181 | 0.275534 | no |
| 42 | 6 | L3 | [3, 4, 5] | [3, 5] | 64 | -0.035889 | 0.245494 | 0.164326 | no |
| 42 | 6 | L4 | [2, 3, 4, 5] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 6 | L5 | [1, 2, 3, 4, 5] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 6 | L6 | [1, 2, 3, 4, 5] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 6 | L7 | [1, 2, 3, 4, 5] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 6 | cumulative | [1, 2, 3, 4, 5] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 7 | L1 | [6] | [] | 0 | — | — | — | no |
| 42 | 7 | L2 | [5, 6] | [5] | 32 | -0.026367 | 0.347181 | 0.275534 | no |
| 42 | 7 | L3 | [4, 5, 6] | [5] | 32 | -0.026367 | 0.347181 | 0.275534 | no |
| 42 | 7 | L4 | [3, 4, 5, 6] | [3, 5] | 64 | -0.035889 | 0.245494 | 0.164326 | no |
| 42 | 7 | L5 | [2, 3, 4, 5, 6] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 7 | L6 | [1, 2, 3, 4, 5, 6] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 7 | L7 | [1, 2, 3, 4, 5, 6] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 7 | cumulative | [1, 2, 3, 4, 5, 6] | [2, 3, 5] | 96 | -0.018392 | 0.200445 | 0.136773 | no |
| 42 | 8 | L1 | [7] | [7] | 32 | 0.037109 | 0.347181 | 0.339011 | no |
| 42 | 8 | L2 | [6, 7] | [7] | 32 | 0.037109 | 0.347181 | 0.339011 | no |
| 42 | 8 | L3 | [5, 6, 7] | [5, 7] | 64 | 0.005371 | 0.245494 | 0.205585 | no |
| 42 | 8 | L4 | [4, 5, 6, 7] | [5, 7] | 64 | 0.005371 | 0.245494 | 0.205585 | no |
| 42 | 8 | L5 | [3, 4, 5, 6, 7] | [3, 5, 7] | 96 | -0.011556 | 0.200445 | 0.143609 | no |
| 42 | 8 | L6 | [2, 3, 4, 5, 6, 7] | [2, 3, 5, 7] | 128 | -0.004517 | 0.173591 | 0.123794 | no |
| 42 | 8 | L7 | [1, 2, 3, 4, 5, 6, 7] | [2, 3, 5, 7] | 128 | -0.004517 | 0.173591 | 0.123794 | no |
| 42 | 8 | cumulative | [1, 2, 3, 4, 5, 6, 7] | [2, 3, 5, 7] | 128 | -0.004517 | 0.173591 | 0.123794 | no |
| 43 | 2 | L1 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L2 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L3 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L4 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L5 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L6 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | L7 | [1] | [] | 0 | — | — | — | no |
| 43 | 2 | cumulative | [1] | [] | 0 | — | — | — | no |
| 43 | 3 | L1 | [2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L2 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L3 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L4 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L5 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L6 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | L7 | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 3 | cumulative | [1, 2] | [2] | 32 | 0.015625 | 0.347181 | 0.317526 | no |
| 43 | 4 | L1 | [3] | [3] | 32 | -0.068848 | 0.347181 | 0.233054 | no |
| 43 | 4 | L2 | [2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | L3 | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | L4 | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | L5 | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | L6 | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | L7 | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 4 | cumulative | [1, 2, 3] | [2, 3] | 64 | -0.026611 | 0.245494 | 0.173603 | no |
| 43 | 5 | L1 | [4] | [4] | 32 | 0.005859 | 0.347181 | 0.307761 | no |
| 43 | 5 | L2 | [3, 4] | [3, 4] | 64 | -0.031494 | 0.245494 | 0.168720 | no |
| 43 | 5 | L3 | [2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 5 | L4 | [1, 2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 5 | L5 | [1, 2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 5 | L6 | [1, 2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 5 | L7 | [1, 2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 5 | cumulative | [1, 2, 3, 4] | [2, 3, 4] | 96 | -0.015788 | 0.200445 | 0.139377 | no |
| 43 | 6 | L1 | [5] | [5] | 32 | -0.017578 | 0.347181 | 0.284323 | no |
| 43 | 6 | L2 | [4, 5] | [4, 5] | 64 | -0.005859 | 0.245494 | 0.194355 | no |
| 43 | 6 | L3 | [3, 4, 5] | [3, 4, 5] | 96 | -0.026855 | 0.200445 | 0.128310 | no |
| 43 | 6 | L4 | [2, 3, 4, 5] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 6 | L5 | [1, 2, 3, 4, 5] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 6 | L6 | [1, 2, 3, 4, 5] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 6 | L7 | [1, 2, 3, 4, 5] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 6 | cumulative | [1, 2, 3, 4, 5] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 7 | L1 | [6] | [] | 0 | — | — | — | no |
| 43 | 7 | L2 | [5, 6] | [5] | 32 | -0.017578 | 0.347181 | 0.284323 | no |
| 43 | 7 | L3 | [4, 5, 6] | [4, 5] | 64 | -0.005859 | 0.245494 | 0.194355 | no |
| 43 | 7 | L4 | [3, 4, 5, 6] | [3, 4, 5] | 96 | -0.026855 | 0.200445 | 0.128310 | no |
| 43 | 7 | L5 | [2, 3, 4, 5, 6] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 7 | L6 | [1, 2, 3, 4, 5, 6] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 7 | L7 | [1, 2, 3, 4, 5, 6] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 7 | cumulative | [1, 2, 3, 4, 5, 6] | [2, 3, 4, 5] | 128 | -0.016235 | 0.173591 | 0.112075 | no |
| 43 | 8 | L1 | [7] | [7] | 32 | 0.070312 | 0.347181 | 0.372214 | no |
| 43 | 8 | L2 | [6, 7] | [7] | 32 | 0.070312 | 0.347181 | 0.372214 | no |
| 43 | 8 | L3 | [5, 6, 7] | [5, 7] | 64 | 0.026367 | 0.245494 | 0.226581 | no |
| 43 | 8 | L4 | [4, 5, 6, 7] | [4, 5, 7] | 96 | 0.019531 | 0.200445 | 0.174696 | no |
| 43 | 8 | L5 | [3, 4, 5, 6, 7] | [3, 4, 5, 7] | 128 | -0.002563 | 0.173591 | 0.125747 | no |
| 43 | 8 | L6 | [2, 3, 4, 5, 6, 7] | [2, 3, 4, 5, 7] | 160 | 0.001074 | 0.155264 | 0.111058 | no |
| 43 | 8 | L7 | [1, 2, 3, 4, 5, 6, 7] | [2, 3, 4, 5, 7] | 160 | 0.001074 | 0.155264 | 0.111058 | no |
| 43 | 8 | cumulative | [1, 2, 3, 4, 5, 6, 7] | [2, 3, 4, 5, 7] | 160 | 0.001074 | 0.155264 | 0.111058 | no |

Every row uses only rounds strictly earlier than its decision round. Missing-pilot rounds remain in the calendar window with zero weight. Unequal pilot counts are weighted observation-by-observation.

## Optimistic numerical STOP candidates

| Seed | Pre-round | Window | n_eff | Net-UCB | Max rho+eta allowed | Post-hoc regret | Post-hoc cost avoided |
|---:|---:|---|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — |

These are diagnostics, not policy decisions. Accuracy regret and avoided historical costs are computed only after identifying a numerical candidate and never enter the stopping calculation.

## Drift sensitivity

| Illustrative d | Numerical STOP combinations | Status |
|---:|---:|---|
| 0.000 | 0 (minimum Net-UCB 0.111058) | illustrative, eta=0; not calibrated |
| 0.005 | 0 (minimum Net-UCB 0.124575) | illustrative, eta=0; not calibrated |
| 0.010 | 0 (minimum Net-UCB 0.137075) | illustrative, eta=0; not calibrated |
| 0.020 | 0 (minimum Net-UCB 0.162075) | illustrative, eta=0; not calibrated |

For the stated stability scenario, each observation pays `d × age`; older windows can reduce beta while increasing rho. Therefore adding temporal drift never tightens the total bound relative to the `d=0` diagnostic. Historical maximum drift is reported in CSV/JSON only and is not used to set d.

## Scientific validity

The simultaneous radius would cover the observation-weighted average of historical conditional means under bounded width-one observations plus an appropriate conditional-independence or martingale concentration assumption. It does not automatically cover next-round gain.

A publishable pre-round theorem additionally needs:

1. a filtration and conditional independence/martingale argument across pilot training runs and rounds;
2. a predeclared temporal stability bound relating each historical conditional mean to the next round;
3. a policy-mismatch bound between historical selected batches and the full next V7 policy;
4. predecision proof that the protocol cost-LCB applies to the candidate intervention;
5. simultaneous coverage for the complete window/decision family without selecting an uncovered minimum.

No non-negative combined `rho+eta` allowance is compatible with STOP. The closest case is seed 43 before R8 with L6: even at `rho+eta=0`, Net-UCB is **+0.111058**, so the formal maximum allowance is **-0.111058** (impossible for a non-negative uncertainty term). Its required cost-LCB is **$0.781692**, versus the audited conditional **$0.226400**.

## Paper conclusion

The result is suitable as a negative feasibility/limitations experiment: historical pooling materially reduces statistical radius, but it does not solve the target-shift problem. Without independently justified rho and eta, it does not support a theorem claiming certified Cost-Aware STOP.

## Outputs and provenance

- Full per-window and sensitivity results are stored in JSON/CSV.
- Historical source artifacts were read-only.
- Sealed/future accuracy was not used in any decision calculation.
- `budget_stop` remains distinct from certified Cost-Aware STOP.
