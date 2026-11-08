# V7 next-round Cost-Aware refactor

## Research objective

The only target objective is whether the expected improvement from the next
V7 active-learning round is worth its expected full-round cost:

```text
J_(t+1) = E[Delta_(t+1) | F_t] - lambda E[C_(t+1) | F_t]
```

The safe stopping certificate is:

```text
U_gain,(t+1) - lambda L_cost,(t+1) <= 0
```

Both bounds must use only information in `F_t`; sealed HANS, future rounds and
retrospectively selected thresholds are forbidden.

## Refactor boundary

The refactor does not alter V7 training, acquisition, Adaptive Verification,
Soft-Pair Selection, Stateful LoRA, AdamW continuity or early stopping.
Frozen v2/v2.1 configs and controllers are retained for reproducibility.

The v4 Adaptive Pilot and v5 Risk-Ceiling experiments are logically archived
under `archive/v7_cost_aware_experiments/README.md`. Their files remain at the
original paths so historical commands and reports do not break. Neither is
referenced by the V7 runtime.

## New minimal decision contract

`src/decision/next_round_cost_aware.py` consumes externally constructed:

- expected gain and expected full-round cost for reporting `J`;
- a valid pre-decision next-round Gain-UCB;
- a valid pre-decision full-round cost lower bound;
- the existing `lambda` supplied by the caller.

It contains no estimator and is not wired into the runtime.

| Condition | Action | Certification |
|---|---|---|
| Valid `U_gain - lambda L_cost <= 0` | STOP | `certified_stop` |
| Positive valid Net-UCB | CONTINUE | `uncertified_continue` |
| Missing/invalid/leaky evidence | CONTINUE | `uncertified_continue` |
| Maximum budget reached | STOP | `budget_stop` |

Point estimates never replace confidence bounds. A positive UCB is not called
a certified CONTINUE; it only means STOP has not been established.

## Minimal future integration point

After a round checkpoint is finalized and before any next-round discovery,
verification, generation or training begins:

1. Freeze `F_t` and the current checkpoint/hash.
2. Ask a future estimator for `E[Delta_(t+1)|F_t]` and valid `U_gain` for the
   exact mixed Soft-Pair round action.
3. Ask cost accounting for `E[C_(t+1)|F_t]` and valid `L_cost` covering the
   complete next round: discovery, verification, generation/QC, selection,
   training and required validation.
4. Pass only those values and provenance to `decide_next_round`.
5. Persist the decision before committing any next-round work.

No estimator is implemented in this refactor. In particular, the integration
must not reuse v4 VOI, fixed-4 behavior, v5 risk ceilings, future-round metrics
or sealed final tests.

## File disposition

### Kept unchanged

- `scripts/run_stage2_closed_loop.py` and the V7 training path.
- `src/decision/v7_minimal_cost_aware.py` for frozen v2/v2.1 behavior.
- `configs/stage2_v7_minimal_cost_aware*.yaml` and historical configs.
- Checkpoints, cached pilots, outputs, logs and historical reports.
- Gain, Range-Aware confidence and cost-accounting utilities.

### Logically archived, retained at original path

- v4 Adaptive Pilot module, simulation, tests, docs and results.
- v5 Pre-Round Risk-Ceiling replay, tests, docs and results.

### Added or changed

- Added `src/decision/next_round_cost_aware.py`.
- Exported its estimator-agnostic types from `src/decision/__init__.py`.
- Added contract tests in `tests/test_v7_next_round_cost_aware.py`.
- Added archive notices to v4/v5 code and documentation.

## Safety and compatibility

The new contract is not imported by the V7 runner and no frozen parameter was
changed. Existing V7 behavior and outputs therefore remain unaffected. Runtime
integration requires a separate reviewed change after valid next-round gain
and cost-bound estimators exist.
