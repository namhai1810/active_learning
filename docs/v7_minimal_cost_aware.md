# V7 Minimal Cost-Aware Safe-Stopping Extension

## Scope

This branch starts from the stable V7 commit `bd08fe8`. The historical V7
configs, outputs, selector, loss, optimizer continuity, early stopping, and
checkpoint restoration are not replaced. Cost-Aware behavior is enabled only
by `configs/stage2_v7_minimal_cost_aware.yaml` and writes to a separate output
root.

## Preserved V7 flow

```text
Round 1
  frozen initial pool -> unchanged soft-pair selection -> 32 examples
  -> unchanged stateful LoRA + AdamW + validation-loss early stopping

Rounds 2-8
  Discover -> target-only adaptive Verify -> retain certified + unresolved
  -> Generate + QC -> unchanged soft-pair selection of 32 examples
  -> minimal Cost-Aware evidence/decision gate
  -> unchanged stateful LoRA + AdamW + validation-loss early stopping
  -> Refresh
```

`unresolved` is retained exactly as in V7. It is never converted to `refuted`.
If any hypothesis remains unresolved, the Cost-Aware gate returns CONTINUE with
reason `unresolved_hypotheses_require_more_evidence`; it does not run pilots or
claim a gain-cost STOP.

Legacy `configs/stage2_v7_early_stopping_verification_on.yaml` contains no
`cost_aware` section and follows the original fixed 4x32 path. Regression tests
lock its budget, selector weights, and training config.

## Mathematical integration boundary

The appendix defines a hypothesis-specific action and

```text
U_t(h) = Delta_hat_t(h) + b_t(h) - lambda C_t(h).
```

The proven V7 selector does not choose one `h`; it greedily constructs one
mixed-hypothesis batch. Replacing it with a per-hypothesis selector would break
the highest-priority preservation requirement. Therefore the minimal
controller defines one action:

```text
pi_t = train on the exact 32-example batch returned by the unchanged V7 selector
G_UCB(pi) = min(M, Delta_hat_t(pi) + b_t(pi))
U_t(pi) = G_UCB(pi) - lambda C_t(pi).
```

The intersection with `M` does not shrink or replace the Hoeffding radius.
Because pilot gain is defined as a difference of risks in `[0,1]`, its mean is
known a priori to satisfy `Delta <= M=1`. Therefore the minimum of the
Hoeffding upper bound and `M` remains a valid upper confidence bound. Both raw
and support-bounded UCBs are logged.

Four temporary pilot clones start from the same current LoRA checkpoint and
AdamW state, train on the exact selected batch with the same early-stopping
policy, and evaluate on four disjoint predeclared 64-example shards. Their
updates are never committed. The main update uses the original V7 seed and is
performed only after CONTINUE.

This is a policy-level bounded-gain UCB, not Theorem 1's guarantee over
robustly verified hypothesis-specific interventions. Every decision and final
result explicitly sets `appendix_theorem_claimed=false`. The code does not
silently reinterpret the mixed selector as a targeted intervention.

## Decision rules

Round 1 is always CONTINUE and has no Cost-Aware decision. From round 2:

1. no active hypotheses: unchanged V7 operational STOP;
2. insufficient synthetic data: unchanged V7 operational STOP;
3. any unresolved hypothesis: CONTINUE for more evidence, no pilots;
4. fewer than four valid pilots: CONTINUE due insufficient pilot evidence;
5. otherwise compute the mixed-policy UCB and:
   - CONTINUE when `U_t(pi) > 0`;
   - STOP with `nonpositive_mixed_policy_optimistic_net_gain` when
     `U_t(pi) <= 0`;
6. after round 8: STOP due maximum budget.

These reasons remain distinct in `stop_reason_category`.

## Confidence accounting and data firewall

- Adaptive verification retains V7's two-look `[20, 30]` schedule.
- `verification.max_rounds=9` covers refreshes before rounds 2-8, the final
  audit, and the reserved family used by the inherited accounting.
- Mixed-policy gain uses `K=1`, `T=7`, `delta=.05`, `M=1`, and `m=4`.
- The 2,850 old gain-report IDs are prediction-blind partitioned before model
  outcomes: `7 * 4 * 64 = 1,792` pilot-eval IDs and 1,058 held-out report IDs.
- Pilot shards are pairwise disjoint and disjoint from the 900 early-stopping
  examples.
- Sealed HANS and MNLI are materialized only after the adaptive loop freezes.

## Cost model and ex-ante scale calibration

The historical USD rates and projections from `configs/main_cost_aware.yaml`
are retained, but the old `lambda=.001` is not: it placed the maximum possible
gain and ordinary round costs on incompatible scales. Lambda is now fixed by a
declared utility assumption made without seed/model/final-test outcomes:

```text
reference gain = 1.0 (complete elimination of binary validation risk)
maximum willingness to pay = USD 5.0
lambda = 1.0 / 5.0 = 0.2 gain units per USD
```

The config loader rejects an enabled run unless `lambda` exactly matches these
two calibration fields. This is a scientific preference/utility assumption,
not an empirically estimated parameter. Changing it would define a different
protocol and requires a new preregistered experiment rather than tuning a run.

The cost ledger includes:

- incurred teacher, generation/QC, selector-evaluation, and pilot resources;
- projected main training and validation resources known before the decision.

Pilot wall-clock proxies are partitioned into disjoint training and evaluation
components. Evaluation accounting includes the pre/post gain measurements and
every epoch-level early-stopping validation; those seconds are not also billed
as LoRA training time.

Round 1 and the frozen initial-pool sunk cost are logged separately from the
round-2+ decision-cost sum. Cost rates are assumptions, not market-price
claims.

## Important limitation: STOP is possible, not guaranteed

With the frozen values,

```text
b = M * sqrt(2 log(2 K T / delta) / m) = 1.6785096966.
```

The raw radius remains wider than the feasible gain interval. Intersecting the
gain UCB with `M=1` makes its largest possible value exactly 1, so the maximum
break-even cost under `lambda=.2` is USD 5. A lower observed gain UCB has a
proportionally lower break-even cost. This makes gain-cost STOP operationally
possible without weakening the confidence statement, but does not guarantee
that any seed stops before round 8. Unresolved hypotheses and insufficient
pilot evidence still force CONTINUE rather than a fabricated gain-cost STOP.

The conditional independence and unbiasedness of pilot gains remain
unverified: pilot shards and training seeds are disjoint, but the runs share a
checkpoint and selected batch. The UCB arithmetic is implemented exactly; the
statistical assumption is not claimed as empirically proven.

## Artifacts

Each seed writes under `outputs/stage2_runs_v7_minimal_cost_aware/seed_N/`:

- `cost_aware/pilot_eval_plan.json`;
- `cost_aware/round_NNN/pilots/*/observation.json`;
- `cost_aware/round_NNN/{gain_estimate,cost,decision}.json`;
- `rounds.jsonl` with states, selection, training, resources, and decisions;
- `cost_aware_stop.json` for a gain-cost STOP;
- `hans_prediction_rule_metrics.json` with Three-way and Direct Binary HANS;
- `result.json`, `V7_MINIMAL_COST_AWARE.md`, and `.complete`.

Final evaluation uses one frozen probability pass to report both
`argmax(P(E),P(N),P(C)) -> binary` and
`argmax(P(E),P(N)+P(C))`. MNLI matched/mismatched, budget, cost, and stop reason
are included in the artifact-only three-seed report. Sealed final-test outputs
never enter STOP/CONTINUE, early stopping, or lambda calibration.

Completed pilot observations are atomic resume boundaries. Resume validates
checkpoint, selected IDs, eval IDs, and training policy before reuse. Partial
pilot directories are rerun; completed V7 rounds continue to use the original
checkpoint/optimizer continuity checks.

## CPU validation

```bash
python scripts/dry_run_v7_minimal_cost_aware.py
python -m pytest -q tests/test_v7_minimal_cost_aware.py
python -m pytest -q
```

These commands do not load a teacher/student model or use a GPU.

## Experiment commands

Check GPUs:

```bash
nvidia-smi
```

Run one seed:

```bash
TEACHER_GPU=1 STUDENT_GPU=2 \
bash run_v7_minimal_cost_aware.sh --seed 42
```

Resume:

```bash
TEACHER_GPU=1 STUDENT_GPU=2 \
bash run_v7_minimal_cost_aware.sh --seed 42 --resume
```

After inspecting seed 42, repeat with `--seed 43` and `--seed 44`. Generate an
artifact-only report with:

```bash
python scripts/report_v7_minimal_cost_aware.py
```

