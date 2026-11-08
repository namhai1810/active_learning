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

`unresolved` is retained exactly as in V7. It is never converted to `refuted`
and continues through adaptive verification and acquisition. Its presence does
not suppress a policy-level pilot estimate when the unchanged selector has at
least one certified hypothesis and a valid mixed batch. This does not turn an
unresolved hypothesis into certified evidence and does not support a claim of
the Appendix hypothesis-specific theorem.

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

Thirty-two temporary pilot clones start from the same current LoRA checkpoint
and AdamW state, train on the exact selected batch with the same early-stopping
policy and distinct predeclared training-randomization seeds, and evaluate on
one predeclared fixed 64-example shard for that round. Their
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
3. no certified support for the unchanged mixed policy: CONTINUE due
   insufficient evidence; unresolved hypotheses remain active;
4. fewer than 32 valid pilots: CONTINUE due insufficient pilot evidence;
5. otherwise compute the mixed-policy UCB even when unresolved hypotheses are
   also present, and:
   - CONTINUE when `U_t(pi) > 0`;
   - STOP with `nonpositive_mixed_policy_optimistic_net_gain` when
     `U_t(pi) <= 0`;
6. after round 8: STOP due maximum budget.

These reasons remain distinct in `stop_reason_category`.

## Confidence accounting and data firewall

- Adaptive verification retains V7's two-look `[20, 30]` schedule.
- `verification.max_rounds=9` covers refreshes before rounds 2-8, the final
  audit, and the reserved family used by the inherited accounting.
- Mixed-policy gain uses `K=1`, `T=7`, `delta=.05`, `M=1`, and `m=32`.
- A fully disjoint `32 * 7 * 64` design would require 14,336 gain-eval IDs and
  is impossible with the frozen 2,850-ID pool. The protocol therefore
  prediction-blind partitions `7 * 64 = 448` pilot-eval IDs (one fixed shard
  per round) and leaves 2,402 held-out report IDs.
- Round shards are mutually disjoint and disjoint from the 900 early-stopping
  examples, sealed HANS/MNLI and held-out gain report. The 64 shared eval rows
  within a round are not counted as 32 independent observations.
- The Hoeffding units are the 32 pilot gains over distinct training
  randomizations, conditional on the fixed empirical-risk shard. Their
  conditional independence/unbiasedness is a stated assumption, not a proven
  property.
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

Protocol `v7_m32_marginal_continue_cost_v2` separates:

- `incurred_exploration_cost`: teacher, generation/QC, selector evaluation and
  completed pilot resources. These are sunk at decision time and do not enter
  `U`;
- `decision_cost`: projected main LoRA training and validation resources that
  occur only if the current intervention CONTINUEs;
- `total_pipeline_cost`: actual incurred exploration plus the decision cost
  only when the main update is committed.

The decision is therefore
`U = min(M, Delta_hat + b) - lambda * decision_cost`. The previous m=32 result
that used total round cost remains historical and is not overwritten.

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
b = M * sqrt(2 log(2 K T / delta) / m) = 0.5934427944.
```

The raw radius remains substantial relative to the feasible gain interval.
Intersecting the gain UCB with `M=1` makes its largest possible value exactly 1, so the maximum
break-even cost under `lambda=.2` is USD 5. A lower observed gain UCB has a
proportionally lower break-even cost. This makes gain-cost STOP operationally
possible without weakening the confidence statement, but does not guarantee
that any seed stops before round 8. Insufficient pilot evidence still forces
CONTINUE rather than a fabricated gain-cost STOP. Unresolved hypotheses remain
represented in V7 but do not automatically suppress a complete mixed-policy
estimate.

The conditional independence and unbiasedness of pilot gains remain
unverified: training seeds differ, but pilots share a checkpoint, selected
batch and fixed evaluation shard. The UCB arithmetic is implemented exactly;
the statistical assumption is not claimed as empirically proven.

## Artifacts

Protocol-v2 runs write under
`outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_N/`.
The original eight-round seed-42 trajectory under
`outputs/stage2_runs_v7_minimal_cost_aware/seed_42/` and the completed m=32
pilot cache under `outputs/stage2_runs_v7_minimal_cost_aware_m32/seed_42/`
are read-only replay inputs:

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

Audit the protocol and immutable seed-42 replay source without loading a model:

```bash
python scripts/audit_v7_minimal_cost_aware.py
bash run_v7_minimal_cost_aware_replay.sh --audit-only
```

Replay only the seed-42 round-2 decision from the 32 saved pilots (CPU-only):

```bash
bash run_v7_minimal_cost_aware_replay.sh --decision-only
```

Then continue seed 42 from the compatible historical trajectory; new pilots
are run only for later decisions that have no compatible cache:

```bash
STUDENT_GPU=2 bash run_v7_minimal_cost_aware_replay.sh --resume
```

The replay validates source config, selected IDs and checkpoint/AdamW hashes;
it runs no teacher stages or main V7 training and stops reusing later rounds if
the new decision stops earlier. For fresh seeds, use

```bash
TEACHER_GPU=1 STUDENT_GPU=2 bash run_v7_minimal_cost_aware.sh --seed 43
TEACHER_GPU=1 STUDENT_GPU=2 bash run_v7_minimal_cost_aware.sh --seed 44
```

Add `--resume` to either fresh-seed command after an interruption. Generate an
artifact-only report with:

```bash
python scripts/report_v7_minimal_cost_aware.py
```

