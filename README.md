# Failure-driven active learning — primary LLM method and oracle references

This repository implements MNLI/HANS loading, a Qwen3-1.7B NLI student, HANS
evaluation, and the Base, Random, and entropy-based Uncertainty acquisition
baselines. Random and Uncertainty share one LoRA training recipe. All baseline
runs use a unified local logger with resolved configuration and round metadata.

The primary research path is now distinct from the HANS oracle/reference path:

```text
Qwen HANS predictions -> deterministic five-way split
-> [discover -> verify + LCB -> generate/QC -> pilot gain UCB
    -> full cost-aware STOP/CONTINUE -> fresh LoRA update] x T
-> one final held-out HANS evaluation after the adaptive loop terminates
```

`configs/main_common.yaml` centralizes the explicit teacher choice. Replace its
model ID with an appropriate local path or Hugging Face model ID when needed. A
missing/failed teacher is an error; the code never silently substitutes the
HANS oracle.

## Setup

```bash
conda activate active_lr
python -m pip install -r requirements.txt
```

## Evaluate HANS

```bash
python scripts/evaluate_hans.py --config configs/hans_qwen17b.yaml
```

The run writes `predictions.jsonl` and `metrics.json` beneath the configured
`outputs/<run_name>/` directory. Set `overwrite: true` in the config only when
an existing run should be replaced.

## Test

```bash
python -m pytest -q
```

## Run acquisition baselines

Run the three evaluations independently. The commands below intentionally
include `--overwrite` because they are for a clean re-evaluation of the named
run directories; omit it when you want the existing-artifact guard instead.

```bash
# 1. Unadapted Base checkpoint
python scripts/run_baseline.py \
  --method base \
  --config configs/baseline_base.yaml \
  --overwrite

# 2. Random MNLI acquisition + LoRA
python scripts/run_baseline.py \
  --method random \
  --config configs/baseline_random.yaml \
  --overwrite

# 3. Entropy-based Uncertainty MNLI acquisition + LoRA
python scripts/run_baseline.py \
  --method uncertainty \
  --config configs/baseline_uncertainty.yaml \
  --overwrite
```

Each run saves `config.yaml`, `run.json`, `rounds.jsonl`, `metrics.json`,
`predictions.jsonl`, and `selected_example_ids.json`. Random and Uncertainty
also save their LoRA adapter. Existing non-empty runs are rejected unless the
command includes the explicit `--overwrite` flag.

After all three commands finish, build the validated comparison report and
visualizations with one command:

```bash
python scripts/report_hans_baselines.py \
  --outputs outputs \
  --report-dir reports/hans_baselines_seed42
```

The same workflow is available as a shell script when you want to run each
stage explicitly:

```bash
bash run_hans_baselines_individual.sh base
bash run_hans_baselines_individual.sh random
bash run_hans_baselines_individual.sh uncertainty
bash run_hans_baselines_individual.sh report
```

Or run the complete sequence:

```bash
bash run_hans_baselines_individual.sh all
```

Set `CUDA_DEVICE=<index>` before the command to select another physical GPU.

The reporter requires all three full HANS artifacts, verifies that they contain
the same example IDs, recomputes metrics from predictions, and refuses partial
or inconsistent inputs. It creates:

- `reports/hans_baselines_seed42/HANS_BASELINE_REPORT.md`;
- `data/run_summary.csv`, `data/group_metrics_long.csv`, and `data/results.json`;
- `figures/overall_accuracy.svg`, `label_heuristic_accuracy.svg`, and
  `subcase_accuracy_heatmap.svg`.

To additionally compare the separately generated `hans_qwen3_1_7b_base`
zero-shot artifact, append `--include-standalone-zero-shot`. For a quick console
table only, `python scripts/summarize_runs.py --outputs outputs` remains
available.

For the full CUDA 0 benchmark suite (zero-shot plus Base, Random, and
Uncertainty), activate `active_lr` and run the non-overwriting orchestrator:

```bash
bash run_hans_baselines_cuda0.sh
```

To resume only selected stages, pass their names explicitly. For example:

```bash
bash run_hans_baselines_cuda0.sh random uncertainty
```

It saves per-stage logs under `logs/hans_baselines_seed42/` and builds a
paper-ready Markdown report, tidy CSV/JSON exports, and SVG vector figures under
`reports/hans_baselines_seed42/`. Completed stages are safely skipped on rerun;
partial run directories cause an explicit failure and are never overwritten.

## Run the primary LLM-based method

Start from an existing HANS evaluation directory containing the unsplit
`predictions.jsonl`. All primary scripts reconstruct and validate the same
12.5/12.5/12.5/12.5/50 split. The compatible main manifest is
`hans_split_v2.json`.

For the proposed multi-round method, configure the teacher and cost rates in
`configs/main_multiround.yaml`, then run:

```bash
python scripts/run_main_active_learning.py \
  --config configs/main_multiround.yaml \
  --run-dir outputs/<existing_run>
```

Every attempted round is saved under `active_learning/rounds/round_NNN/` and
executes discovery, verification, targeted generation/QC, pilot estimation, and
a frozen STOP/CONTINUE decision. CONTINUE trains from that round's starting
checkpoint, and the saved adapter becomes the next round's checkpoint. The loop
stops when the eligible maximum of
`U(h) = delta_hat(h) + radius(h) - lambda_cost*C(h)` is non-positive, no
candidate is eligible, the next candidate exceeds the remaining budget, the
budget is consumed, or `max_rounds` is reached. `rounds.jsonl` and
`summary.json` record checkpoint transitions and cumulative modeled cost.

Final-test rows are reduced to ID-only placeholders in all adaptive round
artifacts. Their text, labels, and previous predictions are materialized only
after termination, when the final checkpoint is evaluated once. Gain-evaluation
IDs are deterministically partitioned into disjoint round shards. Discovery
data may be revisited to diagnose an updated student, but it is never used as
gain evaluation or final test data.

The individual stage commands remain available for compatible one-round runs
and debugging. After setting the same teacher configuration in each inherited
main config, run them in order:

```bash
python scripts/build_hypotheses.py \
  --config configs/main_hypothesis_discovery.yaml \
  --run-dir outputs/<existing_run>

python scripts/verify_hypotheses.py \
  --config configs/main_verification.yaml \
  --run-dir outputs/<existing_run>

python scripts/generate_synthetic_interventions.py \
  --config configs/main_synthetic_generation.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_pilot_gain.py \
  --config configs/main_pilot_gain.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_decision.py \
  --config configs/main_cost_aware.yaml \
  --run-dir outputs/<existing_run>
```

An explicit teacher-critic first estimates
`P(student prediction is wrong | premise, hypothesis, prediction, confidence)`.
The critic never receives gold labels, correctness, HANS taxonomy fields, or
non-discovery records. Discovery then uses configured normalization,
logistic/exponential weighting, or threshold sampling; challenging correct
false alarms are the default controls. The hypothesis prompt still receives
only approved fields and never receives HANS `heuristic` or `subcase`.

Verification records are context only. The teacher generates new fixed sets of
one target, one counterfactual/positive control, and one invariant control. A
separate validation call solves their labels and checks their constraints before
the student is loaded. The student is queried once on the frozen accepted set.
Each binary outcome is one only when the target exhibits the predeclared failure
and both controls are correct. Certification additionally requires enough valid
sets and a union-adjusted Hoeffding LCB; its iid interpretation remains
conditional because generated scenarios cannot prove statistical independence.

Synthetic generation uses only `intervention_pool` seeds and emits new complete
counterfactual and invariant pairs (plus optional reasoning contrasts), with
stable `pair_id`, `pair_type`, and member roles. Every member has an explicit
`entailment`, `neutral`, or `contradiction` label; no binary HANS non-entailment
label is converted automatically. A second teacher QC pass and deterministic
checks reject entire pairs for inconsistent labels/semantics, ambiguity, poor
grammar, irrelevance, weak diversity, duplicates, near duplicates, or overlap
with any permitted pre-final split or generated verification probe. `final_test`
is not materialized at this stage. Other held-out/probe text is used only for a
post-generation local leakage veto and never enters a teacher generation prompt.

Pilot synthetic pairs are disjoint across pilots and are never split across
pilot/full selections. Training remains the existing per-example MNLI
cross-entropy; no unsupported contrastive loss is implied. All pilot models begin
from the same checkpoint, and pilot examples are excluded from the full-data
reserve. `run_main_decision.py` freezes the utility before any full
intervention. STOP performs no training. In standalone mode, CONTINUE runs one
fresh full LoRA and only then evaluates `final_test`. The multi-round controller
uses `--defer-final-test`, records `next_checkpoint.json`, and promotes that
adapter only to the next round.

### Full cost model

The primary method uses `cost.type: full_pipeline`. `C(h)` is a scalar in the
configured currency with a saved component ledger covering:

- measured teacher-LLM input and output tokens;
- automated verification and synthetic generation/QC items;
- measured pilot plus projected full LoRA runs/GPU time; and
- pilot/full evaluation runs, items, and projected GPU time.

The proposed pipeline is fully automated. Its cost schema contains no manual
annotation or review fields, rates, projections, or usage quantities.

Rates live under `cost.rates`; pre-decision projections for resources not yet
committed live under `cost.projections`. Candidate artifacts store quantities,
rates, subtotals, currency, and the total used in the utility. Teacher/pilot
usage is measured, whereas full-intervention GPU time must be projected to keep
the decision frozen. Rates and projections must be fixed without inspecting
final-test or post-intervention outcomes. The historical
`validated_synthetic_examples` mode remains accepted by the standalone script
for reproduction, and oracle debug/baseline paths retain their prior costs.

Teacher request/response envelopes, prompt versions, deterministic seeds,
selected IDs, QC rejections, gain measurements, and decision artifacts are
saved beneath the run directory. Local Hugging Face teacher logs also include
measured input/output token counts and generation time.

The configured `K*T` factor union-adjusts verification and gain intervals over
the predeclared maximum hypotheses per round and maximum rounds. This supports
adaptive selection only if every constituent interval remains valid conditional
on all earlier-round history: observations must be bounded, fresh round outcomes
must satisfy the required conditional independence (or another valid
conditional concentration argument), the conditional target mean must be well
defined, and no current outcome may influence its own probe, applicability, or
sample selection. Disjoint gain shards, disjoint pilot batches, pre-outcome
probe freezing, and round-specific seeds reduce reuse and feedback; they do not
prove these assumptions. Generated scenarios and related LoRA runs may remain
dependent. The intervals, decisions, and stopping rule are therefore
conditional diagnostics, not unconditional, causal, or profitability
guarantees.

## Stage 2 fair benchmark

Stage 2 is deliberately separate from cost-aware stopping. Experiment A fixes
the final-training data budget; Experiment B fixes four acquisitions of 32 new
examples. All methods use the same frozen initial checkpoint, LoRA recipe,
optimizer settings, HANS manifest, and MNLI evaluation implementation.

Prepare the seed-independent HANS split once, before any HANS inference:

```bash
python scripts/prepare_stage2_benchmark.py \
  --config configs/stage2_benchmark.yaml \
  --output-dir outputs/stage2_shared
```

Build the three synthetic acquisition pools for seeds 42, 43, and 44. These
commands use the same intervention seed role, teacher, generation/QC code, and a
global limit of 320 generated examples per method. The no-verification policy
uses discovered hypotheses without certification; the random policy uses a
non-failure-targeted prompt and does not inspect discovered failures.

```bash
for seed in 42 43 44; do
  python scripts/build_stage2_synthetic_pool.py \
    --benchmark-config configs/stage2_benchmark.yaml \
    --method random_synthetic --seed "$seed" \
    --output-dir "outputs/stage2_acquisition/random_synthetic/seed_${seed}"

  python scripts/build_stage2_synthetic_pool.py \
    --benchmark-config configs/stage2_benchmark.yaml \
    --method failure_driven_no_verification --seed "$seed" \
    --output-dir "outputs/stage2_acquisition/failure_driven_no_verification/seed_${seed}"

  python scripts/build_stage2_synthetic_pool.py \
    --benchmark-config configs/stage2_benchmark.yaml \
    --method ours_verified --seed "$seed" \
    --output-dir "outputs/stage2_acquisition/ours_verified/seed_${seed}"
done
```

Experiment A — one fixed-budget update at 32, 64, and 128 examples:

```bash
./run_experiment_A.sh
```

`run_experiment_A.sh` validates the frozen artifacts, builds the nine required
synthetic pools sequentially, runs all 54 method/budget/seed cells, and produces
the combined report plus three SVG visualizations under
`reports/stage2_experiment_A/`. Completed runs are skipped on restart and partial
directories are archived under `logs/stage2_experiment_A/partial_runs/` rather
than deleted. The terminal shows `[POOL x/9]`, `[RUN x/54]`, elapsed time, and
live `tqdm` bars for teacher generation, verification, QC, inference, and LoRA
training. To rebuild only the report after all 54 runs are complete:

```bash
./run_experiment_A.sh --report-only
```

Experiment B — four rounds × 32 fresh examples, total 128:

```bash
python scripts/run_stage2_suite.py \
  --config configs/stage2_benchmark.yaml \
  --experiment B \
  --output-root outputs/stage2_runs
```

Build the report without inventing missing results:

```bash
python scripts/report_stage2_benchmark.py \
  --runs-root outputs/stage2_runs \
  --output-dir reports/stage2
```

The report contains seed-level results plus mean, sample standard deviation, and
a normal-approximation 95% interval. With three seeds, that interval is only a
descriptive uncertainty summary. Accuracy curves use the independent
`gain_eval` role and are evaluated post hoc after all acquisition/training
decisions. The sealed final-test is opened exactly once per run for the frozen
final checkpoint, so no intermediate final-test metric can influence acquisition.

### Stage 2 fairness checklist

- One manifest hash and identical final-test IDs for every method and seed.
- Final-test inputs/labels are absent from adaptive artifacts; overlap auditing is
  post-selection only and never acts as a training-data veto.
- Exact budgets are 32/64/128 individual examples; synthetic budgets are even and
  preserve complete pairs.
- Missing valid synthetic examples produce an explicit shortage and are never
  filled with unrelated data.
- Random and Uncertainty acquire fresh, previously unseen MNLI examples after
  each checkpoint update.
- MNLI baselines are labeled cross-source; equal data access with HANS-guided
  methods is not claimed.
- All trained methods share the same LoRA architecture, optimizer settings, and
  supervised token cross-entropy term. `ours_verified` additionally uses the
  complete-pair structure during training: counterfactual pairs receive a
  contrastive Jensen-Shannon margin loss and invariant pairs receive a
  Jensen-Shannon consistency loss. Random-synthetic and no-verification retain
  per-example cross-entropy so this objective difference is explicit in every
  round artifact and final result.
- Synthetic labels must be explicit entailment/neutral/contradiction labels; no
  binary HANS non-entailment conversion is permitted.
- Fixed-budget acquisition is reported separately from cost-aware stopping.
- Raw measured quantities and assumed USD rates are saved separately.

Remaining limitations include teacher/QC dependence, lexical rather than
semantic near-duplicate auditing, only three experimental seeds, configured cost
rates and projected compute, and filesystem separation rather than an external
access-control service for sealed final-test data.

## Build debug-only HANS oracle hypotheses

The remaining `oracle_hans_*` configs and oracle scripts are debug/reference
experiments, not the primary method. This command reuses an existing run's predictions and deterministically splits
their stable IDs into hypothesis-discovery, verification, intervention-pool,
gain-evaluation, and final-test roles. Only discovery records and metadata are
passed to the oracle generator.

```bash
python scripts/build_oracle_hypotheses.py \
  --config configs/oracle_hans_hypotheses.yaml \
  --run-dir outputs/<existing_run>
```

It adds `hypotheses.json`, `hypothesis_evidence.json`, and
`hans_oracle_split_v2.json` without modifying the older split manifest or any
Milestone 3 artifacts. These are
debug-only candidate hypotheses, not verified or certified failures.

## Verify debug-only HANS oracle hypotheses

```bash
python scripts/verify_oracle_hypotheses.py \
  --config configs/oracle_hans_verification.yaml \
  --run-dir outputs/<existing_run>
```

The verification script reconstructs and validates the versioned five-way split, then
selects fixed non-entailment challenge probes from the verification role before
examining predictions. It writes `verification/probes.jsonl` and
`verification/results.json`. Certification means only that the observed
failure behavior passes the configured reproducibility threshold; it is not a
causal claim and says nothing about intervention value.

## Run targeted pilot gain estimates (oracle debug only)

Milestone 6 assigns HANS examples to five pairwise-disjoint roles:
`hypothesis_discovery`, `verification`, `intervention_pool`, `gain_eval`, and
`final_test`. The manifest is saved as `hans_oracle_split_v2.json`; its
`final_test` IDs are preserved from the earlier 50/50 boundary.

Regenerate the oracle and verification artifacts with their existing commands,
then run only certified hypotheses:

```bash
python scripts/run_pilot_gain.py \
  --config configs/oracle_hans_pilot_gain.yaml \
  --run-dir outputs/<verified_run>
```

Each pilot reloads exactly the same starting Qwen checkpoint, trains a fresh
LoRA intervention on matching `intervention_pool` records, and evaluates risk
before and after on matching `gain_eval` records. It writes
`gain/pilot_observations.jsonl`, `gain/gain_estimates.json`, and isolated pilot
adapters under `gain/pilots/`. The estimate is `mean(risk_before-risk_after)`;
negative values are valid and mean the intervention hurt performance.

The confidence radius assumes independent bounded pilot gains. Fixed HANS data,
related LoRA runs, and unavoidable gain-eval overlap can violate that assumption,
so these oracle-debug intervals are not causal guarantees. Milestone 6 does not
use `final_test` and does not implement UCB or cost-aware stopping.

## Make one cost-aware decision (oracle debug only)

Milestone 7 joins only certified hypotheses to valid pilot-gain estimates. Its
explicit cost is the actual number of matching `intervention_pool` examples
that would be acquired, up to `full_intervention_size`. For each eligible
hypothesis it computes:

```text
J_hat(h) = Delta_hat(h) - lambda_cost * C(h)
U(h)     = Delta_hat(h) + b(h) - lambda_cost * C(h)
```

The command selects `argmax U(h)` and continues only when that maximum is
strictly positive:

```bash
python scripts/run_cost_aware_decision.py \
  --config configs/oracle_hans_cost_aware.yaml \
  --run-dir outputs/<milestone_6_run>
```

It writes `decision/candidates.json` and `decision/decision.json`. A STOP result
does not load the model, acquire examples, or train. A CONTINUE result performs
exactly one fresh full intervention from the same current checkpoint used for
the pilots, saves its isolated adapter under
`decision/full_intervention/<hypothesis_id>/adapter/`, and reports post-decision
risk on `gain_eval` in `decision/post_intervention_metrics.json`. Pilot adapters
are never promoted, and `final_test` is not evaluated.

`lambda_cost` is fixed in the config before selection and must not be tuned on
`final_test` or intervention outcomes. A positive optimistic value does not
guarantee profitability: CONTINUE means only that positive net value cannot yet
be ruled out. Any stopping interpretation also depends on the verification and
gain concentration assumptions, bounded pilot gains, and a correctly specified
cost; this implementation does not itself prove those assumptions or a theorem.

## Run controlled ablations (oracle debug only)

Milestone 8 exposes seven one-round variants through one shared strategy layer
and one runner. All variants read the same checkpoint, five-way split,
hypotheses, verification records, pilot-gain artifacts, intervention budget,
seeds, LoRA recipe, and evaluator. Only the named component changes:

- `full`: LCB verification, gain UCB, acquisition cost, and stopping.
- `no_verification`: accepts every candidate hypothesis.
- `point_verification`: accepts when `nu_hat >= tau_v`.
- `no_gain_ucb`: ranks/stops with `Delta_hat - lambda_cost*C`.
- `no_cost`: ranks/stops with `Delta_hat + b`.
- `random_hypothesis`: seeded uniform selection among accepted hypotheses with
  valid gain estimates; the Full stopping test remains unchanged.
- `fixed_budget`: selects the Full argmax but does not stop merely because its
  optimistic value is non-positive.

Run one variant at a time—the command never launches the matrix automatically:

```bash
python scripts/run_ablation.py \
  --variant full \
  --config configs/oracle_hans_ablation.yaml \
  --run-dir outputs/<milestone_6_run>
```

Replace `full` with `no_verification`, `point_verification`, `no_gain_ucb`,
`no_cost`, `random_hypothesis`, or `fixed_budget` for the other controlled
runs. Each variant writes separately beneath `ablations/<variant>/`; after each
run, `ablations/summary.json` is rebuilt and a compact comparison table is
printed. Existing variant outputs require an explicit `--overwrite` to replace.

The comparison uses `gain_eval` only. `final_test` prediction fields are not
materialized into the ablation strategy or intervention, and no variant is
selected or tuned from final-test performance. Hypotheses without an existing
valid pilot-gain estimate remain in the accepted count but cannot enter the
decision set; obtain comparable pilot estimates before interpreting verification
ablations on a real experiment.

## Validate theoretical assumptions offline (oracle debug only)

Milestone 9 is evaluation-only. It freezes an already-saved Milestone 7
`decision.json` before reading any offline oracle data, then evaluates:

1. verification certification behavior across probe counts;
2. empirical gain-UCB coverage across pilot counts; and
3. when the online action was STOP, the best offline estimated one-step net
   value among robustly valid hypotheses in the existing candidate set.

The full configuration intentionally requires a separate HANS-style JSONL file:

```yaml
reference_data:
  file: /path/to/disjoint_reference.jsonl
  use_precomputed_predictions: false
```

Every row must have a globally disjoint ID and one of two roles:
`verification_reference` or `gain_reference`. The former supplies the large
offline validity reference; the latter is used only to evaluate independent
reference interventions. IDs overlapping any online role are rejected.

After providing that file, the expensive full command is:

```bash
python scripts/run_theory_validation.py \
  --config configs/oracle_hans_theory_validation.yaml \
  --run-dir outputs/<completed_online_run>
```

This full configuration may perform up to 16 online-analysis pilots plus 32
offline reference repetitions per relevant hypothesis, each starting from the
same original checkpoint. It is intentionally not launched automatically.

The small precomputed smoke command is:

```bash
python scripts/run_theory_validation.py \
  --config configs/oracle_hans_theory_validation_smoke.yaml \
  --run-dir outputs/oracle_verification_synthetic_smoke
```

Outputs live under `theory_validation/{verification,gain_ucb,stopping}/` with a
combined `theory_validation/summary.json`. `final_test` is never materialized
into these analyses. The verification reference is an empirical label, the gain
reference approximates rather than equals true gain, and observed coverage or a
passing stop audit supports assumptions only—it does not prove a theorem or
rule out unseen hypotheses, multi-step repairs, or hypothesis interactions.
