# V7 Training Early Stopping + Verification ON/OFF Ablation

This experiment is a training-only extension of the preserved v7 protocol. It
does not overwrite the original v7 seed-42 artifacts or reinterpret the old
65.0000% HANS result.

## Controlled design

Both arms run the same fixed closed loop:

    Discover -> Verify ON/OFF -> Generate -> synthetic QC -> Select -> Train -> Refresh

The controls held equal are:

- frozen seed-42 initial pool and base checkpoint;
- acquisition budget 128 = 4 rounds x 32 examples;
- v7 selector and all selector weights;
- synthetic generator and QC;
- stateful LoRA and full AdamW moments across rounds;
- seed schedule (42 first, then 43 and 44);
- early-stopping training policy.

The only intended ablation variable is hypothesis verification:

- ON runs the existing target-only adaptive verification and retains certified
  plus unresolved hypotheses;
- OFF creates no verification probes and retains every merged hypothesis.
  Synthetic generation still runs the unchanged schema, label, quality, and
  near-duplicate QC.

There is no adaptive acquisition stopping and no cost-aware stopping. Training
early stopping can shorten epochs inside a round, but every valid completed run
still acquires 32 fresh examples in each of four rounds.

## Training policy

The training microbatch is 4 and gradient accumulation is 2, so the effective
batch size is 8. Each 32-example round therefore makes four AdamW updates per
epoch instead of the old single update.

Each round trains for at most 8 epochs with patience 2, min_delta 0.001, and
best-checkpoint restoration.

After every epoch, the runner computes binary HANS negative log-likelihood on
the predeclared 900-example gain-eval stopping subset. Entailment probability
is the MNLI entailment probability; non-entailment probability is neutral plus
contradiction. This subset is now a model-selection set and must not be reported
as untouched test evidence.

The best epoch restores both:

1. all trainable LoRA tensors; and
2. the complete named AdamW state, including moments and step counters.

The next round must start at the restored best optimizer step. The runner
audits adapter serialization and the optimizer step chain on every round and
again on resume.

The sealed 15,000-example HANS final test, gain-report partition, and MNLI
matched/mismatched validation are not used for epoch early stopping. Sealed
HANS and MNLI are materialized for final evaluation only after all training and
the final hypothesis audit are complete.

## Artifacts and logging

Each run writes to a new root:

    outputs/stage2_runs_v7_training_verification_ablation/
      verification_on/seed_42/
      verification_off/seed_42/
      ...

Important artifacts are:

- rounds.jsonl: selection diagnostics, HANS accuracy, hypothesis states, and
  per-phase resource timing for every round;
- training/round_NNN/training_history.json: train/validation loss, validation
  accuracy, optimizer steps, best epoch, patience state, and restore metadata;
- refresh/round_NNN/lifecycle.json: discovery and verification/bypass states;
- result.json: final HANS, MNLI retention, learning curve, continuity audit,
  resource summary, and sealed-test policy;
- reports/stage2_v7_training_verification_ablation/REPORT.md: available-seed
  results and mean +/- sample standard deviation.

Teacher input/output tokens are summed from canonical teacher-call logs. GPU
fields are measured GPU wall-clock proxies by phase. The frozen initial-pool
cost is deliberately excluded as shared sunk cost and should be reported
separately if total historical pipeline cost is required.

## Commands

Activate the project environment and run seed 42 for both arms first:

    conda activate active_lr
    cd /home/llm/hainn/hainn/active_learning

    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification on --seed 42

    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification off --seed 42

Inspect seed 42 before starting the remaining seeds. Then run matched ON/OFF
pairs for seeds 43 and 44:

    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification on --seed 43
    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification off --seed 43

    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification on --seed 44
    TEACHER_GPU=1 STUDENT_GPU=0 \
    ./run_v7_training_verification_ablation.sh --verification off --seed 44

Resume an interrupted run by repeating its command with --resume. Completed
runs are never overwritten by the launcher.

Regenerate the aggregate report independently with:

    python scripts/report_v7_verification_ablation.py \
      --input-root outputs/stage2_runs_v7_training_verification_ablation \
      --output-dir reports/stage2_v7_training_verification_ablation

## Interpretation boundary

This ablation estimates the effect of the verification gate under the new
early-stopping optimizer policy. It must not be compared to old v7 as if
training were held fixed: old v7 used effective batch 32, one epoch, and one
optimizer update per round. Old v7 remains a historical reference, while the
ON/OFF comparison within this experiment is the controlled causal contrast.
