# Stage 2 seed-42 comparison with preserved v6 diagnostic

All rows use seed 42 and configured budget 128, except zero-shot, which has no
training budget. The HANS column is evaluated on the same sealed final split.
Random/uncertainty MNLI are cross-source baselines and are not claimed to have
equal data access to HANS-guided synthetic methods.

| Method | HANS accuracy | HANS non-entailment | MNLI matched | MNLI mismatched | Interpretation |
|---|---:|---:|---:|---:|---|
| zero_shot | 0.638533 | 0.337518 | 0.576566 | 0.579434 | Reference checkpoint |
| random_synthetic | 0.644600 | 0.351700 | 0.583902 | 0.585130 | Fixed-budget comparison |
| random_mnli | 0.652600 | 0.376519 | 0.594906 | 0.597742 | Cross-source fixed-budget baseline |
| uncertainty_mnli | 0.669933 | 0.423000 | 0.610392 | 0.615439 | Cross-source fixed-budget baseline |
| failure_driven_no_verification | 0.654667 | 0.383732 | 0.593377 | 0.596318 | Fixed-budget ablation |
| ours_verified_v3_ce | 0.647533 | 0.361026 | 0.587672 | 0.589402 | Fixed-budget comparison |
| ours_verified_v5_ce | 0.648200 | 0.364766 | 0.588079 | 0.589402 | Fixed-budget comparison |
| **ours_verified_v6_closed_loop** | **0.638533** | **0.337518** | **0.576566** | **0.579434** | **Preserved diagnostic; not fixed-budget comparable** |

## Mandatory v6 note

The v6 row must not be deleted. It records the observed output of the completed
run and is retained for auditability and negative-result reporting. It must not
be included when ranking valid fixed-budget methods or estimating the accuracy
of the intended cumulative closed-loop algorithm.

Although the run processed 128 examples across four rounds, checkpoint auditing
showed that the round-4 adapter did not retain the expected cumulative LoRA
updates from rounds 1-3. Its final predictions on all 15,000 sealed HANS examples
were identical to zero-shot. A controlled reconstruction of round 2 from the
round-1 adapter produced a maximum expected update of approximately `1e-5`,
whereas the saved round-1-to-round-2 LoRA-A difference was approximately
`4.42e-2`. Consequently, the v6 row is a diagnostic implementation outcome, not
an equal-budget comparison against checkpoints trained cumulatively on 128
examples.

The verification lifecycle remains informative: one old hypothesis was
refuted, one new hypothesis was discovered, and selector allocation was more
balanced than v5. Those observations should be reported separately from final
accuracy.

The preserved final adapter and integrity manifest are stored at:

`outputs/preserved_checkpoints/ours_verified_v6_closed_loop/seed_42_budget_128_diagnostic_20261019`

