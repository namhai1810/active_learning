# Failure-driven active learning — Milestone 1

This repository currently implements only the first milestone for MNLI/HANS:
dataset loading, a zero-shot Qwen3-1.7B NLI student, HANS evaluation, local
prediction/metric artifacts, and unit tests.

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
