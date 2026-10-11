"""Uniform random acquisition baseline."""

from __future__ import annotations

import random
from collections.abc import Sequence

from src.baselines.base import Baseline, ProbabilityPredictor, validate_acquisition_request
from src.data.mnli import MNLIExample


class RandomBaseline(Baseline):
    method = "random"

    def acquire(
        self,
        pool: Sequence[MNLIExample],
        budget: int,
        seed: int,
        student: ProbabilityPredictor | None,
        batch_size: int,
    ) -> list[MNLIExample]:
        validate_acquisition_request(pool, budget)
        indices = random.Random(seed).sample(range(len(pool)), budget)
        return [pool[index] for index in indices]
