"""Common baseline interface and no-acquisition baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from src.data.mnli import MNLIExample


class ProbabilityPredictor(Protocol):
    def predict_proba(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int,
    ) -> list[list[float]]: ...


def validate_acquisition_request(
    pool: Sequence[MNLIExample], budget: int
) -> None:
    if budget <= 0:
        raise ValueError("acquisition_budget must be positive")
    if budget > len(pool):
        raise ValueError(
            f"acquisition_budget={budget} exceeds pool size={len(pool)}"
        )
    ids = [example.id for example in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("The acquisition pool must contain unique example IDs")


class Baseline(ABC):
    method: str
    trains_after_acquisition: bool = True

    @abstractmethod
    def acquire(
        self,
        pool: Sequence[MNLIExample],
        budget: int,
        seed: int,
        student: ProbabilityPredictor | None,
        batch_size: int,
    ) -> list[MNLIExample]:
        """Select examples from the acquisition pool."""


class BaseBaseline(Baseline):
    method = "base"
    trains_after_acquisition = False

    def acquire(
        self,
        pool: Sequence[MNLIExample],
        budget: int,
        seed: int,
        student: ProbabilityPredictor | None,
        batch_size: int,
    ) -> list[MNLIExample]:
        if budget <= 0:
            raise ValueError("acquisition_budget must be positive")
        return []
