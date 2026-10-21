"""Published active-learning acquisition functions used in Stage 2."""

from src.baselines.literature.acquire import (
    LITERATURE_STRATEGIES,
    acquire_literature_mnli,
)
from src.baselines.literature.metadata import LITERATURE_METADATA
from src.baselines.literature.synthetic import acquire_literature_synthetic_pairs

__all__ = [
    "LITERATURE_STRATEGIES",
    "LITERATURE_METADATA",
    "acquire_literature_mnli",
    "acquire_literature_synthetic_pairs",
]
