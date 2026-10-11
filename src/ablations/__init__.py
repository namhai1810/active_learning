"""Controlled one-round ablation strategies."""

from src.ablations.runner import ABLATION_VARIANTS, evaluate_ablation
from src.ablations.types import AblationOutcome

__all__ = ("ABLATION_VARIANTS", "AblationOutcome", "evaluate_ablation")
