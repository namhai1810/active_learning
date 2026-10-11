"""Failure-hypothesis representations for main and oracle-debug methods."""

from src.hypotheses.llm_generator import LLMFailureHypothesisGenerator
from src.hypotheses.oracle_hans import HansOracleHypothesisGenerator
from src.hypotheses.types import FailureHypothesis

__all__ = [
    "FailureHypothesis",
    "HansOracleHypothesisGenerator",
    "LLMFailureHypothesisGenerator",
]
