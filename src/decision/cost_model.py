"""Configurable, auditable cost accounting for active-learning interventions.

The decision rule consumes one scalar cost, but retaining the component ledger is
essential: it makes units explicit and prevents a synthetic-example count from
silently standing in for annotation, inference, training, and evaluation costs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_RATE_FIELDS = (
    "human_annotation_per_item",
    "human_review_per_item",
    "llm_input_per_1k_tokens",
    "llm_output_per_1k_tokens",
    "data_generation_per_item",
    "data_verification_per_item",
    "lora_training_per_run",
    "lora_training_per_gpu_hour",
    "evaluation_per_run",
    "evaluation_per_item",
    "evaluation_per_gpu_hour",
)


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class CostRates:
    """Prices in one configured currency/unit for every supported resource."""

    human_annotation_per_item: float = 0.0
    human_review_per_item: float = 0.0
    llm_input_per_1k_tokens: float = 0.0
    llm_output_per_1k_tokens: float = 0.0
    data_generation_per_item: float = 0.0
    data_verification_per_item: float = 0.0
    lora_training_per_run: float = 0.0
    lora_training_per_gpu_hour: float = 0.0
    evaluation_per_run: float = 0.0
    evaluation_per_item: float = 0.0
    evaluation_per_gpu_hour: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CostRates":
        unknown = sorted(set(value) - set(_RATE_FIELDS))
        if unknown:
            raise ValueError(f"Unknown cost rate fields: {unknown}")
        return cls(
            **{
                field: _nonnegative(value.get(field, 0.0), f"cost.rates.{field}")
                for field in _RATE_FIELDS
            }
        )


@dataclass(frozen=True)
class CostUsage:
    """Resource quantities attributable to one candidate intervention."""

    human_annotation_items: float = 0.0
    human_review_items: float = 0.0
    llm_input_tokens: float = 0.0
    llm_output_tokens: float = 0.0
    generated_items: float = 0.0
    verified_items: float = 0.0
    lora_training_runs: float = 0.0
    lora_training_gpu_seconds: float = 0.0
    evaluation_runs: float = 0.0
    evaluation_items: float = 0.0
    evaluation_gpu_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CostUsage":
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - fields)
        if unknown:
            raise ValueError(f"Unknown cost usage fields: {unknown}")
        return cls(
            **{
                field: _nonnegative(value.get(field, 0.0), f"cost.usage.{field}")
                for field in fields
            }
        )

    def plus(self, other: "CostUsage") -> "CostUsage":
        return CostUsage(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class CostBreakdown:
    currency: str
    components: dict[str, float]
    total: float
    usage: CostUsage
    rates: CostRates

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "components": dict(self.components),
            "total": self.total,
            "usage": asdict(self.usage),
            "rates": asdict(self.rates),
        }


def compute_cost(
    usage: CostUsage,
    rates: CostRates,
    *,
    currency: str = "cost_units",
) -> CostBreakdown:
    """Convert a resource ledger into the scalar ``C(h)`` used by selection."""
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("cost.currency must be a non-empty string")
    checked_usage = CostUsage.from_mapping(asdict(usage))
    checked_rates = CostRates.from_mapping(asdict(rates))
    components = {
        "human_annotation": checked_usage.human_annotation_items
        * checked_rates.human_annotation_per_item,
        "human_review": checked_usage.human_review_items
        * checked_rates.human_review_per_item,
        "llm_inference_input": checked_usage.llm_input_tokens
        / 1000.0
        * checked_rates.llm_input_per_1k_tokens,
        "llm_inference_output": checked_usage.llm_output_tokens
        / 1000.0
        * checked_rates.llm_output_per_1k_tokens,
        "data_generation": checked_usage.generated_items
        * checked_rates.data_generation_per_item,
        "data_verification": checked_usage.verified_items
        * checked_rates.data_verification_per_item,
        "lora_training": (
            checked_usage.lora_training_runs * checked_rates.lora_training_per_run
            + checked_usage.lora_training_gpu_seconds
            / 3600.0
            * checked_rates.lora_training_per_gpu_hour
        ),
        "evaluation": (
            checked_usage.evaluation_runs * checked_rates.evaluation_per_run
            + checked_usage.evaluation_items * checked_rates.evaluation_per_item
            + checked_usage.evaluation_gpu_seconds
            / 3600.0
            * checked_rates.evaluation_per_gpu_hour
        ),
    }
    total = math.fsum(components.values())
    return CostBreakdown(
        currency=currency.strip(),
        components=components,
        total=total,
        usage=checked_usage,
        rates=checked_rates,
    )


def cost_rates_from_config(cost_config: Mapping[str, Any]) -> tuple[CostRates, str]:
    """Parse the ``cost.rates`` section used by the full-pipeline cost model."""
    if cost_config.get("type") != "full_pipeline":
        raise ValueError("Composite costing requires cost.type: full_pipeline")
    raw_rates = cost_config.get("rates")
    if not isinstance(raw_rates, Mapping):
        raise ValueError("cost.rates must be a mapping")
    currency = cost_config.get("currency", "cost_units")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("cost.currency must be a non-empty string")
    return CostRates.from_mapping(raw_rates), currency.strip()
