"""Helpers for scraping online LLM model pricing data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelCost:
    """Normalized model pricing in USD per 1M tokens."""

    model_id: str
    model_name: str
    input_cost_per_1m_tokens: float | None
    output_cost_per_1m_tokens: float | None
    context_length: int | None


def _parse_price(pricing: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        raw_value = pricing.get(key)
        if raw_value in (None, ""):
            continue

        try:
            return float(raw_value) * TOKENS_PER_MILLION
        except (TypeError, ValueError):
            continue

    return None


def _load_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310
        payload = response.read().decode("utf-8")

    data = json.loads(payload)
    if not isinstance(data, dict):
        msg = "Expected a JSON object in pricing response"
        raise ValueError(msg)
    return data


def scrape_llm_model_costs(
    endpoint: str = OPENROUTER_MODELS_ENDPOINT,
    timeout: float = 15.0,
    limit: int | None = None,
) -> list[ModelCost]:
    """Fetch and normalize LLM model pricing from an online endpoint.

    The default endpoint is OpenRouter's public models API. Pricing values are
    converted to USD per 1 million tokens.
    """

    if limit is not None and limit < 1:
        msg = "limit must be >= 1"
        raise ValueError(msg)

    data = _load_json(endpoint, timeout=timeout)
    raw_models = data.get("data", [])
    if not isinstance(raw_models, list):
        msg = "Expected 'data' to be a list of models"
        raise ValueError(msg)

    model_costs: list[ModelCost] = []
    for model in raw_models:
        if not isinstance(model, dict):
            continue

        pricing = model.get("pricing", {})
        if not isinstance(pricing, dict):
            pricing = {}

        model_id = str(model.get("id", "")).strip()
        model_name = str(model.get("name", model_id)).strip() or model_id
        if not model_id:
            continue

        context_length_raw = model.get("context_length")
        context_length = (
            int(context_length_raw) if isinstance(context_length_raw, int) else None
        )

        model_costs.append(
            ModelCost(
                model_id=model_id,
                model_name=model_name,
                input_cost_per_1m_tokens=_parse_price(
                    pricing,
                    ("prompt", "input"),
                ),
                output_cost_per_1m_tokens=_parse_price(
                    pricing,
                    ("completion", "output"),
                ),
                context_length=context_length,
            )
        )

    model_costs.sort(key=lambda item: item.model_id)
    if limit is not None:
        return model_costs[:limit]
    return model_costs


__all__ = [
    "ModelCost",
    "OPENROUTER_MODELS_ENDPOINT",
    "scrape_llm_model_costs",
]
