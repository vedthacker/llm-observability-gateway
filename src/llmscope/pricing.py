"""Per-model price sheet (USD per 1M tokens).

The numbers below are ILLUSTRATIVE placeholders so the cost maths is testable. Real prices change; load the
current Vertex AI / Gemini price list with `load_pricing("prices.json")` before trusting any cost figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


DEFAULT_PRICES: dict[str, Price] = {
    "mock-fast": Price(0.30, 2.50),
    "mock-large": Price(1.25, 10.00),
    "gemini-2.5-flash": Price(0.30, 2.50),     # illustrative - verify
    "gemini-2.5-pro": Price(1.25, 10.00),      # illustrative - verify
}


def load_pricing(path: str | Path) -> dict[str, Price]:
    raw = json.loads(Path(path).read_text())
    return {m: Price(float(p["input_per_m"]), float(p["output_per_m"])) for m, p in raw.items()}


def cost_usd(prices: dict[str, Price], model: str, in_tokens: int, out_tokens: int) -> float:
    p = prices.get(model)
    if p is None:
        raise KeyError(f"no price configured for model '{model}'")
    return in_tokens / 1e6 * p.input_per_m + out_tokens / 1e6 * p.output_per_m
