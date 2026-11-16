"""Token estimation. Providers that report usage (Gemini does) override this; the estimate is only
used for budgeting before a call and for the mock provider. ~4 characters per token for English text."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)   # +4 ~ per-message role/format overhead
