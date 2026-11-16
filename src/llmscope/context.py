"""Conversation state compaction.

Every turn re-sends the whole history, so input tokens (and cost, and TTFT - prefill grows with prompt length)
grow with each turn. When the history exceeds `budget_tokens` this keeps:
    1. all system messages, verbatim
    2. the last `keep_last` messages, verbatim (the model needs the recent exchange exactly)
    3. ONE synthetic message summarising everything older
The default summariser is extractive (first sentence of each old message) - cheap and deterministic. Pass
`summarizer=` to use an LLM instead. If still over budget, the oldest "recent" messages are dropped last, but the
final user message is never removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .tokens import estimate_tokens, messages_tokens

Summarizer = Callable[[list[dict]], str]


def extractive_summary(old: list[dict], max_chars: int = 160) -> str:
    parts = []
    for m in old:
        first = re.split(r"(?<=[.!?])\s+", m["content"].strip())[0]
        parts.append(f"{'U' if m['role'] == 'user' else 'A'}: {first[:max_chars]}")
    return "Summary of earlier conversation - " + " | ".join(parts)


@dataclass
class CompactionResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    compacted: bool

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


def compact(messages: list[dict], budget_tokens: int, keep_last: int = 4,
            summarizer: Summarizer = extractive_summary) -> CompactionResult:
    before = messages_tokens(messages)
    if before <= budget_tokens:
        return CompactionResult(messages, before, before, False)
    system = [m for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]
    tail = convo[-keep_last:]
    old = convo[:-keep_last] if len(convo) > keep_last else []
    out = list(system)
    if old:
        summary = summarizer(old)
        max_summary = max(40, budget_tokens // 4)                       # the summary may not eat the whole budget
        out.append({"role": "user", "content": summary[: max_summary * 4]})
    out += tail
    while messages_tokens(out) > budget_tokens and len([m for m in out if m["role"] != "system"]) > 2:
        idx = next(i for i, m in enumerate(out) if m["role"] != "system")
        out.pop(idx)                                                     # drop oldest non-system message
    return CompactionResult(out, before, messages_tokens(out), True)
