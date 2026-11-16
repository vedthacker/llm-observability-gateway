"""Model providers. Both stream, because time-to-first-token can only be measured on a stream.

MockProvider simulates a model with the two numbers that dominate perceived latency:
    TTFT        = base_ttft + prefill_ms_per_token * input_tokens     (prefill grows with prompt length)
    decode rate = tokens_per_sec                                      (steady generation speed)
It is deterministic (jitter is seeded from the prompt) and runs against a Clock, so a VirtualClock makes it instant.
Numbers it produces are SIMULATED - they demonstrate the instrumentation, they are not benchmarks of a real model.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator, Protocol

from .clock import Clock, RealClock
from .tokens import estimate_tokens, messages_tokens


class ProviderError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class StreamChunk:
    text: str = ""
    usage: Usage | None = None     # only on the final chunk


class Provider(Protocol):
    def stream(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[StreamChunk]: ...


@dataclass(frozen=True)
class Profile:
    base_ttft_ms: float
    prefill_ms_per_token: float
    tokens_per_sec: float


PROFILES = {
    "mock-fast": Profile(180, 0.25, 120),
    "mock-large": Profile(450, 0.80, 45),
}
WORDS = ("the policy states that customers should check their plan limits before contacting support and the team "
         "will confirm approval timelines escalation owners and next steps within the agreed response window").split()


def prompt_ref(text: str) -> str:
    """Stable short id of a normalised prompt - lets tests and the benchmark see WHICH prompt a response was made for."""
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()[:8]


class MockProvider:
    def __init__(self, clock: Clock | None = None, profiles: dict[str, Profile] | None = None, speed: float = 1.0):
        self.clock = clock or RealClock()
        self.speed = speed            # >1 makes the simulated model faster (demo server); all latencies scale together
        self.profiles = profiles or PROFILES
        self.fail_next = 0            # test hook: raise ProviderError for the next N calls
        self.calls = 0

    def stream(self, model, messages, max_tokens=256, temperature=0.0):
        self.calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ProviderError("simulated upstream 503")
        prof = self.profiles.get(model)
        if prof is None:
            raise ProviderError(f"unknown mock model '{model}'")
        n_in = messages_tokens(messages)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        rng = random.Random(prompt_ref(last))
        n_out = max(1, min(max_tokens, 40 + rng.randint(0, 120)))
        jitter = 1 + 0.1 * rng.random()
        self.clock.sleep((prof.base_ttft_ms + prof.prefill_ms_per_token * n_in) * jitter / 1000 / self.speed)   # prefill
        words = [f"[ref:{prompt_ref(last)}]"] + [rng.choice(WORDS) for _ in range(n_out)]
        emitted = 0
        for w in words:
            tok = estimate_tokens(w + " ")
            self.clock.sleep(tok / prof.tokens_per_sec / self.speed)                                # decode
            emitted += tok
            yield StreamChunk(text=w + " ")
        yield StreamChunk(usage=Usage(n_in, emitted))


class GeminiProvider:
    """Gemini through the google-genai SDK (Vertex AI or API key). Streams and reports real usage metadata."""

    def __init__(self, project: str | None = None, location: str = "us-central1", vertex: bool = False):
        from google import genai

        self._client = genai.Client(vertexai=True, project=project, location=location) if vertex else genai.Client()

    def stream(self, model, messages, max_tokens=256, temperature=0.0):
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        usage = None
        try:
            for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                                       temperature=temperature)):
                if chunk.text:
                    yield StreamChunk(text=chunk.text)
                if getattr(chunk, "usage_metadata", None):
                    u = chunk.usage_metadata
                    usage = Usage(u.prompt_token_count or 0, u.candidates_token_count or 0)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc
        yield StreamChunk(usage=usage)
