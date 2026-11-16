"""Response cache: exact match first, then (optionally) semantic match.

Safety rules, each of them enforced by a test:
  * scoped by (tenant, model, system prompt): one tenant can never be served another tenant's answer
  * only temperature == 0 requests are cached (a sampled answer is not "the" answer)
  * semantic lookup only for single-turn requests - with history, "same last question" is not "same request"
  * entries expire (TTL) and the cache is size-bounded
A semantic cache can return a *wrong* answer for a near-identical question ("Business tier" vs "Enterprise tier"),
so the threshold is configurable and the benchmark measures how often it happens.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np

from .clock import Clock, RealClock


class HashingEmbedder:
    """Dependency-free text embedding (signed hashed unigrams + bigrams). Catches near-duplicates, NOT paraphrases."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        import re

        toks = re.findall(r"[a-z0-9]+", text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f, tf in Counter(feats).items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1 + math.log(tf))
        n = np.linalg.norm(v)
        return v / n if n else v


class FastEmbedAdapter:
    """Real sentence embeddings (BAAI/bge-small-en-v1.5, ONNX, local). Needs `pip install fastembed`."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._m = TextEmbedding(model_name=model)

    def embed(self, text: str) -> np.ndarray:
        v = np.array(next(iter(self._m.embed([text]))), dtype=np.float32)
        return v / np.linalg.norm(v)


@dataclass
class CacheEntry:
    text: str
    input_tokens: int
    output_tokens: int
    created: float
    last_user: str


def numbers(text: str) -> frozenset[str]:
    """Digit sequences in a prompt ('SEV1' -> {'1'}, 'Tier 2' -> {'2'}, '30 days' -> {'30'})."""
    return frozenset(re.findall(r"\d+(?:[.,]\d+)?", text))


def norm(text: str) -> str:
    return " ".join(text.lower().split())


class ResponseCache:
    def __init__(self, clock: Clock | None = None, ttl_s: float = 3600, max_entries: int = 2000,
                 semantic_threshold: float | None = 0.92, embedder=None, number_guard: bool = True):
        self.clock, self.ttl_s, self.max_entries = clock or RealClock(), ttl_s, max_entries
        self.threshold = semantic_threshold
        self.number_guard = number_guard      # never serve a cached answer whose prompt had different numbers
        self.embedder = embedder or HashingEmbedder()
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._sem: dict[str, list[tuple[np.ndarray, str]]] = {}      # scope -> [(vector, exact key)]

    @staticmethod
    def scope(tenant: str, model: str, system: str) -> str:
        return hashlib.sha256(f"{tenant}\x00{model}\x00{norm(system)}".encode()).hexdigest()[:16]

    @staticmethod
    def _key(scope: str, messages: list[dict]) -> str:
        body = json.dumps([[m["role"], norm(m["content"])] for m in messages if m["role"] != "system"])
        return scope + hashlib.sha256(body.encode()).hexdigest()

    @staticmethod
    def _user_turns(messages: list[dict]) -> int:
        return sum(m["role"] == "user" for m in messages)

    def lookup(self, scope: str, messages: list[dict]) -> tuple[str, CacheEntry, float] | None:
        key = self._key(scope, messages)
        e = self._exact.get(key)
        if e and self.clock.now() - e.created <= self.ttl_s:
            self._exact.move_to_end(key)
            return "exact", e, 1.0
        if self.threshold is None or self._user_turns(messages) != 1:
            return None
        cands = self._sem.get(scope, [])
        if not cands:
            return None
        query = messages[-1]["content"]
        sims = np.stack([v for v, _ in cands]) @ self.embedder.embed(query)
        for i in np.argsort(-sims):                                   # best candidate first
            if sims[i] < self.threshold:
                break
            entry = self._exact.get(cands[i][1])
            if not entry or self.clock.now() - entry.created > self.ttl_s:
                continue
            if self.number_guard and numbers(query) != numbers(entry.last_user):
                continue                                              # "30 days" must not be answered with "7 days"
            return "semantic", entry, float(sims[i])
        return None

    def put(self, scope: str, messages: list[dict], text: str, in_tokens: int, out_tokens: int) -> None:
        key = self._key(scope, messages)
        self._exact[key] = CacheEntry(text, in_tokens, out_tokens, self.clock.now(), messages[-1]["content"])
        self._exact.move_to_end(key)
        if self.threshold is not None and self._user_turns(messages) == 1:
            self._sem.setdefault(scope, []).append((self.embedder.embed(messages[-1]["content"]), key))
        while len(self._exact) > self.max_entries:               # evict least recently used
            old, _ = self._exact.popitem(last=False)
            for lst in self._sem.values():
                lst[:] = [(v, k) for v, k in lst if k != old]

    def __len__(self):
        return len(self._exact)
