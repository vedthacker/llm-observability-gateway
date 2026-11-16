"""Per-tenant guardrails: token-bucket rate limit and a daily USD budget."""

from __future__ import annotations

from .clock import Clock, RealClock


class TokenBucket:
    def __init__(self, per_minute: float, burst: float | None = None, clock: Clock | None = None):
        self.rate = per_minute / 60.0
        self.capacity = burst if burst is not None else per_minute
        self.clock = clock or RealClock()
        self._state: dict[str, tuple[float, float]] = {}          # tenant -> (tokens, last_ts)

    def allow(self, tenant: str) -> bool:
        now = self.clock.now()
        tokens, last = self._state.get(tenant, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        ok = tokens >= 1.0
        self._state[tenant] = (tokens - 1.0 if ok else tokens, now)
        return ok


def utc_day_start(ts: float) -> float:
    return ts - (ts % 86400)
