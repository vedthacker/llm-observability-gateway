"""Time source abstraction.

Latency metrics (TTFT, tokens/sec) are only testable if time is controllable. `RealClock` is for
production; `VirtualClock` lets the simulated provider "sleep" instantly while every timestamp the gateway
records still reflects the simulated model latency, so benchmarks run in milliseconds and tests can assert
exact numbers.
"""

from __future__ import annotations

import time


class RealClock:
    virtual = False

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class VirtualClock:
    virtual = True

    def __init__(self, start: float = 1_700_000_000.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.sleep(seconds)


from typing import Protocol  # noqa: E402


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
