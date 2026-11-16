"""Granular tracing: nested spans (request -> agent step -> LLM call -> tool call) in SQLite, optionally
mirrored to OpenTelemetry using the GenAI semantic-convention attribute names so the same spans can flow to
Cloud Trace / any OTLP backend.

Parenting is implicit through a ContextVar, so application code just nests `with tracer.span(...)` blocks and
`Gateway.chat` attaches its spans under whatever span is current - that is how an agent framework's step
becomes the parent of the LLM calls it makes.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .clock import Clock, RealClock
from .store import Store

_current: ContextVar["Span | None"] = ContextVar("llmscope_span", default=None)

GENAI_ATTRS = {"model": "gen_ai.request.model", "input_tokens": "gen_ai.usage.input_tokens",
               "output_tokens": "gen_ai.usage.output_tokens", "temperature": "gen_ai.request.temperature",
               "max_tokens": "gen_ai.request.max_tokens"}


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ts: float
    attrs: dict = field(default_factory=dict)
    status: str = "ok"

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)


class OTelMirror:
    """Mirrors every span to an OpenTelemetry TracerProvider (e.g. one exporting to Cloud Trace)."""

    def __init__(self, tracer_provider):
        from opentelemetry import trace

        self._trace = trace
        self._tracer = tracer_provider.get_tracer("llmscope")
        self._live: dict[str, object] = {}

    def start(self, span: Span) -> None:
        ctx = self._trace.set_span_in_context(self._live[span.parent_id]) if span.parent_id in self._live else None
        self._live[span.span_id] = self._tracer.start_span(span.name, context=ctx, start_time=int(span.start_ts * 1e9))

    def end(self, span: Span, end_ts: float) -> None:
        o = self._live.pop(span.span_id)
        for k, v in span.attrs.items():
            if isinstance(v, (str, int, float, bool)):
                o.set_attribute(GENAI_ATTRS.get(k, f"llmscope.{k}"), v)
        if span.status != "ok":
            o.set_status(self._trace.Status(self._trace.StatusCode.ERROR))
        o.end(end_time=int(end_ts * 1e9))


class Tracer:
    def __init__(self, store: Store, clock: Clock | None = None, otel: OTelMirror | None = None):
        self.store, self.clock, self.otel = store, clock or RealClock(), otel

    @contextmanager
    def span(self, name: str, **attrs):
        parent = _current.get()
        s = Span(parent.trace_id if parent else uuid.uuid4().hex[:16], uuid.uuid4().hex[:8],
                 parent.span_id if parent else None, name, self.clock.now(), dict(attrs))
        token = _current.set(s)
        if self.otel:
            self.otel.start(s)
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attrs["error"] = f"{type(exc).__name__}: {exc}"[:200]
            raise
        finally:
            end = self.clock.now()
            _current.reset(token)
            self.store.add_span({"trace_id": s.trace_id, "span_id": s.span_id, "parent_id": s.parent_id, "name": s.name,
                                 "start_ts": s.start_ts, "end_ts": end, "status": s.status, "attrs": s.attrs})
            if self.otel:
                self.otel.end(s, end)


def render_tree(spans: list[dict]) -> str:
    """Pretty-print a trace as an indented tree with durations."""
    children: dict[str | None, list[dict]] = {}
    for s in spans:
        children.setdefault(s["parent_id"], []).append(s)
    lines: list[str] = []

    def walk(parent, depth):
        for s in children.get(parent, []):
            ms = (s["end_ts"] - s["start_ts"]) * 1000
            keep = {k: v for k, v in s["attrs"].items() if k in ("model", "input_tokens", "output_tokens", "cache", "ttft_ms", "cost_usd", "error")}
            lines.append(f"{'  ' * depth}{s['name']:<{28 - 2 * depth}} {ms:9.1f} ms  {s['status']:<5} {keep if keep else ''}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)
