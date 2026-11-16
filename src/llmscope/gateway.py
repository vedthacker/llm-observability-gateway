"""The gateway: one call path that adds guardrails, optimisation and measurement around any provider.

    rate limit -> budget -> compact context -> cache lookup -> provider (stream, measure TTFT / tokens-per-sec)
                                                          -> cost -> cache store -> request record + spans

LLM-native metrics recorded per request:
    ttft_ms         time from sending the request to the first streamed token (prefill + queueing)
    tokens_per_sec  output tokens / (end - first token): the steady decode rate, deliberately excluding TTFT
    cost_usd        input/output tokens x price sheet; saved_cost_usd for cache hits
    overhead_ms     real CPU time the gateway itself added (cache lookup, compaction, bookkeeping)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .cache import ResponseCache
from .clock import Clock, RealClock
from .context import compact
from .limits import TokenBucket, utc_day_start
from .pricing import DEFAULT_PRICES, Price, cost_usd
from .providers import Provider, ProviderError
from .store import Store
from .tokens import estimate_tokens, messages_tokens
from .tracing import Tracer


class GatewayError(Exception):
    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code, self.http_status = code, http_status


@dataclass
class ChatRequest:
    messages: list[dict]
    model: str = "mock-fast"
    tenant: str = "default"
    max_tokens: int = 256
    temperature: float = 0.0
    use_cache: bool = True
    compact_budget: int | None = None      # compact history above this many input tokens (None = never)


@dataclass
class ChatResult:
    text: str
    model: str
    cache: str                              # miss | exact | semantic
    input_tokens: int
    output_tokens: int
    cost_usd: float
    saved_tokens: int
    saved_cost_usd: float
    ttft_ms: float | None
    latency_ms: float
    tokens_per_sec: float | None
    overhead_ms: float
    compacted: bool
    trace_id: str
    similarity: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GatewayConfig:
    prices: dict[str, Price] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    rate_limit_per_min: float = 600
    daily_budget_usd: float | None = None   # per tenant; None = unlimited
    max_retries: int = 1


class Gateway:
    def __init__(self, provider: Provider, store: Store | None = None, clock: Clock | None = None,
                 cache: ResponseCache | None = None, config: GatewayConfig | None = None, tracer: Tracer | None = None):
        self.clock = clock or RealClock()
        self.store = store or Store()
        self.provider = provider
        self.cache = cache
        self.cfg = config or GatewayConfig()
        self.tracer = tracer or Tracer(self.store, self.clock)
        self.bucket = TokenBucket(self.cfg.rate_limit_per_min, clock=self.clock)

    # ------------------------------------------------------------------
    def chat(self, req: ChatRequest) -> ChatResult:
        cpu0 = time.perf_counter()
        t0 = self.clock.now()
        with self.tracer.span("gateway.chat", tenant=req.tenant, model=req.model) as root:
            try:
                res = self._chat(req, root, cpu0, t0)
            except GatewayError as exc:
                self._record(req, root.trace_id, status=exc.code, error=str(exc))
                raise
            root.set(cache=res.cache, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                     cost_usd=round(res.cost_usd, 6), ttft_ms=res.ttft_ms)
            return res

    def _chat(self, req: ChatRequest, root, cpu0: float, t0: float) -> ChatResult:
        if req.model not in self.cfg.prices:
            raise GatewayError("unknown_model", f"no price configured for '{req.model}'", 400)
        if not self.bucket.allow(req.tenant):
            raise GatewayError("rate_limited", f"tenant '{req.tenant}' exceeded {self.cfg.rate_limit_per_min}/min", 429)
        if self.cfg.daily_budget_usd is not None:
            spent = self.store.spend_since(req.tenant, utc_day_start(self.clock.now()))
            if spent >= self.cfg.daily_budget_usd:
                raise GatewayError("budget_exceeded", f"tenant '{req.tenant}' spent ${spent:.4f} of ${self.cfg.daily_budget_usd}", 429)

        messages, compacted, saved_by_compaction = req.messages, False, 0
        if req.compact_budget:
            with self.tracer.span("context.compact") as sp:
                c = compact(req.messages, req.compact_budget)
                messages, compacted, saved_by_compaction = c.messages, c.compacted, c.saved
                sp.set(tokens_before=c.tokens_before, tokens_after=c.tokens_after, compacted=c.compacted)

        price = self.cfg.prices[req.model]
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        scope = ResponseCache.scope(req.tenant, req.model, system)
        # `is not None`, not truthiness: ResponseCache defines __len__, so an EMPTY cache is falsy (this bug disabled the cache)
        cacheable = self.cache is not None and req.use_cache and req.temperature == 0

        if cacheable:
            with self.tracer.span("cache.lookup") as sp:
                hit = self.cache.lookup(scope, messages)
                sp.set(cache=hit[0] if hit else "miss")
            if hit:
                kind, e, sim = hit
                overhead = (time.perf_counter() - cpu0) * 1000
                latency = (self.clock.now() - t0) * 1000 + (overhead if self.clock.virtual else 0)
                saved_cost = cost_usd(self.cfg.prices, req.model, e.input_tokens, e.output_tokens)
                # a hit bills zero tokens; what it avoided is recorded separately as saved_tokens / saved_cost
                self._record(req, root.trace_id, cache=kind, saved_tokens=e.input_tokens + e.output_tokens,
                             saved_cost=saved_cost, latency=latency, overhead=overhead)
                return ChatResult(e.text, req.model, kind, e.input_tokens, e.output_tokens, 0.0,
                                  e.input_tokens + e.output_tokens, saved_cost, None, latency, None, overhead,
                                  compacted, root.trace_id, round(sim, 4))

        p0 = time.perf_counter()
        text, usage, ttft_ms, tps = self._call_provider(req, messages)
        provider_ms = (time.perf_counter() - p0) * 1000
        overhead = (time.perf_counter() - cpu0) * 1000 - provider_ms     # gateway-only time: exclude the model call itself
        latency = (self.clock.now() - t0) * 1000
        if self.clock.virtual:
            latency += overhead
        cost = cost_usd(self.cfg.prices, req.model, usage.input_tokens, usage.output_tokens)
        saved_cost = saved_by_compaction / 1e6 * price.input_per_m
        if cacheable:
            self.cache.put(scope, messages, text, usage.input_tokens, usage.output_tokens)
        self._record(req, root.trace_id, cache="miss", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                     saved_tokens=saved_by_compaction, cost=cost, saved_cost=saved_cost, ttft=ttft_ms, latency=latency,
                     tps=tps, overhead=overhead)
        return ChatResult(text, req.model, "miss", usage.input_tokens, usage.output_tokens, cost, saved_by_compaction,
                          saved_cost, ttft_ms, latency, tps, overhead, compacted, root.trace_id)

    def _call_provider(self, req: ChatRequest, messages: list[dict]):
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            with self.tracer.span("llm.call", model=req.model, max_tokens=req.max_tokens, temperature=req.temperature,
                                  attempt=attempt) as sp:
                try:
                    start = self.clock.now()
                    first = last = None
                    parts: list[str] = []
                    usage = None
                    for chunk in self.provider.stream(req.model, messages, req.max_tokens, req.temperature):
                        if chunk.text:
                            if first is None:
                                first = self.clock.now()
                            parts.append(chunk.text)
                            last = self.clock.now()
                        if chunk.usage:
                            usage = chunk.usage
                    text = "".join(parts).strip()
                    if usage is None:                                   # provider gave no usage: fall back to estimates
                        from .providers import Usage

                        usage = Usage(messages_tokens(messages), estimate_tokens(text))
                    ttft_ms = (first - start) * 1000 if first is not None else None
                    decode_s = (last - first) if first is not None and last is not None else 0
                    tps = round(usage.output_tokens / decode_s, 2) if decode_s > 0 else None
                    sp.set(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                           ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None, tokens_per_sec=tps)
                    return text, usage, ttft_ms, tps
                except ProviderError as exc:
                    sp.status = "error"                                  # this attempt failed, even though we handle it
                    sp.set(error=str(exc)[:200])
                    last_err = exc
                    if attempt < self.cfg.max_retries:
                        self.clock.sleep(0.2 * (2 ** attempt))          # exponential backoff between attempts
        raise GatewayError("upstream_error", f"provider failed after {self.cfg.max_retries + 1} attempt(s): {last_err}", 502)

    def _record(self, req, trace_id, status="ok", cache="none", input_tokens=0, output_tokens=0, saved_tokens=0, cost=0.0,
                saved_cost=0.0, ttft=None, latency=0.0, tps=None, overhead=0.0, error=None):
        self.store.add_request(trace_id=trace_id, ts=self.clock.now(), tenant=req.tenant, model=req.model, status=status,
                               cache=cache, input_tokens=input_tokens, output_tokens=output_tokens,
                               saved_tokens=saved_tokens, cost_usd=cost, saved_cost_usd=saved_cost, ttft_ms=ttft,
                               latency_ms=latency, tokens_per_sec=tps, overhead_ms=overhead, error=error)
