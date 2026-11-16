"""FastAPI surface: /v1/chat plus the observability endpoints and a self-contained dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .cache import ResponseCache
from .gateway import ChatRequest, Gateway, GatewayConfig, GatewayError
from .providers import GeminiProvider, MockProvider
from .store import Store
from .tracing import render_tree

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=200_000)


class ChatBody(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = "mock-fast"
    tenant: str = Field("default", max_length=64)
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.0, ge=0, le=2)
    use_cache: bool = True
    compact_budget: int | None = Field(None, ge=200)


def default_gateway() -> Gateway:
    if os.getenv("LLMSCOPE_PROVIDER", "mock") == "gemini":
        provider = GeminiProvider(os.getenv("GOOGLE_CLOUD_PROJECT"), os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                                  vertex=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true"})
    else:
        provider = MockProvider(speed=float(os.getenv("LLMSCOPE_MOCK_SPEED", "10")))   # demo server: simulated model runs 10x faster
    budget = float(os.environ["LLMSCOPE_DAILY_BUDGET_USD"]) if "LLMSCOPE_DAILY_BUDGET_USD" in os.environ else None
    return Gateway(provider, Store(os.getenv("LLMSCOPE_DB", "llmscope.db")), cache=ResponseCache(),
                   config=GatewayConfig(daily_budget_usd=budget))


def create_app(gateway: Gateway | None = None) -> FastAPI:
    gw = gateway or default_gateway()
    app = FastAPI(title="llmscope", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat")
    def chat(body: ChatBody):
        try:
            return gw.chat(ChatRequest(messages=[m.model_dump() for m in body.messages], model=body.model,
                                       tenant=body.tenant, max_tokens=body.max_tokens, temperature=body.temperature,
                                       use_cache=body.use_cache, compact_budget=body.compact_budget)).to_dict()
        except GatewayError as exc:
            raise HTTPException(exc.http_status, {"error": exc.code, "message": str(exc)})

    @app.get("/metrics")
    def metrics():
        return gw.store.summary()

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prometheus():
        s = gw.store.summary()
        lines = [f"llmscope_requests_total {s['requests']}", f"llmscope_errors_total {s['errors']}",
                 f"llmscope_cache_hit_ratio {s['cache_hit_rate']}", f"llmscope_cost_usd_total {s['cost_usd']}",
                 f"llmscope_tokens_saved_total {s['tokens_saved']}"]
        for q in ("p50", "p95"):
            for name, key in (("latency_ms", "latency_ms"), ("ttft_ms", "ttft_ms")):
                if s[key][q] is not None:
                    lines.append(f'llmscope_{name}{{quantile="{q}"}} {s[key][q]}')
        return "\n".join(lines) + "\n"

    @app.get("/requests")
    def requests(limit: int = 25):
        return gw.store.recent(max(1, min(limit, 200)))

    @app.get("/traces/{trace_id}")
    def trace(trace_id: str, tree: bool = False):
        spans = gw.store.spans(trace_id)
        if not spans:
            raise HTTPException(404, "unknown trace")
        return PlainTextResponse(render_tree(spans)) if tree else spans

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "dashboard.html").read_text()

    return app


def __getattr__(name):      # `uvicorn llmscope.api:app` builds the default app lazily
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
