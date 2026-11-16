from fastapi.testclient import TestClient

from llmscope.api import create_app
from llmscope.bench import cache_report, make_stream, run_compaction


def client(make_gateway, **cfg):
    gw, _ = make_gateway(**cfg)
    return TestClient(create_app(gw)), gw


def test_chat_metrics_trace_and_dashboard_endpoints(make_gateway):
    c, _ = client(make_gateway)
    body = {"messages": [{"role": "user", "content": "What is the refund window?"}], "tenant": "acme"}
    first, second = c.post("/v1/chat", json=body).json(), c.post("/v1/chat", json=body).json()
    assert first["cache"] == "miss" and second["cache"] == "exact" and first["ttft_ms"] > 0
    m = c.get("/metrics").json()
    assert m["requests"] == 2 and m["cache_exact"] == 1
    assert "llmscope_cache_hit_ratio 0.5" in c.get("/metrics/prometheus").text
    tree = c.get(f"/traces/{first['trace_id']}?tree=true").text
    assert "gateway.chat" in tree and "llm.call" in tree
    assert c.get("/traces/nope").status_code == 404
    assert "llmscope" in c.get("/dashboard").text and len(c.get("/requests?limit=5").json()) == 2


def test_errors_map_to_http_status_and_input_is_validated(make_gateway):
    c, _ = client(make_gateway, rate_limit_per_min=1)
    ok = {"messages": [{"role": "user", "content": "one"}]}
    assert c.post("/v1/chat", json=ok).status_code == 200
    r = c.post("/v1/chat", json={"messages": [{"role": "user", "content": "two"}]})
    assert r.status_code == 429 and r.json()["detail"]["error"] == "rate_limited"
    assert c.post("/v1/chat", json={"messages": []}).status_code == 422
    assert c.post("/v1/chat", json={"messages": [{"role": "root", "content": "x"}]}).status_code == 422
    assert c.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}], "model": "nope"}).status_code in (400, 429)


def test_benchmark_is_deterministic_and_cache_beats_baseline():
    rows = {r["config"]: r for r in cache_report(n=200, thresholds=(0.9,))}
    assert rows["no cache"]["hit_rate"] == 0
    assert rows["exact only"]["cost_usd"] < rows["no cache"]["cost_usd"] * 0.6
    stable = lambda rs: [{k: v for k, v in r.items() if k not in ("overhead_p95_ms", "mean_latency_ms", "p95_latency_ms")} for r in rs]
    assert stable(cache_report(n=200, thresholds=(0.9,))) == stable(rows.values())   # real CPU timings legitimately vary
    assert len(make_stream(50)) == 50


def test_compaction_benchmark_saves_tokens():
    base, small = run_compaction(None, conversations=3, turns=10), run_compaction(800, conversations=3, turns=10)
    assert small["input_tokens"] < base["input_tokens"] * 0.6 and small["ttft_p95_ms"] < base["ttft_p95_ms"]
