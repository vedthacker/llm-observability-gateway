import pytest

from conftest import ask
from llmscope import ChatRequest, GatewayError, Tracer
from llmscope.gateway import GatewayConfig
from llmscope.tracing import OTelMirror, render_tree


def test_rate_limit_uses_token_bucket_and_refills(make_gateway, clock):
    gw, _ = make_gateway(cache=False, rate_limit_per_min=3)
    for i in range(3):
        ask(gw, f"q{i}")
    with pytest.raises(GatewayError) as e:
        ask(gw, "q-over")
    assert e.value.code == "rate_limited" and e.value.http_status == 429
    ask(gw, "other tenant ok", tenant="someone-else")
    clock.advance(60)
    ask(gw, "after refill")


def test_daily_budget_blocks_a_tenant_until_the_next_day(make_gateway, clock):
    gw, _ = make_gateway(cache=False, daily_budget_usd=0.0005)
    with pytest.raises(GatewayError) as e:
        for i in range(50):
            ask(gw, f"expensive question {i} " + "pad " * 300, model="mock-large", tenant="acme")
    assert e.value.code == "budget_exceeded"
    ask(gw, "different tenant is fine", tenant="globex", model="mock-large")
    clock.advance(86400)
    ask(gw, "new day", tenant="acme", model="mock-large")
    assert any(r["status"] == "budget_exceeded" for r in gw.store.recent(100))   # rejections are recorded


def test_upstream_failure_is_retried_with_a_span_per_attempt(make_gateway):
    gw, provider = make_gateway(cache=False)
    provider.fail_next = 1
    r = ask(gw, "flaky upstream")
    spans = [s for s in gw.store.spans(r.trace_id) if s["name"] == "llm.call"]
    assert len(spans) == 2 and [s["status"] for s in sorted(spans, key=lambda s: s["attrs"]["attempt"])] == ["error", "ok"]


def test_persistent_failure_becomes_a_502_and_is_recorded(make_gateway):
    gw, provider = make_gateway(cache=False)
    provider.fail_next = 10
    with pytest.raises(GatewayError) as e:
        ask(gw, "down")
    assert e.value.http_status == 502 and gw.store.summary()["errors"] == 1


def test_spans_nest_under_the_calling_application_span(make_gateway):
    gw, _ = make_gateway(cache=False)
    with gw.tracer.span("agent.step", step="research") as step:
        with gw.tracer.span("tool.search"):
            pass
        r = ask(gw, "summarise the findings")
    spans = gw.store.spans(step.trace_id)
    by_name = {s["name"]: s for s in spans}
    assert r.trace_id == step.trace_id
    assert by_name["tool.search"]["parent_id"] == by_name["agent.step"]["span_id"]
    assert by_name["gateway.chat"]["parent_id"] == by_name["agent.step"]["span_id"]
    assert by_name["llm.call"]["parent_id"] == by_name["gateway.chat"]["span_id"]
    tree = render_tree(spans)
    assert tree.index("agent.step") < tree.index("gateway.chat") < tree.index("llm.call")


def test_exception_inside_a_span_marks_it_as_error(make_gateway):
    gw, _ = make_gateway()
    with pytest.raises(ValueError):
        with gw.tracer.span("boom") as s:
            raise ValueError("bad")
    sp = gw.store.spans(s.trace_id)[0]
    assert sp["status"] == "error" and "bad" in sp["attrs"]["error"]


def test_otel_mirror_exports_genai_attributes_and_parenting(make_gateway, clock):
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    gw, _ = make_gateway(cache=False)
    gw.tracer = Tracer(gw.store, clock, OTelMirror(provider))
    r = ask(gw, "hello otel", model="mock-fast")
    done = {s.name: s for s in exporter.get_finished_spans()}
    llm = done["llm.call"]
    assert llm.attributes["gen_ai.request.model"] == "mock-fast"
    assert llm.attributes["gen_ai.usage.input_tokens"] == r.input_tokens
    assert llm.attributes["gen_ai.usage.output_tokens"] == r.output_tokens
    assert llm.parent.span_id == done["gateway.chat"].context.span_id
    assert (llm.end_time - llm.start_time) / 1e6 > 100          # timestamps come from the (virtual) clock, not wall time
