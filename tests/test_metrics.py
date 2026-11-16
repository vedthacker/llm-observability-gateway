import pytest

from conftest import ask
from llmscope.pricing import DEFAULT_PRICES, cost_usd
from llmscope.providers import PROFILES
from llmscope.store import percentile


def test_ttft_tokens_per_sec_and_cost_follow_the_model_profile(make_gateway):
    gw, _ = make_gateway(cache=False)
    r = ask(gw, "Explain the escalation policy for a regional outage in detail please.", model="mock-fast", max_tokens=200)
    prof = PROFILES["mock-fast"]
    floor = prof.base_ttft_ms + prof.prefill_ms_per_token * r.input_tokens
    assert floor <= r.ttft_ms <= floor * 1.11 + 50                  # +10% jitter, + decoding of the first token itself
    assert r.tokens_per_sec == pytest.approx(prof.tokens_per_sec, rel=0.15)
    assert r.latency_ms >= r.ttft_ms + (r.output_tokens - 1) / prof.tokens_per_sec * 1000 * 0.85
    assert r.cost_usd == pytest.approx(cost_usd(DEFAULT_PRICES, "mock-fast", r.input_tokens, r.output_tokens))


def test_larger_model_is_slower_and_costlier(make_gateway):
    gw, _ = make_gateway(cache=False)
    fast, large = ask(gw, "same question", model="mock-fast"), ask(gw, "same question", model="mock-large")
    assert large.ttft_ms > fast.ttft_ms and large.tokens_per_sec < fast.tokens_per_sec
    assert large.cost_usd > fast.cost_usd


def test_longer_prompts_have_longer_ttft(make_gateway):
    gw, _ = make_gateway(cache=False)
    short, long = ask(gw, "hi there"), ask(gw, "word " * 2000)
    assert long.ttft_ms > short.ttft_ms + 100                         # prefill grows with prompt length


def test_summary_aggregates_requests(make_gateway):
    gw, _ = make_gateway()
    for q in ["alpha question one", "alpha question one", "beta question two"]:
        ask(gw, q)
    s = gw.store.summary()
    assert s["requests"] == 3 and s["cache_exact"] == 1 and s["cache_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["cost_saved_usd"] > 0 and s["ttft_ms"]["p50"] is not None and s["by_model"]["mock-fast"]["requests"] == 3


def test_percentile_is_nearest_rank():
    assert percentile(list(range(1, 101)), 95) == 95 and percentile([5], 99) == 5 and percentile([], 50) is None


def test_unknown_model_is_rejected_before_any_call(make_gateway):
    from llmscope import GatewayError

    gw, provider = make_gateway()
    with pytest.raises(GatewayError) as e:
        ask(gw, "x", model="gpt-nope")
    assert e.value.code == "unknown_model" and provider.calls == 0


def test_gateway_overhead_excludes_provider_time_on_a_real_clock():
    """Regression: overhead once included the model call (dashboard showed ~1000 ms 'gateway overhead')."""
    from llmscope import Gateway
    from llmscope.clock import RealClock
    from llmscope.providers import MockProvider, Profile
    from llmscope.store import Store

    slow = MockProvider(RealClock(), profiles={"mock-fast": Profile(120, 0, 2000)})     # ~120 ms of real waiting
    gw = Gateway(slow, Store(), RealClock())
    r = ask(gw, "hello")
    assert r.latency_ms > 100 and r.overhead_ms < 25
