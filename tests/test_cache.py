from conftest import ask
from llmscope.cache import ResponseCache


def test_exact_hit_is_free_and_records_savings(make_gateway):
    gw, provider = make_gateway()
    first, second = ask(gw, "What is the refund window?"), ask(gw, "  what is the REFUND window? ".strip().replace("?", "?"))
    third = ask(gw, "What is the refund window?")
    assert first.cache == "miss" and third.cache == "exact" and third.cost_usd == 0 and third.saved_cost_usd > 0
    assert third.text == first.text and provider.calls <= 2


def test_empty_cache_object_is_still_used(make_gateway):
    """Regression: ResponseCache defines __len__, so `bool(empty_cache)` is False - the gateway once skipped it."""
    cache = ResponseCache(semantic_threshold=None)
    assert len(cache) == 0 and not cache
    gw, provider = make_gateway(cache=cache)
    ask(gw, "same"); ask(gw, "same")
    assert provider.calls == 1


def test_tenants_never_share_cached_answers(make_gateway):
    gw, provider = make_gateway()
    ask(gw, "What is our margin?", tenant="acme")
    other = ask(gw, "What is our margin?", tenant="globex")
    assert other.cache == "miss" and provider.calls == 2


def test_system_prompt_and_model_are_part_of_the_cache_scope(make_gateway):
    gw, provider = make_gateway()
    msgs = lambda sys: [{"role": "system", "content": sys}, {"role": "user", "content": "hello there"}]
    from llmscope import ChatRequest

    gw.chat(ChatRequest(msgs("You are formal.")))
    assert gw.chat(ChatRequest(msgs("You are casual."))).cache == "miss"
    assert gw.chat(ChatRequest(msgs("You are formal."), model="mock-large")).cache == "miss"
    assert gw.chat(ChatRequest(msgs("You are formal."))).cache == "exact"


def test_sampled_requests_are_not_cached(make_gateway):
    gw, provider = make_gateway()
    ask(gw, "write a poem", temperature=0.9); ask(gw, "write a poem", temperature=0.9)
    assert provider.calls == 2


def test_semantic_hit_for_near_duplicate_but_not_for_different_question(make_gateway):
    gw, _ = make_gateway()
    ask(gw, "What is the rate limit for the Business plan?")
    assert ask(gw, "what is the rate limit for the business plan").cache == "semantic"
    assert ask(gw, "How do I rotate my service keys?").cache == "miss"


def test_number_guard_blocks_different_numbers(make_gateway):
    gw, _ = make_gateway(cache=ResponseCache(semantic_threshold=0.6))
    ask(gw, "What is the refund policy after 30 days of delivery?")
    assert ask(gw, "What is the refund policy after 7 days of delivery?").cache == "miss"
    gw2, _ = make_gateway(cache=ResponseCache(semantic_threshold=0.6, number_guard=False))
    ask(gw2, "What is the refund policy after 30 days of delivery?")
    assert ask(gw2, "What is the refund policy after 7 days of delivery?").cache == "semantic"     # why the guard exists


def test_multi_turn_requests_only_use_exact_matching(make_gateway):
    from llmscope import ChatRequest

    gw, _ = make_gateway()
    turn = lambda q: [{"role": "user", "content": "context question"}, {"role": "assistant", "content": "ok"},
                      {"role": "user", "content": q}]
    gw.chat(ChatRequest(turn("what is the rate limit for the business plan")))
    assert gw.chat(ChatRequest(turn("What is the rate limit for the business plan?"))).cache == "miss"


def test_entries_expire_and_cache_is_bounded(make_gateway, clock):
    gw, provider = make_gateway(cache=ResponseCache(clock, ttl_s=60, max_entries=2, semantic_threshold=None))
    ask(gw, "question a"); clock.advance(61)
    assert ask(gw, "question a").cache == "miss"                         # expired
    ask(gw, "question b"); ask(gw, "question c")
    assert len(gw.cache) == 2 and ask(gw, "question a").cache == "miss"  # evicted
