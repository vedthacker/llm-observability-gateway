from llmscope.context import compact
from llmscope.tokens import messages_tokens


def convo(n, words=80):
    msgs = [{"role": "system", "content": "You are a support bot."}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}. " + "detail " * words})
    return msgs


def test_short_conversations_are_untouched():
    m = convo(3, words=5)
    r = compact(m, 2000)
    assert r.messages == m and not r.compacted and r.saved == 0


def test_long_history_is_compacted_within_budget_and_keeps_what_matters():
    m = convo(30)
    r = compact(m, 800, keep_last=4)
    assert r.compacted and r.tokens_after <= 800 < r.tokens_before and r.saved > 0
    assert r.messages[0]["role"] == "system" and r.messages[0]["content"] == m[0]["content"]
    assert r.messages[-1] == m[-1] and r.messages[-4:] == m[-4:]
    assert any(x["content"].startswith("Summary of earlier conversation") for x in r.messages)


def test_last_user_message_is_never_dropped_even_under_a_tiny_budget():
    m = convo(21)
    m.append({"role": "user", "content": "FINAL QUESTION " + "x " * 50})
    r = compact(m, 100)
    assert r.messages[-1]["content"].startswith("FINAL QUESTION")


def test_custom_summarizer_is_used():
    r = compact(convo(30), 800, summarizer=lambda old: f"LLM summary of {len(old)} messages")
    assert any("LLM summary of" in x["content"] for x in r.messages)


def test_compaction_reduces_input_tokens_seen_by_the_provider(make_gateway):
    from llmscope import ChatRequest

    gw, _ = make_gateway(cache=False)
    plain = gw.chat(ChatRequest(convo(30)))
    small = gw.chat(ChatRequest(convo(30), compact_budget=800))
    assert small.compacted and small.input_tokens < plain.input_tokens * 0.5
    assert small.saved_tokens == messages_tokens(convo(30)) - small.input_tokens or small.saved_tokens > 0
    assert small.ttft_ms < plain.ttft_ms                              # shorter prompt -> shorter prefill
