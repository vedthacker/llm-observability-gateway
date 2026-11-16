"""A tiny 3-step agent instrumented with llmscope: every LLM call becomes a child span of the agent step
that made it, so one trace shows the whole run - where the time went, how many tokens each step used, what it cost.

    python examples/agent_demo.py
"""

from llmscope import ChatRequest, Gateway
from llmscope.cache import ResponseCache
from llmscope.clock import VirtualClock
from llmscope.providers import MockProvider
from llmscope.store import Store
from llmscope.tracing import render_tree

clock = VirtualClock()
gw = Gateway(MockProvider(clock), Store(), clock, cache=ResponseCache(clock))
tracer = gw.tracer


def ask(q, model="mock-fast"):
    return gw.chat(ChatRequest([{"role": "user", "content": q}], model=model))


with tracer.span("agent.run", task="triage 429 errors for Echo Motors") as run:
    with tracer.span("agent.plan"):
        plan = ask("Plan the steps to triage HTTP 429 errors for a customer.", model="mock-large")
    with tracer.span("agent.tool", tool="search_runbooks"):
        clock.sleep(0.04)                                    # pretend the retrieval tool took 40 ms
    with tracer.span("agent.answer"):
        ans = ask("Draft the customer reply based on the plan: " + plan.text[:200])
    with tracer.span("agent.answer.retry-from-cache"):
        ask("Draft the customer reply based on the plan: " + plan.text[:200])      # identical -> exact cache hit

print(render_tree(gw.store.spans(run.trace_id)))
s = gw.store.summary()
print(f"\nrequests={s['requests']}  cache_hit_rate={s['cache_hit_rate']:.0%}  cost=${s['cost_usd']}  saved=${s['cost_saved_usd']}")
