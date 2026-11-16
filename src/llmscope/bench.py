"""Benchmarks for the two optimisations, on a SIMULATED model (VirtualClock + MockProvider).

What is real here: token counts, cost arithmetic, cache behaviour (hits, wrong hits), compaction savings, the
gateway's own overhead. What is simulated: model latency (TTFT / tokens-per-sec come from the profile in
providers.py). So read the latency columns as "what the instrumentation reports for a model with that profile",
not as a claim about any real model.
"""

from __future__ import annotations

import random
import statistics

from .cache import FastEmbedAdapter, HashingEmbedder, ResponseCache
from .clock import VirtualClock
from .gateway import ChatRequest, Gateway, GatewayConfig
from .providers import MockProvider
from .store import Store, percentile

# ---------------------------------------------------------------- workload
STANDALONE = {   # intent -> phrasings (first is canonical, last is a real paraphrase)
    "refund_window": ["How many days do customers have to ask for a refund?", "What is the time limit for requesting my money back?"],
    "password_len": ["How long must passwords be?", "What is the minimum password length?"],
    "key_rotation": ["How often do service account keys need to be rotated?", "What is the rotation period for service credentials?"],
    "log_retention": ["How long are application logs kept?", "For how long do you retain app logs?"],
    "webhook_retry": ["How many times are webhooks retried?", "How many retry attempts do webhook deliveries get?"],
    "postmortem": ["How soon must a postmortem be published?", "What is the deadline for publishing a postmortem?"],
    "lost_parcel": ["After how many days is a parcel declared lost?", "When does a missing parcel count as lost?"],
    "deletion": ["How fast must a data deletion request be completed?", "What is the deadline for handling a deletion request?"],
    "learning_budget": ["What is the yearly learning budget for engineers?", "How much can an engineer spend on learning per year?"],
    "dr_drill": ["How often are disaster recovery drills run?", "What is the frequency of DR drills?"],
    "idempotency": ["How do I avoid duplicate shipments when retrying a POST?", "Which header prevents duplicate creation on POST retries?"],
    "backoff": ["What should clients do when they receive HTTP 429?", "How should an API client react to a 429 response?"],
}
METRICS = ["rate limit", "first response time", "uptime commitment"]
TIERS = ["Basic", "Business", "Enterprise"]
SEVS = ["SEV1", "SEV2", "SEV3", "SEV4"]
RTO = ["Tier 1", "Tier 2", "Tier 3"]
POLITE = ["Could you tell me {q}", "Please answer this: {q}", "Quick question - {q}"]


def build_intents() -> dict[str, list[str]]:
    intents = {k: list(v) for k, v in STANDALONE.items()}
    for m in METRICS:                           # hard negatives: identical except for one word
        for t in TIERS:
            intents[f"{m}|{t}"] = [f"What is the {m} for the {t} plan?", f"Tell me the {m} of the {t} plan."]
    for s in SEVS:
        intents[f"ack|{s}"] = [f"How quickly must a {s} be acknowledged?", f"What is the acknowledgement deadline for a {s}?"]
    for t in RTO:
        intents[f"rto|{t}"] = [f"What is the RTO for a {t} service?", f"How long can a {t} service stay down (RTO)?"]
    return intents


def variants(phrasings: list[str], rng: random.Random) -> str:
    """Pick a phrasing: canonical, near-duplicate (case/punctuation/politeness), or true paraphrase."""
    kind = rng.choices(["canonical", "near_dup", "paraphrase"], [0.4, 0.4, 0.2])[0]
    base = phrasings[0] if kind != "paraphrase" else phrasings[-1]
    if kind == "near_dup":
        r = rng.random()
        if r < 0.34:
            return base.lower().rstrip("?.")
        if r < 0.67:
            return rng.choice(POLITE).format(q=base[0].lower() + base[1:])
        return base + " Thanks!"
    return base


def make_stream(n: int, seed: int = 11) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    intents = build_intents()
    keys = list(intents)
    weights = [1 / (i + 1) ** 0.8 for i in range(len(keys))]          # a few intents are asked far more often
    return [(k, variants(intents[k], rng)) for k in rng.choices(keys, weights, k=n)]


# ---------------------------------------------------------------- cache benchmark
def run_cache(stream, cache: ResponseCache | None, label: str) -> dict:
    clock = VirtualClock()
    if cache:
        cache.clock = clock
    gw = Gateway(MockProvider(clock), Store(), clock, cache=cache, config=GatewayConfig(rate_limit_per_min=1e9))
    ref_intent: dict[str, str] = {}
    from .providers import prompt_ref

    wrong = 0
    lat, cost = [], 0.0
    for intent, text in stream:
        ref_intent.setdefault(prompt_ref(text), intent)
        r = gw.chat(ChatRequest([{"role": "user", "content": text}]))
        lat.append(r.latency_ms)
        cost += r.cost_usd
        if r.cache != "miss":
            served_for = ref_intent.get(r.text.split("]")[0].replace("[ref:", ""))
            wrong += served_for != intent                      # answer was produced for a different question
    s = gw.store.summary()
    hits = s["cache_exact"] + s["cache_semantic"]
    return {"config": label, "requests": len(stream), "hit_rate": s["cache_hit_rate"], "exact": s["cache_exact"],
            "semantic": s["cache_semantic"], "wrong_hits": wrong,
            "wrong_hit_pct_of_hits": round(100 * wrong / hits, 1) if hits else 0.0,
            "cost_usd": round(cost, 5), "mean_latency_ms": round(statistics.mean(lat)),
            "p95_latency_ms": round(percentile(lat, 95)), "overhead_p95_ms": round(s["gateway_overhead_ms"]["p95"], 2)}


def cache_report(n: int = 600, embedder: str = "hashing", thresholds=(0.80, 0.85, 0.90, 0.95)) -> list[dict]:
    stream = make_stream(n)
    emb = FastEmbedAdapter() if embedder == "fastembed" else HashingEmbedder()
    rows = [run_cache(stream, None, "no cache"), run_cache(stream, ResponseCache(semantic_threshold=None), "exact only")]
    for th in thresholds:
        rows.append(run_cache(stream, ResponseCache(semantic_threshold=th, embedder=emb), f"exact + semantic >= {th} ({embedder})"))
    rows.append(run_cache(stream, ResponseCache(semantic_threshold=0.85, embedder=emb, number_guard=False),
                          f"exact + semantic >= 0.85 ({embedder}), NO number guard"))
    return rows


# ---------------------------------------------------------------- compaction benchmark
def run_compaction(budget: int | None, conversations: int = 20, turns: int = 14, seed: int = 5) -> dict:
    rng = random.Random(seed)
    clock = VirtualClock()
    gw = Gateway(MockProvider(clock), Store(), clock, cache=None, config=GatewayConfig(rate_limit_per_min=1e9))
    ttfts, in_tok, cost, saved, compacted_calls = [], 0, 0.0, 0, 0
    words = ("order invoice shipment customs carrier refund escalation region warehouse manifest delay tracking "
             "approval ticket backlog severity customer contract renewal").split()
    for _ in range(conversations):
        history = [{"role": "system", "content": "You are a logistics support assistant."}]
        for _t in range(turns):
            history.append({"role": "user", "content": " ".join(rng.choices(words, k=60)) + "?"})
            r = gw.chat(ChatRequest(list(history), compact_budget=budget, max_tokens=120))
            history.append({"role": "assistant", "content": r.text})
            ttfts.append(r.ttft_ms)
            in_tok += r.input_tokens
            cost += r.cost_usd
            saved += r.saved_tokens
            compacted_calls += r.compacted
    return {"budget": budget or "none", "calls": len(ttfts), "input_tokens": in_tok, "cost_usd": round(cost, 4),
            "ttft_p50_ms": round(percentile(ttfts, 50)), "ttft_p95_ms": round(percentile(ttfts, 95)),
            "compacted_calls": compacted_calls, "tokens_saved": saved}


def compaction_report() -> list[dict]:
    return [run_compaction(None), run_compaction(1500), run_compaction(800)]


# ---------------------------------------------------------------- markdown
def to_markdown() -> str:
    out = ["# llmscope benchmark (simulated model)\n", "## Response cache, 600 requests, 28 intents (12 standalone + 16 hard-negative siblings)\n"]
    for emb in ("hashing", "fastembed"):
        try:
            rows = cache_report(embedder=emb)
        except Exception as exc:                                  # fastembed not installed
            out.append(f"_{emb} embedder skipped: {exc}_\n")
            continue
        if emb == "hashing":
            out.append("| config | hit rate | exact | semantic | wrong hits | wrong % of hits | cost USD | mean latency ms | p95 latency ms | gateway overhead p95 ms |\n|---|---|---|---|---|---|---|---|---|---|")
            base = rows
        else:
            rows = rows[2:]                                       # baseline rows already shown
        for r in rows:
            out.append(f"| {r['config']} | {r['hit_rate']:.1%} | {r['exact']} | {r['semantic']} | {r['wrong_hits']} | {r['wrong_hit_pct_of_hits']}% | "
                       f"{r['cost_usd']} | {r['mean_latency_ms']} | {r['p95_latency_ms']} | {r['overhead_p95_ms']} |")
    out.append("\n## Context compaction, 20 conversations x 14 turns\n")
    out.append("| history budget (tokens) | input tokens | cost USD | TTFT p50 ms | TTFT p95 ms | calls compacted | tokens saved |\n|---|---|---|---|---|---|---|")
    base_in = None
    for r in compaction_report():
        base_in = base_in or r["input_tokens"]
        out.append(f"| {r['budget']} | {r['input_tokens']} ({r['input_tokens'] / base_in:.0%}) | {r['cost_usd']} | {r['ttft_p50_ms']} | "
                   f"{r['ttft_p95_ms']} | {r['compacted_calls']}/{r['calls']} | {r['tokens_saved']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(to_markdown())
