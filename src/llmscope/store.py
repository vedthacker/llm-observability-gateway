"""SQLite store for request records and spans, plus the aggregation the dashboard needs."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, ts REAL, tenant TEXT, model TEXT, status TEXT,
    cache TEXT, input_tokens INTEGER, output_tokens INTEGER, saved_tokens INTEGER, cost_usd REAL,
    saved_cost_usd REAL, ttft_ms REAL, latency_ms REAL, tokens_per_sec REAL, overhead_ms REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT, span_id TEXT, parent_id TEXT, name TEXT, start_ts REAL, end_ts REAL, status TEXT, attrs TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (what most dashboards mean by p95)."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(-(-p / 100 * len(s) // 1)) - 1))
    return s[k]


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript(SCHEMA)

    def add_request(self, **r) -> None:
        cols = ",".join(r)
        with self._lock:
            self._con.execute(f"INSERT INTO requests ({cols}) VALUES ({','.join('?' * len(r))})", tuple(r.values()))
            self._con.commit()

    def add_span(self, s: dict) -> None:
        with self._lock:
            self._con.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)",
                              (s["trace_id"], s["span_id"], s["parent_id"], s["name"], s["start_ts"], s["end_ts"],
                               s["status"], json.dumps(s["attrs"])))
            self._con.commit()

    def spans(self, trace_id: str) -> list[dict]:
        with self._lock:
            rows = self._con.execute("SELECT * FROM spans WHERE trace_id=? ORDER BY start_ts, rowid", (trace_id,)).fetchall()
        return [{**dict(r), "attrs": json.loads(r["attrs"])} for r in rows]

    def recent(self, limit: int = 25) -> list[dict]:
        with self._lock:
            rows = self._con.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def spend_since(self, tenant: str, since_ts: float) -> float:
        with self._lock:
            return self._con.execute("SELECT COALESCE(SUM(cost_usd),0) FROM requests WHERE tenant=? AND ts>=?",
                                     (tenant, since_ts)).fetchone()[0]

    def summary(self) -> dict:
        with self._lock:
            rows = [dict(r) for r in self._con.execute("SELECT * FROM requests").fetchall()]
        ok = [r for r in rows if r["status"] == "ok"]
        live = [r for r in ok if r["cache"] == "miss"]            # requests that actually reached a model
        hits = [r for r in ok if r["cache"] in ("exact", "semantic")]

        def col(rs, k):
            return [r[k] for r in rs if r[k] is not None]

        by_model: dict[str, dict] = {}
        for r in ok:
            m = by_model.setdefault(r["model"], {"requests": 0, "cost_usd": 0.0, "output_tokens": 0})
            m["requests"] += 1
            m["cost_usd"] = round(m["cost_usd"] + r["cost_usd"], 6)
            m["output_tokens"] += r["output_tokens"]
        return {
            "requests": len(rows), "errors": sum(r["status"] != "ok" for r in rows),
            "rejected": sum(r["status"] in ("budget_exceeded", "rate_limited") for r in rows),
            "cache_hit_rate": round(len(hits) / len(ok), 4) if ok else 0.0,
            "cache_exact": sum(r["cache"] == "exact" for r in ok), "cache_semantic": sum(r["cache"] == "semantic" for r in ok),
            "input_tokens": sum(r["input_tokens"] for r in ok), "output_tokens": sum(r["output_tokens"] for r in ok),
            "tokens_saved": sum(r["saved_tokens"] for r in ok),
            "cost_usd": round(sum(r["cost_usd"] for r in ok), 6),
            "cost_saved_usd": round(sum(r["saved_cost_usd"] for r in ok), 6),
            "latency_ms": {"p50": percentile(col(ok, "latency_ms"), 50), "p95": percentile(col(ok, "latency_ms"), 95)},
            "ttft_ms": {"p50": percentile(col(live, "ttft_ms"), 50), "p95": percentile(col(live, "ttft_ms"), 95)},
            "tokens_per_sec": {"p50": percentile(col(live, "tokens_per_sec"), 50), "p5": percentile(col(live, "tokens_per_sec"), 5)},
            "gateway_overhead_ms": {"p50": percentile(col(ok, "overhead_ms"), 50), "p95": percentile(col(ok, "overhead_ms"), 95)},
            "by_model": by_model,
            "avg_cost_per_request_usd": round(sum(r["cost_usd"] for r in ok) / len(ok), 6) if ok else 0.0,
            "latency_histogram": _histogram(col(ok, "latency_ms")),
        }


def _histogram(values: list[float], bins: int = 12) -> list[dict]:
    if not values:
        return []
    hi = max(values)
    width = max(hi / bins, 1.0)
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int(v / width))] += 1
    return [{"from_ms": round(i * width), "to_ms": round((i + 1) * width), "count": c} for i, c in enumerate(counts)]
