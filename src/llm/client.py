"""
Provider-agnostic LLM client with caching, retries, and cost tracking.

Three things here exist because of how the experiment is structured:

  CACHING -- responses are cached on disk keyed by (model, prompt). Prompt
  iteration means re-running the same postings many times; without a cache
  you pay for every rerun. With one, only genuinely new calls cost money.

  RETRIES -- rate limits and overload errors are normal at concurrency, not
  exceptional. Exponential backoff with jitter prevents a burst of retries
  from synchronising into another burst.

  COST + LATENCY -- these are reported metrics, not incidentals. "Method B
  is 2 points better but 1.5x the cost" is the finding this project exists
  to produce, so per-call tokens and wall time are recorded on every call.

REVISION NOTE:
  The first version shared ONE sqlite3 connection across an 8-thread pool.
  check_same_thread=False permits cross-thread use but does not make
  concurrent writes safe, and ~2% of calls died with "cannot commit - no
  transaction is active" and "bad parameter or other API misuse". Those
  were SQLite errors wearing an API-error costume: paid-for responses were
  being thrown away by a caching bug. Fixed with thread-local connections
  plus WAL journaling, and cache errors are now non-fatal -- a cache is an
  optimisation, so its failure should cost speed, never correctness.

Run:  python src/llm/client.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIStatusError, RateLimitError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MTOK = 1.00
OUTPUT_COST_PER_MTOK = 5.00

MAX_RETRIES = 5
BASE_BACKOFF_S = 2.0


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """One call's result plus everything needed to evaluate it."""

    text: str
    parsed: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False
    parse_error: str | None = None
    attempts: int = 1

    @property
    def cost(self) -> float:
        """Cached calls cost nothing -- important for honest cost reporting."""
        if self.cached:
            return 0.0
        return (self.input_tokens / 1e6 * INPUT_COST_PER_MTOK
                + self.output_tokens / 1e6 * OUTPUT_COST_PER_MTOK)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Models wrap JSON in markdown fences even when told not to. Observed on the
# very first test call, so this is handled rather than assumed away.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed_dict, error). Never raises.

    A parse failure is a real result worth counting -- valid-JSON rate is one
    of the reliability metrics -- so failures are returned, not thrown.
    """
    cleaned = _FENCE.sub("", text).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, f"expected object, got {type(obj).__name__}"
    except json.JSONDecodeError as exc:
        first_error = str(exc)

    # Fallback: grab the outermost {...} in case of surrounding prose.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass

    return None, first_error


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """SQLite cache keyed by a hash of (model, prompt).

    Thread-local connections: a single sqlite3 connection cannot safely take
    concurrent writes from multiple threads, even with check_same_thread
    disabled. Each thread gets its own connection instead, and WAL journaling
    lets readers and writers proceed without blocking each other.

    SQLite rather than a dict so the cache survives between runs, and rather
    than one file per response so it stays a single tidy artifact.
    """

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()

        # Create the schema once on a throwaway connection.
        setup = sqlite3.connect(path)
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key           TEXT PRIMARY KEY,
                response      TEXT NOT NULL,
                input_tokens  INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at    REAL NOT NULL
            )
        """)
        setup.commit()
        setup.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """One connection per thread, created lazily.

        timeout=30 lets a blocked writer wait for the lock instead of
        failing immediately.
        """
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.path, timeout=30)
        return self._local.conn

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def get(self, key: str) -> tuple[str, int, int] | None:
        row = self.conn.execute(
            "SELECT response, input_tokens, output_tokens FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        return row if row else None

    def put(self, key: str, response: str, in_tok: int, out_tok: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
            (key, response, in_tok, out_tok, time.time()),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        n, tin, tout = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0) FROM cache"
        ).fetchone()
        return {"entries": n, "input_tokens": tin, "output_tokens": tout}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, model: str = MODEL, use_cache: bool = True,
                 cache_path: Path | None = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Check .env in the project root.")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.cache = (ResponseCache(cache_path or PROCESSED / "llm_cache.sqlite")
                      if use_cache else None)

        # Running totals for the session summary. Guarded because the thread
        # pool increments them concurrently.
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.total_cost = 0.0
        self.errors: list[str] = []

    def complete(self, prompt: str, max_tokens: int = 800) -> LLMResponse:
        """One completion, with cache lookup and retry on transient failures."""
        key = ResponseCache.make_key(self.model, prompt) if self.cache else None

        if key:
            try:
                hit = self.cache.get(key)
            except sqlite3.Error:
                hit = None  # a cache miss is never worth failing a call over
            if hit:
                with self._lock:
                    self.cache_hits += 1
                text, in_tok, out_tok = hit
                parsed, err = parse_json(text)
                return LLMResponse(text, parsed, in_tok, out_tok,
                                   latency_s=0.0, cached=True, parse_error=err)

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                started = time.perf_counter()
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0,  # deterministic: measuring prompts, not sampling
                    messages=[{"role": "user", "content": prompt}],
                )
                latency = time.perf_counter() - started

                text = resp.content[0].text
                in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens

                if key:
                    try:
                        self.cache.put(key, text, in_tok, out_tok)
                    except sqlite3.Error as exc:
                        # Non-fatal by design: we already paid for this
                        # response, so a cache failure must not discard it.
                        print(f"    cache write failed (non-fatal): {exc}")

                parsed, err = parse_json(text)
                result = LLMResponse(text, parsed, in_tok, out_tok, latency,
                                     parse_error=err, attempts=attempt)

                with self._lock:
                    self.calls += 1
                    self.total_cost += result.cost
                return result

            except (RateLimitError, APIStatusError) as exc:
                last_error = exc
                # Jitter matters: without it, concurrent workers that hit a
                # rate limit together retry together and hit it again.
                delay = BASE_BACKOFF_S * (2 ** (attempt - 1)) + random.uniform(0, 1)
                if attempt < MAX_RETRIES:
                    print(f"    retry {attempt}/{MAX_RETRIES} in {delay:.1f}s "
                          f"({type(exc).__name__})")
                    time.sleep(delay)

        with self._lock:
            self.errors.append(str(last_error))
        raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_error}")

    def summary(self) -> str:
        cache_info = f", {self.cache_hits} cache hits" if self.cache else ""
        return (f"{self.calls} API calls{cache_info} | "
                f"${self.total_cost:.4f} | {len(self.errors)} errors")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import duckdb

    client = LLMClient()

    row = duckdb.connect().execute(f"""
        SELECT job_id, title, description
        FROM '{PROCESSED / "benchmark.parquet"}'
        WHERE stratum = 'labeled_stated' AND desc_len BETWEEN 1500 AND 3000
        LIMIT 1
    """).fetchdf().iloc[0]

    prompt = (
        'Return ONLY valid JSON: {"salary_min": number|null, "pay_period": '
        '"HOURLY"|"WEEKLY"|"MONTHLY"|"YEARLY"|null}\n\n'
        f"POSTING:\n{row.description}"
    )

    print("First call")
    r1 = client.complete(prompt)
    print(f"  parsed : {r1.parsed}")
    print(f"  cached : {r1.cached}  latency {r1.latency_s:.2f}s  cost ${r1.cost:.5f}")

    print("\nSecond identical call (expect a cache hit)")
    r2 = client.complete(prompt)
    print(f"  parsed : {r2.parsed}")
    print(f"  cached : {r2.cached}  latency {r2.latency_s:.2f}s  cost ${r2.cost:.5f}")

    print("\nFence-stripping check")
    for sample in ['```json\n{"a": 1}\n```', '{"b": 2}', 'Sure!\n{"c": 3}', 'not json']:
        parsed, err = parse_json(sample)
        print(f"  {sample[:24]!r:30s} -> {parsed} {f'({err[:30]})' if err else ''}")

    print("\nConcurrent cache check (8 threads)")
    from concurrent.futures import ThreadPoolExecutor

    def hammer(n: int) -> bool:
        try:
            k = ResponseCache.make_key("test-model", f"prompt-{n}")
            client.cache.put(k, f'{{"v": {n}}}', 10, 5)
            return client.cache.get(k) is not None
        except sqlite3.Error:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = sum(pool.map(hammer, range(200)))
    print(f"  200 concurrent writes: {ok}/200 succeeded")

    print(f"\n{client.summary()}")
    print(f"cache: {client.cache.stats()}")