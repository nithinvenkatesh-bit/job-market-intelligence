"""
Run all prompt variants over the same postings and record everything.

Design decisions that matter:

  PAIRED DESIGN -- every variant sees the SAME postings. Comparing methods
  on different samples would confound prompt quality with sampling noise;
  two independent 2,000-row baseline draws already showed ~1.3pp of drift.
  Pairing removes that and makes McNemar's test applicable.

  PINNED SAMPLE -- the exact job_ids are stored in
  config/experiment_sample.json rather than re-drawn each run. See
  load_sample() for why this is not optional.

  CACHING -- reruns are free. If this crashes at posting 300, rerunning
  replays the first 300 from cache and only pays for the rest.

  CONCURRENCY -- 1,600 sequential calls at ~1s each is 27 minutes. Eight
  workers brings it under 5. The client's retry-with-jitter handles the
  rate limits this provokes.

  EVERYTHING RECORDED -- tokens, latency, cost, attempts, valid-JSON status.
  The finding this project exists to produce is a cost/quality tradeoff, so
  cost and latency are first-class results, not afterthoughts.

Run:  python src/llm/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.client import LLMClient  # noqa: E402
from src.llm.prompts import VARIANTS, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"

MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def load_sample(dataset: str, limit: int) -> pd.DataFrame:
    """Load the experiment sample.

    Prefers an explicit manifest at config/experiment_sample.json over
    drawing a fresh sample.

    This exists because of a real and expensive bug: DuckDB's setseed() did
    NOT make `ORDER BY random()` reproducible across sessions. Every run
    therefore drew a different 400 postings, which silently invalidated the
    LLM response cache and re-billed for extractions already paid for.

    A pinned, version-controlled sample is the right answer regardless of
    that bug. It makes results reproducible for anyone cloning the repo, and
    it means the pipeline can be re-run any number of times at zero cost.

    The fallback path orders by hash(job_id) rather than random(): still
    pseudo-random, but stable across sessions and machines.
    """
    manifest = CONFIG / "experiment_sample.json"

    if manifest.exists():
        pinned = json.loads(manifest.read_text()).get(dataset)
        if pinned:
            id_list = ",".join(str(i) for i in pinned)
            df = duckdb.connect().execute(f"""
                SELECT job_id, title, description, stratum, pay_bucket
                FROM '{PROCESSED / f"{dataset}.parquet"}'
                WHERE job_id IN ({id_list})
                ORDER BY job_id
            """).fetchdf()
            print(f"Using pinned sample: {len(df)} of {len(pinned)} postings found")
            if len(df) < len(pinned):
                print("  WARNING: some pinned ids are missing from the dataset.")
                print("  The dataset was probably rebuilt with different sampling.")
            return df

    print(f"No manifest at {manifest}; drawing a deterministic sample.")
    return duckdb.connect().execute(f"""
        WITH ranked AS (
            SELECT job_id, title, description, stratum, pay_bucket,
                   row_number() OVER (PARTITION BY stratum, pay_bucket
                                      ORDER BY hash(job_id)) AS rn,
                   COUNT(*)   OVER (PARTITION BY stratum, pay_bucket) AS cell_n,
                   COUNT(*)   OVER ()                                 AS total_n
            FROM '{PROCESSED / f"{dataset}.parquet"}'
        )
        SELECT job_id, title, description, stratum, pay_bucket
        FROM ranked
        WHERE rn <= GREATEST(1, CAST(ROUND({limit} * cell_n / total_n) AS INTEGER))
        ORDER BY job_id
    """).fetchdf()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _as_list(value) -> list[str]:
    """Models occasionally return a comma-joined string instead of a list."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [p.strip() for p in value.split(",") if p.strip()]
    return []


def _as_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def normalise(job_id: int, variant: str, response) -> dict:
    """Flatten one LLM response into the same shape as baseline output.

    Matching the baseline schema is what allows one evaluation script to
    score both methods without special-casing either.
    """
    parsed = response.parsed or {}
    evidence = parsed.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}

    return {
        "job_id": job_id,
        "variant": variant,
        "salary_min": _as_number(parsed.get("salary_min")),
        "salary_max": _as_number(parsed.get("salary_max")),
        "pay_period": (str(parsed["pay_period"]).upper()
                       if parsed.get("pay_period") else None),
        "seniority": parsed.get("seniority") or None,
        "years_experience_min": _as_number(parsed.get("years_experience_min")),
        "required_skills": _as_list(parsed.get("required_skills")),
        "preferred_skills": _as_list(parsed.get("preferred_skills")),
        "n_required": len(_as_list(parsed.get("required_skills"))),
        "n_preferred": len(_as_list(parsed.get("preferred_skills"))),
        "evidence_salary": evidence.get("salary") or None,
        "evidence_seniority": evidence.get("seniority") or None,
        "evidence_years": evidence.get("years_experience") or None,
        # Reliability + cost telemetry
        "valid_json": response.parsed is not None,
        "parse_error": response.parse_error,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_s": response.latency_s,
        "cost": response.cost,
        "cached": response.cached,
        "attempts": response.attempts,
        "raw_text": response.text[:2000],  # kept for error analysis
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_variant(client: LLMClient, variant: str, sample: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    failures = 0

    def one(record) -> dict | None:
        prompt = build(variant, record.title, record.description)
        try:
            resp = client.complete(prompt, max_tokens=1200)
            return normalise(record.job_id, variant, resp)
        except Exception as exc:  # noqa: BLE001 -- a dead call is a result too
            print(f"    job {record.job_id} failed: {exc}")
            return None

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(one, r): r.job_id
                   for r in sample.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result is None:
                failures += 1
            else:
                rows.append(result)
            if i % 50 == 0:
                print(f"    {i}/{len(sample)}  (${client.total_cost:.3f} so far)")

    elapsed = time.perf_counter() - started
    valid = sum(r["valid_json"] for r in rows)
    print(f"  {variant}: {len(rows)} ok, {failures} failed, "
          f"{valid}/{len(rows)} valid JSON, {elapsed:.0f}s")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark",
                        choices=["benchmark", "holdout"])
    parser.add_argument("--limit", type=int, default=400,
                        help="postings per variant (ignored if a manifest exists)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate cost without calling the API")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting a larger existing results file")
    args = parser.parse_args()

    sample = load_sample(args.dataset, args.limit)
    print(f"Sample: {len(sample)} postings from {args.dataset}")
    print(sample.groupby(["stratum", "pay_bucket"]).size().to_string())

    if args.dry_run:
        # ~4 chars per token is the standard rough estimate.
        est_in = sum(len(build(v, r.title, r.description)) / 4
                     for v in args.variants
                     for r in sample.itertuples(index=False))
        est_out = len(sample) * len(args.variants) * 250
        cost = est_in / 1e6 * 1.00 + est_out / 1e6 * 5.00
        print(f"\nEstimated: {est_in:,.0f} input + {est_out:,.0f} output tokens")
        print(f"Estimated cost: ${cost:.2f}")
        print("(Cached responses are free, so actual cost will be lower.)")
        return

    client = LLMClient()
    all_rows: list[dict] = []

    for variant in args.variants:
        print(f"\n=== {variant} ===")
        all_rows.extend(run_variant(client, variant, sample))

    out = pd.DataFrame(all_rows)
    path = PROCESSED / f"llm_extractions_{args.dataset}.parquet"

    # Refuse to shrink an existing results file. A partial run once replaced
    # 1,600 paid-for extractions with 120 because the benchmark had been
    # rebuilt underneath it; a printed warning was not enough.
    if path.exists() and not args.force:
        existing = len(pd.read_parquet(path))
        if len(out) < existing:
            print(f"\nREFUSING TO WRITE: {len(out)} rows would replace {existing}.")
            print("Nothing written. Rebuild the benchmark so the pinned sample")
            print("is present, or pass --force if the shrink is intended.")
            return

    out.to_parquet(path, index=False)

    print(f"\n{'=' * 62}")
    print(client.summary())
    print("\nPer-variant summary")
    print(out.groupby("variant").agg(
        n=("job_id", "count"),
        valid_json=("valid_json", "mean"),
        mean_in_tok=("input_tokens", "mean"),
        mean_out_tok=("output_tokens", "mean"),
        mean_latency=("latency_s", "mean"),
        total_cost=("cost", "sum"),
    ).round(3).to_string())

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()