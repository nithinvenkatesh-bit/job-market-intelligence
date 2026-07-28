"""
Run the LLM extractors over the hand-annotated gold set.

The main experiment ran on the benchmark sample, which barely overlaps the
gold set (1 posting of 80). So the gold postings need their own extraction
pass before anything can be scored against them.

All four variants are run, not just the winner. The original prompt
experiment could only measure salary and seniority -- the fields with
provider-supplied reference labels. Skills, work arrangement and years of
experience were unmeasurable until now. Running all four tests whether
schema_rules still leads on fields its rules were never written for, which
is a stronger claim than winning on the fields it was tuned against.

Cost: ~320 calls, roughly $0.90. Cached, so re-runs are free.

Run:  python src/llm/extract_gold.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.client import LLMClient  # noqa: E402
from src.llm.prompts import VARIANTS, build  # noqa: E402
from src.llm.run_experiment import normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

MAX_WORKERS = 8


def main() -> None:
    con = duckdb.connect()
    sample = con.execute(f"""
        SELECT job_id, title, description, role_family
        FROM '{PROCESSED / "gold_seed.parquet"}'
        ORDER BY job_id
    """).fetchdf()

    print(f"Gold set: {len(sample)} postings")
    print(sample.role_family.value_counts().to_string())

    client = LLMClient()
    rows: list[dict] = []

    for variant in VARIANTS:
        print(f"\n=== {variant} ===")

        def one(record):
            prompt = build(variant, record.title, record.description)
            try:
                return normalise(record.job_id, variant,
                                 client.complete(prompt, max_tokens=1200))
            except Exception as exc:  # noqa: BLE001
                print(f"    job {record.job_id} failed: {exc}")
                return None

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(one, r) for r in sample.itertuples(index=False)]
            results = [f.result() for f in as_completed(futures)]

        ok = [r for r in results if r is not None]
        rows.extend(ok)
        print(f"  {len(ok)}/{len(sample)} ok, "
              f"{sum(r['valid_json'] for r in ok)} valid JSON, "
              f"{time.perf_counter() - started:.0f}s, "
              f"${client.total_cost:.3f} so far")

    out = pd.DataFrame(rows)
    path = PROCESSED / "llm_extractions_gold.parquet"

    # Same guard as the main experiment: never shrink a results file.
    if path.exists():
        existing = len(pd.read_parquet(path))
        if len(out) < existing:
            print(f"\nREFUSING TO WRITE: {len(out)} rows would replace {existing}.")
            return

    out.to_parquet(path, index=False)

    print(f"\n{'=' * 60}")
    print(client.summary())
    print("\nPer-variant")
    print(out.groupby("variant").agg(
        n=("job_id", "count"),
        valid_json=("valid_json", "mean"),
        mean_required=("n_required", "mean"),
        mean_preferred=("n_preferred", "mean"),
        cost=("cost", "sum"),
    ).round(3).to_string())
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()