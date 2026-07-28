"""
Extract work arrangement for the manually annotated gold set.

This uses the versioned gold-only prompts in gold_arrangement_prompts.py and
writes a separate sidecar file. It does not modify the original benchmark
prompts or llm_extractions_gold.parquet.

Dry run:
    python src/llm/extract_gold_arrangement.py --dry-run

Full run:
    python src/llm/extract_gold_arrangement.py
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
from src.llm.gold_arrangement_prompts import (  # noqa: E402
    ARRANGEMENT_CLASSES,
    VARIANTS,
    build,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

FULL_OUTPUT = PROCESSED / "llm_extractions_gold_arrangement.parquet"
DRY_RUN_OUTPUT = (
    PROCESSED / "llm_extractions_gold_arrangement_dry_run.parquet"
)

LABEL_ALIASES = {
    "remote": "remote",
    "fully remote": "remote",
    "work from home": "remote",
    "work-from-home": "remote",
    "virtual": "remote",
    "hybrid": "hybrid",
    "on site": "onsite",
    "on-site": "onsite",
    "onsite": "onsite",
    "in office": "onsite",
    "in-office": "onsite",
    "office based": "onsite",
    "office-based": "onsite",
    "unclear": "unclear",
    "unknown": "unclear",
    "not specified": "unclear",
    "not stated": "unclear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two diagnostic postings across all four variants.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of postings.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANTS),
        help="Run only the selected variant. May be repeated.",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        type=int,
        help="Run only a specific job ID. May be repeated.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def load_gold_labels() -> list[dict]:
    path = PROCESSED / "gold_labels.json"

    with path.open() as file:
        payload = json.load(file)

    return payload["labels"] if isinstance(payload, dict) else payload


def load_postings() -> pd.DataFrame:
    return duckdb.connect().execute(
        f"""
        SELECT job_id, title, description, role_family
        FROM '{PROCESSED / "gold_seed.parquet"}'
        ORDER BY job_id
        """
    ).fetchdf()


def choose_dry_run_postings(
    postings: pd.DataFrame,
    gold_labels: list[dict],
) -> pd.DataFrame:
    """Choose one clear hybrid and one difficult unclear example."""

    gold_map = {
        int(row["job_id"]): row["work_arrangement"]
        for row in gold_labels
    }

    work = postings.copy()
    work["gold_work_arrangement"] = (
        work["job_id"].astype(int).map(gold_map)
    )

    hybrid = work[
        (work["gold_work_arrangement"] == "hybrid")
        & work["description"].str.contains(
            r"\bhybrid\b",
            case=False,
            na=False,
        )
    ].head(1)

    difficult_unclear = work[
        (work["gold_work_arrangement"] == "unclear")
        & work["description"].str.contains(
            r"onsite|on-site|remote|hybrid",
            case=False,
            na=False,
        )
    ].head(1)

    chosen = pd.concat(
        [hybrid, difficult_unclear],
        ignore_index=True,
    ).drop_duplicates("job_id")

    if len(chosen) < 2:
        fallback = work[
            work["gold_work_arrangement"].isin(["hybrid", "unclear"])
        ].drop_duplicates("gold_work_arrangement")

        chosen = pd.concat(
            [chosen, fallback],
            ignore_index=True,
        ).drop_duplicates("job_id").head(2)

    if len(chosen) != 2:
        raise RuntimeError(
            "Could not select two diagnostic dry-run postings."
        )

    return chosen.drop(columns=["gold_work_arrangement"])


def normalise_arrangement(value) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if not text:
        return None

    if text in ARRANGEMENT_CLASSES:
        return text

    return LABEL_ALIASES.get(text)


def normalise_response(
    job_id: int,
    variant: str,
    response,
) -> dict:
    parsed = response.parsed or {}

    raw_label = parsed.get("work_arrangement")
    label = normalise_arrangement(raw_label)

    evidence = parsed.get("evidence_work_arrangement")

    if not evidence:
        stage1 = parsed.get("stage1_evidence")

        if isinstance(stage1, dict):
            evidence = stage1.get(
                "role_specific_arrangement_text"
            )

    # An unclear label means no role-specific arrangement evidence exists.
    # Discard generic company, facility, or policy text that the model may quote.
    if label == "unclear":
        evidence = None

    validation_error = None

    if response.parsed is None:
        validation_error = response.parse_error or "Invalid JSON"
    elif label is None:
        validation_error = (
            f"Invalid work_arrangement value: {raw_label!r}"
        )

    return {
        "job_id": int(job_id),
        "variant": variant,
        "work_arrangement": label,
        "evidence_work_arrangement": evidence or None,
        "valid_json": response.parsed is not None,
        "valid_arrangement": label in ARRANGEMENT_CLASSES,
        "parse_error": response.parse_error,
        "validation_error": validation_error,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_s": response.latency_s,
        "cost": response.cost,
        "cached": response.cached,
        "attempts": response.attempts,
        "raw_text": response.text[:2000],
    }


def main() -> None:
    args = parse_args()

    postings = load_postings()
    gold_labels = load_gold_labels()

    if args.job_id:
        requested = set(args.job_id)

        postings = postings[
            postings["job_id"].astype(int).isin(requested)
        ]

        missing = requested - set(
            postings["job_id"].astype(int)
        )

        if missing:
            raise SystemExit(
                f"Job IDs not found: {sorted(missing)}"
            )

    elif args.dry_run:
        postings = choose_dry_run_postings(
            postings,
            gold_labels,
        )

    elif args.limit is not None:
        postings = postings.head(args.limit)

    variants = args.variant or list(VARIANTS)

    output_path = (
        args.output
        if args.output is not None
        else DRY_RUN_OUTPUT
        if args.dry_run
        else FULL_OUTPUT
    )

    print(f"Postings: {len(postings)}")
    print(f"Variants: {variants}")
    print(f"Expected calls: {len(postings) * len(variants)}")
    print(f"Output: {output_path}")

    client = LLMClient()
    rows: list[dict] = []

    def run_one(record, variant: str) -> dict | None:
        prompt = build(
            variant,
            record.title,
            record.description,
        )

        try:
            response = client.complete(
                prompt,
                max_tokens=250,
            )

            return normalise_response(
                record.job_id,
                variant,
                response,
            )

        except Exception as exc:  # noqa: BLE001
            print(
                f"FAILED job={record.job_id} "
                f"variant={variant}: {exc}"
            )
            return None

    tasks = [
        (record, variant)
        for variant in variants
        for record in postings.itertuples(index=False)
    ]

    started = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as pool:
        futures = {
            pool.submit(run_one, record, variant): (
                int(record.job_id),
                variant,
            )
            for record, variant in tasks
        }

        for future in as_completed(futures):
            result = future.result()

            if result is not None:
                rows.append(result)

    expected_rows = len(tasks)

    if len(rows) != expected_rows:
        raise SystemExit(
            f"Incomplete extraction: received {len(rows)} "
            f"of {expected_rows} rows. "
            "Nothing was written; successful calls remain cached."
        )

    output = (
        pd.DataFrame(rows)
        .sort_values(["variant", "job_id"])
        .reset_index(drop=True)
    )

    duplicates = output.duplicated(
        ["job_id", "variant"]
    )

    if duplicates.any():
        raise SystemExit(
            "Duplicate (job_id, variant) pairs found. "
            "Nothing was written."
        )

    if (
        output_path.exists()
        and not args.dry_run
    ):
        existing_rows = len(
            pd.read_parquet(output_path)
        )

        if len(output) < existing_rows:
            raise SystemExit(
                f"Refusing to replace {existing_rows} rows "
                f"with {len(output)} rows."
            )

    output.to_parquet(
        output_path,
        index=False,
    )

    elapsed = time.perf_counter() - started

    print("\nExtraction summary")

    summary = output.groupby("variant").agg(
        rows=("job_id", "count"),
        valid_json=("valid_json", "mean"),
        valid_arrangement=("valid_arrangement", "mean"),
        cached=("cached", "mean"),
        cost=("cost", "sum"),
        mean_latency_s=("latency_s", "mean"),
    ).round(4)

    print(summary.to_string())
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Session cost: ${output['cost'].sum():.4f}")
    print(f"Wrote: {output_path}")

    if args.dry_run:
        gold_map = {
            int(row["job_id"]): row["work_arrangement"]
            for row in gold_labels
        }

        review = output[
            [
                "job_id",
                "variant",
                "work_arrangement",
                "evidence_work_arrangement",
                "valid_arrangement",
            ]
        ].copy()

        review.insert(
            2,
            "gold_work_arrangement",
            review["job_id"].map(gold_map),
        )

        print("\nDry-run predictions")
        print(review.to_string(index=False))


if __name__ == "__main__":
    main()
