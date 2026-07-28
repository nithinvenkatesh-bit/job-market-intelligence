"""Export sanitized evaluation outputs for the public Streamlit dashboard.

The source CSV and Parquet files under data/processed are intentionally ignored
by Git. This script converts the dashboard inputs into compact JSON artifacts
that can be committed and deployed without raw job descriptions or API output.

Run:
    python dashboard/export_data.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "dashboard" / "data"

METHOD_ORDER = [
    "rules",
    "zero_shot",
    "few_shot",
    "schema_rules",
    "decomposed",
]

REQUIRED_INPUTS = {
    "method_summary": PROCESSED / "gold_method_summary.csv",
    "pairwise_tests": PROCESSED / "gold_pairwise_tests.csv",
    "item_scores": PROCESSED / "gold_item_scores.parquet",
    "skill_errors": PROCESSED / "gold_skill_errors.csv",
    "llm_extractions": PROCESSED / "llm_extractions_gold.parquet",
}


def json_safe(value: Any) -> Any:
    """Convert pandas/NumPy values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if pd.isna(value):
        return None

    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(column): json_safe(value)
            for column, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]


def write_json(name: str, payload: Any) -> None:
    path = OUTPUT / name
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {path.relative_to(ROOT)}")


def validate_inputs() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED_INPUTS.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required dashboard inputs:\n- "
            + "\n- ".join(missing)
        )


def method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def build_llm_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize reliability, caching, latency, tokens, and model cost.

    Recorded run cost reflects incremental spend for this particular run.
    Estimated model cost prices every row from its token usage, including
    cached responses, to represent the cost of a fresh uncached run.
    """

    input_cost_per_mtok = 1.00
    output_cost_per_mtok = 5.00

    llm = df.copy()

    llm["variant"] = llm["variant"].astype(str)
    llm["valid_json"] = (
        llm["valid_json"]
        .fillna(False)
        .astype(bool)
    )
    llm["cached"] = (
        llm["cached"]
        .fillna(False)
        .astype(bool)
    )

    llm["estimated_model_cost_usd"] = (
        llm["input_tokens"].fillna(0)
        / 1_000_000
        * input_cost_per_mtok
        + llm["output_tokens"].fillna(0)
        / 1_000_000
        * output_cost_per_mtok
    )

    rows: list[dict[str, Any]] = []

    for variant, group in llm.groupby(
        "variant",
        sort=False,
    ):
        uncached = group.loc[~group["cached"]]

        rows.append(
            {
                "variant": variant,
                "n_predictions": int(len(group)),
                "actual_api_calls": int(len(uncached)),
                "cached_rows": int(group["cached"].sum()),
                "cache_rate": float(group["cached"].mean()),
                "valid_json_rate": float(
                    group["valid_json"].mean()
                ),
                "mean_uncached_latency_s": (
                    float(uncached["latency_s"].mean())
                    if not uncached.empty
                    else np.nan
                ),
                "median_uncached_latency_s": (
                    float(uncached["latency_s"].median())
                    if not uncached.empty
                    else np.nan
                ),
                "total_input_tokens": int(
                    group["input_tokens"].fillna(0).sum()
                ),
                "total_output_tokens": int(
                    group["output_tokens"].fillna(0).sum()
                ),
                "recorded_run_cost_usd": float(
                    group["cost"].fillna(0).sum()
                ),
                "estimated_model_cost_usd": float(
                    group["estimated_model_cost_usd"].sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["estimated_cost_per_1k_predictions"] = (
        summary["estimated_model_cost_usd"]
        / summary["n_predictions"]
        * 1000
    )

    summary["method_rank"] = (
        summary["variant"].map(method_rank)
    )
    summary = (
        summary
        .sort_values("method_rank")
        .drop(columns="method_rank")
    )

    return summary


def main() -> None:
    validate_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    method_summary = pd.read_csv(
        REQUIRED_INPUTS["method_summary"]
    )
    pairwise = pd.read_csv(
        REQUIRED_INPUTS["pairwise_tests"]
    )
    item_scores = pd.read_parquet(
        REQUIRED_INPUTS["item_scores"]
    )
    skill_errors = pd.read_csv(
        REQUIRED_INPUTS["skill_errors"]
    )
    llm_extractions = pd.read_parquet(
        REQUIRED_INPUTS["llm_extractions"]
    )

    method_summary["method_rank"] = (
        method_summary["method"].map(method_rank)
    )
    method_summary = (
        method_summary
        .sort_values("method_rank")
        .drop(columns="method_rank")
    )

    pairwise["method_rank"] = pairwise["method"].map(method_rank)
    pairwise = (
        pairwise
        .sort_values(["field", "metric", "method_rank"])
        .drop(columns="method_rank")
    )

    item_scores["method_rank"] = item_scores["method"].map(method_rank)
    item_scores = (
        item_scores
        .sort_values(["job_id", "method_rank"])
        .drop(columns="method_rank")
    )

    skill_errors["method_rank"] = skill_errors["method"].map(method_rank)
    skill_errors = (
        skill_errors
        .sort_values(["job_id", "method_rank"])
        .drop(columns="method_rank")
    )

    llm_operations = build_llm_operations(llm_extractions)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_postings": int(item_scores["job_id"].nunique()),
        "methods": METHOD_ORDER,
        "item_score_rows": int(len(item_scores)),
        "pairwise_test_rows": int(len(pairwise)),
        "skill_error_rows": int(len(skill_errors)),
        "notes": [
            "Gold evaluation uses 80 manually reviewed postings.",
            "The deterministic work-arrangement baseline is gold-informed and diagnostic.",
            "One malformed schema-rules arrangement response is counted as InvalidPrediction.",
            "No raw job descriptions or API responses are included in dashboard exports.",
        ],
    }

    write_json("metadata.json", metadata)
    write_json("method_summary.json", records(method_summary))
    write_json("pairwise_tests.json", records(pairwise))
    write_json("item_scores.json", records(item_scores))
    write_json("skill_errors.json", records(skill_errors))
    write_json("llm_operations.json", records(llm_operations))

    print("\nDashboard export complete.")
    print(f"Gold postings: {metadata['gold_postings']}")
    print(f"Methods: {len(METHOD_ORDER)}")
    print(f"Item-score rows: {len(item_scores)}")


if __name__ == "__main__":
    main()
