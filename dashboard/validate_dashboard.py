"""Validate dashboard exports against their source evaluation outputs.

Run:
    python dashboard/validate_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA = ROOT / "dashboard" / "data"
PROCESSED = ROOT / "data" / "processed"

METHODS = [
    "rules",
    "zero_shot",
    "few_shot",
    "schema_rules",
    "decomposed",
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        failures.append(message)


def load_json(name: str):
    return json.loads(
        (DASHBOARD_DATA / name).read_text(encoding="utf-8")
    )


def numeric_equal(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = 1e-10,
) -> bool:
    return np.allclose(
        pd.to_numeric(left, errors="coerce"),
        pd.to_numeric(right, errors="coerce"),
        equal_nan=True,
        atol=atol,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Load source and exported data
# ---------------------------------------------------------------------------

metadata = load_json("metadata.json")
method_export = pd.DataFrame(load_json("method_summary.json"))
pairwise_export = pd.DataFrame(load_json("pairwise_tests.json"))
item_export = pd.DataFrame(load_json("item_scores.json"))
operations_export = pd.DataFrame(load_json("llm_operations.json"))

method_source = pd.read_csv(
    PROCESSED / "gold_method_summary.csv"
)
pairwise_source = pd.read_csv(
    PROCESSED / "gold_pairwise_tests.csv"
)
item_source = pd.read_parquet(
    PROCESSED / "gold_item_scores.parquet"
)
llm_source = pd.read_parquet(
    PROCESSED / "llm_extractions_gold.parquet"
)


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

check(
    metadata["gold_postings"] == 80,
    "Metadata reports 80 gold postings",
)

check(
    metadata["item_score_rows"] == 400,
    "Metadata reports 400 method-posting rows",
)

check(
    metadata["pairwise_test_rows"] == 24,
    "Metadata reports 24 paired comparisons",
)

check(
    metadata["methods"] == METHODS,
    "Method order is correct",
)

check(
    len(item_export) == 400,
    "Dashboard item export contains 400 rows",
)

check(
    item_export["job_id"].nunique() == 80,
    "Dashboard item export contains 80 unique jobs",
)

check(
    not item_export.duplicated(
        ["job_id", "method"]
    ).any(),
    "Every job-method key is unique",
)

method_counts = (
    item_export["method"]
    .value_counts()
    .reindex(METHODS)
)

check(
    method_counts.eq(80).all(),
    "Every method has exactly 80 item-score rows",
)


# ---------------------------------------------------------------------------
# Method-summary reconciliation
# ---------------------------------------------------------------------------

method_export = (
    method_export
    .sort_values("method")
    .reset_index(drop=True)
)

method_source = (
    method_source
    .sort_values("method")
    .reset_index(drop=True)
)

check(
    method_export["method"].tolist()
    == method_source["method"].tolist(),
    "Method-summary method names match source",
)

common_numeric = [
    column
    for column in method_source.columns
    if column != "method"
    and column in method_export.columns
]

for column in common_numeric:
    check(
        numeric_equal(
            method_export[column],
            method_source[column],
        ),
        f"Method-summary metric matches: {column}",
    )


# ---------------------------------------------------------------------------
# Headline metric verification
# ---------------------------------------------------------------------------

summary = method_source.set_index("method")

check(
    summary["any_skill_f1"].idxmax() == "rules",
    "Rules have the best any-skill F1",
)

check(
    summary["work_accuracy"].idxmax() == "few_shot",
    "Few-shot has the best work-arrangement accuracy",
)

best_years = set(
    summary.index[
        summary["years_exact"].eq(
            summary["years_exact"].max()
        )
    ]
)

check(
    best_years == {"zero_shot", "few_shot"},
    "Zero-shot and few-shot tie for best years-exact accuracy",
)

check(
    np.isclose(
        summary.loc["rules", "any_skill_f1"],
        0.926,
        atol=0.001,
    ),
    "Rules any-skill F1 is approximately 0.926",
)

check(
    np.isclose(
        summary.loc["few_shot", "work_accuracy"],
        0.850,
    ),
    "Few-shot work accuracy is 85.0%",
)


# ---------------------------------------------------------------------------
# Paired-test reconciliation
# ---------------------------------------------------------------------------

pairwise_keys = [
    "method",
    "baseline",
    "field",
    "metric",
]

pairwise_export = pairwise_export.sort_values(
    pairwise_keys
).reset_index(drop=True)

pairwise_source = pairwise_source.sort_values(
    pairwise_keys
).reset_index(drop=True)

check(
    pairwise_export[pairwise_keys].equals(
        pairwise_source[pairwise_keys]
    ),
    "Paired-test keys match source",
)

for column in [
    "difference",
    "ci_low",
    "ci_high",
    "p_raw",
    "p_holm",
]:
    check(
        numeric_equal(
            pairwise_export[column],
            pairwise_source[column],
        ),
        f"Paired-test values match: {column}",
    )

skill_tests = pairwise_source[
    pairwise_source["field"].eq("skills")
]

skill_ci_excludes_zero = (
    skill_tests["ci_low"].gt(0)
    | skill_tests["ci_high"].lt(0)
)

check(
    len(skill_tests) == 12
    and skill_ci_excludes_zero.all(),
    "All 12 skill bootstrap CIs exclude zero",
)

holm_tests = pairwise_source[
    pairwise_source["p_holm"].notna()
]

check(
    len(holm_tests) == 12,
    "Exactly 12 work/years tests received Holm correction",
)

check(
    holm_tests["p_holm"].lt(0.05).all(),
    "All 12 corrected work/years tests remain significant",
)


# ---------------------------------------------------------------------------
# Invalid-prediction handling
# ---------------------------------------------------------------------------

invalid_row = item_source[
    item_source["job_id"].astype(int).eq(3904946292)
    & item_source["method"].eq("schema_rules")
]

check(
    len(invalid_row) == 1,
    "Known malformed schema-rules row exists once",
)

if len(invalid_row) == 1:
    row = invalid_row.iloc[0]

    check(
        row["pred_work_arrangement"]
        == "InvalidPrediction",
        "Malformed response is represented as InvalidPrediction",
    )

    check(
        not bool(row["work_correct"]),
        "Malformed response is scored as incorrect",
    )


# ---------------------------------------------------------------------------
# Operations reconciliation
# ---------------------------------------------------------------------------

llm = llm_source.copy()

llm["cached"] = (
    llm["cached"]
    .fillna(False)
    .astype(bool)
)

llm["valid_json"] = (
    llm["valid_json"]
    .fillna(False)
    .astype(bool)
)

llm["estimated_cost"] = (
    llm["input_tokens"].fillna(0)
    / 1_000_000
    * 1.00
    + llm["output_tokens"].fillna(0)
    / 1_000_000
    * 5.00
)

recalculated_rows = []

for variant, group in llm.groupby("variant"):
    uncached = group[~group["cached"]]

    recalculated_rows.append(
        {
            "variant": variant,
            "n_predictions": len(group),
            "actual_api_calls": len(uncached),
            "cache_rate": group["cached"].mean(),
            "valid_json_rate": group["valid_json"].mean(),
            "total_input_tokens": group["input_tokens"].sum(),
            "total_output_tokens": group["output_tokens"].sum(),
            "estimated_model_cost_usd": (
                group["estimated_cost"].sum()
            ),
            "recorded_run_cost_usd": group["cost"].sum(),
        }
    )

operations_source = pd.DataFrame(
    recalculated_rows
).sort_values("variant").reset_index(drop=True)

operations_export = (
    operations_export
    .sort_values("variant")
    .reset_index(drop=True)
)

check(
    operations_export["variant"].tolist()
    == operations_source["variant"].tolist(),
    "Operations prompt variants match source",
)

for column in [
    "n_predictions",
    "actual_api_calls",
    "cache_rate",
    "valid_json_rate",
    "total_input_tokens",
    "total_output_tokens",
    "estimated_model_cost_usd",
    "recorded_run_cost_usd",
]:
    check(
        numeric_equal(
            operations_export[column],
            operations_source[column],
        ),
        f"Operations value matches: {column}",
    )

check(
    int(
        operations_export[
            "actual_api_calls"
        ].sum()
    )
    == 237,
    "Recorded actual API calls equal 237",
)

overall_cache_rate = (
    (
        operations_export["cache_rate"]
        * operations_export["n_predictions"]
    ).sum()
    / operations_export["n_predictions"].sum()
)

check(
    np.isclose(
        overall_cache_rate,
        0.259375,
    ),
    "Overall cache rate is approximately 25.9%",
)

check(
    np.isclose(
        operations_export[
            "estimated_model_cost_usd"
        ].sum(),
        1.079,
        atol=0.002,
    ),
    "Estimated fresh-run cost is approximately $1.079",
)


# ---------------------------------------------------------------------------
# Privacy and deployability
# ---------------------------------------------------------------------------

combined_json = "\n".join(
    path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    for path in DASHBOARD_DATA.glob("*.json")
)

for forbidden in [
    '"description"',
    '"raw_text"',
    "sk-ant-",
    "ANTHROPIC_API_KEY",
]:
    check(
        forbidden not in combined_json,
        f"Dashboard exports exclude sensitive content: {forbidden}",
    )

total_size = sum(
    path.stat().st_size
    for path in DASHBOARD_DATA.glob("*.json")
)

check(
    total_size < 2_000_000,
    "Dashboard JSON exports remain below 2 MB",
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)

if failures:
    print(
        f"DASHBOARD VALIDATION FAILED: "
        f"{len(failures)} check(s) failed."
    )

    for failure in failures:
        print(f"- {failure}")

    sys.exit(1)

print("DASHBOARD DATA VALIDATION PASSED")
print("All automated checks completed successfully.")
