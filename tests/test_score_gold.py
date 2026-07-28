from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.score_gold import (
    bootstrap_micro_f1,
    canonical_skill_set,
    counts_for_sets,
    normalise_work,
    normalise_years,
    score_items,
    standardise,
    ResolvedColumns,
    aggregate_method,
    read_table,
    resolve_columns,
    validate_gold_skill_buckets,
)


def test_skill_parser_and_aliases() -> None:
    assert canonical_skill_set('["PowerBI", "Structured Query Language", "Postgres"]') == {
        "power bi", "sql", "postgresql"
    }
    assert canonical_skill_set("Python; dbt | Snowflake") == {"python", "dbt", "snowflake"}


def test_skill_counts_penalise_required_preferred_swap() -> None:
    required = counts_for_sets({"sql"}, set())
    preferred = counts_for_sets(set(), {"sql"})
    assert (required.tp, required.fp, required.fn) == (0, 0, 1)
    assert (preferred.tp, preferred.fp, preferred.fn) == (0, 1, 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fully remote", "Remote"),
        ("Hybrid - 2 days in office", "Hybrid"),
        ("On-site", "Onsite"),
        (None, "Unclear"),
    ],
)
def test_work_normalisation(raw, expected) -> None:
    assert normalise_work(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5+ years", 5.0), ("3-5 years", 3.0), (4, 4.0), (None, None), ("not stated", None)],
)
def test_year_normalisation(raw, expected) -> None:
    assert normalise_years(raw) == expected


def _standard(df: pd.DataFrame, method: str | None = None, variant: bool = False) -> pd.DataFrame:
    return standardise(
        df,
        ResolvedColumns(
            job_id="job_id",
            required_skills="required_skills",
            preferred_skills="preferred_skills",
            work_arrangement="work_arrangement",
            years_experience_min="years_experience_min",
            variant="variant" if variant else None,
        ),
        method=method,
    )


def test_end_to_end_item_scoring_and_aggregation() -> None:
    gold_raw = pd.DataFrame(
        [
            {"job_id": 1, "required_skills": ["SQL", "Python"], "preferred_skills": ["dbt"], "work_arrangement": "Hybrid", "years_experience_min": 5},
            {"job_id": 2, "required_skills": [], "preferred_skills": [], "work_arrangement": "Remote", "years_experience_min": None},
        ]
    )
    pred_raw = pd.DataFrame(
        [
            {"job_id": 1, "required_skills": ["SQL"], "preferred_skills": ["Python", "dbt"], "work_arrangement": "hybrid", "years_experience_min": 4},
            {"job_id": 2, "required_skills": ["Excel"], "preferred_skills": [], "work_arrangement": "onsite", "years_experience_min": 2},
        ]
    )
    gold = _standard(gold_raw, method="gold").drop(columns="method")
    pred = _standard(pred_raw, method="rules")
    items = score_items(gold, pred)

    row1 = items[items.job_id == 1].iloc[0]
    assert row1.required_tp == 1
    assert row1.required_fn == 1
    assert row1.preferred_tp == 1
    assert row1.preferred_fp == 1
    assert row1.skill_type_errors == ["python"]
    assert bool(row1.work_correct)
    assert not bool(row1.years_exact)
    assert bool(row1.years_within_1)

    row2 = items[items.job_id == 2].iloc[0]
    assert bool(row2.years_false_positive)
    assert not bool(row2.work_correct)

    summary, _ = aggregate_method(items)
    assert summary["required_skill_precision"] == pytest.approx(0.5)
    assert summary["required_skill_recall"] == pytest.approx(0.5)
    assert summary["years_within_1"] == pytest.approx(0.5)
    assert summary["work_accuracy"] == pytest.approx(0.5)


def test_bootstrap_micro_f1_is_paired_and_directional() -> None:
    better = pd.DataFrame({
        "job_id": [1, 2, 3],
        "required_tp": [1, 1, 1],
        "required_fp": [0, 0, 0],
        "required_fn": [0, 0, 0],
    })
    worse = pd.DataFrame({
        "job_id": [1, 2, 3],
        "required_tp": [0, 0, 1],
        "required_fp": [1, 1, 0],
        "required_fn": [1, 1, 0],
    })
    diff, low, high = bootstrap_micro_f1(better, worse, "required", n_iter=1000, seed=7)
    assert diff > 0
    assert high > 0
    assert not np.isnan(low)


def test_gold_skill_buckets_must_be_disjoint() -> None:
    gold = pd.DataFrame({
        "job_id": [1],
        "required_skills": [{"sql"}],
        "preferred_skills": [{"sql"}],
    })
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_gold_skill_buckets(gold)


def test_nested_gold_json_is_loaded(tmp_path) -> None:
    path = tmp_path / "gold_labels.json"
    path.write_text(
        '{"n": 1, "labels": [{"job_id": 1, "required_skills": ["SQL"], '
        '"preferred_skills": [], "work_arrangement": "hybrid", '
        '"years_experience_min": 3}]}'
    )
    df = read_table(path)
    assert len(df) == 1
    assert df.iloc[0]["job_id"] == 1


def test_missing_prediction_work_arrangement_is_skipped() -> None:
    gold_raw = pd.DataFrame([
        {"job_id": 1, "required_skills": ["SQL"], "preferred_skills": [],
         "work_arrangement": "Hybrid", "years_experience_min": 3}
    ])
    pred_raw = pd.DataFrame([
        {"job_id": 1, "required_skills": ["SQL"], "preferred_skills": [],
         "years_experience_min": 3}
    ])
    gold_columns = resolve_columns(
        gold_raw, {}, needs_variant=False, require_work_arrangement=True
    )
    pred_columns = resolve_columns(
        pred_raw, {}, needs_variant=False, require_work_arrangement=False
    )
    gold = standardise(gold_raw, gold_columns, method="gold").drop(columns="method")
    pred = standardise(pred_raw, pred_columns, method="rules")
    items = score_items(gold, pred)
    assert not bool(items.iloc[0]["work_scored"])
    assert pd.isna(items.iloc[0]["work_correct"])
    summary, _ = aggregate_method(items)
    assert pd.isna(summary["work_accuracy"])
    assert summary["work_n_scored"] == 0
