"""Score rules and LLM extractors against the manually annotated gold set.

The scorer is deliberately field-specific:

* skills: entity-level micro precision/recall/F1, set exact match, per-posting
  macro F1, and required-vs-preferred classification errors;
* work arrangement: strict accuracy and macro F1;
* minimum years: strict exact match, within-one-year accuracy, MAE on answered
  numeric rows, coverage, and abstention behaviour.

Every method is evaluated on the same job_ids. Missing predictions count as
misses; rows are never silently dropped. Outputs are designed both for the
README and for a dashboard.

Default file names (all under data/processed):

    gold_labels.json | gold_labels.parquet | gold_labels.csv
    llm_extractions_gold.parquet | llm_extractions_gold.csv
    baseline_gold_seed.parquet | baseline_gold.parquet | baseline_gold.csv

Work arrangement is optional for prediction files. If the existing rules/LLM
outputs do not contain it, the scorer reports skills and years and leaves work
metrics blank instead of fabricating errors.

The command accepts explicit paths when local names differ:

    python src/evaluation/score_gold.py \
        --gold data/processed/gold_annotations.csv \
        --llm data/processed/llm_extractions_gold.parquet \
        --rules data/processed/baseline_gold.parquet

The input loader accepts a small set of column aliases and prints the resolved
schema before scoring. Use --column-map JSON to override ambiguous names.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Methods emitted by the existing experiment runner. Additional variants are
# accepted and appear after this preferred order.
PREFERRED_METHOD_ORDER = [
    "rules",
    "zero_shot",
    "few_shot",
    "schema_rules",
    "decomposed",
]

# Canonicalisation is intentionally conservative: only obvious aliases are
# merged. The raw and canonical values are both retained in item-level output.
SKILL_ALIASES = {
    "structured query language": "sql",
    "sql server": "microsoft sql server",
    "ms sql server": "microsoft sql server",
    "mssql": "microsoft sql server",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "google big query": "bigquery",
    "google bigquery": "bigquery",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "apache spark": "spark",
    "pyspark": "pyspark",
    "apache airflow": "airflow",
    "apache kafka": "kafka",
    "snowflake data cloud": "snowflake",
    "data build tool": "dbt",
    "tableau desktop": "tableau",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "github actions": "github actions",
    "git hub": "github",
    "k8s": "kubernetes",
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "job_id": ["job_id", "id", "posting_id"],
    "variant": ["variant", "method", "prompt_variant", "extractor"],
    "required_skills": [
        "required_skills",
        "gold_required_skills",
        "required_skills_gold",
        "skills_required",
        "required",
    ],
    "preferred_skills": [
        "preferred_skills",
        "gold_preferred_skills",
        "preferred_skills_gold",
        "skills_preferred",
        "preferred",
        "nice_to_have_skills",
    ],
    "work_arrangement": [
        "work_arrangement",
        "gold_work_arrangement",
        "work_type",
        "location_type",
        "remote_status",
        "workplace_type",
    ],
    "years_experience_min": [
        "years_experience_min",
        "gold_years_experience_min",
        "minimum_years_experience",
        "min_years_experience",
        "years_min",
        "years",
    ],
}

WORK_ALIASES = {
    "remote": "Remote",
    "fully remote": "Remote",
    "100% remote": "Remote",
    "work from home": "Remote",
    "wfh": "Remote",
    "hybrid": "Hybrid",
    "hybrid remote": "Hybrid",
    "hybrid/remote": "Hybrid",
    "onsite": "Onsite",
    "on site": "Onsite",
    "on-site": "Onsite",
    "in office": "Onsite",
    "office": "Onsite",
    "not stated": "Unclear",
    "unknown": "Unclear",
    "unclear": "Unclear",
    "ambiguous": "Unclear",
    "none": "Unclear",
    "null": "Unclear",
    "": "Unclear",
}


@dataclass(frozen=True)
class ResolvedColumns:
    job_id: str
    required_skills: str
    preferred_skills: str
    work_arrangement: str | None
    years_experience_min: str
    variant: str | None = None


@dataclass(frozen=True)
class SkillCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return f1_from_counts(self.tp, self.fp, self.fn)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_from_counts(tp: int | float, fp: int | float, fn: int | float) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 1.0


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (np.ndarray, pd.Series)) else False
    except (TypeError, ValueError):
        return False


def parse_list(value: Any) -> list[str]:
    """Parse list-like values from parquet objects, JSON, Python literals or CSV."""
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(v).strip() for v in value if not is_missing(v) and str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan", "[]"}:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                pass
        # Annotation exports commonly use commas, semicolons, pipes or newlines.
        return [piece.strip() for piece in re.split(r"\s*(?:,|;|\||\n)\s*", text) if piece.strip()]
    return [str(value).strip()] if str(value).strip() else []


def canonical_skill(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[™®]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;()[]{}")
    # Remove a trailing version only for well-known versioned tool names. This
    # avoids turning meaningful names such as "R" into an empty string.
    text = re.sub(r"^(python|tableau|excel|spark)\s+\d+(?:\.\d+)*$", r"\1", text)
    return SKILL_ALIASES.get(text, text)


def canonical_skill_set(value: Any) -> set[str]:
    return {skill for item in parse_list(value) if (skill := canonical_skill(item))}


def normalise_work(value: Any) -> str:
    if is_missing(value):
        return "Unclear"
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if text in WORK_ALIASES:
        return WORK_ALIASES[text]
    # Resolve verbose model outputs without letting "remote" override hybrid.
    if "hybrid" in text:
        return "Hybrid"
    if any(token in text for token in ("on-site", "onsite", "on site", "in-office", "in office")):
        return "Onsite"
    if any(token in text for token in ("remote", "work from home", "wfh")):
        return "Remote"
    return "Unclear"


def normalise_years(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return None if math.isnan(number) else number
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "not stated", "unknown", "unclear"}:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text())
        # The project's annotation export stores records under a top-level
        # ``labels`` key alongside metadata such as n and exported_at.
        if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
            return pd.DataFrame(payload["labels"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported input type {suffix!r}: {path}")


def write_item_scores(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write rich item-level rows, preferring Parquet with a CSV fallback.

    The project requirements include pyarrow, but the fallback keeps schema
    validation and local smoke tests usable in lightweight environments.
    """
    parquet_path = output_dir / "gold_item_scores.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_path = output_dir / "gold_item_scores.csv"
        serialisable = df.copy()
        for column in serialisable.columns:
            if serialisable[column].map(lambda value: isinstance(value, (list, dict, set, tuple))).any():
                serialisable[column] = serialisable[column].map(
                    lambda value: json.dumps(sorted(value) if isinstance(value, set) else value)
                    if isinstance(value, (list, dict, set, tuple)) else value
                )
        serialisable.to_csv(csv_path, index=False)
        print("WARNING: pyarrow/fastparquet unavailable; wrote item scores as CSV")
        return csv_path


def find_default(stem_candidates: Sequence[str]) -> Path:
    extensions = (".parquet", ".csv", ".jsonl", ".json")
    for stem in stem_candidates:
        for extension in extensions:
            candidate = PROCESSED / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    tried = ", ".join(str(PROCESSED / f"{s}{e}") for s in stem_candidates for e in extensions)
    raise FileNotFoundError(f"Could not find a default input. Tried: {tried}")


def parse_column_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    candidate = Path(raw)
    payload = candidate.read_text() if candidate.exists() else raw
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--column-map must be a JSON object")
    return {str(k): str(v) for k, v in parsed.items()}


def resolve_column(df: pd.DataFrame, logical: str, overrides: Mapping[str, str], *, required: bool = True) -> str | None:
    if logical in overrides:
        chosen = overrides[logical]
        if chosen not in df.columns:
            raise KeyError(f"Column override {logical}={chosen!r} is absent. Available: {list(df.columns)}")
        return chosen
    for candidate in COLUMN_ALIASES[logical]:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(
            f"Could not resolve logical column {logical!r}. "
            f"Accepted aliases: {COLUMN_ALIASES[logical]}. Available: {list(df.columns)}"
        )
    return None


def resolve_columns(
    df: pd.DataFrame,
    overrides: Mapping[str, str],
    *,
    needs_variant: bool,
    require_work_arrangement: bool,
) -> ResolvedColumns:
    work_column = resolve_column(
        df,
        "work_arrangement",
        overrides,
        required=require_work_arrangement,
    )
    return ResolvedColumns(
        job_id=str(resolve_column(df, "job_id", overrides)),
        required_skills=str(resolve_column(df, "required_skills", overrides)),
        preferred_skills=str(resolve_column(df, "preferred_skills", overrides)),
        work_arrangement=str(work_column) if work_column is not None else None,
        years_experience_min=str(resolve_column(df, "years_experience_min", overrides)),
        variant=resolve_column(df, "variant", overrides, required=needs_variant),
    )


def standardise(df: pd.DataFrame, columns: ResolvedColumns, *, method: str | None = None) -> pd.DataFrame:
    work_raw = (
        df[columns.work_arrangement]
        if columns.work_arrangement is not None
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    )
    out = pd.DataFrame(
        {
            "job_id": df[columns.job_id],
            "required_skills_raw": df[columns.required_skills],
            "preferred_skills_raw": df[columns.preferred_skills],
            "work_arrangement_raw": work_raw,
            "work_arrangement_available": columns.work_arrangement is not None,
            "years_experience_min_raw": df[columns.years_experience_min],
        }
    )
    if columns.variant:
        out["method"] = df[columns.variant].astype(str)
    elif method:
        out["method"] = method
    else:
        raise ValueError("A method name is required when the input has no variant column")
    out["required_skills"] = out.required_skills_raw.map(canonical_skill_set)
    out["preferred_skills"] = out.preferred_skills_raw.map(canonical_skill_set)
    out["work_arrangement"] = out.work_arrangement_raw.map(normalise_work)

    # A missing prediction is an extraction failure, not an "Unclear"
    # prediction. Keep gold normalization unchanged, but assign prediction
    # rows an explicit invalid sentinel so malformed or absent model output
    # is counted as incorrect.
    is_prediction_table = columns.variant is not None or method not in {None, "gold"}
    if is_prediction_table:
        missing_work_prediction = out["work_arrangement_raw"].map(is_missing)
        out.loc[
            missing_work_prediction,
            "work_arrangement",
        ] = "InvalidPrediction"

    out["years_experience_min"] = out.years_experience_min_raw.map(normalise_years)
    return out


def validate_gold_skill_buckets(gold: pd.DataFrame) -> None:
    overlap = gold.apply(
        lambda row: row.required_skills & row.preferred_skills,
        axis=1,
    )
    invalid = overlap.map(bool)
    if invalid.any():
        examples = [
            {"job_id": gold.loc[index, "job_id"], "overlap": sorted(overlap.loc[index])}
            for index in gold.index[invalid][:10]
        ]
        raise ValueError(
            "Gold required/preferred skill buckets must be disjoint. "
            f"Examples: {examples}"
        )


def validate_unique(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = df.duplicated(list(keys), keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, list(keys)].head(10).to_dict("records")
        raise ValueError(f"{name} contains duplicate keys {list(keys)}. Examples: {sample}")


def counts_for_sets(truth: set[str], pred: set[str]) -> SkillCounts:
    return SkillCounts(tp=len(truth & pred), fp=len(pred - truth), fn=len(truth - pred))


def skill_item_metrics(truth: set[str], pred: set[str], prefix: str) -> dict[str, Any]:
    counts = counts_for_sets(truth, pred)
    return {
        f"{prefix}_tp": counts.tp,
        f"{prefix}_fp": counts.fp,
        f"{prefix}_fn": counts.fn,
        f"{prefix}_precision": counts.precision,
        f"{prefix}_recall": counts.recall,
        f"{prefix}_f1": counts.f1,
        f"{prefix}_exact": truth == pred,
        f"{prefix}_gold_count": len(truth),
        f"{prefix}_pred_count": len(pred),
        f"{prefix}_missed": sorted(truth - pred),
        f"{prefix}_extra": sorted(pred - truth),
    }


def score_items(gold: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = gold.merge(predictions, on="job_id", how="left", suffixes=("_gold", "_pred"), validate="one_to_many")
    missing_methods = merged.method.isna()
    if missing_methods.any():
        missing_ids = merged.loc[missing_methods, "job_id"].head(10).tolist()
        raise ValueError(f"Predictions are missing for gold job_ids, examples: {missing_ids}")

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        required_gold = row.required_skills_gold
        required_pred = row.required_skills_pred
        preferred_gold = row.preferred_skills_gold
        preferred_pred = row.preferred_skills_pred
        any_gold = required_gold | preferred_gold
        any_pred = required_pred | preferred_pred

        result: dict[str, Any] = {
            "job_id": row.job_id,
            "method": row.method,
            "gold_required_skills": sorted(required_gold),
            "pred_required_skills": sorted(required_pred),
            "gold_preferred_skills": sorted(preferred_gold),
            "pred_preferred_skills": sorted(preferred_pred),
            "gold_work_arrangement": row.work_arrangement_gold,
            "pred_work_arrangement": row.work_arrangement_pred,
            "gold_years_experience_min": row.years_experience_min_gold,
            "pred_years_experience_min": row.years_experience_min_pred,
        }
        result.update(skill_item_metrics(required_gold, required_pred, "required"))
        result.update(skill_item_metrics(preferred_gold, preferred_pred, "preferred"))
        result.update(skill_item_metrics(any_gold, any_pred, "any_skill"))

        # Same skill found but assigned to the wrong requirement bucket.
        gold_type = {skill: "required" for skill in required_gold}
        gold_type.update({skill: "preferred" for skill in preferred_gold if skill not in gold_type})
        pred_type = {skill: "required" for skill in required_pred}
        pred_type.update({skill: "preferred" for skill in preferred_pred if skill not in pred_type})
        overlap = set(gold_type) & set(pred_type)
        type_correct = sum(gold_type[s] == pred_type[s] for s in overlap)
        result["skill_type_overlap"] = len(overlap)
        result["skill_type_correct"] = type_correct
        result["skill_type_accuracy"] = safe_div(type_correct, len(overlap)) if overlap else np.nan
        result["skill_type_errors"] = sorted(s for s in overlap if gold_type[s] != pred_type[s])

        work_scored = bool(row.work_arrangement_available_pred)
        result["work_scored"] = work_scored
        result["work_correct"] = (
            row.work_arrangement_gold == row.work_arrangement_pred
            if work_scored
            else np.nan
        )

        gold_years = None if is_missing(row.years_experience_min_gold) else float(row.years_experience_min_gold)
        pred_years = None if is_missing(row.years_experience_min_pred) else float(row.years_experience_min_pred)
        # pandas promotes nullable numeric columns to float and represents nulls
        # as NaN after the merge. Convert those back to explicit None before
        # evaluating abstention and false-positive behaviour.
        both_missing = gold_years is None and pred_years is None
        both_numeric = gold_years is not None and pred_years is not None
        result["years_exact"] = bool(both_missing or (both_numeric and gold_years == pred_years))
        result["years_within_1"] = bool(both_missing or (both_numeric and abs(gold_years - pred_years) <= 1.0))
        result["years_abs_error"] = abs(gold_years - pred_years) if both_numeric else np.nan
        result["years_gold_stated"] = gold_years is not None
        result["years_pred_answered"] = pred_years is not None
        result["years_correct_abstention"] = bool(gold_years is None and pred_years is None)
        result["years_false_positive"] = bool(gold_years is None and pred_years is not None)
        rows.append(result)

    return pd.DataFrame(rows)


def aggregate_skill(item: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    tp = int(item[f"{prefix}_tp"].sum())
    fp = int(item[f"{prefix}_fp"].sum())
    fn = int(item[f"{prefix}_fn"].sum())
    nonempty = (item[f"{prefix}_gold_count"] + item[f"{prefix}_pred_count"]) > 0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": f1_from_counts(tp, fp, fn),
        "macro_f1_all": float(item[f"{prefix}_f1"].mean()),
        "macro_f1_nonempty": float(item.loc[nonempty, f"{prefix}_f1"].mean()) if nonempty.any() else np.nan,
        "exact_match": float(item[f"{prefix}_exact"].mean()),
        "avg_gold_count": float(item[f"{prefix}_gold_count"].mean()),
        "avg_pred_count": float(item[f"{prefix}_pred_count"].mean()),
    }


def macro_f1_labels(truth: pd.Series, pred: pd.Series, labels: Iterable[str]) -> float:
    scores = []
    for label in labels:
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        scores.append(f1_from_counts(tp, fp, fn))
    return float(np.mean(scores)) if scores else np.nan


def aggregate_method(item: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method = str(item.method.iloc[0])
    required = aggregate_skill(item, "required")
    preferred = aggregate_skill(item, "preferred")
    any_skill = aggregate_skill(item, "any_skill")
    work_item = item[item.work_scored.astype(bool)]
    # Evaluate macro-F1 over the gold label space. Invalid predictions count
    # as false negatives for the true class, but do not create an artificial
    # fifth work-arrangement class.
    work_labels = (
        sorted(set(work_item.gold_work_arrangement))
        if not work_item.empty
        else []
    )
    numeric_truth = item.years_gold_stated
    null_truth = ~numeric_truth
    answered_numeric = numeric_truth & item.years_pred_answered
    overlap = int(item.skill_type_overlap.sum())
    type_correct = int(item.skill_type_correct.sum())

    summary = {
        "method": method,
        "n_jobs": len(item),
        "required_skill_precision": required["precision"],
        "required_skill_recall": required["recall"],
        "required_skill_f1": required["f1"],
        "required_avg_gold_count": required["avg_gold_count"],
        "required_avg_pred_count": required["avg_pred_count"],
        "preferred_skill_precision": preferred["precision"],
        "preferred_skill_recall": preferred["recall"],
        "preferred_skill_f1": preferred["f1"],
        "any_skill_precision": any_skill["precision"],
        "any_skill_recall": any_skill["recall"],
        "any_skill_f1": any_skill["f1"],
        "skill_type_accuracy": safe_div(type_correct, overlap) if overlap else np.nan,
        "work_accuracy": float(work_item.work_correct.mean()) if not work_item.empty else np.nan,
        "work_macro_f1": (
            macro_f1_labels(work_item.gold_work_arrangement, work_item.pred_work_arrangement, work_labels)
            if not work_item.empty
            else np.nan
        ),
        "work_n_scored": int(len(work_item)),
        "years_exact": float(item.years_exact.mean()),
        "years_within_1": float(item.years_within_1.mean()),
        "years_mae_answered": float(item.loc[answered_numeric, "years_abs_error"].mean()) if answered_numeric.any() else np.nan,
        "years_answer_rate_when_stated": float(item.loc[numeric_truth, "years_pred_answered"].mean()) if numeric_truth.any() else np.nan,
        "years_correct_abstention": float(item.loc[null_truth, "years_correct_abstention"].mean()) if null_truth.any() else np.nan,
        "years_false_positive_rate": float(item.loc[null_truth, "years_false_positive"].mean()) if null_truth.any() else np.nan,
    }

    long_rows: list[dict[str, Any]] = []
    for scope, metrics in (("required", required), ("preferred", preferred), ("any_skill", any_skill)):
        for metric, value in metrics.items():
            long_rows.append({"method": method, "field": "skills", "scope": scope, "metric": metric, "value": value, "n": len(item)})
    for metric in ("skill_type_accuracy", "work_accuracy", "work_macro_f1", "years_exact", "years_within_1", "years_mae_answered", "years_answer_rate_when_stated", "years_correct_abstention", "years_false_positive_rate"):
        field = "work_arrangement" if metric.startswith("work_") else "years" if metric.startswith("years_") else "skills"
        long_rows.append({"method": method, "field": field, "scope": "overall", "metric": metric, "value": summary[metric], "n": len(item)})
    return summary, long_rows


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, float(stats.binomtest(a_only, discordant, 0.5).pvalue)


def paired_bootstrap_mean(a: np.ndarray, b: np.ndarray, *, n_iter: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired arrays must have equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(n_iter, len(a)))
    differences = a[indices].mean(axis=1) - b[indices].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def bootstrap_micro_f1(item_a: pd.DataFrame, item_b: pd.DataFrame, prefix: str, *, n_iter: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    cols = [f"{prefix}_tp", f"{prefix}_fp", f"{prefix}_fn"]
    a = item_a.sort_values("job_id")[cols].to_numpy(dtype=float)
    b = item_b.sort_values("job_id")[cols].to_numpy(dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired methods have different row counts")
    rng = np.random.default_rng(seed)
    observed = f1_from_counts(*a.sum(axis=0)) - f1_from_counts(*b.sum(axis=0))
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(a), size=len(a))
        diffs[i] = f1_from_counts(*a[idx].sum(axis=0)) - f1_from_counts(*b[idx].sum(axis=0))
    return float(observed), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def pairwise_tests(item_scores: pd.DataFrame, baseline: str = "rules") -> pd.DataFrame:
    methods = list(dict.fromkeys(item_scores.method.astype(str)))
    if baseline not in methods:
        return pd.DataFrame()
    base = item_scores[item_scores.method == baseline].sort_values("job_id")
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == baseline:
            continue
        candidate = item_scores[item_scores.method == method].sort_values("job_id")
        if candidate.job_id.tolist() != base.job_id.tolist():
            raise ValueError(f"Method {method!r} is not paired to {baseline!r} on identical job_ids")
        for scope in ("required", "preferred", "any_skill"):
            diff, lo, hi = bootstrap_micro_f1(candidate, base, scope)
            rows.append({
                "method": method,
                "baseline": baseline,
                "field": "skills",
                "metric": f"{scope}_micro_f1",
                "difference": diff,
                "ci_low": lo,
                "ci_high": hi,
                "test": "paired job bootstrap",
                "p_raw": np.nan,
            })
        for field, column in (
            ("work_arrangement", "work_correct"),
            ("years", "years_exact"),
            ("years", "years_within_1"),
        ):
            paired = pd.DataFrame({
                "candidate": candidate[column].to_numpy(),
                "baseline": base[column].to_numpy(),
            }).dropna()
            if paired.empty:
                continue
            a = paired["candidate"].astype(bool).to_numpy()
            b = paired["baseline"].astype(bool).to_numpy()
            difference, lo, hi = paired_bootstrap_mean(a, b)
            a_only, b_only, pvalue = mcnemar(a, b)
            rows.append({
                "method": method,
                "baseline": baseline,
                "field": field,
                "metric": column,
                "difference": difference,
                "ci_low": lo,
                "ci_high": hi,
                "test": "McNemar exact",
                "candidate_only": a_only,
                "baseline_only": b_only,
                "p_raw": pvalue,
            })
    result = pd.DataFrame(rows)
    testable = result.p_raw.notna()
    if testable.any():
        result.loc[testable, "p_holm"] = holm_adjust(result.loc[testable, "p_raw"].tolist())
        result.loc[testable, "significant_holm_005"] = result.loc[testable, "p_holm"] <= 0.05
    return result


def method_order(methods: Iterable[str]) -> list[str]:
    unique = list(dict.fromkeys(str(method) for method in methods))
    preferred = [method for method in PREFERRED_METHOD_ORDER if method in unique]
    return preferred + sorted(set(unique) - set(preferred))


def format_percent(value: Any) -> str:
    return "—" if is_missing(value) else f"{100 * float(value):.1f}%"


def build_markdown(summary: pd.DataFrame, tests: pd.DataFrame, gold_path: Path, llm_path: Path, rules_path: Path) -> str:
    order = method_order(summary.method)
    indexed = summary.set_index("method").loc[order]
    lines = [
        "# Gold-set evaluation",
        "",
        f"Evaluated **{int(indexed.n_jobs.iloc[0])} manually annotated postings**. Missing predictions count as errors.",
        "",
        "## Headline metrics",
        "",
        "| Method | Required skill F1 | Preferred skill F1 | Any-skill F1 | Work accuracy | Work macro F1 | Years exact | Years ±1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in indexed.iterrows():
        lines.append(
            f"| {method} | {row.required_skill_f1:.3f} | {row.preferred_skill_f1:.3f} | "
            f"{row.any_skill_f1:.3f} | {format_percent(row.work_accuracy)} | "
            f"{row.work_macro_f1:.3f} | {format_percent(row.years_exact)} | "
            f"{format_percent(row.years_within_1)} |"
        )
    lines.extend([
        "",
        "## Extraction behaviour",
        "",
        "| Method | Required P | Required R | Avg predicted required | Avg gold required | Skill type accuracy | Years answer rate | Years false-positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method, row in indexed.iterrows():
        lines.append(
            f"| {method} | {row.required_skill_precision:.3f} | {row.required_skill_recall:.3f} | "
            f"{row.required_avg_pred_count:.2f} | {row.required_avg_gold_count:.2f} | "
            f"{format_percent(row.skill_type_accuracy)} | "
            f"{format_percent(row.years_answer_rate_when_stated)} | {format_percent(row.years_false_positive_rate)} |"
        )
    if not tests.empty:
        lines.extend([
            "",
            "## Paired comparisons versus rules",
            "",
            "Skill F1 differences use a paired job-level bootstrap. Work and years use McNemar's exact test; Holm-adjusted p-values are reported across those binary tests.",
            "",
            "| Method | Metric | Difference | 95% CI | Test | Holm p |",
            "|---|---|---:|---:|---|---:|",
        ])
        for row in tests.itertuples(index=False):
            p = "—" if is_missing(row.p_raw) else f"{row.p_holm:.4f}"
            lines.append(
                f"| {row.method} | {row.metric} | {100 * row.difference:+.1f}pp | "
                f"[{100 * row.ci_low:+.1f}, {100 * row.ci_high:+.1f}] | {row.test} | {p} |"
            )
    lines.extend([
        "",
        "## Inputs",
        "",
        f"- Gold labels: `{gold_path}`",
        f"- LLM predictions: `{llm_path}`",
        f"- Rules predictions: `{rules_path}`",
        "",
        "## Interpretation guardrails",
        "",
        "- This is a difficulty-enriched gold set, not a population estimate for all job postings.",
        "- One annotator and exposure to rules suggestions on some rows create possible anchoring bias.",
        "- Exact skill matching depends on the canonical alias map in `score_gold.py`; inspect the error export before changing aliases.",
        "- Work-arrangement metrics are blank when the prediction files do not contain that field.",
        "- Lead with effect sizes and confidence intervals. A leaderboard rank without uncertainty is not enough.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, help="Gold annotations (.parquet/.csv/.jsonl)")
    parser.add_argument("--llm", type=Path, help="LLM gold-set extractions")
    parser.add_argument("--rules", type=Path, help="Rules gold-set predictions")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED, help="Directory for score outputs")
    parser.add_argument(
        "--column-map",
        help=(
            "JSON object or path to JSON. Logical keys: job_id, variant, required_skills, "
            "preferred_skills, work_arrangement, years_experience_min"
        ),
    )
    parser.add_argument("--baseline", default="rules", help="Baseline method for paired comparisons")
    parser.add_argument("--no-tests", action="store_true", help="Skip bootstrap and significance tests")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        gold_path = args.gold or find_default(("gold_labels", "gold_annotations", "gold_set"))
        llm_path = args.llm or find_default(("llm_extractions_gold", "gold_llm_extractions"))
        rules_path = args.rules or find_default(("baseline_gold_seed", "baseline_gold", "gold_baseline", "rules_gold"))
        overrides = parse_column_map(args.column_map)

        gold_raw = read_table(gold_path)
        llm_raw = read_table(llm_path)
        rules_raw = read_table(rules_path)

        gold_columns = resolve_columns(
            gold_raw, overrides, needs_variant=False, require_work_arrangement=True
        )
        llm_columns = resolve_columns(
            llm_raw, overrides, needs_variant=True, require_work_arrangement=False
        )
        rules_columns = resolve_columns(
            rules_raw, overrides, needs_variant=False, require_work_arrangement=False
        )

        print("Resolved columns:")
        print(f"  gold : {gold_columns}")
        print(f"  llm  : {llm_columns}")
        print(f"  rules: {rules_columns}")
        if llm_columns.work_arrangement is None or rules_columns.work_arrangement is None:
            print(
                "WARNING: work_arrangement is absent from one or more prediction files; "
                "work metrics will be left blank. Skills and years will still be scored."
            )

        gold = standardise(gold_raw, gold_columns, method="gold").drop(columns="method")
        llm = standardise(llm_raw, llm_columns)
        rules = standardise(rules_raw, rules_columns, method="rules")

        validate_unique(gold, ["job_id"], "gold")
        validate_gold_skill_buckets(gold)
        validate_unique(llm, ["job_id", "method"], "llm")
        validate_unique(rules, ["job_id", "method"], "rules")

        gold_ids = set(gold.job_id)
        prediction_ids = set(llm.job_id) | set(rules.job_id)
        extra = prediction_ids - gold_ids
        if extra:
            print(f"WARNING: ignoring {len(extra)} prediction job_ids not present in gold")
        llm = llm[llm.job_id.isin(gold_ids)]
        rules = rules[rules.job_id.isin(gold_ids)]
        predictions = pd.concat([rules, llm], ignore_index=True)

        expected_methods = method_order(predictions.method)
        coverage = predictions.groupby("method").job_id.nunique().reindex(expected_methods, fill_value=0)
        incomplete = coverage[coverage != len(gold)]
        if not incomplete.empty:
            raise ValueError(
                f"Every method must cover all {len(gold)} gold rows. Coverage: {coverage.to_dict()}"
            )

        item_scores = score_items(gold, predictions)
        summaries: list[dict[str, Any]] = []
        long_rows: list[dict[str, Any]] = []
        for method in expected_methods:
            summary, rows = aggregate_method(item_scores[item_scores.method == method])
            summaries.append(summary)
            long_rows.extend(rows)
        summary_df = pd.DataFrame(summaries)
        long_df = pd.DataFrame(long_rows)
        tests_df = pd.DataFrame() if args.no_tests else pairwise_tests(item_scores, baseline=args.baseline)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "gold_method_summary.csv"
        long_path = args.output_dir / "gold_metrics_long.csv"
        tests_path = args.output_dir / "gold_pairwise_tests.csv"
        errors_path = args.output_dir / "gold_skill_errors.csv"
        report_path = args.output_dir / "gold_evaluation.md"

        item_path = write_item_scores(item_scores, args.output_dir)
        summary_df.to_csv(summary_path, index=False)
        long_df.to_csv(long_path, index=False)
        if not tests_df.empty:
            tests_df.to_csv(tests_path, index=False)
        error_mask = (
            (item_scores.required_fp + item_scores.required_fn + item_scores.preferred_fp + item_scores.preferred_fn) > 0
        )
        error_columns = [
            "job_id", "method",
            "gold_required_skills", "pred_required_skills", "required_missed", "required_extra",
            "gold_preferred_skills", "pred_preferred_skills", "preferred_missed", "preferred_extra",
            "skill_type_errors",
        ]
        item_scores.loc[error_mask, error_columns].to_csv(errors_path, index=False)
        report_path.write_text(build_markdown(summary_df, tests_df, gold_path, llm_path, rules_path))

        print("\nGold-set results")
        display_columns = [
            "method", "required_skill_precision", "required_skill_recall", "required_skill_f1",
            "preferred_skill_f1", "any_skill_f1", "work_accuracy", "work_macro_f1",
            "years_exact", "years_within_1", "years_mae_answered",
        ]
        print(summary_df[display_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nWrote:")
        for path in (item_path, summary_path, long_path, errors_path, report_path):
            print(f"  {path}")
        if not tests_df.empty:
            print(f"  {tests_path}")
        return 0
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
