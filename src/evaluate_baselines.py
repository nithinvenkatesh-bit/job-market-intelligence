"""
Score the rule-based extractions against ground truth.

Measurement decisions that matter here:

  * Abstentions count as misses. A rule that answers 38% of the time and is
    right when it answers is NOT a 90%-accurate rule. Strict metrics include
    every labeled row; conditional metrics are reported alongside, labelled
    as such.

  * Salary is compared after annualisation. "$60" read as yearly instead of
    hourly is a 2,080x error, not a rounding difference -- comparing raw
    figures would score it as perfect.

  * Median-only labels are scored by containment. ~17% of rows carry only
    med_salary. Comparing a predicted range's endpoints against a midpoint
    manufactures error where the extraction was correct.

Run:  python src/evaluate_baselines.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Hours per year for hourly->annual conversion. Documented because it is an
# assumption, not a fact: 40h x 52w. Part-time roles will be overstated.
HOURS_PER_YEAR = 2080
ANNUALISE = {
    "HOURLY": HOURS_PER_YEAR,
    "WEEKLY": 52,
    "BIWEEKLY": 26,
    "MONTHLY": 12,
    "YEARLY": 1,
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="benchmark",
                    choices=["benchmark", "holdout"])
args = parser.parse_args()
DATASET = args.dataset

con = duckdb.connect()

print(f"Evaluating on: {DATASET}")

df = con.execute(f"""
    SELECT
      b.job_id,
      b.stratum,
      b.pay_period            AS gt_pay_period,
      b.min_salary            AS gt_min_raw,
      b.med_salary            AS gt_med_raw,
      b.max_salary            AS gt_max_raw,
      b.exp_level             AS gt_seniority,
      e.salary_min            AS pred_min,
      e.salary_max            AS pred_max,
      e.pay_period            AS pred_pay_period,
      e.seniority             AS pred_seniority,
      e.years_experience_min  AS pred_years
    FROM '{PROCESSED / f"{DATASET}.parquet"}' b
    LEFT JOIN '{PROCESSED / f"baseline_{DATASET}.parquet"}' e USING (job_id)
""").fetchdf()

# Label shape: a true range, or a single midpoint?
df["label_type"] = np.where(
    df.gt_min_raw.notna(), "range",
    np.where(df.gt_med_raw.notna(), "point", "none"),
)
df["gt_min"] = df.gt_min_raw.fillna(df.gt_med_raw)
df["gt_max"] = df.gt_max_raw.fillna(df.gt_med_raw)

stated = df[df.stratum == "labeled_stated"].copy()
not_stated = df[df.stratum == "labeled_not_stated"].copy()

print(f"Benchmark: {len(df):,} rows "
      f"({len(stated):,} salary-in-text, {len(not_stated):,} not-in-text)")
print(f"Label shapes: {df.label_type.value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}\nSALARY (descriptions that DO state pay)\n{'=' * 64}")

attempted = stated[stated.pred_min.notna()].copy()
recall = len(attempted) / len(stated)
print(f"Attempted: {len(attempted):,} / {len(stated):,}  ({100 * recall:.1f}%)")


def annualise(value: float, period: str | None) -> float:
    if pd.isna(value) or period not in ANNUALISE:
        return np.nan
    return value * ANNUALISE[period]


attempted["gt_annual"] = [
    annualise(v, p) for v, p in zip(attempted.gt_min, attempted.gt_pay_period)
]
attempted["pred_annual"] = [
    annualise(v, p) for v, p in zip(attempted.pred_min, attempted.pred_pay_period)
]

# --- raw figure, ignoring period (the flattering view) ---
rng = attempted[attempted.label_type == "range"].copy()
rng["err_raw"] = (rng.pred_min - rng.gt_min).abs() / rng.gt_min.replace(0, np.nan)

print("\n  RAW min figure, period ignored  (range-labelled rows only)")
for tol in (0.05, 0.10, 0.20):
    print(f"    within {int(tol * 100):>2}%   {100 * (rng.err_raw <= tol).mean():5.1f}%")

# --- annualised, which is what actually matters ---
ann = attempted[attempted.gt_annual.notna() & attempted.pred_annual.notna()].copy()
ann["err_annual"] = (ann.pred_annual - ann.gt_annual).abs() / ann.gt_annual.replace(0, np.nan)

print(f"\n  ANNUALISED  (n={len(ann):,})  <- the honest number")
for tol in (0.05, 0.10, 0.20):
    pct = 100 * (ann.err_annual <= tol).mean()
    e2e = 100 * (ann.err_annual <= tol).sum() / len(stated)
    print(f"    within {int(tol * 100):>2}%   {pct:5.1f}% of attempted   "
          f"|  {e2e:5.1f}% end-to-end")
print(f"    median error  {100 * ann.err_annual.median():.1f}%")

# --- point labels: is the midpoint inside the predicted range? ---
pts = attempted[(attempted.label_type == "point") & attempted.pred_max.notna()]
if len(pts):
    inside = ((pts.gt_min >= pts.pred_min * 0.98) &
              (pts.gt_min <= pts.pred_max * 1.02))
    print(f"\n  POINT labels ({len(pts):,}): midpoint inside predicted range "
          f"{100 * inside.mean():.1f}%")

# --- pay period ---
pp = attempted[attempted.gt_pay_period.notna()]
pp_acc = (pp.pred_pay_period == pp.gt_pay_period).mean()
print(f"\n  PAY PERIOD accuracy: {100 * pp_acc:.1f}%  (n={len(pp):,})")
print("\n  confusion (rows=truth, cols=predicted)")
print(pd.crosstab(pp.gt_pay_period, pp.pred_pay_period).to_string())

# --- combined: both the figure AND the period right ---
combined = ann[(ann.err_annual <= 0.10) &
               (ann.pred_pay_period == ann.gt_pay_period)]
print(f"\n  COMBINED (period correct AND annualised within 10%)")
print(f"    {100 * len(combined) / len(attempted):5.1f}% of attempted")
print(f"    {100 * len(combined) / len(stated):5.1f}% end-to-end")


# ---------------------------------------------------------------------------
# Abstention -- the hallucination test
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}\nABSTENTION (descriptions that do NOT state pay)\n{'=' * 64}")

abstained = not_stated.pred_min.isna().sum()
print(f"Correctly returned null: {abstained:,} / {len(not_stated):,} "
      f"({100 * abstained / len(not_stated):.1f}%)")

fp = not_stated[not_stated.pred_min.notna()]
if len(fp):
    print(f"\nFalse positives ({len(fp)}) -- likely bonuses, budgets, fees:")
    print(fp[["job_id", "pred_min", "pred_max", "pred_pay_period"]]
          .head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Seniority -- strict, abstentions count as misses
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}\nSENIORITY (vs formatted_experience_level -- WEAK label)\n{'=' * 64}")

sen = df[df.gt_seniority.notna()].copy()
answered = sen[sen.pred_seniority.notna()]

print(f"Labelled rows: {len(sen):,}")
print(f"Rule answered: {len(answered):,} ({100 * len(answered) / len(sen):.1f}%)")
print(f"Accuracy when answered (conditional): "
      f"{100 * (answered.pred_seniority == answered.gt_seniority).mean():.1f}%")
print(f"Accuracy over ALL labelled rows (strict): "
      f"{100 * (sen.pred_seniority == sen.gt_seniority).mean():.1f}%")

labels = sorted(set(sen.gt_seniority.dropna()) | set(sen.pred_seniority.dropna()))
rows = []
for lab in labels:
    # Computed over ALL labelled rows: an abstention is a false negative,
    # because pred_seniority is NaN and NaN != lab evaluates True.
    tp = ((sen.pred_seniority == lab) & (sen.gt_seniority == lab)).sum()
    fp_ = ((sen.pred_seniority == lab) & (sen.gt_seniority != lab)).sum()
    fn = ((sen.pred_seniority != lab) & (sen.gt_seniority == lab)).sum()
    prec = tp / (tp + fp_) if tp + fp_ else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows.append({"label": lab, "support": int((sen.gt_seniority == lab).sum()),
                 "precision": round(prec, 3), "recall": round(rec, 3),
                 "f1": round(f1, 3)})

per_class = pd.DataFrame(rows)
print("\nPer-class (strict -- abstentions count against recall)")
print(per_class.to_string(index=False))
print(f"\nMacro F1 (strict): {per_class.f1.mean():.3f}")

print("\nConfusion, answered rows only (rows=truth, cols=predicted)")
print(pd.crosstab(answered.gt_seniority, answered.pred_seniority).to_string())


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

print(f"\n{'=' * 64}\nBASELINE SUMMARY -- the bar the LLM must clear\n{'=' * 64}")

summary = pd.DataFrame([
    {"field": "salary", "metric": "annualised within 10% (end-to-end)",
     "score": f"{100 * (ann.err_annual <= 0.10).sum() / len(stated):.1f}%"},
    {"field": "salary", "metric": "annualised within 10% (conditional)",
     "score": f"{100 * (ann.err_annual <= 0.10).mean():.1f}%"},
    {"field": "salary", "metric": "combined period+value (end-to-end)",
     "score": f"{100 * len(combined) / len(stated):.1f}%"},
    {"field": "salary", "metric": "correct abstention",
     "score": f"{100 * abstained / len(not_stated):.1f}%"},
    {"field": "pay_period", "metric": "accuracy",
     "score": f"{100 * pp_acc:.1f}%"},
    {"field": "seniority", "metric": "macro F1 (strict)",
     "score": f"{per_class.f1.mean():.3f}"},
    {"field": "seniority", "metric": "accuracy (strict)",
     "score": f"{100 * (sen.pred_seniority == sen.gt_seniority).mean():.1f}%"},
    {"field": "seniority", "metric": "answer rate",
     "score": f"{100 * len(answered) / len(sen):.1f}%"},
])
print(summary.to_string(index=False))

summary.to_csv(PROCESSED / f"baseline_summary_{DATASET}.csv", index=False)
print(f"\nWrote baseline_summary_{DATASET}.csv")
