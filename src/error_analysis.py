"""
Inspect baseline failures and categorise them.

The output of this feeds directly into prompt design: every recurring
failure mode here becomes an explicit instruction in the LLM prompt, and
every one of them becomes a row in the final rules-vs-LLM comparison.

Run:  python src/error_analysis.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

pd.set_option("display.max_colwidth", 90)

HOURS_PER_YEAR = 2080
ANNUALISE = {"HOURLY": HOURS_PER_YEAR, "WEEKLY": 52,
             "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}

con = duckdb.connect()

df = con.execute(f"""
    SELECT
      b.job_id, b.title, b.description, b.stratum,
      b.pay_period AS gt_pay_period,
      b.min_salary AS gt_min_raw, b.med_salary AS gt_med_raw,
      b.exp_level  AS gt_seniority,
      e.salary_min AS pred_min, e.salary_max AS pred_max,
      e.pay_period AS pred_pay_period,
      e.seniority  AS pred_seniority,
      e.evidence_salary, e.evidence_seniority
    FROM '{PROCESSED / "benchmark.parquet"}' b
    LEFT JOIN '{PROCESSED / "baseline_extractions.parquet"}' e USING (job_id)
""").fetchdf()

df["gt_min"] = df.gt_min_raw.fillna(df.gt_med_raw)


def annual(v, p):
    return np.nan if pd.isna(v) or p not in ANNUALISE else v * ANNUALISE[p]


df["gt_annual"] = [annual(v, p) for v, p in zip(df.gt_min, df.gt_pay_period)]
df["pred_annual"] = [annual(v, p) for v, p in zip(df.pred_min, df.pred_pay_period)]
df["err"] = (df.pred_annual - df.gt_annual).abs() / df.gt_annual.replace(0, np.nan)


def context(row, width: int = 130) -> str:
    """Show the text around the matched evidence, so we can see what the
    regex actually latched onto."""
    desc, ev = row.description, row.evidence_salary
    if not isinstance(desc, str) or not isinstance(ev, str):
        return ""
    i = desc.find(ev)
    if i < 0:
        return ev
    s, e = max(0, i - width), min(len(desc), i + len(ev) + width)
    return " ".join(desc[s:e].split())


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n1. WORST SALARY ERRORS (annualised)\n{'=' * 78}")

worst = (df[(df.stratum == "labeled_stated") & df.err.notna()]
         .nlargest(15, "err"))

for _, r in worst.iterrows():
    print(f"\n  job {r.job_id} | error {r.err:,.1f}x")
    print(f"  truth: {r.gt_min:>12,.0f} {r.gt_pay_period:<8} "
          f"-> {r.gt_annual:>12,.0f}/yr")
    print(f"  pred : {r.pred_min:>12,.0f} {str(r.pred_pay_period):<8} "
          f"-> {r.pred_annual:>12,.0f}/yr")
    print(f"  matched: {r.evidence_salary!r}")
    print(f"  context: ...{context(r)}...")


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n2. FALSE POSITIVES (no salary in text, but extracted)\n{'=' * 78}")

for _, r in df[(df.stratum == "labeled_not_stated") & df.pred_min.notna()].iterrows():
    print(f"\n  job {r.job_id} | pred {r.pred_min:,.0f}-{r.pred_max:,.0f} "
          f"{r.pred_pay_period}")
    print(f"  title  : {r.title}")
    print(f"  matched: {r.evidence_salary!r}")
    print(f"  context: ...{context(r)}...")


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n3. PAY PERIOD FAILURES\n{'=' * 78}")

pp = df[(df.stratum == "labeled_stated") &
        df.gt_pay_period.notna() & df.pred_pay_period.notna()]
wrong = pp[pp.gt_pay_period != pp.pred_pay_period]

print(f"\n{len(wrong):,} of {len(pp):,} wrong ({100 * len(wrong) / len(pp):.1f}%)\n")
print(wrong.groupby(["gt_pay_period", "pred_pay_period"])
      .size().sort_values(ascending=False).to_string())

for truth in ("MONTHLY", "WEEKLY", "YEARLY"):
    sub = wrong[wrong.gt_pay_period == truth].head(3)
    if len(sub):
        print(f"\n  --- truth = {truth} ---")
        for _, r in sub.iterrows():
            print(f"  job {r.job_id}: matched {r.evidence_salary!r} "
                  f"-> called it {r.pred_pay_period}")
            print(f"    ...{context(r, 90)}...")


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n4. SENIORITY: EXECUTIVE FALSE POSITIVES\n{'=' * 78}")

exec_fp = df[(df.pred_seniority == "Executive") &
             (df.gt_seniority.notna()) &
             (df.gt_seniority != "Executive")]

print(f"\n{len(exec_fp)} postings called Executive that are not:\n")
print(exec_fp[["title", "gt_seniority", "evidence_seniority"]]
      .head(20).to_string(index=False))


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n5. SENIORITY: ENTRY-LEVEL MISSES\n{'=' * 78}")

entry_missed = df[(df.gt_seniority == "Entry level") & df.pred_seniority.isna()]
print(f"\n{len(entry_missed)} entry-level postings the rule abstained on.")
print("Sample titles -- note how few contain any seniority keyword:\n")
print(entry_missed.title.head(25).to_string(index=False))


# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}\n6. SENIORITY DISAGREEMENTS (rule answered, truth differs)\n{'=' * 78}")

dis = df[df.pred_seniority.notna() & df.gt_seniority.notna() &
         (df.pred_seniority != df.gt_seniority)]
print(f"\n{len(dis)} disagreements. Sample:\n")
print(dis[["title", "gt_seniority", "pred_seniority"]]
      .head(20).to_string(index=False))

print("\nNote: formatted_experience_level is a WEAK label. Some of these are")
print("LinkedIn mis-tagging, not rule failures. Worth eyeballing before")
print("attributing every disagreement to the extractor.\n")
