"""
Hybrid routing: use each method where it measurably wins.

The comparison established, with paired significance tests:
  salary, pay period, abstention -> rules (free, and better)
  seniority                      -> LLM (+30pp)

And a subtler result: on seniority the rules are MORE accurate than the LLM
when they fire (77.5% vs 59.8% conditional) -- they simply abstain on 64% of
postings. So the right policy is not "use the LLM for seniority", it is
"use the rules when they speak, and the LLM only when they are silent".

This script builds three policies and measures them against the same
ground truth, so the choice is evidence-based rather than assumed.

Run:  python src/evaluation/hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

LLM_VARIANT = "schema_rules"          # winner of the prompt experiment
COST_PER_LLM_CALL = 2.39 / 1000       # measured $/posting
SENIORITY_LABELS = ["Internship", "Entry level", "Associate",
                    "Mid-Senior level", "Director", "Executive"]


def macro_f1(truth: pd.Series, pred: pd.Series) -> float:
    scores = []
    for lab in SENIORITY_LABELS:
        tp = int(((pred == lab) & (truth == lab)).sum())
        fp = int(((pred == lab) & (truth != lab)).sum())
        fn = int(((pred != lab) & (truth == lab)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores))


def mcnemar(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if x + y == 0 else stats.binomtest(x, x + y, 0.5).pvalue


def bootstrap_diff(a, b, n_iter=5000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(n_iter, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


con = duckdb.connect()

llm = con.execute(f"""
    SELECT job_id, seniority AS llm_seniority
    FROM '{PROCESSED / "llm_extractions_benchmark.parquet"}'
    WHERE variant = '{LLM_VARIANT}'
""").fetchdf()

job_ids = tuple(llm.job_id.tolist())

rules = con.execute(f"""
    SELECT job_id, seniority AS rules_seniority
    FROM '{PROCESSED / "baseline_benchmark.parquet"}'
    WHERE job_id IN {job_ids}
""").fetchdf()

truth = con.execute(f"""
    SELECT job_id, exp_level AS gt_seniority
    FROM '{PROCESSED / "benchmark.parquet"}'
    WHERE job_id IN {job_ids}
""").fetchdf()

df = truth.merge(rules, on="job_id").merge(llm, on="job_id")
df = df[df.gt_seniority.notna()].copy()

print(f"Postings with a seniority label: {len(df)}\n")

# --- three policies ---
df["policy_rules"] = df.rules_seniority
df["policy_llm"] = df.llm_seniority
# Rules first: they are more precise when they fire. LLM fills the silence.
df["policy_hybrid"] = df.rules_seniority.fillna(df.llm_seniority)

policies = {
    "rules only": "policy_rules",
    "LLM only": "policy_llm",
    "hybrid (rules first)": "policy_hybrid",
}

rows = []
hits = {}
for name, col in policies.items():
    hit = (df[col] == df.gt_seniority).to_numpy()
    hits[name] = hit
    llm_calls = 0 if name == "rules only" else (
        len(df) if name == "LLM only" else int(df.rules_seniority.isna().sum())
    )
    rows.append({
        "policy": name,
        "accuracy": f"{100 * hit.mean():.1f}%",
        "macro F1": f"{macro_f1(df.gt_seniority, df[col].fillna('__none__')):.3f}",
        "answer rate": f"{100 * df[col].notna().mean():.1f}%",
        "LLM calls": llm_calls,
        "cost/1k": f"${1000 * llm_calls / len(df) * COST_PER_LLM_CALL:.2f}",
    })

print("=" * 78)
print("SENIORITY: ROUTING POLICIES")
print("=" * 78)
print(pd.DataFrame(rows).to_string(index=False))

print(f"\n{'=' * 78}")
print("HYBRID vs EACH SINGLE METHOD (paired)")
print("=" * 78)

comparisons = []
for other in ("rules only", "LLM only"):
    diff, lo, hi = bootstrap_diff(hits["hybrid (rules first)"], hits[other])
    p = mcnemar(hits["hybrid (rules first)"], hits[other])
    comparisons.append({
        "vs": other,
        "diff": f"{100 * diff:+.1f}pp",
        "95% CI": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
        "p": f"{p:.4f}" if p >= 0.0001 else "<0.0001",
        "significant": "yes" if p < 0.05 else "no",
    })
print(pd.DataFrame(comparisons).to_string(index=False))

# --- why it works: accuracy split by whether the rules fired ---
print(f"\n{'=' * 78}")
print("WHY: accuracy on the rows each method actually handles")
print("=" * 78)

fired = df.rules_seniority.notna()
print(f"\nRules fired on {fired.sum()} postings ({100 * fired.mean():.1f}%)")
print(f"  rules accuracy there : {100 * (df.loc[fired, 'rules_seniority'] == df.loc[fired, 'gt_seniority']).mean():.1f}%")
print(f"  LLM accuracy there   : {100 * (df.loc[fired, 'llm_seniority'] == df.loc[fired, 'gt_seniority']).mean():.1f}%")

print(f"\nRules abstained on {(~fired).sum()} postings ({100 * (~fired).mean():.1f}%)")
print(f"  LLM accuracy there   : {100 * (df.loc[~fired, 'llm_seniority'] == df.loc[~fired, 'gt_seniority']).mean():.1f}%")

print(f"\n{'=' * 78}")
print("FINAL RECOMMENDED PIPELINE")
print("=" * 78)
print("""
  salary, pay period   -> rules   (better AND free)
  abstention           -> rules   (LLM never correctly abstained where rules did)
  seniority            -> rules first, LLM (schema_rules prompt) on abstention
""")

pd.DataFrame(rows).to_csv(PROCESSED / "hybrid_policies.csv", index=False)
print(f"Wrote hybrid_policies.csv")