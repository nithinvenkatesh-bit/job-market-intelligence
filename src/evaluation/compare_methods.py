"""
Head-to-head: rules baseline vs four LLM prompt variants.

Measurement principles carried over from the baseline evaluation:

  STRICT METRICS -- abstentions count as misses. A method answering 38% of
  the time and being right when it answers is not a 74%-accurate method.

  ANNUALISED SALARY -- "$60" read as yearly instead of hourly is a 2,080x
  error, not a rounding difference.

  PAIRED TESTS -- every method saw the same postings, so McNemar's exact
  test and paired bootstrap apply. This matters: unpaired at n=400, even a
  7pp difference can fail to reach significance; paired, ~2pp is detectable.

  MULTIPLE COMPARISONS -- this script runs ~24 tests. At alpha=0.05 that is
  roughly one false positive expected by chance, so raw p-values overstate
  confidence. Holm-Bonferroni corrections are reported at the end, under two
  framings that disagree.

  COST FROM TOKENS -- reruns replay from cache, so recorded cost is 0.
  Token counts survive caching, so cost is recomputed from them.

Run:  python src/evaluation/compare_methods.py
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

INPUT_COST_PER_MTOK = 1.00
OUTPUT_COST_PER_MTOK = 5.00

HOURS_PER_YEAR = 2080
ANNUALISE = {"HOURLY": HOURS_PER_YEAR, "WEEKLY": 52,
             "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}

SENIORITY_LABELS = ["Internship", "Entry level", "Associate",
                    "Mid-Senior level", "Director", "Executive"]

# Measured during the first (uncached) run. Latency cannot be recovered from
# cached responses, so it is recorded here rather than silently reported as 0.
MEASURED_LATENCY_S = {
    "rules": 0.0004,      # regex over 2,000 postings, per posting
    "zero_shot": 2.13,
    "schema_rules": 2.09,
    "few_shot": 2.01,
    "decomposed": 2.72,
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes.

    Only the DISCORDANT pairs carry information: postings where one method
    succeeded and the other failed. Agreements tell us nothing about which
    is better. The exact binomial form is used rather than the chi-square
    approximation because discordant counts here are often small.
    """
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, stats.binomtest(a_only, n, 0.5).pvalue


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_iter: int = 5000,
                   seed: int = 42) -> tuple[float, float, float]:
    """Paired percentile bootstrap CI for mean(a) - mean(b).

    Postings are resampled together, preserving pairing. A CI that spans
    zero means the observed difference is compatible with no difference.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(n_iter, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def holm(pvalues: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction.

    Returns (adjusted p-values, rejected) in the original order.

    Uniformly more powerful than Bonferroni at the same family-wise error
    rate. The running max enforces monotonicity: an adjusted p-value can
    never fall below that of a test with a smaller raw p-value.
    """
    m = len(pvalues)
    order_idx = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order_idx):
        running_max = max(running_max, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running_max
    return adjusted, [adjusted[i] <= alpha for i in range(m)]


def macro_f1(truth: pd.Series, pred: pd.Series, labels: list[str]) -> float:
    """Macro F1 over ALL rows -- a null prediction is a false negative.

    Macro rather than micro so rare classes (Executive, Internship) cannot
    be hidden behind good performance on the dominant class.
    """
    scores = []
    for lab in labels:
        tp = int(((pred == lab) & (truth == lab)).sum())
        fp = int(((pred == lab) & (truth != lab)).sum())
        fn = int(((pred != lab) & (truth == lab)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores))


def annualise(value, period) -> float:
    if pd.isna(value) or period not in ANNUALISE:
        return np.nan
    return float(value) * ANNUALISE[period]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

con = duckdb.connect()

llm = con.execute(f"""
    SELECT * FROM '{PROCESSED / "llm_extractions_benchmark.parquet"}'
""").fetchdf()

# The LLM ran on a 400-posting subsample; restrict everything to those ids
# so all methods are compared on identical items.
job_ids = sorted(llm.job_id.unique())
print(f"Evaluating {len(job_ids)} postings x {llm.variant.nunique()} LLM variants + rules baseline")

truth = con.execute(f"""
    SELECT job_id, stratum, pay_period AS gt_pay_period,
           min_salary AS gt_min_raw, med_salary AS gt_med_raw,
           exp_level AS gt_seniority
    FROM '{PROCESSED / "benchmark.parquet"}'
""").fetchdf()
truth = truth[truth.job_id.isin(job_ids)].copy()
truth["gt_min"] = truth.gt_min_raw.fillna(truth.gt_med_raw)
truth["gt_annual"] = [annualise(v, p) for v, p in zip(truth.gt_min, truth.gt_pay_period)]

rules = con.execute(f"""
    SELECT job_id, salary_min, salary_max, pay_period, seniority,
           years_experience_min, n_required, n_preferred
    FROM '{PROCESSED / "baseline_benchmark.parquet"}'
""").fetchdf()
rules = rules[rules.job_id.isin(job_ids)].copy()
rules["variant"] = "rules"
rules["input_tokens"] = 0
rules["output_tokens"] = 0
rules["valid_json"] = True
rules["evidence_salary"] = None

methods = pd.concat([rules, llm], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-posting correctness vectors
# ---------------------------------------------------------------------------


def score_method(df: pd.DataFrame) -> dict:
    """Compute every metric for one method, plus per-posting hit vectors.

    The vectors are what the significance tests consume -- aggregate scores
    alone cannot tell you whether a difference is real.
    """
    m = truth.merge(df, on="job_id", how="left")
    m["pred_annual"] = [annualise(v, p) for v, p in zip(m.salary_min, m.pay_period)]

    stated = m.stratum == "labeled_stated"
    not_stated = m.stratum == "labeled_not_stated"

    # --- salary: correct only if within 10% AFTER annualisation ---
    err = (m.pred_annual - m.gt_annual).abs() / m.gt_annual.replace(0, np.nan)
    salary_hit = (stated & (err <= 0.10)).fillna(False).to_numpy()

    # --- abstention: on postings with no pay in the text, silence is right ---
    abstain_hit = (not_stated & m.salary_min.isna()).to_numpy()

    # --- pay period, over all stated rows (null counts as wrong) ---
    period_hit = (stated & (m.pay_period == m.gt_pay_period)).fillna(False).to_numpy()

    # --- seniority, strict ---
    labelled = m.gt_seniority.notna()
    seniority_hit = (labelled & (m.seniority == m.gt_seniority)).fillna(False).to_numpy()

    sub = m[labelled]
    f1 = macro_f1(sub.gt_seniority, sub.seniority.fillna("__none__"), SENIORITY_LABELS)

    in_tok, out_tok = m.input_tokens.mean(), m.output_tokens.mean()
    cost_per_1k = (in_tok / 1e6 * INPUT_COST_PER_MTOK
                   + out_tok / 1e6 * OUTPUT_COST_PER_MTOK) * 1000

    return {
        "n_stated": int(stated.sum()),
        "n_not_stated": int(not_stated.sum()),
        "n_labelled_seniority": int(labelled.sum()),
        "salary_10pct": salary_hit[stated.to_numpy()].mean(),
        "abstention": abstain_hit[not_stated.to_numpy()].mean(),
        "pay_period": period_hit[stated.to_numpy()].mean(),
        "seniority_acc": seniority_hit[labelled.to_numpy()].mean(),
        "seniority_f1": f1,
        "seniority_answer_rate": float(sub.seniority.notna().mean()),
        "valid_json": float(m.valid_json.mean()),
        "cost_per_1k_postings": cost_per_1k,
        # vectors for paired tests
        "_salary": salary_hit,
        "_abstain": abstain_hit,
        "_period": period_hit,
        "_seniority": seniority_hit,
        "_stated_mask": stated.to_numpy(),
        "_not_stated_mask": not_stated.to_numpy(),
        "_labelled_mask": labelled.to_numpy(),
    }


scored = {name: score_method(g) for name, g in methods.groupby("variant")}


# ---------------------------------------------------------------------------
# Headline table
# ---------------------------------------------------------------------------

print(f"\n{'=' * 96}\nACCURACY BY METHOD\n{'=' * 96}")

order = ["rules", "zero_shot", "few_shot", "schema_rules", "decomposed"]
order = [o for o in order if o in scored]

table = pd.DataFrame([
    {
        "method": name,
        "salary ±10%": f"{100 * s['salary_10pct']:.1f}%",
        "pay period": f"{100 * s['pay_period']:.1f}%",
        "abstention": f"{100 * s['abstention']:.1f}%",
        "seniority F1": f"{s['seniority_f1']:.3f}",
        "seniority acc": f"{100 * s['seniority_acc']:.1f}%",
        "answer rate": f"{100 * s['seniority_answer_rate']:.1f}%",
        "valid JSON": f"{100 * s['valid_json']:.1f}%",
    }
    for name, s in ((o, scored[o]) for o in order)
])
print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Cost and latency
# ---------------------------------------------------------------------------

print(f"\n{'=' * 96}\nCOST AND LATENCY (per 1,000 postings)\n{'=' * 96}")

cost_table = pd.DataFrame([
    {
        "method": name,
        "cost/1k": f"${scored[name]['cost_per_1k_postings']:.2f}",
        "latency": f"{MEASURED_LATENCY_S.get(name, float('nan')):.2f}s",
        "hours/1k (8 workers)": f"{MEASURED_LATENCY_S.get(name, 0) * 1000 / 8 / 3600:.2f}",
        "salary ±10%": f"{100 * scored[name]['salary_10pct']:.1f}%",
        "seniority F1": f"{scored[name]['seniority_f1']:.3f}",
    }
    for name in order
])
print(cost_table.to_string(index=False))


# ---------------------------------------------------------------------------
# Significance tests vs the rules baseline
# ---------------------------------------------------------------------------

print(f"\n{'=' * 96}\nSIGNIFICANCE vs RULES BASELINE (paired, same postings)\n{'=' * 96}")
print("Raw p-values -- see the correction section below before quoting these.")
print("CI spanning zero = difference indistinguishable from noise\n")

base = scored["rules"]
rows = []

for name in order:
    if name == "rules":
        continue
    s = scored[name]
    for metric, key, mask_key in [
        ("salary ±10%", "_salary", "_stated_mask"),
        ("pay period", "_period", "_stated_mask"),
        ("abstention", "_abstain", "_not_stated_mask"),
        ("seniority", "_seniority", "_labelled_mask"),
    ]:
        mask = base[mask_key]
        a, b = s[key][mask], base[key][mask]
        diff, lo, hi = bootstrap_diff(a, b)
        a_only, b_only, p = mcnemar(a, b)
        rows.append({
            "method": name,
            "metric": metric,
            "diff": f"{100 * diff:+.1f}pp",
            "95% CI": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
            "LLM only": a_only,
            "rules only": b_only,
            "p_raw": p,
            "p": f"{p:.4f}" if p >= 0.0001 else "<0.0001",
        })

sig = pd.DataFrame(rows)
for metric in sig.metric.unique():
    print(f"--- {metric} ---")
    print(sig[sig.metric == metric]
          .drop(columns=["metric", "p_raw"]).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Best LLM variant vs the others
# ---------------------------------------------------------------------------

llm_only = [o for o in order if o != "rules"]
best = max(llm_only, key=lambda n: scored[n]["seniority_f1"] + scored[n]["salary_10pct"])

print(f"{'=' * 96}\nBEST VARIANT ({best}) vs OTHER PROMPTS\n{'=' * 96}\n")

prompt_rows = []
for name in llm_only:
    if name == best:
        continue
    for metric, key, mask_key in [("salary ±10%", "_salary", "_stated_mask"),
                                  ("seniority", "_seniority", "_labelled_mask")]:
        mask = base[mask_key]
        a, b = scored[best][key][mask], scored[name][key][mask]
        diff, lo, hi = bootstrap_diff(a, b)
        _, _, p = mcnemar(a, b)
        prompt_rows.append({"vs": name, "metric": metric,
                            "diff": f"{100 * diff:+.1f}pp",
                            "95% CI": f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                            "p_raw": p,
                            "p": f"{p:.4f}"})
prompts = pd.DataFrame(prompt_rows)
print(prompts.drop(columns="p_raw").to_string(index=False))


# ---------------------------------------------------------------------------
# Evidence-value consistency
# ---------------------------------------------------------------------------

print(f"\n{'=' * 96}\nEVIDENCE-VALUE CONSISTENCY\n{'=' * 96}")
print("Cases where the model quoted salary evidence but returned a null value.")
print("Observed on the very first test call, so it is measured, not assumed.\n")

rows = []
for name in llm_only:
    g = llm[llm.variant == name]
    quoted = g.evidence_salary.notna() & (g.evidence_salary.astype(str).str.strip() != "")
    contradiction = quoted & g.salary_min.isna()
    rows.append({
        "variant": name,
        "quoted evidence": int(quoted.sum()),
        "but value null": int(contradiction.sum()),
        "inconsistency": f"{100 * contradiction.sum() / max(1, quoted.sum()):.1f}%",
    })
print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

print(f"\n{'=' * 96}\nMULTIPLE-COMPARISON CORRECTION (Holm-Bonferroni)\n{'=' * 96}")
print("""
This analysis runs ~24 significance tests. At alpha=0.05 roughly one false
positive is expected by chance alone, so raw p-values overstate confidence.

Two framings are reported because they disagree, and choosing between them is
a judgement call rather than a fact:

  WITHIN-FAMILY   Each metric is treated as a separate question, so the
                  correction is applied within metric. More powerful -- but
                  the families were defined AFTER seeing results, which is a
                  real limitation and is stated as one.

  WHOLE-FAMILY    All tests corrected together. Conservative, assumption-free,
                  and the safer number to quote.

Where the two disagree, prefer the conservative reading, and lead with the
effect size and confidence interval rather than the p-value.
""")

# Combine both families of tests into one frame.
corr = pd.concat([
    sig.assign(comparison=sig.method + " vs rules")[
        ["comparison", "metric", "diff", "95% CI", "p", "p_raw"]],
    prompts.assign(comparison=f"{best} vs " + prompts["vs"])[
        ["comparison", "metric", "diff", "95% CI", "p", "p_raw"]],
], ignore_index=True)

# Within-metric correction.
parts = []
for metric, group in corr.groupby("metric", sort=False):
    adj, rej = holm(group["p_raw"].tolist())
    g = group.copy()
    g["holm_within"] = [f"{a:.4f}" for a in adj]
    g["sig_within"] = ["yes" if r else "no" for r in rej]
    parts.append(g)
corr = pd.concat(parts).sort_index()

# Whole-family correction across every test above.
adj_all, rej_all = holm(corr["p_raw"].tolist())
corr["holm_all"] = [f"{a:.4f}" for a in adj_all]
corr["sig_all"] = ["yes" if r else "no" for r in rej_all]

print(corr[["comparison", "metric", "diff", "95% CI", "p",
            "holm_within", "sig_within", "holm_all", "sig_all"]].to_string(index=False))

n_within = (corr.sig_within == "yes").sum()
n_all = (corr.sig_all == "yes").sum()
print(f"\nSurvive within-family correction: {n_within}/{len(corr)}")
print(f"Survive whole-family correction : {n_all}/{len(corr)}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

table.to_csv(PROCESSED / "method_comparison.csv", index=False)
sig.drop(columns="p_raw").to_csv(PROCESSED / "significance_tests.csv", index=False)
corr.drop(columns="p_raw").to_csv(PROCESSED / "significance_corrected.csv", index=False)
print("\nWrote method_comparison.csv, significance_tests.csv, "
      "significance_corrected.csv")