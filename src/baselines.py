"""
Deterministic (non-LLM) extractors: regex and keyword rules.

These are the baseline the LLM must beat. Building them first means we have
a working pipeline before spending a cent on inference, and it turns the
final question from "which prompt won?" into the better one: "is the LLM
worth its cost, field by field?"

Every extractor returns an evidence span -- the exact text it matched. That
mirrors the contract we'll impose on the LLM, so the two are directly
comparable, and it lets us check whether an answer was grounded or invented.

Run:  python src/baselines.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Output contract -- shared by baselines and (later) the LLM extractor
# ---------------------------------------------------------------------------


@dataclass
class Extraction:
    job_id: int
    salary_min: float | None = None
    salary_max: float | None = None
    pay_period: str | None = None
    years_experience_min: int | None = None
    seniority: str | None = None
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    method: str = "rules"


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

# A dollar figure: $140,000 / $140,000.50 / $140K / $65.00
_AMOUNT = r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*([kK])?"

# A range: two figures joined by a dash, en-dash, or the word "to".
_SALARY_RANGE = re.compile(_AMOUNT + r"\s*(?:-|–|—|to|through)\s*" + _AMOUNT)
_SALARY_SINGLE = re.compile(_AMOUNT)

_HOURLY_HINT = re.compile(r"per\s+hour|hourly|/\s?hr\b|an\s+hour|/hour", re.I)
_YEARLY_HINT = re.compile(r"per\s+year|annual|annually|/\s?yr\b|a\s+year|per\s+annum", re.I)
_MONTHLY_HINT = re.compile(r"per\s+month|monthly|/\s?mo\b", re.I)


def _to_float(number: str, k_suffix: str | None) -> float:
    """'140,000' -> 140000.0 ; '140' with 'K' -> 140000.0"""
    value = float(number.replace(",", ""))
    if k_suffix:
        value *= 1_000
    return value


def extract_salary(text: str) -> tuple[float | None, float | None, str | None, str | None]:
    """Return (min, max, pay_period, evidence).

    Strategy: prefer an explicit range, fall back to a single figure. Pay
    period comes from words near the match; when absent we infer from
    magnitude -- under $500 is almost certainly an hourly rate, not a salary.
    That heuristic is documented rather than hidden because it is the kind
    of assumption that quietly distorts downstream numbers.
    """
    if not text:
        return None, None, None, None

    match = _SALARY_RANGE.search(text)
    if match:
        low = _to_float(match.group(1), match.group(2))
        high = _to_float(match.group(3), match.group(4))
        evidence = match.group(0)
    else:
        match = _SALARY_SINGLE.search(text)
        if not match:
            return None, None, None, None
        low = high = _to_float(match.group(1), match.group(2))
        evidence = match.group(0)

    # Look at a window around the match for period words.
    start, end = max(0, match.start() - 120), min(len(text), match.end() + 120)
    window = text[start:end]

    if _HOURLY_HINT.search(window):
        period = "HOURLY"
    elif _YEARLY_HINT.search(window):
        period = "YEARLY"
    elif _MONTHLY_HINT.search(window):
        period = "MONTHLY"
    else:
        period = "HOURLY" if low < 500 else "YEARLY"

    if low > high:  # ranges occasionally appear reversed
        low, high = high, low

    return low, high, period, evidence.strip()


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

# Output uses LinkedIn's own label space so it can be scored directly
# against formatted_experience_level. Order matters -- first match wins.
_SENIORITY_RULES: list[tuple[str, str]] = [
    (r"\b(intern|internship|co-?op)\b", "Internship"),
    (r"\b(chief|c[teof]o\b|vice president|\bvp\b|head of|executive)\b", "Executive"),
    (r"\bdirector\b", "Director"),
    (r"\b(senior|sr\.?|lead|principal|staff|manager|mgr\.?)\b", "Mid-Senior level"),
    (r"\b(junior|jr\.?|entry[- ]level|new grad|graduate|trainee|apprentice)\b", "Entry level"),
    (r"\bassociate\b", "Associate"),
]

_SENIORITY = [(re.compile(p, re.I), label) for p, label in _SENIORITY_RULES]


def classify_seniority(title: str, description: str = "") -> tuple[str | None, str | None]:
    """Return (level, evidence).

    Title first -- it is far more reliable than the body text, where words
    like "senior leadership" appear constantly without describing the role.
    Returns None rather than guessing when no keyword matches, so that
    abstention shows up honestly in the metrics instead of being hidden
    behind a default label.
    """
    for pattern, label in _SENIORITY:
        m = pattern.search(title or "")
        if m:
            return label, m.group(0)
    return None, None


# ---------------------------------------------------------------------------
# Years of experience
# ---------------------------------------------------------------------------

_YEARS = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:year|yr)s?\b", re.I)


def extract_years_experience(text: str) -> tuple[int | None, str | None]:
    """Return (minimum years, evidence).

    A bare "5 years" is ambiguous -- it might be "5 years of company history".
    So we only accept a match when the word 'experience' appears within an
    80-character window, and we take the smallest qualifying number, since
    job ads state a floor ("5+ years") far more often than a ceiling.
    """
    if not text:
        return None, None

    best: int | None = None
    best_evidence: str | None = None

    for m in _YEARS.finditer(text):
        start, end = max(0, m.start() - 80), min(len(text), m.end() + 80)
        if "experience" not in text[start:end].lower():
            continue
        years = int(m.group(1))
        if years > 30:  # "20 years in business" style noise
            continue
        if best is None or years < best:
            best = years
            best_evidence = text[max(0, m.start() - 30):min(len(text), m.end() + 30)].strip()

    return best, best_evidence


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

# Canonical name -> regex alternatives. Aliases are folded here so that
# "PowerBI", "Power-BI" and "Power BI" all normalise to one entity; without
# canonicalisation the skill counts fragment and precision/recall break.
_SKILL_PATTERNS: dict[str, str] = {
    "SQL": r"\bsql\b",
    "Python": r"\bpython\b",
    "Excel": r"\bexcel\b",
    "Tableau": r"\btableau\b",
    "Power BI": r"\bpower\s?-?bi\b|\bpowerbi\b",
    "Looker": r"\blooker\b",
    "Snowflake": r"\bsnowflake\b",
    "Redshift": r"\bredshift\b",
    "BigQuery": r"\bbig\s?query\b",
    "Databricks": r"\bdatabricks\b",
    "dbt": r"\bdbt\b",
    "Airflow": r"\bairflow\b",
    "Spark": r"\b(apache\s+)?spark\b",
    "Hadoop": r"\bhadoop\b",
    "Kafka": r"\bkafka\b",
    "AWS": r"\baws\b|\bamazon web services\b",
    "Azure": r"\bazure\b",
    "GCP": r"\bgcp\b|\bgoogle cloud\b",
    "Java": r"\bjava\b(?!script)",
    "Scala": r"\bscala\b",
    "SAS": r"\bsas\b",
    "SPSS": r"\bspss\b",
    "Git": r"\bgit\b|\bgithub\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "ETL": r"\betl\b|\belt\b",
    "Machine Learning": r"\bmachine learning\b|\bml\b",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "Scikit-learn": r"\bscikit-?learn\b|\bsklearn\b",
    "TensorFlow": r"\btensorflow\b",
    "PyTorch": r"\bpytorch\b",
    "PostgreSQL": r"\bpostgre?s(ql)?\b",
    "MySQL": r"\bmysql\b",
    "MongoDB": r"\bmongo\s?db\b",
    "Oracle": r"\boracle\b",
    "Alteryx": r"\balteryx\b",
    "VBA": r"\bvba\b",
    "JavaScript": r"\bjavascript\b",
    "Salesforce": r"\bsalesforce\b",
    "Snowpark": r"\bsnowpark\b",
    "SSIS": r"\bssis\b",
    "SSRS": r"\bssrs\b",
}

# Note: bare "R" is deliberately excluded. Matching a single capital letter
# produces far too many false positives (bullet markers, "R&D", initials).
# It is a known gap and a good example of where an LLM should win.

_SKILLS = {name: re.compile(p, re.I) for name, p in _SKILL_PATTERNS.items()}

# Words that signal a skill is optional rather than mandatory.
_PREFERRED_HINT = re.compile(
    r"preferred|nice to have|nice-to-have|a plus|bonus|desirable|would be great|"
    r"ideally|familiarity with|exposure to|good to have",
    re.I,
)


def extract_skills(text: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (required, preferred, evidence-by-skill).

    Required vs preferred is decided by looking at a window around each
    match for hedging language. This is crude on purpose -- it is precisely
    the distinction a keyword matcher cannot really make, and therefore the
    clearest place to see whether the LLM earns its cost.
    """
    if not text:
        return [], [], {}

    required: list[str] = []
    preferred: list[str] = []
    evidence: dict[str, str] = {}

    for name, pattern in _SKILLS.items():
        m = pattern.search(text)
        if not m:
            continue

        start, end = max(0, m.start() - 200), min(len(text), m.end() + 100)
        window = text[start:end]

        if _PREFERRED_HINT.search(window):
            preferred.append(name)
        else:
            required.append(name)

        evidence[name] = text[max(0, m.start() - 40):min(len(text), m.end() + 40)].strip()

    return sorted(required), sorted(preferred), evidence


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def extract_all(job_id: int, title: str, description: str) -> Extraction:
    """Run every rule-based extractor over one posting."""
    s_min, s_max, period, s_ev = extract_salary(description)
    years, y_ev = extract_years_experience(description)
    level, l_ev = classify_seniority(title, description)
    required, preferred, skill_ev = extract_skills(description)

    evidence: dict[str, str] = {}
    if s_ev:
        evidence["salary"] = s_ev
    if y_ev:
        evidence["years_experience"] = y_ev
    if l_ev:
        evidence["seniority"] = l_ev
    evidence.update({f"skill:{k}": v for k, v in skill_ev.items()})

    return Extraction(
        job_id=job_id,
        salary_min=s_min,
        salary_max=s_max,
        pay_period=period,
        years_experience_min=years,
        seniority=level,
        required_skills=required,
        preferred_skills=preferred,
        evidence=evidence,
    )


def run_on_benchmark() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"SELECT job_id, title, description FROM '{PROCESSED / 'benchmark.parquet'}'"
    ).fetchdf()

    rows = [
        extract_all(r.job_id, r.title, r.description)
        for r in df.itertuples(index=False)
    ]

    out = pd.DataFrame([
        {
            "job_id": e.job_id,
            "salary_min": e.salary_min,
            "salary_max": e.salary_max,
            "pay_period": e.pay_period,
            "years_experience_min": e.years_experience_min,
            "seniority": e.seniority,
            "required_skills": e.required_skills,
            "preferred_skills": e.preferred_skills,
            "n_required": len(e.required_skills),
            "n_preferred": len(e.preferred_skills),
            "evidence_salary": e.evidence.get("salary"),
            "evidence_years": e.evidence.get("years_experience"),
            "evidence_seniority": e.evidence.get("seniority"),
            "method": e.method,
        }
        for e in rows
    ])

    path = PROCESSED / "baseline_extractions.parquet"
    out.to_parquet(path, index=False)
    return out


if __name__ == "__main__":
    result = run_on_benchmark()
    total = len(result)

    print(f"\nExtracted {total:,} postings with rule-based methods\n")
    print("Coverage (% of postings where the rule produced an answer)")
    print("-" * 55)
    for col in ("salary_min", "pay_period", "years_experience_min", "seniority"):
        pct = 100 * result[col].notna().mean()
        print(f"  {col:24s} {pct:5.1f}%")

    has_skill = (result["n_required"] + result["n_preferred"]) > 0
    print(f"  {'any skill found':24s} {100 * has_skill.mean():5.1f}%")

    print(f"\nMean skills per posting: "
          f"{result['n_required'].mean():.1f} required, "
          f"{result['n_preferred'].mean():.1f} preferred")

    print("\nPay period distribution")
    print(result["pay_period"].value_counts(dropna=False).to_string())

    print("\nSeniority distribution")
    print(result["seniority"].value_counts(dropna=False).to_string())

    print("\nSample extractions")
    print(result[["job_id", "salary_min", "salary_max", "pay_period",
                  "years_experience_min", "seniority", "n_required"]].head(10).to_string(index=False))

    print(f"\nWrote {PROCESSED / 'baseline_extractions.parquet'}")