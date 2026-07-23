"""
Deterministic (non-LLM) extractors: regex and keyword rules.

These are the baseline the LLM must beat. Building them first means we have
a working pipeline before spending a cent on inference, and it turns the
final question from "which prompt won?" into the better one: "is the LLM
worth its cost, field by field?"

Every extractor returns an evidence span -- the exact text it matched. That
mirrors the contract we'll impose on the LLM, so the two are directly
comparable, and it lets us check whether an answer was grounded or invented.

REVISION NOTE (after error analysis on 2,000 postings):
  Four bugs were found by inspecting the worst failures and are fixed here.
  Each is marked FIX-n below with the real posting that exposed it. The
  baseline had to be corrected before comparing against an LLM -- measuring
  a strong model against a broken baseline would flatter the model.

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

# FIX-1: number truncation.
# The old pattern was \d{1,3}(?:,\d{3})*  -- the * allows ZERO comma groups,
# so "163680.00" matched the first three digits and stopped, yielding 163.
# Requiring + on the comma-separated branch means an unformatted number must
# fall through to the plain-digits branch and be captured whole.
#   Broke on: job 3906254848 "$ 163680.00" -> 163
#             job 3886851948 "$1840.90"    -> 184
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*([kK])?"

_AMOUNT = r"\$\s*" + _NUM          # requires a dollar sign
_AMOUNT_OPT = r"\$?\s*" + _NUM     # dollar sign optional (second half of a range)

# FIX-2: ranges where only the first figure carries a dollar sign.
#   Broke on: job 3895206655 "$50-60K" -> matched only "$50"
_SALARY_RANGE = re.compile(_AMOUNT + r"\s*(?:-|–|—|to|through)\s*" + _AMOUNT_OPT)
_SALARY_SINGLE = re.compile(_AMOUNT)

# FIX-3: scale words. "$30 billion" is company revenue, not compensation.
#   Broke on: job 3903456111 "$224 billion (retail investment client assets)"
_SCALE_WORD = re.compile(r"^\s*(billion|million|trillion|bn|mm)\b", re.I)

# Ordered so that more specific periods are tried first; ties are broken by
# proximity, not by list position.
_PERIOD_HINTS: list[tuple[str, re.Pattern]] = [
    ("HOURLY",   re.compile(r"per\s+hour|hourly|/\s?hr\b|an\s+hour|/hour|\bph\b", re.I)),
    ("WEEKLY",   re.compile(r"per\s+week\b|/\s?wk\b|(?:gross\s+)?weekly\s+(?:pay|rate|salary|earnings)", re.I)),
    ("BIWEEKLY", re.compile(r"bi-?weekly|every\s+two\s+weeks", re.I)),
    ("MONTHLY",  re.compile(r"per\s+month|monthly|/\s?mo\b|a\s+month", re.I)),
    ("YEARLY",   re.compile(r"per\s+year|annual|annually|/\s?yr\b|a\s+year|per\s+annum", re.I)),
]


def _to_float(number: str, k_suffix: str | None) -> float:
    """'140,000' -> 140000.0 ; '140' with 'K' -> 140000.0"""
    value = float(number.replace(",", ""))
    if k_suffix:
        value *= 1_000
    return value


def _nearest_period(text: str, start: int, end: int) -> str | None:
    """Find the period word CLOSEST to the matched figure.

    FIX-4: postings routinely quote both an hourly and an annual rate in the
    same paragraph. The old code scanned a +/-120 char window and took the
    first hint it found, which meant it picked whichever word appeared
    earlier in the string rather than whichever describes THIS number. That
    single bug produced roughly 9 of the 15 worst errors.
      Broke on: "Salary Grade Minimum - Annual $108,400 ... Hourly $52.12"
                "Salary/Hourly Rate $130k - $180k Annually"
    """
    lo, hi = max(0, start - 150), min(len(text), end + 150)
    window = text[lo:hi]
    best_period: str | None = None
    best_distance = 10**9

    for period, pattern in _PERIOD_HINTS:
        for m in pattern.finditer(window):
            pos = lo + m.start()
            # Distance 0 if the hint sits inside the match itself.
            distance = 0 if start <= pos <= end else min(abs(pos - end), abs(start - pos))
            if distance < best_distance:
                best_period, best_distance = period, distance

    return best_period


def _find_amount(text: str) -> tuple[re.Match | None, bool]:
    """First plausible money match, skipping scale-word figures.

    Returns (match, is_range).
    """
    for m in _SALARY_RANGE.finditer(text):
        if not _SCALE_WORD.match(text[m.end():m.end() + 12]):
            return m, True
    for m in _SALARY_SINGLE.finditer(text):
        if not _SCALE_WORD.match(text[m.end():m.end() + 12]):
            return m, False
    return None, False


def extract_salary(text: str) -> tuple[float | None, float | None, str | None, str | None]:
    """Return (min, max, pay_period, evidence).

    Known limitation, deliberately left in: this takes the FIRST money
    figure in the description. When a posting leads with a benefit
    ("$5,250 annually toward education") the extractor grabs that instead
    of the salary. A regex has no way to tell which figure is compensation.
    That is precisely the judgement an LLM should provide, so it stays as a
    documented gap rather than being papered over with more heuristics.
    """
    if not text:
        return None, None, None, None

    match, is_range = _find_amount(text)
    if match is None:
        return None, None, None, None

    if is_range:
        low = _to_float(match.group(1), match.group(2))
        high = _to_float(match.group(3), match.group(4))
        # "$50-60K": the K binds only to the second figure. Propagate it when
        # the first is implausibly small relative to the second.
        if match.group(4) and not match.group(2) and low < high / 100:
            low *= 1_000
    else:
        low = high = _to_float(match.group(1), match.group(2))

    period = _nearest_period(text, match.start(), match.end())

    # Magnitude sanity check. An "hourly" rate of $108,400 or a "yearly"
    # salary of $33 is a misread, not a real figure. Stated openly because
    # it is an assumption, and assumptions should be visible in the code.
    # Magnitude sanity checks. Stated openly because these are assumptions.
    if period == "HOURLY" and low > 1_000:
        period = "YEARLY"
    elif period == "YEARLY" and low < 200:
        period = "HOURLY"
    # HYPOTHESIS (validate on holdout): US federal minimum wage full-time is
    # ~$15,080/yr, so a "yearly" figure between $1k and $20k is far more
    # likely a monthly salary -- the pattern in state-government postings
    # that quote "$3,640.00 - $4,561.00" with no period word at all.
    elif period == "YEARLY" and 1_000 < low < 20_000:
        period = "MONTHLY"
    elif period is None:
        period = "HOURLY" if low < 500 else "YEARLY"

    if low > high:  # ranges occasionally appear reversed
        low, high = high, low

    return low, high, period, match.group(0).strip()


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

# FIX-5: titles where a seniority keyword does NOT indicate seniority.
# "Account Executive" is a salesperson. "Executive Assistant" supports an
# executive. Error analysis found 39 false Executive labels, and these two
# patterns accounted for nearly all of them.
_SENIORITY_EXCLUSIONS = re.compile(
    r"(account|sales|client|enterprise|outside|inside|healthcare|"
    r"business\s+development)\s+executive"
    r"|executive\s+(assistant|administrative|admin|support|secretary)"
    r"|assistant\s+to\s+(the\s+)?(ceo|president|vice\s+president|vp|leadership)"
    r"|support\s+(for|to)\s+(the\s+)?(ceo|cto|cfo|coo|president|vp)",
    re.I,
)

# When the title is an assistant/support role, the Executive rule is
# suppressed entirely -- otherwise "Executive Assistant to the Vice
# President" still matches on the leftover "Vice President".
_ASSISTANT_ROLE = re.compile(
    r"\b(assistant|secretary|receptionist|coordinator)\b|support\s+(for|to)\b",
    re.I,
)

# Output uses LinkedIn's own label space so it can be scored directly
# against formatted_experience_level. Order matters -- first match wins.
_SENIORITY_RULES: list[tuple[str, str]] = [
    (r"\b(intern|internship|co-?op)\b", "Internship"),
    # Restricted to genuine C-suite. Bare "chief" was catching "Division
    # Chief" and "Planning Chief", which are middle-management titles.
    (r"\bc[teofim]o\b|\bchief\s+\w+\s+officer\b|\bvice\s+president\b|\bvp\b|\bhead\s+of\b",
     "Executive"),
    (r"\bdirector\b", "Director"),
    (r"\b(senior|sr\.?|lead|principal|staff|manager|mgr\.?)\b", "Mid-Senior level"),
    (r"\b(junior|jr\.?|entry[- ]level|new grad|graduate|trainee|apprentice)\b", "Entry level"),
    (r"\bassociate\b", "Associate"),
]

_SENIORITY = [(re.compile(p, re.I), label) for p, label in _SENIORITY_RULES]


def classify_seniority(title: str, description: str = "") -> tuple[str | None, str | None]:
    """Return (level, evidence).

    Title only -- the body text says "senior leadership" constantly without
    describing the role being advertised. Returns None rather than guessing
    when no keyword matches, so abstention shows up honestly in the metrics
    instead of hiding behind a default label.

    Known limitation: 444 postings labelled Entry level by LinkedIn carry no
    seniority keyword at all (File Clerk, Pharmacy Technician, Dental
    Assistant). No title-based rule can reach them. This is the single
    clearest place for an LLM to earn its cost, and it is left unfixed on
    purpose so the comparison is honest.
    """
    original = title or ""
    working = original
    suppress_executive = False

    if _SENIORITY_EXCLUSIONS.search(working):
        working = _SENIORITY_EXCLUSIONS.sub(" ", working)
        if _ASSISTANT_ROLE.search(original):
            suppress_executive = True

    for pattern, label in _SENIORITY:
        if label == "Executive" and suppress_executive:
            continue
        m = pattern.search(working)
        if m:
            return label, m.group(0)
    return None, None


# ---------------------------------------------------------------------------
# Years of experience
# ---------------------------------------------------------------------------

_YEARS = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:year|yr)s?\b",
    re.I,
)


def extract_years_experience(text: str) -> tuple[int | None, str | None]:
    """Return (minimum years, evidence).

    A bare "5 years" is ambiguous -- it might be "5 years in business". So a
    match only counts when "experience" appears within an 80-character
    window, and we take the smallest qualifying number, since job ads state
    a floor ("5+ years") far more often than a ceiling.
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

# Canonical name -> regex alternatives. Aliases fold here so "PowerBI",
# "Power-BI" and "Power BI" all normalise to one entity; without
# canonicalisation the counts fragment and precision/recall break.
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
# catches bullet markers, "R&D", and initials. It is a known gap and a clean
# example of where an LLM should win.

_SKILLS = {name: re.compile(p, re.I) for name, p in _SKILL_PATTERNS.items()}

# Words signalling a skill is optional rather than mandatory.
_PREFERRED_HINT = re.compile(
    r"preferred|nice to have|nice-to-have|a plus|bonus|desirable|would be great|"
    r"ideally|familiarity with|exposure to|good to have",
    re.I,
)


def extract_skills(text: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (required, preferred, evidence-by-skill).

    Required vs preferred is decided by scanning a window around each match
    for hedging language. This is crude on purpose -- descriptions routinely
    say "Python required; Spark preferred" in one sentence, which a window
    cannot separate. It is the clearest place to see whether the LLM earns
    its cost.
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


def run_on_benchmark(dataset: str = "benchmark") -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"SELECT job_id, title, description FROM '{PROCESSED / f'{dataset}.parquet'}'"
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

    path = PROCESSED / f"baseline_{dataset}.parquet"
    out.to_parquet(path, index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark",
                        choices=["benchmark", "holdout"],
                        help="benchmark = tuning set; holdout = unseen validation set")
    args = parser.parse_args()

    result = run_on_benchmark(args.dataset)
    total = len(result)

    print(f"\nDataset: {args.dataset}")
    print(f"Extracted {total:,} postings with rule-based methods\n")
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

    print(f"\nWrote {PROCESSED / 'baseline_extractions.parquet'}")
