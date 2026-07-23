"""
Four prompt strategies for the extraction experiment.

All four return the SAME schema, so differences in output are attributable
to the prompt rather than to formatting. Each represents a distinct real
strategy, ordered by increasing cost:

  A  zero_shot     -- minimal instruction. The floor.
  B  few_shot      -- worked examples covering hard distinctions.
  C  schema_rules  -- explicit JSON schema plus rules derived from the
                      baseline error analysis.
  D  decomposed    -- find evidence first, then extract from that evidence.
                      Auditable intermediates, more output tokens.

Variant D deliberately replaces "think step by step". Free-form reasoning is
hard to evaluate; forcing the model to surface evidence spans first gives an
intermediate artifact we can actually check, which is closer to how this
would be built in production.

Few-shot examples are HAND-WRITTEN, not drawn from the benchmark. Sampling
examples from the evaluation set would leak test data into the prompt and
inflate the score.

Run:  python src/llm/prompts.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared output contract
# ---------------------------------------------------------------------------

SENIORITY_LEVELS = [
    "Internship", "Entry level", "Associate",
    "Mid-Senior level", "Director", "Executive",
]
PAY_PERIODS = ["HOURLY", "WEEKLY", "BIWEEKLY", "MONTHLY", "YEARLY"]

SCHEMA_BLOCK = """{
  "salary_min": number or null,
  "salary_max": number or null,
  "pay_period": "HOURLY" | "WEEKLY" | "BIWEEKLY" | "MONTHLY" | "YEARLY" | null,
  "seniority": "Internship" | "Entry level" | "Associate" | "Mid-Senior level" | "Director" | "Executive" | null,
  "years_experience_min": integer or null,
  "required_skills": [list of technology/tool names],
  "preferred_skills": [list of technology/tool names],
  "evidence": {
    "salary": "exact quote or null",
    "seniority": "exact quote or null",
    "years_experience": "exact quote or null"
  }
}"""

# Rules distilled from the baseline error analysis. Every one of these
# corresponds to an observed failure on real postings -- they are not
# guesses about what a model might get wrong.
EXTRACTION_RULES = """RULES:
1. Extract ONLY base compensation for THIS role. Ignore signing bonuses,
   referral bonuses, tuition or education stipends, 401k match, housing
   stipends, and company revenue figures.
2. Postings often quote BOTH an hourly and an annual rate. Report the figure
   and the period that belong together. Do not mix them.
3. If a figure is stated per year, use YEARLY. Per hour, HOURLY. Do not
   annualise or convert anything yourself.
4. If pay is not stated in the text, return null for salary_min, salary_max
   and pay_period. Do not estimate from the job title or the market.
5. Seniority: judge the actual level of THIS role. "Account Executive" and
   "Sales Executive" are individual contributors, not Executive. "Executive
   Assistant" is not Executive. "Division Chief" or "Planning Chief" is
   usually Mid-Senior level, not Executive.
6. When the title carries no seniority word, infer from the description --
   required years, scope, and reporting line. Roles needing no prior
   experience are Entry level.
7. years_experience_min is the MINIMUM required years. "5+ years" -> 5.
   "3-5 years" -> 3. Ignore company age or product age.
8. Skills: name only concrete technologies and tools (SQL, Python, Tableau,
   Excel, AWS). Exclude soft skills. A skill is "preferred" only when the
   text hedges it -- preferred, a plus, nice to have, familiarity with.
   Otherwise it is required.
9. Every non-null value must be supported by text you can quote. If you
   cannot quote it, return null."""


# ---------------------------------------------------------------------------
# A -- zero-shot
# ---------------------------------------------------------------------------

def zero_shot(title: str, description: str) -> str:
    return f"""Extract structured information from this job posting.
Return ONLY valid JSON matching this shape, with no other text:

{SCHEMA_BLOCK}

JOB TITLE: {title}

JOB POSTING:
{description}"""


# ---------------------------------------------------------------------------
# B -- few-shot
# ---------------------------------------------------------------------------

# Each example targets a failure mode found in the baseline:
#   1  benefit figure before the real salary; hedged skills
#   2  both hourly and annual quoted; no seniority word in title
#   3  no pay stated -> must abstain; misleading "Executive" title
_FEW_SHOT_EXAMPLES = """EXAMPLE 1
TITLE: Senior Financial Analyst
POSTING: We offer $5,250 annually toward continuing education and a 401k
match. The salary range for this position is $95,000 - $120,000 per year.
Requires 5+ years of FP&A experience. Advanced Excel required; SQL a plus.
OUTPUT:
{"salary_min": 95000, "salary_max": 120000, "pay_period": "YEARLY",
 "seniority": "Mid-Senior level", "years_experience_min": 5,
 "required_skills": ["Excel"], "preferred_skills": ["SQL"],
 "evidence": {"salary": "The salary range for this position is $95,000 - $120,000 per year",
              "seniority": "Senior Financial Analyst",
              "years_experience": "Requires 5+ years of FP&A experience"}}

EXAMPLE 2
TITLE: Pharmacy Technician
POSTING: Pay Range: Minimum - Hourly $18.50, Maximum - Hourly $24.00.
Annualized equivalent $38,480 - $49,920. No prior experience necessary;
we provide full training. Familiarity with pharmacy software helpful.
OUTPUT:
{"salary_min": 18.50, "salary_max": 24.00, "pay_period": "HOURLY",
 "seniority": "Entry level", "years_experience_min": null,
 "required_skills": [], "preferred_skills": [],
 "evidence": {"salary": "Minimum - Hourly $18.50, Maximum - Hourly $24.00",
              "seniority": "No prior experience necessary; we provide full training",
              "years_experience": null}}

EXAMPLE 3
TITLE: Enterprise Account Executive
POSTING: Join a $30 billion global leader. You will own a territory and
carry a quota. 3-5 years of B2B software sales experience required.
Salesforce proficiency expected. Compensation discussed during interview.
OUTPUT:
{"salary_min": null, "salary_max": null, "pay_period": null,
 "seniority": "Mid-Senior level", "years_experience_min": 3,
 "required_skills": ["Salesforce"], "preferred_skills": [],
 "evidence": {"salary": null,
              "seniority": "You will own a territory and carry a quota",
              "years_experience": "3-5 years of B2B software sales experience required"}}"""


def few_shot(title: str, description: str) -> str:
    return f"""Extract structured information from job postings.
Return ONLY valid JSON matching this shape:

{SCHEMA_BLOCK}

{_FEW_SHOT_EXAMPLES}

Now extract from this posting.

TITLE: {title}

POSTING:
{description}

OUTPUT:"""


# ---------------------------------------------------------------------------
# C -- schema + explicit rules
# ---------------------------------------------------------------------------

def schema_rules(title: str, description: str) -> str:
    return f"""You are extracting structured data from job postings for a
labour-market database. Accuracy matters more than completeness: a null is
better than a guess.

Return ONLY valid JSON matching this schema exactly:

{SCHEMA_BLOCK}

{EXTRACTION_RULES}

JOB TITLE: {title}

JOB POSTING:
{description}"""


# ---------------------------------------------------------------------------
# D -- decomposed (evidence first)
# ---------------------------------------------------------------------------

def decomposed(title: str, description: str) -> str:
    return f"""Extract structured data from this job posting in two stages.

STAGE 1 -- Locate evidence. Quote the exact sentences that state
compensation, seniority, required experience, and required tools. If a
topic is not covered in the posting, write null for it.

STAGE 2 -- Extract values, using ONLY the text you quoted in stage 1.

{EXTRACTION_RULES}

Return ONLY valid JSON in this shape:

{{
  "stage1_evidence": {{
    "compensation_text": "exact quote or null",
    "seniority_text": "exact quote or null",
    "experience_text": "exact quote or null",
    "skills_text": "exact quote or null"
  }},
  "salary_min": number or null,
  "salary_max": number or null,
  "pay_period": "HOURLY" | "WEEKLY" | "BIWEEKLY" | "MONTHLY" | "YEARLY" | null,
  "seniority": "Internship" | "Entry level" | "Associate" | "Mid-Senior level" | "Director" | "Executive" | null,
  "years_experience_min": integer or null,
  "required_skills": [list],
  "preferred_skills": [list],
  "evidence": {{
    "salary": "exact quote or null",
    "seniority": "exact quote or null",
    "years_experience": "exact quote or null"
  }}
}}

JOB TITLE: {title}

JOB POSTING:
{description}"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANTS: dict[str, callable] = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "schema_rules": schema_rules,
    "decomposed": decomposed,
}


def build(variant: str, title: str, description: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    return VARIANTS[variant](title, description)


if __name__ == "__main__":
    import duckdb
    from pathlib import Path

    PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

    row = duckdb.connect().execute(f"""
        SELECT title, description FROM '{PROCESSED / "benchmark.parquet"}'
        WHERE stratum = 'labeled_stated' AND desc_len BETWEEN 1500 AND 2500
        LIMIT 1
    """).fetchdf().iloc[0]

    print(f"{'variant':<14} {'prompt chars':>12} {'overhead':>10}")
    print("-" * 40)
    base = len(row.description) + len(row.title)
    for name in VARIANTS:
        p = build(name, row.title, row.description)
        print(f"{name:<14} {len(p):>12,} {len(p) - base:>10,}")

    print("\n--- schema_rules prompt (first 700 chars) ---")
    print(build("schema_rules", row.title, row.description)[:700])