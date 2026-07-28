"""
Gold-set-only prompt variants for work-arrangement extraction.

This module is intentionally separate from src/llm/prompts.py. The original
prompt contract produced the benchmark results and must remain unchanged for
reproducibility.

All four variants return the same core fields:

{
  "work_arrangement": "remote" | "hybrid" | "onsite" | "unclear",
  "evidence_work_arrangement": "exact quote or null"
}
"""

from __future__ import annotations


ARRANGEMENT_CLASSES = [
    "remote",
    "hybrid",
    "onsite",
    "unclear",
]

SCHEMA_BLOCK = """{
  "work_arrangement": "remote" | "hybrid" | "onsite" | "unclear",
  "evidence_work_arrangement": "exact quote or null"
}"""


ARRANGEMENT_RULES = """RULES:
1. Classify the working arrangement for THIS specific job, not the company,
   a facility, another team, clients, or employees generally.

2. remote:
   The posting explicitly says this role is remote, fully remote,
   work-from-home, virtual, or may work remotely.
   Geographic restrictions do not change the label:
   "remote within California" is still remote.

3. hybrid:
   The posting explicitly says this role is hybrid or requires some recurring
   office attendance while allowing some remote work.
   Examples: "three days per week in office", "8-12 office days per month".

4. onsite:
   The posting explicitly says this role is onsite, on-site, in person,
   office-based, or requires full-time presence at a named workplace.

5. unclear:
   Use unclear when the role's arrangement is not explicitly stated.
   A city, office address, headquarters, or job location alone is not proof
   that the role is onsite.

6. Ignore incidental uses of arrangement words, including:
   - an onsite manufacturing facility
   - onsite/offshore engineering teams
   - client-site work described only as a possibility
   - a company described as having a hybrid or distributed workforce
   - generic boilerplate explaining what happens "if this role is remote,
     hybrid, or onsite"
   These do not classify the role unless the text explicitly connects the
   arrangement to this position.

7. If several arrangements are mentioned, use the explicit arrangement
   assigned to this role. For example, "Hybrid On-Site" or a stated office-day
   requirement is hybrid, not onsite.

8. Return an exact supporting quote. For unclear, return null unless the text
   explicitly says the arrangement will be determined later.

9. Return only valid JSON."""


# ---------------------------------------------------------------------------
# A — zero-shot
# ---------------------------------------------------------------------------

def zero_shot(title: str, description: str) -> str:
    return f"""Extract the work arrangement for this job posting.

Return ONLY valid JSON matching this shape:

{SCHEMA_BLOCK}

JOB TITLE: {title}

JOB POSTING:
{description}"""


# ---------------------------------------------------------------------------
# B — few-shot
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """EXAMPLE 1
TITLE: Data Analyst
POSTING: This position is fully remote within California. Employees must
reside in the state.
OUTPUT:
{"work_arrangement": "remote",
 "evidence_work_arrangement": "This position is fully remote within California"}

EXAMPLE 2
TITLE: Business Intelligence Analyst
POSTING: The position is based in Austin, Texas. Compensation varies by
location. No working model is stated.
OUTPUT:
{"work_arrangement": "unclear",
 "evidence_work_arrangement": null}

EXAMPLE 3
TITLE: Senior Analytics Engineer
POSTING: This role has been designated hybrid and requires employees to work
from the office at least three days each week.
OUTPUT:
{"work_arrangement": "hybrid",
 "evidence_work_arrangement": "This role has been designated hybrid and requires employees to work from the office at least three days each week"}

EXAMPLE 4
TITLE: Principal Data Scientist
POSTING: Our company operates an onsite manufacturing facility. This posting
does not state where the Principal Data Scientist will work.
OUTPUT:
{"work_arrangement": "unclear",
 "evidence_work_arrangement": null}

EXAMPLE 5
TITLE: Reporting Analyst
POSTING: This is an onsite role at our Chicago office. Remote work is not
available.
OUTPUT:
{"work_arrangement": "onsite",
 "evidence_work_arrangement": "This is an onsite role at our Chicago office"}"""


def few_shot(title: str, description: str) -> str:
    return f"""Extract the work arrangement from job postings.

Return ONLY valid JSON matching this shape:

{SCHEMA_BLOCK}

{_FEW_SHOT_EXAMPLES}

Now extract this posting.

JOB TITLE: {title}

JOB POSTING:
{description}

OUTPUT:"""


# ---------------------------------------------------------------------------
# C — schema and explicit rules
# ---------------------------------------------------------------------------

def schema_rules(title: str, description: str) -> str:
    return f"""You are extracting work-arrangement information for a
labour-market database. Accuracy matters more than guessing.

Return ONLY valid JSON matching this schema exactly:

{SCHEMA_BLOCK}

{ARRANGEMENT_RULES}

JOB TITLE: {title}

JOB POSTING:
{description}"""


# ---------------------------------------------------------------------------
# D — decomposed, evidence first
# ---------------------------------------------------------------------------

def decomposed(title: str, description: str) -> str:
    return f"""Extract the work arrangement from this job posting in two stages.

STAGE 1:
Locate an exact quote that explicitly assigns a remote, hybrid, or onsite
working model to THIS role. Ignore company background, facilities, other teams,
job-location metadata, and generic policy boilerplate.

STAGE 2:
Classify the role using only that evidence. If no role-specific evidence
exists, return unclear.

{ARRANGEMENT_RULES}

Return ONLY valid JSON in this shape:

{{
  "stage1_evidence": {{
    "role_specific_arrangement_text": "exact quote or null"
  }},
  "work_arrangement": "remote" | "hybrid" | "onsite" | "unclear",
  "evidence_work_arrangement": "exact quote or null"
}}

JOB TITLE: {title}

JOB POSTING:
{description}"""


VARIANTS: dict[str, callable] = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "schema_rules": schema_rules,
    "decomposed": decomposed,
}


def build(variant: str, title: str, description: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}"
        )

    return VARIANTS[variant](title, description)


if __name__ == "__main__":
    example_title = "Senior Data Analyst"
    example_description = (
        "This role is hybrid and requires three days each week in our office."
    )

    for name in VARIANTS:
        prompt = build(name, example_title, example_description)
        print(f"{name:<14} {len(prompt):>6} characters")
