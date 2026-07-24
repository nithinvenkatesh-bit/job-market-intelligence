"""
Regression tests for the rule-based extractors.

Every test here corresponds to a bug found by inspecting real failures on the
2,000-posting benchmark. The job id that exposed each one is cited, so a
future change that reintroduces the bug fails loudly instead of quietly
degrading a metric.

These tests exercise pure functions only -- no data files, no network, no API
key -- so they run in CI in seconds.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (  # noqa: E402
    classify_seniority,
    extract_salary,
    extract_skills,
    extract_years_experience,
)


# ---------------------------------------------------------------------------
# Salary: number parsing
# ---------------------------------------------------------------------------


def test_unformatted_number_not_truncated():
    """job 3906254848: '$ 163680.00' was parsed as 163.

    The old pattern used (?:,\\d{3})* -- zero comma groups allowed -- so an
    unformatted number matched its first three digits and stopped.
    """
    low, high, _, _ = extract_salary(
        "Pay Range: The pay range for this role is: $ 109120.00 to $ 163680.00"
    )
    assert low == 109120.0
    assert high == 163680.0


def test_weekly_figure_not_truncated():
    """job 3886851948: '$1840.90' was parsed as 184."""
    low, _, period, _ = extract_salary("Gross Weekly Pay: $1840.90 for this contract")
    assert low == 1840.90
    assert period == "WEEKLY"


def test_k_suffix_propagates_across_range():
    """job 3895206655: '$50-60K' yielded min=50, not 50,000.

    The K binds only to the second figure; it has to propagate backwards when
    the first is implausibly small by comparison.
    """
    low, high, _, _ = extract_salary("paying in the $50-60K range depending on experience")
    assert low == 50_000.0
    assert high == 60_000.0


def test_range_without_second_dollar_sign():
    """'$20.55 - 23.00 per hour' -- only the first figure carries a $."""
    low, high, period, _ = extract_salary("Base Rate: $20.55 - 23.00 per hour")
    assert low == 20.55
    assert high == 23.00
    assert period == "HOURLY"


# ---------------------------------------------------------------------------
# Salary: what is NOT compensation
# ---------------------------------------------------------------------------


def test_scale_words_are_not_salary():
    """job 3903456111: '$224 billion' in client assets was read as pay."""
    assert extract_salary("NTT DATA, a $30 billion trusted global innovator") == (
        None, None, None, None
    )


def test_scale_word_skipped_but_real_salary_found():
    low, high, period, _ = extract_salary(
        "$224 billion in client assets. The salary range is $85,000 - $95,000 annually."
    )
    assert low == 85_000.0
    assert high == 95_000.0
    assert period == "YEARLY"


def test_abstains_when_no_figure_present():
    assert extract_salary("Competitive compensation discussed during interview.") == (
        None, None, None, None
    )


def test_abstains_on_empty_input():
    assert extract_salary("") == (None, None, None, None)
    assert extract_salary(None) == (None, None, None, None)


# ---------------------------------------------------------------------------
# Salary: pay period must come from the NEAREST hint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # job 3902864225 -- both annual and hourly quoted in one block
        ("Salary Grade Minimum - Annual $108,400.00 ... Minimum - Hourly $52.12", "YEARLY"),
        # job 3902321362 -- "Hourly" precedes, "Annually" follows and is nearer
        ("Salary/Hourly Rate $130k - $180k Annually based on experience", "YEARLY"),
        ("salary range: $72,000 - $115,000. Note that hourly rates vary.", "YEARLY"),
        ("a rate of $33 per hour, W2", "HOURLY"),
        ("Compensation Ranges: $20 - $25 an hour", "HOURLY"),
    ],
)
def test_nearest_period_hint_wins(text, expected):
    """The old code scanned a window and took the FIRST hint it found, so it
    picked whichever word appeared earlier in the string rather than whichever
    describes this number. That single bug produced roughly 9 of the 15 worst
    salary errors."""
    _, _, period, _ = extract_salary(text)
    assert period == expected


def test_weekly_hint_ignores_schedule_statements():
    """'40 hours a week' is a schedule, not a pay basis. Matching it produced
    44 false WEEKLY labels against 11 true ones."""
    _, _, period, _ = extract_salary("$85,000 per year. Standard 40 hours a week.")
    assert period == "YEARLY"


def test_magnitude_sanity_overrides_bad_hint():
    """A 'yearly' salary of $33 is a misread, not a real figure."""
    _, _, period, _ = extract_salary("compensation of $33 annually per hour worked")
    assert period == "HOURLY"


# ---------------------------------------------------------------------------
# Seniority: titles where a keyword does NOT indicate seniority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Account Executive",
        "Sales Executive",
        "Enterprise Account Executive",
        "Healthcare Account Executive",
        "Executive Assistant",
        "Executive Administrative Assistant",
    ],
)
def test_executive_false_positives(title):
    """Error analysis found 39 postings mislabelled Executive. An Account
    Executive is a salesperson; an Executive Assistant supports one."""
    level, _ = classify_seniority(title)
    assert level != "Executive"


def test_senior_account_executive_resolves_to_mid_senior():
    """Stripping the misleading phrase must not discard genuine signal."""
    level, _ = classify_seniority("Senior Account Executive, Influencer")
    assert level == "Mid-Senior level"


def test_division_chief_is_not_c_suite():
    """Bare 'chief' matched 'Division Chief' and 'Planning Chief' -- middle
    management, not the C-suite."""
    level, _ = classify_seniority("Division Chief, Classification & Compensation")
    assert level != "Executive"


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Chief Financial Officer", "Executive"),
        ("Vice President Compliance", "Executive"),
        ("Director of Engineering", "Director"),
        ("Senior Data Analyst", "Mid-Senior level"),
        ("Junior Data Analyst", "Entry level"),
        ("Marketing Intern", "Internship"),
    ],
)
def test_seniority_true_positives(title, expected):
    level, _ = classify_seniority(title)
    assert level == expected


def test_abstains_rather_than_guessing():
    """444 entry-level postings carry no seniority keyword at all. The rules
    return None rather than defaulting, so abstention shows up honestly in the
    metrics instead of hiding behind a label. This is the documented gap the
    LLM fills (+30.6pp)."""
    level, evidence = classify_seniority("Pharmacy Technician")
    assert level is None
    assert evidence is None


# ---------------------------------------------------------------------------
# Years of experience
# ---------------------------------------------------------------------------


def test_takes_the_minimum_of_a_range():
    years, _ = extract_years_experience("Requires 3-5 years of relevant experience")
    assert years == 3


def test_plus_notation():
    years, _ = extract_years_experience("5+ years of FP&A experience required")
    assert years == 5


def test_requires_the_word_experience_nearby():
    """A bare '10 years' might describe company history, not a requirement."""
    years, _ = extract_years_experience("We have served customers for 10 years.")
    assert years is None


def test_ignores_implausible_durations():
    years, _ = extract_years_experience("40 years of combined team experience")
    assert years is None


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_required_vs_preferred_split():
    required, preferred, _ = extract_skills(
        "Advanced Excel required for this role. Tableau is a plus."
    )
    assert "Excel" in required
    assert "Tableau" in preferred


@pytest.mark.parametrize("spelling", ["PowerBI", "Power-BI", "Power BI"])
def test_alias_canonicalisation(spelling):
    """PowerBI / Power-BI / Power BI must fold to one entity, or the counts
    fragment and precision/recall break."""
    required, preferred, _ = extract_skills(f"Experience with {spelling} needed")
    assert "Power BI" in required + preferred


def test_javascript_does_not_match_java():
    required, preferred, _ = extract_skills("Strong JavaScript skills required")
    assert "Java" not in required + preferred


def test_evidence_is_returned_for_every_skill():
    """Evidence spans mirror the contract imposed on the LLM, which is what
    makes the two methods directly comparable."""
    required, preferred, evidence = extract_skills("SQL and Python are required")
    for skill in required + preferred:
        assert skill in evidence
        assert evidence[skill]


def test_empty_input_returns_empty_results():
    assert extract_skills("") == ([], [], {})
    assert extract_years_experience("") == (None, None)
