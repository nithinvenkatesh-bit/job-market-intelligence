"""
Verify API connectivity and measure real cost on a single posting.

Run this before building anything else. Auth and quota failures are far
easier to diagnose on one call than buried inside a 1,600-call experiment.

Run:  python src/llm/test_connection.py
"""

import os
from pathlib import Path

import duckdb
from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

load_dotenv(ROOT / ".env")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("No ANTHROPIC_API_KEY found. Check .env is in the project root.")
print(f"Key loaded: {api_key[:12]}...{api_key[-4:]}")

# Haiku 4.5: $1 / $5 per million tokens. Chosen over Sonnet because
# structured extraction is exactly what Haiku handles well, at a third of
# the cost -- and cost per posting is one of the metrics we report.
MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MTOK = 1.00
OUTPUT_COST_PER_MTOK = 5.00

client = Anthropic(api_key=api_key)

row = duckdb.connect().execute(f"""
    SELECT job_id, title, description
    FROM '{PROCESSED / "benchmark.parquet"}'
    WHERE stratum = 'labeled_stated' AND desc_len BETWEEN 1500 AND 3000
    LIMIT 1
""").fetchdf().iloc[0]

print(f"\nTesting on job {row.job_id}: {row.title}")
print(f"Description length: {len(row.description):,} chars\n")

prompt = f"""Extract the following from this job posting. Return ONLY valid JSON, no other text.

{{"salary_min": number or null,
 "salary_max": number or null,
 "pay_period": "HOURLY"|"WEEKLY"|"MONTHLY"|"YEARLY"|null,
 "seniority": "Internship"|"Entry level"|"Associate"|"Mid-Senior level"|"Director"|"Executive"|null,
 "evidence_salary": "exact quote from the posting, or null"}}

Use only information explicitly stated. Return null when a value is absent.

JOB TITLE: {row.title}

JOB POSTING:
{row.description}"""

response = client.messages.create(
    model=MODEL,
    max_tokens=500,
    temperature=0,  # deterministic: we're measuring prompts, not sampling
    messages=[{"role": "user", "content": prompt}],
)

print("RESPONSE:")
print(response.content[0].text)

usage = response.usage
cost = (usage.input_tokens / 1e6 * INPUT_COST_PER_MTOK +
        usage.output_tokens / 1e6 * OUTPUT_COST_PER_MTOK)

print(f"\nTokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out")
print(f"Cost this call: ${cost:.5f}")
print(f"Projected 400 postings x 4 prompts: ${cost * 1600:.2f}")
print("\nConnection OK.")
