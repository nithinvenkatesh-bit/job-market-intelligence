"""
Build curated datasets from the raw postings file.

Produces three parquet files:
  benchmark.parquet   -- stratified sample for method comparison (all roles)
  data_roles.parquet  -- data/analytics postings for the market dashboard
  gold_seed.parquet   -- rows to hand-annotate for unlabeled fields

Why the benchmark spans all roles, not just data roles: extracting salary
and seniority from prose is domain-agnostic, and there are only ~650
data-role postings with salary labels. Benchmarking on the full population
gives real statistical power; the winning method is then applied to the
data-role slice for the dashboard.

TWO REPRODUCIBILITY FIXES, both learned the expensive way:

1. Sampling orders by hash(job_id), NOT random(). DuckDB's setseed() does
   not make ORDER BY random() stable across sessions -- every run drew a
   different sample, which silently invalidated the LLM response cache and
   re-billed for extractions already paid for.

2. Postings named in config/experiment_sample.json are sorted FIRST within
   their stratum, so a rebuild can never drop them. The LLM experiment cost
   real money and its results are keyed to specific job_ids; if those rows
   vanish from the benchmark, the saved extractions can no longer be joined
   to ground truth.

Run:  python src/build_datasets.py
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
PROCESSED.mkdir(parents=True, exist_ok=True)

MIN_DESC_CHARS = 500
GOLD_N = 150

# Postings whose LLM extractions we have already paid for.
manifest = CONFIG / "experiment_sample.json"
PINNED: list[int] = (
    json.loads(manifest.read_text()).get("benchmark", [])
    if manifest.exists() else []
)
# SQL fragment: 1 for pinned rows, 0 otherwise. "(-1)" keeps the IN clause
# valid when there is no manifest yet.
PINNED_SQL = f"CASE WHEN job_id IN ({','.join(str(i) for i in PINNED) or '-1'}) THEN 1 ELSE 0 END"

con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE VIEW postings AS
    SELECT * FROM read_csv_auto('{RAW / "postings.csv"}',
                                header=true, sample_size=-1)
""")

# Base view: typed flags computed once so every downstream query agrees.
con.execute(f"""
    CREATE OR REPLACE VIEW base AS
    SELECT
      job_id,
      company_name,
      title,
      description,
      location,
      min_salary,
      med_salary,
      max_salary,
      normalized_salary,
      pay_period,
      currency,
      formatted_work_type              AS work_type,
      formatted_experience_level       AS exp_level,
      remote_allowed,
      to_timestamp(original_listed_time / 1000)::DATE AS listed_date,
      LENGTH(description)              AS desc_len,

      -- Does a structured salary label exist at all?
      (min_salary IS NOT NULL OR med_salary IS NOT NULL) AS has_salary_label,

      -- Does the prose actually contain a dollar figure? This decides
      -- whether extraction is even possible for a given row.
      -- \\s? matters: postings write "$ 20.47" with a space, and requiring a
      -- digit immediately after $ misclassified those as "no salary stated".
      regexp_matches(description, '\\$\\s?[0-9]')         AS salary_in_text,

      CASE
        WHEN regexp_matches(lower(title), 'analytics engineer') THEN 'Analytics Engineer'
        WHEN regexp_matches(lower(title), 'data engineer|etl developer|data platform') THEN 'Data Engineer'
        WHEN regexp_matches(lower(title), 'data scientist|machine learning engineer|ml engineer') THEN 'Data Scientist'
        WHEN regexp_matches(lower(title), 'data analyst|data analytics') THEN 'Data Analyst'
        WHEN regexp_matches(lower(title), 'business intelligence|bi analyst|bi developer|reporting analyst') THEN 'BI Analyst'
        WHEN regexp_matches(lower(title), 'business analyst') THEN 'Business Analyst'
        WHEN regexp_matches(lower(title), 'product analyst') THEN 'Product Analyst'
      END AS role_family

    FROM postings
    WHERE description IS NOT NULL
      AND LENGTH(description) >= {MIN_DESC_CHARS}
""")

# ---------------------------------------------------------------------------
# Benchmark: stratified on the two dimensions that drive difficulty
#   stratum    -- is the salary visible in the text at all?
#   pay_bucket -- yearly vs hourly (very different formats in prose)
#
# The 'labeled_not_stated' stratum is deliberate. Those rows carry a salary
# label the description never mentions, so the correct model behaviour is to
# return null. Including them lets us measure hallucination -- without them
# we would only ever reward extraction, never correct abstention.
#
# ORDER BY is_pinned DESC first: pinned rows sort to the top of their cell,
# so they are always inside the row_number() cutoff. Everything else is
# ordered by hash(job_id) -- pseudo-random but stable across sessions.
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE TABLE benchmark AS
    WITH pool AS (
        SELECT *,
          CASE WHEN salary_in_text THEN 'labeled_stated'
               ELSE 'labeled_not_stated' END AS stratum,
          CASE WHEN pay_period IN ('YEARLY', 'HOURLY') THEN pay_period
               ELSE 'OTHER' END AS pay_bucket,
          {PINNED_SQL} AS is_pinned
        FROM base
        WHERE has_salary_label
    ),
    ranked AS (
        SELECT *,
          row_number() OVER (PARTITION BY stratum, pay_bucket
                             ORDER BY is_pinned DESC, hash(job_id)) AS rn
        FROM pool
    )
    SELECT * EXCLUDE (rn, is_pinned) FROM ranked
    WHERE rn <= CASE
        WHEN stratum = 'labeled_stated'     AND pay_bucket = 'YEARLY' THEN 800
        WHEN stratum = 'labeled_stated'     AND pay_bucket = 'HOURLY' THEN 700
        WHEN stratum = 'labeled_stated'                               THEN 100
        WHEN stratum = 'labeled_not_stated' AND pay_bucket = 'YEARLY' THEN 200
        WHEN stratum = 'labeled_not_stated' AND pay_bucket = 'HOURLY' THEN 150
        ELSE 50
    END
""")

# Dashboard population: every data-role posting, labeled or not.
con.execute("""
    CREATE OR REPLACE TABLE data_roles AS
    SELECT * FROM base WHERE role_family IS NOT NULL
""")

# Gold seed: rows to hand-annotate for the fields with no free label
# (years of experience, required vs preferred skills, work arrangement).
con.execute(f"""
    CREATE OR REPLACE TABLE gold_seed AS
    SELECT * FROM benchmark
    ORDER BY hash(job_id)
    LIMIT {GOLD_N}
""")

for name in ("benchmark", "data_roles", "gold_seed"):
    out = PROCESSED / f"{name}.parquet"
    con.execute(f"COPY {name} TO '{out}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"{name:12s} {n:>6,} rows  ->  {out.name}")

# Verification: the whole point of the pinning logic.
if PINNED:
    kept = con.execute(f"""
        SELECT COUNT(*) FROM benchmark
        WHERE job_id IN ({','.join(str(i) for i in PINNED)})
    """).fetchone()[0]
    status = "OK" if kept == len(PINNED) else "MISSING ROWS"
    print(f"\npinned experiment sample: {kept}/{len(PINNED)} present  [{status}]")

print("\nBenchmark composition")
print(con.execute("""
    SELECT stratum, pay_bucket, COUNT(*) AS n
    FROM benchmark GROUP BY 1, 2 ORDER BY 1, 2
""").fetchdf().to_string(index=False))

print("\nData roles by family")
print(con.execute("""
    SELECT role_family, COUNT(*) AS n,
           SUM(CASE WHEN has_salary_label THEN 1 ELSE 0 END) AS with_salary
    FROM data_roles GROUP BY 1 ORDER BY n DESC
""").fetchdf().to_string(index=False))

print("\nDone.")