"""
Build a held-out benchmark from postings never used for tuning.

Why this exists: the rule-based baseline was corrected twice using error
analysis on benchmark.parquet. Any further score improvement on that same
file is partly tuning, not capability. Reporting final numbers on unseen
postings separates the two -- and makes the rules-vs-LLM comparison clean.

REPRODUCIBILITY NOTE:
  Sampling orders by hash(job_id), NOT random(). DuckDB's setseed() does not
  make ORDER BY random() stable across sessions, so every run produced a
  different holdout set. hash(job_id) is pseudo-random but deterministic:
  the same job_id always sorts to the same position, on any machine.

Run:  python src/build_holdout.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MIN_DESC_CHARS = 500

con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE VIEW postings AS
    SELECT * FROM read_csv_auto('{RAW / "postings.csv"}',
                                header=true, sample_size=-1)
""")

con.execute(f"""
    CREATE OR REPLACE VIEW used AS
    SELECT job_id FROM '{PROCESSED / "benchmark.parquet"}'
""")

con.execute(f"""
    CREATE OR REPLACE VIEW base AS
    SELECT
      job_id, company_name, title, description, location,
      min_salary, med_salary, max_salary, normalized_salary,
      pay_period, currency,
      formatted_work_type        AS work_type,
      formatted_experience_level AS exp_level,
      remote_allowed,
      LENGTH(description)        AS desc_len,
      (min_salary IS NOT NULL OR med_salary IS NOT NULL) AS has_salary_label,
      -- \\s? matters: "$ 20.47" with a space is common, and requiring a digit
      -- immediately after $ misclassified those rows as "no salary stated".
      regexp_matches(description, '\\$\\s?[0-9]')          AS salary_in_text
    FROM postings
    WHERE description IS NOT NULL
      AND LENGTH(description) >= {MIN_DESC_CHARS}
      AND job_id NOT IN (SELECT job_id FROM used)
""")

# Same stratification as the tuning benchmark, so the two are comparable.
con.execute("""
    CREATE OR REPLACE TABLE holdout AS
    WITH pool AS (
        SELECT *,
          CASE WHEN salary_in_text THEN 'labeled_stated'
               ELSE 'labeled_not_stated' END AS stratum,
          CASE WHEN pay_period IN ('YEARLY', 'HOURLY') THEN pay_period
               ELSE 'OTHER' END AS pay_bucket
        FROM base
        WHERE has_salary_label
    ),
    ranked AS (
        SELECT *, row_number() OVER (PARTITION BY stratum, pay_bucket
                                     ORDER BY hash(job_id)) AS rn
        FROM pool
    )
    SELECT * EXCLUDE (rn) FROM ranked
    WHERE rn <= CASE
        WHEN stratum = 'labeled_stated'     AND pay_bucket = 'YEARLY' THEN 800
        WHEN stratum = 'labeled_stated'     AND pay_bucket = 'HOURLY' THEN 700
        WHEN stratum = 'labeled_stated'                               THEN 100
        WHEN stratum = 'labeled_not_stated' AND pay_bucket = 'YEARLY' THEN 200
        WHEN stratum = 'labeled_not_stated' AND pay_bucket = 'HOURLY' THEN 150
        ELSE 50
    END
""")

out = PROCESSED / "holdout.parquet"
con.execute(f"COPY holdout TO '{out}' (FORMAT PARQUET)")

n = con.execute("SELECT COUNT(*) FROM holdout").fetchone()[0]
overlap = con.execute("""
    SELECT COUNT(*) FROM holdout WHERE job_id IN (SELECT job_id FROM used)
""").fetchone()[0]

print(f"holdout: {n:,} rows -> {out.name}")
print(f"overlap with tuning benchmark: {overlap} (must be 0)")
print(con.execute("""
    SELECT stratum, pay_bucket, COUNT(*) AS n
    FROM holdout GROUP BY 1, 2 ORDER BY 1, 2
""").fetchdf().to_string(index=False))