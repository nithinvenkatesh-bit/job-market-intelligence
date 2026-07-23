"""
Second-pass profiling.

Answers the questions that decide the evaluation design:
  1. How many data-role postings exist under a broader title filter?
  2. Can remote_allowed serve as a binary label? (expected: no)
  3. When a salary LABEL exists, is salary actually IN the description?
  4. Does jobs/salaries.csv cover postings the main table leaves unlabeled?

Question 3 is the critical one. If the structured salary field is entered
separately by the employer and never appears in the prose, then scoring the
model on extracting it would punish it for correctly returning null.

Run:  python src/explore_roles.py
"""

from pathlib import Path

import duckdb

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
POSTINGS = RAW / "postings.csv"
SALARIES = RAW / "jobs" / "salaries.csv"

con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE VIEW postings AS
    SELECT * FROM read_csv_auto('{POSTINGS}', header=true, sample_size=-1)
""")

con.execute(f"""
    CREATE OR REPLACE VIEW salaries AS
    SELECT * FROM read_csv_auto('{SALARIES}', header=true, sample_size=-1)
""")

# One shared definition of role_family so every query agrees.
# Order matters -- first match wins, so specific patterns must precede
# the generic '%analyst%' catch-all.
con.execute("""
    CREATE OR REPLACE VIEW roles AS
    SELECT *,
      CASE
        WHEN regexp_matches(lower(title), 'analytics engineer') THEN 'Analytics Engineer'
        WHEN regexp_matches(lower(title), 'data engineer|etl developer|data platform') THEN 'Data Engineer'
        WHEN regexp_matches(lower(title), 'data scientist|machine learning engineer|ml engineer') THEN 'Data Scientist'
        WHEN regexp_matches(lower(title), 'data analyst|data analytics') THEN 'Data Analyst'
        WHEN regexp_matches(lower(title), 'business intelligence|bi analyst|bi developer|reporting analyst') THEN 'BI Analyst'
        WHEN regexp_matches(lower(title), 'business analyst') THEN 'Business Analyst'
        WHEN regexp_matches(lower(title), 'product analyst') THEN 'Product Analyst'
        WHEN regexp_matches(lower(title), 'analyst') THEN 'Other Analyst'
      END AS role_family
    FROM postings
""")


def show(title: str, sql: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(con.execute(sql).fetchdf().to_string(index=False))


# --- Role families ------------------------------------------------------

show("Broadened role families", """
    SELECT role_family, COUNT(*) AS n
    FROM roles WHERE role_family IS NOT NULL
    GROUP BY 1 ORDER BY n DESC
""")

show("Label coverage within data roles", """
    SELECT
      COUNT(*)                                                    AS data_roles,
      COUNT(min_salary)                                           AS has_min_salary,
      COUNT(med_salary)                                           AS has_med_salary,
      COUNT(normalized_salary)                                    AS has_norm_salary,
      COUNT(formatted_experience_level)                           AS has_exp_level,
      SUM(CASE WHEN LENGTH(description) >= 500 THEN 1 ELSE 0 END) AS desc_500plus
    FROM roles
    WHERE role_family IS NOT NULL AND role_family <> 'Other Analyst'
""")

# Review bucket -- inspect before promoting anything into a real family.
show("Top 'Other Analyst' titles", """
    SELECT title, COUNT(*) AS n
    FROM roles WHERE role_family = 'Other Analyst'
    GROUP BY 1 ORDER BY n DESC LIMIT 20
""")


# --- Label quality ------------------------------------------------------

# If only 1 and NULL appear, NULL means "unknown", not "not remote".
show("remote_allowed distinct values", """
    SELECT remote_allowed, COUNT(*) AS n
    FROM postings GROUP BY 1 ORDER BY n DESC
""")

# THE decisive check.
show("Is salary actually IN the description text?", """
    SELECT
      COUNT(*) AS with_salary_label,
      SUM(CASE WHEN regexp_matches(description, '\\$[0-9]')
               THEN 1 ELSE 0 END) AS mentions_dollar_amount,
      SUM(CASE WHEN regexp_matches(lower(description),
               'salary|compensation|per hour|hourly rate|/hr|per year|annually|pay range')
               THEN 1 ELSE 0 END) AS mentions_pay_language
    FROM postings
    WHERE (min_salary IS NOT NULL OR med_salary IS NOT NULL)
      AND LENGTH(description) >= 500
""")

# The inverse: pay stated in prose but no structured label.
# Invisible to a label-only evaluation.
show("Salary in text but NO structured label", """
    SELECT COUNT(*) AS unlabeled_but_stated
    FROM postings
    WHERE min_salary IS NULL AND med_salary IS NULL
      AND regexp_matches(description, '\\$[0-9]')
      AND LENGTH(description) >= 500
""")


# --- salaries.csv: a second, possibly richer salary source --------------

# If rows > distinct jobs, a job carries multiple compensation records
# and any join must aggregate first or it will fan out.
show("salaries.csv shape", """
    SELECT
      COUNT(*)               AS rows,
      COUNT(DISTINCT job_id) AS distinct_jobs,
      COUNT(min_salary)      AS has_min,
      COUNT(med_salary)      AS has_med,
      COUNT(max_salary)      AS has_max
    FROM salaries
""")

show("Compensation types", """
    SELECT compensation_type, pay_period, COUNT(*) AS n
    FROM salaries GROUP BY 1, 2 ORDER BY n DESC
""")

show("Salary coverage: side file vs main table", """
    SELECT
      COUNT(*)                                                       AS total_postings,
      SUM(CASE WHEN p.min_salary IS NOT NULL OR p.med_salary IS NOT NULL
               THEN 1 ELSE 0 END)                                    AS labeled_in_postings,
      SUM(CASE WHEN s.job_id IS NOT NULL THEN 1 ELSE 0 END)          AS labeled_in_salaries,
      SUM(CASE WHEN p.min_salary IS NULL AND p.med_salary IS NULL
                AND s.job_id IS NOT NULL THEN 1 ELSE 0 END)          AS only_in_salaries
    FROM postings p
    LEFT JOIN (SELECT DISTINCT job_id FROM salaries) s USING (job_id)
""")


# --- Benchmark pool -----------------------------------------------------

show("All salary columns in postings", """
    SELECT
      COUNT(min_salary)        AS min_sal,
      COUNT(med_salary)        AS med_sal,
      COUNT(max_salary)        AS max_sal,
      COUNT(normalized_salary) AS normalized_sal
    FROM postings
""")

show("Benchmark pool (min OR med salary + usable description)", """
    SELECT
      COUNT(*)          AS eligible,
      COUNT(pay_period) AS has_pay_period,
      COUNT(currency)   AS has_currency
    FROM postings
    WHERE (min_salary IS NOT NULL OR med_salary IS NOT NULL)
      AND LENGTH(description) >= 500
""")

print("\nDone.")