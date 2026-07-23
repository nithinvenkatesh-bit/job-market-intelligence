"""
Profile the raw LinkedIn posting data.

Run: python src/explore.py

Why DuckDB and not pandas: postings.csv is 493MB and description is huge text column
pandas would pull he entire file into memory (several GB).
DuckDB reads the file on disk and only materialises what each query asks
for -- so this runs comfortably on a laptop.
 """

from pathlib import Path
import duckdb

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
POSTINGS = RAW / "postings.csv"

con = duckdb.connect()

# Register the CSV as a queryable view. sample_size=-1 makes DuckDB scan the
# whole file to infer column types -- slower, but this data is messy enough
# that a small sample guesses wrong.
con.execute(f"""
CREATE OR REPLACE VIEW postings AS
SELECT * FROM read_csv_auto('{POSTINGS}', header=true, sample_size=-1)
""")

def show(title: str, sql: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(con.execute(sql).fetchdf().to_string(index=False))

show("Row count", "SELECT COUNT(*) AS total_postings FROM postings")

# Coverage tells us how many rows are actually usable as reference labels.
# A field that is 90% null cannot anchor an evaluation.
show("Field coverage (% populated)", """
    SELECT
      ROUND(100.0 * COUNT(description)                / COUNT(*), 1) AS description,
      ROUND(100.0 * COUNT(min_salary)                 / COUNT(*), 1) AS min_salary,
      ROUND(100.0 * COUNT(max_salary)                 / COUNT(*), 1) AS max_salary,
      ROUND(100.0 * COUNT(pay_period)                 / COUNT(*), 1) AS pay_period,
      ROUND(100.0 * COUNT(currency)                   / COUNT(*), 1) AS currency,
      ROUND(100.0 * COUNT(formatted_experience_level) / COUNT(*), 1) AS exp_level,
      ROUND(100.0 * COUNT(formatted_work_type)        / COUNT(*), 1) AS work_type,
      ROUND(100.0 * COUNT(remote_allowed)             / COUNT(*), 1) AS remote_allowed,
      ROUND(100.0 * COUNT(skills_desc)                / COUNT(*), 1) AS skills_desc
    FROM postings
""")

show("Experience level values", """
    SELECT formatted_experience_level AS level, COUNT(*) AS n
    FROM postings GROUP BY 1 ORDER BY n DESC
""")

show("Pay period values", """
    SELECT pay_period, COUNT(*) AS n
    FROM postings GROUP BY 1 ORDER BY n DESC
""")

show("Work type values", """
    SELECT formatted_work_type AS work_type, COUNT(*) AS n
    FROM postings GROUP BY 1 ORDER BY n DESC
""")

# Filtering on title, not the ANLS skill tag -- that tag is too sparse.
show("Data-role postings by family", """
    SELECT
      CASE
        WHEN lower(title) LIKE '%analytics engineer%'    THEN 'Analytics Engineer'
        WHEN lower(title) LIKE '%data engineer%'         THEN 'Data Engineer'
        WHEN lower(title) LIKE '%data scientist%'        THEN 'Data Scientist'
        WHEN lower(title) LIKE '%data analyst%'          THEN 'Data Analyst'
        WHEN lower(title) LIKE '%business intelligence%'
          OR lower(title) LIKE '%bi analyst%'            THEN 'BI Analyst'
        WHEN lower(title) LIKE '%business analyst%'      THEN 'Business Analyst'
        WHEN lower(title) LIKE '%product analyst%'       THEN 'Product Analyst'
      END AS role_family,
      COUNT(*) AS n
    FROM postings
    WHERE role_family IS NOT NULL
    GROUP BY 1 ORDER BY n DESC
""")

show("Description length (characters)", """
    SELECT
      ROUND(AVG(LENGTH(description)))                                        AS avg_len,
      MEDIAN(LENGTH(description))                                            AS median_len,
      MIN(LENGTH(description))                                               AS min_len,
      MAX(LENGTH(description))                                               AS max_len,
      SUM(CASE WHEN LENGTH(description) < 200 THEN 1 ELSE 0 END)             AS under_200_chars
    FROM postings
    WHERE description IS NOT NULL
""")

print("\nDone.")


