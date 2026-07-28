"""
Select postings for hand-annotation.

Random sampling would be a mistake here. 80 postings is a small budget, and
most job descriptions are unambiguous -- annotating them teaches the
evaluation nothing. This script scores each data-role posting on how hard it
is to extract from, then samples the difficult end of the distribution,
stratified across role families.

The difficulty signals are the failure modes already observed in error
analysis, plus the ones the unmeasured fields are most likely to hit:

  * both "required" and "preferred" language present  -> tests the split
  * several different year-counts mentioned            -> which is the floor?
  * remote AND hybrid AND onsite words all present     -> work arrangement
  * multiple distinct salary figures                   -> which is pay?
  * tech skills near company-description language      -> stack vs requirement

Deliberately biased toward hard cases. That means the resulting accuracy
figures are a LOWER bound, not a representative estimate -- which is the
honest direction to be wrong in, and is stated wherever they are reported.

Run:  python src/build_gold_set.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

GOLD_N = 80
MIN_SKILLS = 2       # postings with no tech mentions teach nothing about skills
MIN_DESC_CHARS = 800 # very short postings carry too little to annotate

con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE VIEW roles AS
    SELECT * FROM '{PROCESSED / "data_roles.parquet"}'
    WHERE LENGTH(description) >= {MIN_DESC_CHARS}
""")

# One row per posting with every difficulty signal scored.
con.execute("""
    CREATE OR REPLACE VIEW scored AS
    SELECT
      job_id,
      company_name,
      title,
      description,
      location,
      role_family,
      desc_len,

      -- How many distinct tech skills appear at all. Below MIN_SKILLS the
      -- posting cannot exercise the required-vs-preferred distinction.
      (CASE WHEN regexp_matches(lower(description), '\\bsql\\b')     THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bpython\\b')  THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bexcel\\b')   THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\btableau\\b') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), 'power\\s?-?bi|powerbi') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\blooker\\b')  THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bsnowflake\\b') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bdbt\\b')     THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bairflow\\b') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bspark\\b')   THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\baws\\b')     THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bazure\\b')   THEN 1 ELSE 0 END
      ) AS n_skills,

      -- SIGNAL 1: both required and preferred language present.
      (CASE WHEN regexp_matches(lower(description),
              'required|must have|essential') THEN 1 ELSE 0 END
       * CASE WHEN regexp_matches(lower(description),
              'preferred|a plus|nice to have|desirable|bonus') THEN 1 ELSE 0 END
      ) AS has_both_hedges,

      -- SIGNAL 2: several different year-counts. Which one is the floor?
      LEAST(3, LENGTH(description)
        - LENGTH(regexp_replace(lower(description), '[0-9]+\\+?\\s*(year|yr)', '', 'g'))
      ) AS year_mention_density,

      -- SIGNAL 3: work-arrangement words that conflict.
      (CASE WHEN regexp_matches(lower(description), '\\bremote\\b') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), '\\bhybrid\\b') THEN 1 ELSE 0 END +
       CASE WHEN regexp_matches(lower(description), 'on-?site|in-?office') THEN 1 ELSE 0 END
      ) AS n_arrangement_words,

      -- SIGNAL 4: more than one dollar figure -- which is the salary?
      LEAST(3, (LENGTH(description)
        - LENGTH(regexp_replace(description, '\\$\\s?[0-9]', '', 'g'))) / 2
      ) AS n_dollar_figures,

      -- SIGNAL 5: company-stack language, where skills appear as background
      -- rather than as a candidate requirement.
      CASE WHEN regexp_matches(lower(description),
             'our (stack|tech|technology)|we use|built (on|with)|our platform')
           THEN 1 ELSE 0 END AS has_company_stack_language

    FROM roles
""")

# Weighted difficulty. Weights reflect which unmeasured field each signal
# stresses; they are a judgement call, not a fitted model.
con.execute(f"""
    CREATE OR REPLACE TABLE gold_seed AS
    WITH deduped AS (
        -- Companies repost identical descriptions under different job_ids.
        -- 13 of the first 80 selected were duplicates -- 16% of a hand-
        -- annotation budget spent labelling the same text twice. Keep the
        -- lowest job_id per distinct description.
        SELECT * EXCLUDE (dup_rn) FROM (
            SELECT *,
              row_number() OVER (PARTITION BY md5(description) ORDER BY job_id) AS dup_rn
            FROM scored
            WHERE n_skills >= {MIN_SKILLS}
        ) WHERE dup_rn = 1
    ),
    ranked AS (
        SELECT *,
          (3 * has_both_hedges
           + 2 * year_mention_density
           + 2 * n_arrangement_words
           + 1 * n_dollar_figures
           + 2 * has_company_stack_language
           + LEAST(2, n_skills / 3)) AS difficulty,
          row_number() OVER (
              PARTITION BY role_family
              ORDER BY (3 * has_both_hedges
                        + 2 * year_mention_density
                        + 2 * n_arrangement_words
                        + 1 * n_dollar_figures
                        + 2 * has_company_stack_language
                        + LEAST(2, n_skills / 3)) DESC,
                       hash(job_id)
          ) AS rn
        FROM deduped
    )
    -- Proportional-ish across families, with a floor so small families
    -- (Analytics Engineer, Product Analyst) are not squeezed out entirely.
    SELECT * EXCLUDE (rn) FROM ranked
    WHERE rn <= CASE role_family
        WHEN 'Data Analyst'       THEN 20
        WHEN 'Business Analyst'   THEN 16
        WHEN 'Data Engineer'      THEN 14
        WHEN 'Data Scientist'     THEN 12
        WHEN 'BI Analyst'         THEN 10
        WHEN 'Analytics Engineer' THEN 4
        WHEN 'Product Analyst'    THEN 4
        ELSE 0
    END
""")

out = PROCESSED / "gold_seed.parquet"
con.execute(f"COPY gold_seed TO '{out}' (FORMAT PARQUET)")

n = con.execute("SELECT COUNT(*) FROM gold_seed").fetchone()[0]
print(f"gold_seed: {n} postings -> {out.name}\n")

print("By role family")
print(con.execute("""
    SELECT role_family, COUNT(*) AS n,
           ROUND(AVG(difficulty), 1) AS avg_difficulty,
           ROUND(AVG(n_skills), 1)   AS avg_skills,
           ROUND(AVG(desc_len))      AS avg_chars
    FROM gold_seed GROUP BY 1 ORDER BY n DESC
""").fetchdf().to_string(index=False))

print("\nDifficulty signals present (% of selected)")
print(con.execute("""
    SELECT
      ROUND(100.0 * AVG(has_both_hedges), 0)            AS both_hedges,
      ROUND(100.0 * AVG(CASE WHEN year_mention_density > 1 THEN 1 ELSE 0 END), 0)
                                                        AS multi_year,
      ROUND(100.0 * AVG(CASE WHEN n_arrangement_words > 1 THEN 1 ELSE 0 END), 0)
                                                        AS multi_arrangement,
      ROUND(100.0 * AVG(CASE WHEN n_dollar_figures > 1 THEN 1 ELSE 0 END), 0)
                                                        AS multi_salary,
      ROUND(100.0 * AVG(has_company_stack_language), 0)  AS company_stack
    FROM gold_seed
""").fetchdf().to_string(index=False))

print("\nComparison: difficulty of selected vs all eligible")
print(con.execute(f"""
    SELECT 'selected' AS pool, ROUND(AVG(difficulty), 2) AS avg_difficulty
    FROM gold_seed
    UNION ALL
    SELECT 'all eligible',
           ROUND(AVG(3 * has_both_hedges + 2 * year_mention_density
                     + 2 * n_arrangement_words + 1 * n_dollar_figures
                     + 2 * has_company_stack_language
                     + LEAST(2, n_skills / 3)), 2)
    FROM scored WHERE n_skills >= {MIN_SKILLS}
""").fetchdf().to_string(index=False))

print("\nSample of what was selected")
print(con.execute("""
    SELECT role_family, difficulty, n_skills, LEFT(title, 42) AS title
    FROM gold_seed ORDER BY difficulty DESC LIMIT 10
""").fetchdf().to_string(index=False))