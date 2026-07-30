# Looker Studio Market Dashboard

**Public dashboard:**
https://datastudio.google.com/reporting/077e88eb-7839-4e72-9c3c-51a75f8ed288

This directory contains aggregated CSV exports used by the seven-page Looker
Studio dashboard. Raw job descriptions and model responses are not included.

## Dashboard pages

1. Market Overview
2. Market Breakdown
3. Market Categories
4. Company Activity
5. Salary Analysis
6. Data Role Demand
7. Data Role Trends

## Data sources

| CSV file | Primary use |
|---|---|
| `market_overview.csv` | Headline posting, company, salary, and remote-tagged metrics |
| `market_daily.csv` | Daily posting activity and rolling trends |
| `market_industries.csv` | Industry volume and salary analysis |
| `market_skill_categories.csv` | Broad LinkedIn job-function categories |
| `market_companies.csv` | Company posting-volume analysis |
| `data_role_family.csv` | Data-role demand, salary, and coverage comparisons |
| `data_role_daily.csv` | Daily trends by data-role family |
| `data_role_skills.csv` | Technology-description mentions by role family |

## Definitions and limitations

### Historical scope

The exports describe 123,849 valid unique postings dated from December 5, 2023
through April 19, 2024. This is a historical source snapshot, not the current
job market or a complete longitudinal census.

### Salary

Salary measures use valid USD `normalized_salary` values between $10,000 and
$500,000 after supported pay periods are annualized. Salary relationships are
descriptive and should not be interpreted causally.

### Remote status

The source `remote_allowed` field is one-or-null:

- `1` means remote tagged
- null means unknown

Null is not interpreted as onsite.

### Industries

One posting may map to multiple industries. Industry counts must not be summed
to reconstruct the total number of unique postings.

### Technology mentions

Technology demand is based on deterministic regular-expression matches in job
descriptions. These are description mentions, not manually verified required
skills, and may include contextual references.

### Data-role classification

The data-role slice contains 1,977 postings across seven title-based families:

- Analytics Engineer
- BI Analyst
- Business Analyst
- Data Analyst
- Data Engineer
- Data Scientist
- Product Analyst

Small role families and low salary-support counts produce volatile estimates.

## Refresh process

From the repository root:

    cd dbt
    dbt build
    cd ..

    python dashboard/export_market_data.py
    python dashboard/validate_market_data.py

The export script writes:

- Streamlit JSON files under `dashboard/data/`
- Looker Studio CSV files under `looker_studio/data/`

Commit changed exports only after validation passes.
