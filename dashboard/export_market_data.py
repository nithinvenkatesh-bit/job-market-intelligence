"""Export sanitized market marts for Streamlit and Looker Studio.

The public exports contain only aggregated market information. They exclude
job descriptions, application URLs, raw API responses, and secrets.

Run:
    python dashboard/export_market_data.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "processed" / "jmi.duckdb"
DASHBOARD_OUTPUT = ROOT / "dashboard" / "data"
LOOKER_OUTPUT = ROOT / "looker_studio" / "data"

TABLES = {
    "market_overview": "main_marts.agg_market_overview",
    "market_daily": "main_marts.agg_market_daily",
    "market_companies": "main_marts.agg_market_companies",
    "market_industries": "main_marts.agg_market_industries",
    "market_skill_categories": (
        "main_marts.agg_market_skill_categories"
    ),
    "data_role_family": "main_marts.agg_data_role_family",
    "data_role_daily": "main_marts.agg_data_role_daily",
    "data_role_skills": "main_marts.agg_data_role_skills",
}

ORDER_BY = {
    "market_overview": "",
    "market_daily": "ORDER BY posting_date",
    "market_companies": (
        "ORDER BY postings DESC, company_name, company_id"
    ),
    "market_industries": (
        "ORDER BY postings DESC, industry_name"
    ),
    "market_skill_categories": (
        "ORDER BY postings DESC, skill_name"
    ),
    "data_role_family": (
        "ORDER BY postings DESC, role_family"
    ),
    "data_role_daily": (
        "ORDER BY posting_date, role_family"
    ),
    "data_role_skills": (
        """
        ORDER BY
            CASE
                WHEN role_family = 'All data roles' THEN 0
                ELSE 1
            END,
            role_family,
            postings DESC,
            display_order
        """
    ),
}


def json_safe(value: Any) -> Any:
    """Convert database and pandas values to JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(column): json_safe(value)
            for column, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]


def write_json(filename: str, payload: Any) -> None:
    path = DASHBOARD_OUTPUT / filename
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {path.relative_to(ROOT)}")


def write_csv(filename: str, df: pd.DataFrame) -> None:
    path = LOOKER_OUTPUT / filename
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)}")


def verify_database() -> None:
    if not DATABASE.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DATABASE}. "
            "Run `cd dbt && dbt build && cd ..` first."
        )


def load_tables() -> dict[str, pd.DataFrame]:
    con = duckdb.connect(str(DATABASE), read_only=True)

    try:
        available = {
            row[0]
            for row in con.execute(
                """
                SELECT
                    table_schema || '.' || table_name
                FROM information_schema.tables
                """
            ).fetchall()
        }

        missing = [
            table
            for table in TABLES.values()
            if table.replace("main_", "main_") not in available
        ]

        # DuckDB information_schema returns names without the database
        # catalog prefix, for example main_marts.agg_market_overview.
        if missing:
            raise RuntimeError(
                "Missing required market marts:\n- "
                + "\n- ".join(missing)
            )

        frames: dict[str, pd.DataFrame] = {}

        for name, table in TABLES.items():
            query = f"""
                SELECT *
                FROM {table}
                {ORDER_BY[name]}
            """

            frames[name] = con.execute(query).fetchdf()

        return frames

    finally:
        con.close()


def main() -> None:
    verify_database()

    DASHBOARD_OUTPUT.mkdir(parents=True, exist_ok=True)
    LOOKER_OUTPUT.mkdir(parents=True, exist_ok=True)

    frames = load_tables()
    overview = frames["market_overview"].iloc[0]

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "LinkedIn Job Postings 2023–2024 snapshot",
        "total_postings": int(overview["total_postings"]),
        "earliest_posting_date": str(
            pd.to_datetime(
                overview["earliest_posting_date"]
            ).date()
        ),
        "latest_posting_date": str(
            pd.to_datetime(
                overview["latest_posting_date"]
            ).date()
        ),
        "valid_salary_postings": int(
            overview["valid_salary_postings"]
        ),
        "salary_coverage_pct": float(
            overview["salary_coverage_pct"]
        ),
        "data_role_postings": int(
            frames["data_role_family"]["postings"].sum()
        ),
        "data_role_families": int(
            frames["data_role_family"]["role_family"].nunique()
        ),
        "technology_patterns": int(
            frames["data_role_skills"]["skill_name"].nunique()
        ),
        "dashboard_company_limit": 1000,
        "notes": [
            (
                "original_listed_time is used as the initial "
                "posting date."
            ),
            (
                "Remote status is Remote tagged or Unknown / "
                "not supplied; unknown is not treated as onsite."
            ),
            (
                "Salary metrics use annualized USD values from "
                "YEARLY, HOURLY, MONTHLY, and WEEKLY postings "
                "between $10,000 and $500,000."
            ),
            (
                "Technology demand represents deterministic "
                "description mentions, not manually verified "
                "required skills."
            ),
            (
                "No descriptions, URLs, API responses, or secrets "
                "are included."
            ),
        ],
    }

    write_json("market_metadata.json", metadata)
    write_json(
        "market_overview.json",
        records(frames["market_overview"]),
    )
    write_json(
        "market_daily.json",
        records(frames["market_daily"]),
    )

    # Keep the deployed dashboard compact while preserving the complete
    # company table in the Looker Studio CSV export.
    write_json(
        "market_companies.json",
        records(frames["market_companies"].head(1000)),
    )

    write_json(
        "market_industries.json",
        records(frames["market_industries"]),
    )
    write_json(
        "market_skill_categories.json",
        records(frames["market_skill_categories"]),
    )
    write_json(
        "data_role_family.json",
        records(frames["data_role_family"]),
    )
    write_json(
        "data_role_daily.json",
        records(frames["data_role_daily"]),
    )
    write_json(
        "data_role_skills.json",
        records(frames["data_role_skills"]),
    )

    for name, frame in frames.items():
        write_csv(f"{name}.csv", frame)

    print("\nMarket export complete.")
    print(
        "Dashboard postings:",
        f"{metadata['total_postings']:,}",
    )
    print(
        "Data-role postings:",
        f"{metadata['data_role_postings']:,}",
    )
    print(
        "Looker company rows:",
        f"{len(frames['market_companies']):,}",
    )


if __name__ == "__main__":
    main()
