"""Validate committed Streamlit and Looker Studio market exports.

The validator always checks the committed export files for internal
consistency, privacy, uniqueness, and reasonable size.

When the local DuckDB database exists, it additionally reconciles every
export against the dbt market marts.

Run:
    python dashboard/validate_market_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "processed" / "jmi.duckdb"
DASHBOARD_DIR = ROOT / "dashboard" / "data"
LOOKER_DIR = ROOT / "looker_studio" / "data"

DASHBOARD_FILES = {
    "metadata": "market_metadata.json",
    "overview": "market_overview.json",
    "daily": "market_daily.json",
    "companies": "market_companies.json",
    "industries": "market_industries.json",
    "skill_categories": "market_skill_categories.json",
    "role_family": "data_role_family.json",
    "role_daily": "data_role_daily.json",
    "role_skills": "data_role_skills.json",
}

LOOKER_FILES = {
    "overview": "market_overview.csv",
    "daily": "market_daily.csv",
    "companies": "market_companies.csv",
    "industries": "market_industries.csv",
    "skill_categories": "market_skill_categories.csv",
    "role_family": "data_role_family.csv",
    "role_daily": "data_role_daily.csv",
    "role_skills": "data_role_skills.csv",
}

MARTS = {
    "overview": "main_marts.agg_market_overview",
    "daily": "main_marts.agg_market_daily",
    "companies": "main_marts.agg_market_companies",
    "industries": "main_marts.agg_market_industries",
    "skill_categories": "main_marts.agg_market_skill_categories",
    "role_family": "main_marts.agg_data_role_family",
    "role_daily": "main_marts.agg_data_role_daily",
    "role_skills": "main_marts.agg_data_role_skills",
}

BANNED_COLUMNS = {
    "description",
    "raw_text",
    "job_posting_url",
    "application_url",
    "posting_domain",
    "skills_desc",
}

BANNED_TEXT = {
    "sk-ant-",
    "anthropic_api_key",
    "api_key=",
    "password=",
}


class Validation:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0
        self.skips = 0

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes += 1
            print(f"PASS  {message}")
        else:
            self.failures.append(message)
            print(f"FAIL  {message}")

    def skip(self, message: str) -> None:
        self.skips += 1
        print(f"SKIP  {message}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if column.endswith("_date") or column == "posting_date":
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.date

    return result


def load_dashboard_exports(
    validation: Validation,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    for filename in DASHBOARD_FILES.values():
        path = DASHBOARD_DIR / filename
        validation.check(
            path.exists(),
            f"Dashboard export exists: {filename}",
        )

    if validation.failures:
        raise FileNotFoundError(
            "Required dashboard market exports are missing."
        )

    metadata = read_json(
        DASHBOARD_DIR / DASHBOARD_FILES["metadata"]
    )

    frames: dict[str, pd.DataFrame] = {}

    for name, filename in DASHBOARD_FILES.items():
        if name == "metadata":
            continue

        frames[name] = normalize_date_columns(
            pd.DataFrame(
                read_json(DASHBOARD_DIR / filename)
            )
        )

    return metadata, frames


def load_looker_exports(
    validation: Validation,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    for name, filename in LOOKER_FILES.items():
        path = LOOKER_DIR / filename

        validation.check(
            path.exists(),
            f"Looker export exists: {filename}",
        )

        if path.exists():
            frames[name] = normalize_date_columns(
                pd.read_csv(path)
            )

    if len(frames) != len(LOOKER_FILES):
        raise FileNotFoundError(
            "Required Looker Studio exports are missing."
        )

    return frames


def validate_internal_consistency(
    validation: Validation,
    metadata: dict[str, Any],
    dashboard: dict[str, pd.DataFrame],
    looker: dict[str, pd.DataFrame],
) -> None:
    overview = dashboard["overview"]
    daily = dashboard["daily"]
    companies = dashboard["companies"]
    industries = dashboard["industries"]
    categories = dashboard["skill_categories"]
    role_family = dashboard["role_family"]
    role_daily = dashboard["role_daily"]
    role_skills = dashboard["role_skills"]

    validation.check(
        len(overview) == 1,
        "Dashboard overview contains one row",
    )

    overview_row = overview.iloc[0]

    validation.check(
        int(metadata["total_postings"]) == 123_849,
        "Metadata reports 123,849 market postings",
    )
    validation.check(
        int(overview_row["total_postings"])
        == int(metadata["total_postings"]),
        "Overview posting count matches metadata",
    )
    validation.check(
        int(daily["postings"].sum())
        == int(metadata["total_postings"]),
        "Daily posting counts reconcile to market total",
    )

    validation.check(
        len(daily) == 137,
        "Full-market daily export contains 137 calendar days",
    )
    validation.check(
        daily["posting_date"].nunique() == len(daily),
        "Full-market daily dates are unique",
    )
    validation.check(
        int((daily["postings"] == 0).sum()) == 77,
        "Full-market daily export retains 77 zero-posting days",
    )

    validation.check(
        metadata["earliest_posting_date"]
        == str(daily["posting_date"].min()),
        "Earliest posting date matches daily export",
    )
    validation.check(
        metadata["latest_posting_date"]
        == str(daily["posting_date"].max()),
        "Latest posting date matches daily export",
    )

    validation.check(
        len(companies) == 1_000,
        "Streamlit company export is limited to 1,000 rows",
    )
    validation.check(
        len(looker["companies"]) == 24_474,
        "Looker company export contains all 24,474 rows",
    )
    validation.check(
        companies["company_id"].is_unique,
        "Streamlit company IDs are unique",
    )
    validation.check(
        looker["companies"]["company_id"].is_unique,
        "Looker company IDs are unique",
    )

    validation.check(
        len(industries) == 387,
        "Industry export contains 387 rows",
    )
    validation.check(
        industries["industry_id"].is_unique,
        "Industry IDs are unique",
    )

    validation.check(
        len(categories) == 35,
        "Broad job-function export contains 35 categories",
    )
    validation.check(
        categories["skill_abr"].is_unique,
        "Broad job-function abbreviations are unique",
    )

    validation.check(
        int(role_family["postings"].sum()) == 1_977,
        "Role-family postings reconcile to 1,977",
    )
    validation.check(
        len(role_family) == 7,
        "Role-family export contains seven roles",
    )
    validation.check(
        role_family["role_family"].is_unique,
        "Role-family values are unique",
    )

    validation.check(
        len(role_daily) == 315,
        "Data-role daily grid contains 315 rows",
    )
    validation.check(
        role_daily["posting_date"].nunique() == 45,
        "Data-role grid contains 45 calendar days",
    )
    validation.check(
        role_daily["role_family"].nunique() == 7,
        "Data-role grid contains seven role families",
    )
    validation.check(
        int(role_daily["postings"].sum()) == 1_977,
        "Data-role daily postings reconcile to 1,977",
    )
    validation.check(
        not role_daily.duplicated(
            ["posting_date", "role_family"]
        ).any(),
        "Data-role date-family keys are unique",
    )

    validation.check(
        role_skills["skill_name"].nunique() == 39,
        "Technology-demand export contains 39 technologies",
    )
    validation.check(
        not role_skills.duplicated(
            ["skill_name", "role_family"]
        ).any(),
        "Technology and role-family keys are unique",
    )

    validation.check(
        int(metadata["data_role_postings"]) == 1_977,
        "Metadata reports 1,977 data-role postings",
    )
    validation.check(
        int(metadata["data_role_families"]) == 7,
        "Metadata reports seven data-role families",
    )
    validation.check(
        int(metadata["technology_patterns"]) == 39,
        "Metadata reports 39 technology patterns",
    )

    for name, dashboard_frame in dashboard.items():
        if name == "companies":
            continue

        looker_frame = looker[name]

        validation.check(
            len(dashboard_frame) == len(looker_frame),
            f"Dashboard and Looker row counts match: {name}",
        )

        validation.check(
            list(dashboard_frame.columns)
            == list(looker_frame.columns),
            f"Dashboard and Looker columns match: {name}",
        )


def validate_privacy_and_size(
    validation: Validation,
) -> None:
    files = [
        *(DASHBOARD_DIR / name
          for name in DASHBOARD_FILES.values()),
        *(LOOKER_DIR / name
          for name in LOOKER_FILES.values()),
    ]

    for path in files:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).lower()

        for banned in BANNED_TEXT:
            validation.check(
                banned not in text,
                f"{path.name} excludes sensitive text: {banned}",
            )

        size_mb = path.stat().st_size / 1024 / 1024
        maximum_mb = 2.0 if path.suffix == ".json" else 5.0

        validation.check(
            size_mb < maximum_mb,
            (
                f"{path.name} remains below "
                f"{maximum_mb:.0f} MB"
            ),
        )

    for filename in [
        *DASHBOARD_FILES.values(),
        *LOOKER_FILES.values(),
    ]:
        path = (
            DASHBOARD_DIR / filename
            if filename.endswith(".json")
            else LOOKER_DIR / filename
        )

        if filename.endswith(".json"):
            payload = read_json(path)

            if isinstance(payload, list) and payload:
                columns = set(payload[0])
            elif isinstance(payload, dict):
                columns = set(payload)
            else:
                columns = set()
        else:
            columns = set(
                pd.read_csv(path, nrows=0).columns
            )

        forbidden = columns & BANNED_COLUMNS

        validation.check(
            not forbidden,
            f"{filename} excludes banned columns",
        )


def validate_against_duckdb(
    validation: Validation,
    dashboard: dict[str, pd.DataFrame],
    looker: dict[str, pd.DataFrame],
) -> None:
    if not DATABASE.exists():
        validation.skip(
            "DuckDB reconciliation unavailable because "
            "data/processed/jmi.duckdb is absent"
        )
        return

    con = duckdb.connect(
        str(DATABASE),
        read_only=True,
    )

    try:
        for name, table in MARTS.items():
            source_count = con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            validation.check(
                len(looker[name]) == source_count,
                f"Looker row count matches dbt mart: {name}",
            )

            expected_dashboard_count = (
                min(source_count, 1_000)
                if name == "companies"
                else source_count
            )

            validation.check(
                len(dashboard[name])
                == expected_dashboard_count,
                f"Dashboard row count matches export policy: {name}",
            )

        source_market_total = con.execute(
            """
            SELECT SUM(postings)
            FROM main_marts.agg_market_daily
            """
        ).fetchone()[0]

        source_role_total = con.execute(
            """
            SELECT SUM(postings)
            FROM main_marts.agg_data_role_daily
            """
        ).fetchone()[0]

        validation.check(
            int(source_market_total)
            == int(dashboard["daily"]["postings"].sum()),
            "Dashboard market total matches DuckDB",
        )
        validation.check(
            int(source_role_total)
            == int(dashboard["role_daily"]["postings"].sum()),
            "Dashboard data-role total matches DuckDB",
        )

    finally:
        con.close()


def main() -> None:
    validation = Validation()

    try:
        metadata, dashboard = load_dashboard_exports(
            validation
        )
        looker = load_looker_exports(validation)

        validate_internal_consistency(
            validation,
            metadata,
            dashboard,
            looker,
        )
        validate_privacy_and_size(validation)
        validate_against_duckdb(
            validation,
            dashboard,
            looker,
        )

    except Exception as exc:
        print(f"\nERROR {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\n" + "=" * 72)

    if validation.failures:
        print("MARKET EXPORT VALIDATION FAILED")
        print(
            f"{validation.passes} passed, "
            f"{len(validation.failures)} failed, "
            f"{validation.skips} skipped"
        )

        for failure in validation.failures:
            print(f"- {failure}")

        sys.exit(1)

    print("MARKET EXPORT VALIDATION PASSED")
    print(
        f"{validation.passes} passed, "
        f"0 failed, "
        f"{validation.skips} skipped"
    )
    print("All market export checks completed successfully.")


if __name__ == "__main__":
    main()
