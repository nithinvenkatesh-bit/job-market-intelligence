"""Load committed dashboard JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_FILES = [
    "metadata.json",
    "method_summary.json",
    "pairwise_tests.json",
    "item_scores.json",
    "skill_errors.json",
    "llm_operations.json",
]


def data_signature() -> tuple[tuple[str, int, int], ...]:
    """Return file metadata used to invalidate Streamlit's cache."""

    signature = []

    for filename in DATA_FILES:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dashboard data file not found: {path}. "
                "Run `python dashboard/export_data.py` first."
            )

        stat = path.stat()
        signature.append(
            (
                filename,
                stat.st_mtime_ns,
                stat.st_size,
            )
        )

    return tuple(signature)


def read_json(filename: str) -> Any:
    path = DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_cached_dashboard_data(
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Load dashboard data for one exact set of file versions."""

    # The signature is intentionally unused inside the function body.
    # It exists so changes to file timestamps or sizes invalidate the cache.
    del signature

    return {
        "metadata": read_json("metadata.json"),
        "method_summary": pd.DataFrame(
            read_json("method_summary.json")
        ),
        "pairwise_tests": pd.DataFrame(
            read_json("pairwise_tests.json")
        ),
        "item_scores": pd.DataFrame(
            read_json("item_scores.json")
        ),
        "skill_errors": pd.DataFrame(
            read_json("skill_errors.json")
        ),
        "llm_operations": pd.DataFrame(
            read_json("llm_operations.json")
        ),
    }


def load_dashboard_data() -> dict[str, Any]:
    return load_cached_dashboard_data(
        data_signature()
    )
