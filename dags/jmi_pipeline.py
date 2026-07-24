"""
Job Market Intelligence pipeline.

Orchestrates the full flow: build datasets -> extract (rules + LLM) ->
evaluate -> transform (dbt) -> test.

DESIGN NOTE -- why BashOperator and conda run:
Airflow lives in its own environment because it pins dozens of dependencies
and would downgrade pandas and duckdb in the project environment. Tasks
therefore shell out to `conda run -n jmi`, which is also how this works in
production: the scheduler's environment is not the task's environment.

DESIGN NOTE -- why the LLM step is idempotent:
Extraction is cached on disk by (model, prompt). A re-run replays from cache
at zero cost, so a failed downstream task can be retried without re-billing
for inference. That is what makes the whole DAG safe to retry.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT_ROOT = Path.home() / "projects" / "job-market-intelligence"
CONDA_ENV = "jmi"

# Every task runs through this so the project environment is used, not
# Airflow's. `set -e` makes the shell fail fast rather than reporting
# success on a failed intermediate command.
def run_in_project(command: str) -> str:
    return f"set -e; cd {PROJECT_ROOT} && conda run -n {CONDA_ENV} --no-capture-output {command}"


default_args = {
    "owner": "nithin",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    # Failures here are usually transient (API rate limits), so retry rather
    # than fail the run outright.
    "depends_on_past": False,
}

with DAG(
    dag_id="jmi_pipeline",
    description="Job posting extraction, evaluation, and transformation",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,  # no backfilling: this is not a time-partitioned pipeline
    max_active_runs=1,  # the DuckDB file is a single writer
    tags=["jmi", "llm", "dbt"],
) as dag:

    start = EmptyOperator(task_id="start")

    # --- 1. Build curated datasets from raw parquet -----------------------
    build_datasets = BashOperator(
        task_id="build_datasets",
        bash_command=run_in_project("python src/build_datasets.py"),
        doc_md="Stratified benchmark, data-role slice, and gold seed.",
    )

    build_holdout = BashOperator(
        task_id="build_holdout",
        bash_command=run_in_project("python src/build_holdout.py"),
        doc_md="Unseen validation set with zero overlap with the benchmark.",
    )

    # --- 2. Extraction ----------------------------------------------------
    # Rules and LLM are independent, so they run in parallel.
    extract_rules_benchmark = BashOperator(
        task_id="extract_rules_benchmark",
        bash_command=run_in_project("python src/baselines.py --dataset benchmark"),
    )

    extract_rules_holdout = BashOperator(
        task_id="extract_rules_holdout",
        bash_command=run_in_project("python src/baselines.py --dataset holdout"),
    )

    extract_llm = BashOperator(
        task_id="extract_llm",
        bash_command=run_in_project("python src/llm/run_experiment.py --limit 400"),
        doc_md=(
            "Four prompt variants over the same 400 postings. Cached by "
            "(model, prompt), so re-runs cost nothing."
        ),
        # LLM calls are the slowest and most failure-prone step.
        retries=3,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(minutes=30),
    )

    # --- 3. Evaluation ----------------------------------------------------
    evaluate_baseline = BashOperator(
        task_id="evaluate_baseline",
        bash_command=run_in_project("python src/evaluate_baselines.py --dataset holdout"),
    )

    compare_methods = BashOperator(
        task_id="compare_methods",
        bash_command=run_in_project("python src/evaluation/compare_methods.py"),
        doc_md="Paired significance tests: rules vs each LLM variant.",
    )

    hybrid_routing = BashOperator(
        task_id="hybrid_routing",
        bash_command=run_in_project("python src/evaluation/hybrid.py"),
        doc_md="Tests rules-first-then-LLM routing against each method alone.",
    )

    # --- 4. Transform -----------------------------------------------------
    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=run_in_project("cd dbt && dbt run"),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=run_in_project("cd dbt && dbt test"),
        doc_md=(
            "Data quality gate. Runs AFTER dbt_run so a broken model fails "
            "the pipeline rather than silently publishing bad marts."
        ),
    )

    end = EmptyOperator(task_id="end")

    # --- Dependencies -----------------------------------------------------
    start >> [build_datasets, build_holdout]

    build_datasets >> [extract_rules_benchmark, extract_llm]
    build_holdout >> extract_rules_holdout

    extract_rules_holdout >> evaluate_baseline
    [extract_rules_benchmark, extract_llm] >> compare_methods
    compare_methods >> hybrid_routing

    [evaluate_baseline, hybrid_routing] >> dbt_run >> dbt_test >> end