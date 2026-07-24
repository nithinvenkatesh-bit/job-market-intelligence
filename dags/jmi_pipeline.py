"""
Job Market Intelligence pipeline.

Orchestrates the full flow: build datasets -> extract (rules + LLM) ->
evaluate -> transform (dbt) -> test.

DESIGN NOTE -- why absolute interpreter paths:
Airflow runs in its own conda environment because it pins dozens of
dependencies and would downgrade pandas and duckdb in the project env. Tasks
therefore invoke the project interpreter directly, and dbt runs with the
project env's bin prepended to PATH.

`conda run` was the first approach and it failed in an instructive way. It
works for simple commands, but `conda run -n jmi ... cd dbt && dbt run` made
conda try to execute `cd` as a program, then ran `dbt` in the OUTER shell
where it was not on PATH -- exit 127, from a cause two layers away from where
the error appeared. Absolute paths remove the indirection entirely, and they
mirror production, where the scheduler's environment is not the task's.

DESIGN NOTE -- why the LLM step is idempotent:
Extraction is cached on disk by (model, prompt), and the exact sample is
pinned in config/experiment_sample.json. A re-run replays from cache at zero
cost, so a failed downstream task can be retried without re-billing for
inference. That is what makes the whole DAG safe to retry.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT_ROOT = Path.home() / "projects" / "job-market-intelligence"
JMI_BIN = Path("/opt/anaconda3/envs/jmi/bin")
PYTHON = JMI_BIN / "python"


def run_py(script: str) -> str:
    """Run a project script with the project interpreter.

    `set -e` makes the shell fail fast rather than reporting success after a
    failed intermediate command.
    """
    return f"set -e; cd {PROJECT_ROOT} && {PYTHON} {script}"


def run_dbt(command: str) -> str:
    """Run a dbt command with the project env's bin on PATH.

    dbt resolves adapters and spawns subprocesses that expect their own
    environment to be discoverable, so prepending bin is more robust than
    invoking the binary by absolute path alone.
    """
    return (
        f'set -e; export PATH="{JMI_BIN}:$PATH"; '
        f"cd {PROJECT_ROOT}/dbt && dbt {command}"
    )


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
    catchup=False,       # no backfilling: this is not a time-partitioned pipeline
    max_active_runs=1,   # the DuckDB file is a single writer
    tags=["jmi", "llm", "dbt"],
) as dag:

    start = EmptyOperator(task_id="start")

    # --- 1. Build curated datasets ---------------------------------------
    build_datasets = BashOperator(
        task_id="build_datasets",
        bash_command=run_py("src/build_datasets.py"),
        doc_md="Stratified benchmark, data-role slice, and gold seed.",
    )

    build_holdout = BashOperator(
        task_id="build_holdout",
        bash_command=run_py("src/build_holdout.py"),
        doc_md="Unseen validation set with zero overlap with the benchmark.",
    )

    # --- 2. Extraction (rules and LLM are independent) --------------------
    extract_rules_benchmark = BashOperator(
        task_id="extract_rules_benchmark",
        bash_command=run_py("src/baselines.py --dataset benchmark"),
    )

    extract_rules_holdout = BashOperator(
        task_id="extract_rules_holdout",
        bash_command=run_py("src/baselines.py --dataset holdout"),
    )

    extract_llm = BashOperator(
        task_id="extract_llm",
        bash_command=run_py("src/llm/run_experiment.py --limit 400"),
        doc_md=(
            "Four prompt variants over the same 400 postings. Cached by "
            "(model, prompt), so re-runs cost nothing."
        ),
        # The slowest and most failure-prone step.
        retries=3,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(minutes=30),
    )

    # --- 3. Evaluation ----------------------------------------------------
    evaluate_baseline = BashOperator(
        task_id="evaluate_baseline",
        bash_command=run_py("src/evaluate_baselines.py --dataset holdout"),
    )

    compare_methods = BashOperator(
        task_id="compare_methods",
        bash_command=run_py("src/evaluation/compare_methods.py"),
        doc_md="Paired significance tests: rules vs each LLM variant.",
    )

    hybrid_routing = BashOperator(
        task_id="hybrid_routing",
        bash_command=run_py("src/evaluation/hybrid.py"),
        doc_md="Tests rules-first-then-LLM routing against each method alone.",
    )

    # --- 4. Transform and gate --------------------------------------------
    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=run_dbt("run"),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=run_dbt("test"),
        doc_md=(
            "Data quality gate. Runs after dbt_run so a broken model fails "
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