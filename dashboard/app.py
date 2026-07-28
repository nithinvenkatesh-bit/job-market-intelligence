"""Job Market Intelligence evaluation dashboard.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_dashboard_data
from market_tabs import (
    data_role_deep_dive_tab,
    market_intelligence_tab,
)

METHOD_LABELS = {
    "rules": "Rules",
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "schema_rules": "Schema-rules",
    "decomposed": "Decomposed",
}

METHOD_ORDER = [
    "Rules",
    "Zero-shot",
    "Few-shot",
    "Schema-rules",
    "Decomposed",
]

METRIC_LABELS = {
    "required_skill_precision": "Required-skill precision",
    "required_skill_recall": "Required-skill recall",
    "required_skill_f1": "Required-skill F1",
    "preferred_skill_f1": "Preferred-skill F1",
    "any_skill_f1": "Any-skill F1",
    "work_accuracy": "Work-arrangement accuracy",
    "work_macro_f1": "Work-arrangement macro-F1",
    "years_exact": "Years exact",
    "years_within_1": "Years within ±1",
}

FIELD_METRICS = {
    "Skills": "any_skill_f1",
    "Work arrangement": "work_accuracy",
    "Years of experience": "years_exact",
}


def method_name(value: str) -> str:
    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())


def percent(value: float) -> str:
    return f"{value:.1%}"


def score(value: float) -> str:
    return f"{value:.3f}"


def configure_page() -> None:
    st.set_page_config(
        page_title="Job Market Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.75rem;
            padding: 0.8rem 1rem;
        }

        .dashboard-note {
            border-left: 4px solid #808080;
            padding: 0.75rem 1rem;
            background: rgba(128, 128, 128, 0.08);
            border-radius: 0.25rem;
            margin: 0.5rem 0 1.25rem 0;
        }

        .small-muted {
            color: #777;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["method_display"] = result["method"].map(method_name)
    result["method_display"] = pd.Categorical(
        result["method_display"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return result.sort_values("method_display")


def executive_metrics(
    metadata: dict,
    summary: pd.DataFrame,
) -> None:
    best_skills = summary.loc[summary["any_skill_f1"].idxmax()]
    best_work = summary.loc[summary["work_accuracy"].idxmax()]
    best_years = summary.loc[summary["years_exact"].idxmax()]

    columns = st.columns(5)

    columns[0].metric(
        "Gold postings",
        f"{metadata['gold_postings']}",
    )
    columns[1].metric(
        "Methods evaluated",
        f"{len(metadata['methods'])}",
    )
    columns[2].metric(
        "Best any-skill F1",
        score(best_skills["any_skill_f1"]),
        method_name(best_skills["method"]),
    )
    columns[3].metric(
        "Best work accuracy",
        percent(best_work["work_accuracy"]),
        method_name(best_work["method"]),
    )
    best_years_value = summary["years_exact"].max()
    best_years_methods = (
        summary.loc[
            summary["years_exact"].eq(best_years_value),
            "method",
        ]
        .map(method_name)
        .tolist()
    )

    columns[4].metric(
        "Best years exact",
        percent(best_years_value),
        " / ".join(best_years_methods),
    )


def headline_chart(summary: pd.DataFrame):
    chart = summary[
        [
            "method_display",
            "any_skill_f1",
            "work_accuracy",
            "years_exact",
        ]
    ].melt(
        id_vars="method_display",
        var_name="metric",
        value_name="value",
    )

    chart["field"] = chart["metric"].map(
        {
            "any_skill_f1": "Skills: any-skill F1",
            "work_accuracy": "Work arrangement: accuracy",
            "years_exact": "Experience: exact match",
        }
    )

    figure = px.bar(
        chart,
        x="method_display",
        y="value",
        color="field",
        barmode="group",
        text_auto=".1%",
        labels={
            "method_display": "Method",
            "value": "Score",
            "field": "Evaluation field",
        },
    )

    figure.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )
    figure.update_layout(
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
        height=470,
    )

    return figure


def routing_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for field, metric in FIELD_METRICS.items():
        winner = summary.loc[summary[metric].idxmax()]

        note = {
            "Skills": "Rules dominate skill extraction.",
            "Work arrangement": (
                "Few-shot leads accuracy; schema-rules leads macro-F1."
            ),
            "Years of experience": (
                "Zero-shot and few-shot tie on exact accuracy."
            ),
        }[field]

        rows.append(
            {
                "Field": field,
                "Operational choice": method_name(winner["method"]),
                "Primary score": percent(winner[metric]),
                "Interpretation": note,
            }
        )

    return pd.DataFrame(rows)


def overview_tab(
    metadata: dict,
    summary: pd.DataFrame,
) -> None:
    st.subheader("Field-level performance")

    st.plotly_chart(
        headline_chart(summary),
        use_container_width=True,
    )

    st.subheader("Recommended field-level routing")

    st.dataframe(
        routing_table(summary),
        hide_index=True,
        use_container_width=True,
        height=145,
        column_config={
            "Field": st.column_config.TextColumn(
                width="medium",
            ),
            "Operational choice": st.column_config.TextColumn(
                width="medium",
            ),
            "Primary score": st.column_config.TextColumn(
                width="small",
            ),
            "Interpretation": st.column_config.TextColumn(
                width="large",
            ),
        },
    )

    st.subheader("Primary conclusion")
    st.markdown(
        """
        **No method wins every field.**

        Deterministic rules deliver the strongest technology-skill
        extraction. LLM prompts perform substantially better on minimum
        experience and work arrangement.

        The evidence supports a **field-level routing policy**, rather
        than replacing the entire deterministic pipeline with an LLM.
        """
    )

    st.markdown(
        """
        <div class="dashboard-note">
        <strong>Evaluation scope:</strong> 80 manually reviewed postings,
        processed by one deterministic baseline and four Claude Haiku prompt
        strategies. Every method is evaluated on the same postings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Important methodological qualifications"):
        for note in metadata["notes"]:
            st.markdown(f"- {note}")


def method_comparison_tab(summary: pd.DataFrame) -> None:
    st.subheader("Compare methods and metrics")

    available = list(METRIC_LABELS)

    selected = st.multiselect(
        "Metrics",
        options=available,
        default=[
            "required_skill_f1",
            "preferred_skill_f1",
            "any_skill_f1",
            "work_accuracy",
            "work_macro_f1",
            "years_exact",
        ],
        format_func=lambda value: METRIC_LABELS[value],
    )

    if not selected:
        st.info("Select at least one metric.")
        return

    chart = summary[
        ["method_display", *selected]
    ].melt(
        id_vars="method_display",
        var_name="metric",
        value_name="value",
    )

    chart["metric_display"] = chart["metric"].map(METRIC_LABELS)

    figure = px.bar(
        chart,
        x="method_display",
        y="value",
        color="metric_display",
        barmode="group",
        text_auto=".1%",
        labels={
            "method_display": "Method",
            "value": "Score",
            "metric_display": "Metric",
        },
    )

    figure.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )
    figure.update_layout(
        legend_title_text="",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    table_columns = [
        "method_display",
        "required_skill_precision",
        "required_skill_recall",
        "required_skill_f1",
        "preferred_skill_f1",
        "any_skill_f1",
        "work_accuracy",
        "work_macro_f1",
        "years_exact",
        "years_within_1",
        "years_mae_answered",
    ]

    table = summary[table_columns].copy()
    table = table.rename(
        columns={
            "method_display": "Method",
            "required_skill_precision": "Required P",
            "required_skill_recall": "Required R",
            "required_skill_f1": "Required F1",
            "preferred_skill_f1": "Preferred F1",
            "any_skill_f1": "Any-skill F1",
            "work_accuracy": "Work accuracy",
            "work_macro_f1": "Work macro-F1",
            "years_exact": "Years exact",
            "years_within_1": "Years ±1",
            "years_mae_answered": "Years MAE",
        }
    )

    percent_columns = [
        "Required P",
        "Required R",
        "Required F1",
        "Preferred F1",
        "Any-skill F1",
        "Work accuracy",
        "Work macro-F1",
        "Years exact",
        "Years ±1",
    ]

    st.dataframe(
        table.style.format(
            {
                **{
                    column: "{:.1%}"
                    for column in percent_columns
                },
                "Years MAE": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def error_analysis_tab(
    item_scores: pd.DataFrame,
) -> None:
    """Explore method-level and posting-level extraction failures."""

    st.subheader("Error analysis")

    st.markdown(
        """
        Aggregate scores explain which method performs best. Error analysis
        explains **why**: missed technologies, over-extracted capabilities,
        work-arrangement confusion, and incorrect experience thresholds.
        """
    )

    scores = item_scores.copy()

    scores["method_display"] = scores["method"].map(method_name)
    scores["method_display"] = pd.Categorical(
        scores["method_display"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    boolean_columns = [
        "required_exact",
        "preferred_exact",
        "any_skill_exact",
        "work_correct",
        "years_exact",
    ]

    for column in boolean_columns:
        scores[column] = (
            scores[column]
            .fillna(False)
            .astype(bool)
        )

    error_summary = (
        scores.groupby(
            ["method", "method_display"],
            observed=True,
            as_index=False,
        )
        .agg(
            postings=("job_id", "size"),
            required_errors=(
                "required_exact",
                lambda values: int((~values).sum()),
            ),
            preferred_errors=(
                "preferred_exact",
                lambda values: int((~values).sum()),
            ),
            any_skill_errors=(
                "any_skill_exact",
                lambda values: int((~values).sum()),
            ),
            work_errors=(
                "work_correct",
                lambda values: int((~values).sum()),
            ),
            years_errors=(
                "years_exact",
                lambda values: int((~values).sum()),
            ),
        )
        .sort_values("method_display")
    )

    for column in [
        "required_errors",
        "preferred_errors",
        "any_skill_errors",
        "work_errors",
        "years_errors",
    ]:
        error_summary[
            column.replace("_errors", "_error_rate")
        ] = (
            error_summary[column]
            / error_summary["postings"]
        )

    best_skill = error_summary.loc[
        error_summary["any_skill_errors"].idxmin()
    ]
    best_work = error_summary.loc[
        error_summary["work_errors"].idxmin()
    ]

    best_years_count = error_summary["years_errors"].min()
    best_years_methods = (
        error_summary.loc[
            error_summary["years_errors"].eq(
                best_years_count
            ),
            "method_display",
        ]
        .astype(str)
        .tolist()
    )

    cards = st.columns(3)

    cards[0].metric(
        "Fewest any-skill errors",
        f"{int(best_skill['any_skill_errors'])}",
        str(best_skill["method_display"]),
    )
    cards[1].metric(
        "Fewest work errors",
        f"{int(best_work['work_errors'])}",
        str(best_work["method_display"]),
    )
    cards[2].metric(
        "Fewest years errors",
        f"{int(best_years_count)}",
        " / ".join(best_years_methods),
    )

    error_rates = error_summary[
        [
            "method_display",
            "any_skill_error_rate",
            "work_error_rate",
            "years_error_rate",
        ]
    ].melt(
        id_vars="method_display",
        var_name="error_type",
        value_name="error_rate",
    )

    error_rates["Error type"] = error_rates[
        "error_type"
    ].map(
        {
            "any_skill_error_rate": "Any-skill exact error",
            "work_error_rate": "Work-arrangement error",
            "years_error_rate": "Years exact error",
        }
    )

    error_figure = px.bar(
        error_rates,
        x="method_display",
        y="error_rate",
        color="Error type",
        barmode="group",
        text_auto=".1%",
        labels={
            "method_display": "Method",
            "error_rate": "Error rate",
        },
        title="Strict posting-level exact-match error rates",
    )

    error_figure.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )
    error_figure.update_layout(
        legend_title_text="",
        height=470,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        error_figure,
        use_container_width=True,
    )

    st.caption(
        "Skill errors use strict set-level exact matching: one missed or "
        "extra skill makes the posting an exact-match error. This is not "
        "the inverse of micro-F1."
    )

    selected_method_display = st.selectbox(
        "Method to inspect",
        options=METHOD_ORDER,
        index=0,
    )

    display_to_method = {
        display: method
        for method, display in METHOD_LABELS.items()
    }

    selected_method = display_to_method[
        selected_method_display
    ]

    selected_scores = scores[
        scores["method"].eq(selected_method)
    ].copy()

    (
        work_section,
        skills_section,
        years_section,
        posting_section,
    ) = st.tabs(
        [
            "Work confusion",
            "Skill errors",
            "Years errors",
            "Posting drilldown",
        ]
    )

    with work_section:
        st.markdown(
            f"#### {selected_method_display}: work-arrangement confusion"
        )

        confusion = pd.crosstab(
            selected_scores["gold_work_arrangement"],
            selected_scores["pred_work_arrangement"],
            dropna=False,
        )

        gold_order = [
            "Remote",
            "Hybrid",
            "Onsite",
            "Unclear",
        ]

        prediction_order = [
            "Remote",
            "Hybrid",
            "Onsite",
            "Unclear",
            "InvalidPrediction",
        ]

        gold_labels = [
            label
            for label in gold_order
            if label in confusion.index
        ]

        prediction_labels = [
            label
            for label in prediction_order
            if label in confusion.columns
        ]

        confusion = confusion.reindex(
            index=gold_labels,
            columns=prediction_labels,
            fill_value=0,
        )

        confusion_figure = go.Figure(
            data=go.Heatmap(
                z=confusion.to_numpy(),
                x=confusion.columns.tolist(),
                y=confusion.index.tolist(),
                text=confusion.to_numpy(),
                texttemplate="%{text}",
                hovertemplate=(
                    "Gold: %{y}<br>"
                    "Prediction: %{x}<br>"
                    "Postings: %{z}"
                    "<extra></extra>"
                ),
            )
        )

        confusion_figure.update_layout(
            xaxis_title="Predicted arrangement",
            yaxis_title="Gold arrangement",
            yaxis={"autorange": "reversed"},
            height=440,
            margin=dict(l=10, r=10, t=30, b=20),
        )

        st.plotly_chart(
            confusion_figure,
            use_container_width=True,
        )

        work_errors = selected_scores[
            ~selected_scores["work_correct"]
        ][
            [
                "job_id",
                "gold_work_arrangement",
                "pred_work_arrangement",
            ]
        ].rename(
            columns={
                "job_id": "Job ID",
                "gold_work_arrangement": "Gold",
                "pred_work_arrangement": "Prediction",
            }
        )

        st.dataframe(
            work_errors,
            hide_index=True,
            use_container_width=True,
            height=300,
        )

    with skills_section:
        st.markdown(
            f"#### {selected_method_display}: skill extraction failures"
        )

        skill_bucket = st.radio(
            "Skill bucket",
            options=["Required", "Preferred"],
            horizontal=True,
        )

        prefix = skill_bucket.lower()

        def skill_frequency(column: str) -> pd.DataFrame:
            exploded = (
                selected_scores[[column]]
                .explode(column)
                .dropna(subset=[column])
            )

            if exploded.empty:
                return pd.DataFrame(
                    columns=["skill", "count"]
                )

            return (
                exploded[column]
                .astype(str)
                .value_counts()
                .rename_axis("skill")
                .reset_index(name="count")
                .head(12)
            )

        missed = skill_frequency(
            f"{prefix}_missed"
        )
        extra = skill_frequency(
            f"{prefix}_extra"
        )

        left, right = st.columns(2)

        with left:
            st.markdown("##### Most frequently missed")

            if missed.empty:
                st.info("No missed skills for this selection.")
            else:
                missed_plot = missed.sort_values(
                    "count"
                )

                missed_figure = px.bar(
                    missed_plot,
                    x="count",
                    y="skill",
                    orientation="h",
                    text="count",
                    labels={
                        "count": "Postings",
                        "skill": "Skill",
                    },
                )
                missed_figure.update_layout(
                    showlegend=False,
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=20,
                    ),
                )
                st.plotly_chart(
                    missed_figure,
                    use_container_width=True,
                )

        with right:
            st.markdown("##### Most frequently over-extracted")

            if extra.empty:
                st.info(
                    "No over-extracted skills for this selection."
                )
            else:
                extra_plot = extra.sort_values(
                    "count"
                )

                extra_figure = px.bar(
                    extra_plot,
                    x="count",
                    y="skill",
                    orientation="h",
                    text="count",
                    labels={
                        "count": "Postings",
                        "skill": "Skill",
                    },
                )
                extra_figure.update_layout(
                    showlegend=False,
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=20,
                    ),
                )
                st.plotly_chart(
                    extra_figure,
                    use_container_width=True,
                )

        st.caption(
            "Missed means present in the manual gold annotation but absent "
            "from the prediction. Over-extracted means predicted but not "
            "supported by the gold annotation."
        )

    with years_section:
        st.markdown(
            f"#### {selected_method_display}: years-of-experience errors"
        )

        years_errors = (
            selected_scores[
                selected_scores["years_abs_error"].notna()
                & selected_scores["years_abs_error"].gt(0)
            ][
                [
                    "job_id",
                    "gold_years_experience_min",
                    "pred_years_experience_min",
                    "years_abs_error",
                ]
            ]
            .sort_values(
                "years_abs_error",
                ascending=False,
            )
        )

        if years_errors.empty:
            st.success(
                "No numeric years-of-experience errors."
            )
        else:
            scatter = px.scatter(
                years_errors,
                x="gold_years_experience_min",
                y="pred_years_experience_min",
                size="years_abs_error",
                hover_data=["job_id", "years_abs_error"],
                labels={
                    "gold_years_experience_min": "Gold minimum years",
                    "pred_years_experience_min": "Predicted minimum years",
                    "years_abs_error": "Absolute error",
                },
                title="Incorrect numeric experience predictions",
            )

            maximum = max(
                float(
                    years_errors[
                        "gold_years_experience_min"
                    ].max()
                ),
                float(
                    years_errors[
                        "pred_years_experience_min"
                    ].max()
                ),
            )

            scatter.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=maximum,
                y1=maximum,
                line={
                    "dash": "dash",
                },
            )

            scatter.update_layout(
                height=440,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=20,
                ),
            )

            st.plotly_chart(
                scatter,
                use_container_width=True,
            )

            years_table = years_errors.rename(
                columns={
                    "job_id": "Job ID",
                    "gold_years_experience_min": "Gold years",
                    "pred_years_experience_min": "Predicted years",
                    "years_abs_error": "Absolute error",
                }
            )

            st.dataframe(
                years_table,
                hide_index=True,
                use_container_width=True,
                height=330,
            )

    with posting_section:
        st.markdown(
            f"#### {selected_method_display}: posting-level drilldown"
        )

        drilldown_type = st.selectbox(
            "Error type",
            options=[
                "Any-skill error",
                "Work-arrangement error",
                "Years exact error",
            ],
        )

        if drilldown_type == "Any-skill error":
            mask = ~selected_scores["any_skill_exact"]
        elif drilldown_type == "Work-arrangement error":
            mask = ~selected_scores["work_correct"]
        else:
            mask = ~selected_scores["years_exact"]

        available_jobs = (
            selected_scores.loc[mask, "job_id"]
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

        if not available_jobs:
            st.success(
                "No postings match the selected error type."
            )
        else:
            selected_job = st.selectbox(
                "Job ID",
                options=available_jobs,
            )

            row = selected_scores[
                selected_scores["job_id"]
                .astype(int)
                .eq(int(selected_job))
            ].iloc[0]

            def format_list(value) -> str:
                if isinstance(value, list) and value:
                    return ", ".join(
                        str(item)
                        for item in value
                    )
                return "None"

            left, right = st.columns(2)

            with left:
                st.markdown("##### Gold")
                st.markdown(
                    "**Required skills:** "
                    + format_list(
                        row["gold_required_skills"]
                    )
                )
                st.markdown(
                    "**Preferred skills:** "
                    + format_list(
                        row["gold_preferred_skills"]
                    )
                )
                st.markdown(
                    "**Work arrangement:** "
                    + str(
                        row["gold_work_arrangement"]
                    )
                )
                st.markdown(
                    "**Minimum experience:** "
                    + str(
                        row[
                            "gold_years_experience_min"
                        ]
                    )
                )

            with right:
                st.markdown("##### Prediction")
                st.markdown(
                    "**Required skills:** "
                    + format_list(
                        row["pred_required_skills"]
                    )
                )
                st.markdown(
                    "**Preferred skills:** "
                    + format_list(
                        row["pred_preferred_skills"]
                    )
                )
                st.markdown(
                    "**Work arrangement:** "
                    + str(
                        row["pred_work_arrangement"]
                    )
                )
                st.markdown(
                    "**Minimum experience:** "
                    + str(
                        row[
                            "pred_years_experience_min"
                        ]
                    )
                )

            st.markdown("##### Differences")

            differences = pd.DataFrame(
                [
                    {
                        "Category": "Required missed",
                        "Values": format_list(
                            row["required_missed"]
                        ),
                    },
                    {
                        "Category": "Required extra",
                        "Values": format_list(
                            row["required_extra"]
                        ),
                    },
                    {
                        "Category": "Preferred missed",
                        "Values": format_list(
                            row["preferred_missed"]
                        ),
                    },
                    {
                        "Category": "Preferred extra",
                        "Values": format_list(
                            row["preferred_extra"]
                        ),
                    },
                    {
                        "Category": "Skill type errors",
                        "Values": format_list(
                            row["skill_type_errors"]
                        ),
                    },
                ]
            )

            st.dataframe(
                differences,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Raw job descriptions are intentionally excluded from the "
                "public dashboard export."
            )


def statistical_evidence_tab(
    pairwise: pd.DataFrame,
) -> None:
    """Display paired effect sizes and statistical evidence."""

    st.subheader("Paired statistical evidence")

    st.markdown(
        """
        Every candidate method processed the same 80 postings as the rules
        baseline. Positive differences favor the LLM method; negative
        differences favor deterministic rules.
        """
    )

    tests = pairwise.copy()

    numeric_columns = [
        "difference",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_holm",
    ]

    for column in numeric_columns:
        tests[column] = pd.to_numeric(
            tests[column],
            errors="coerce",
        )

    tests["method_display"] = tests["method"].map(method_name)

    field_labels = {
        "skills": "Skills",
        "work_arrangement": "Work arrangement",
        "years": "Years of experience",
    }

    metric_labels = {
        "required_micro_f1": "Required-skill micro-F1",
        "preferred_micro_f1": "Preferred-skill micro-F1",
        "any_skill_micro_f1": "Any-skill micro-F1",
        "work_correct": "Work-arrangement accuracy",
        "years_exact": "Years exact accuracy",
        "years_within_1": "Years within ±1 accuracy",
    }

    tests["field_display"] = tests["field"].map(field_labels)
    tests["metric_display"] = tests["metric"].map(metric_labels)

    tests["difference_pp"] = tests["difference"] * 100
    tests["ci_low_pp"] = tests["ci_low"] * 100
    tests["ci_high_pp"] = tests["ci_high"] * 100

    tests["ci_excludes_zero"] = (
        (tests["ci_low"] > 0)
        | (tests["ci_high"] < 0)
    )

    # Missing Holm values must remain missing. They are not significant.
    tests["holm_significant"] = (
        tests["p_holm"].notna()
        & tests["p_holm"].lt(0.05)
    )

    skill_tests = tests[
        tests["field"].eq("skills")
    ]
    corrected_tests = tests[
        tests["p_holm"].notna()
    ]

    cards = st.columns(3)

    cards[0].metric(
        "Skill CIs excluding zero",
        (
            f"{int(skill_tests['ci_excludes_zero'].sum())}"
            f" / {len(skill_tests)}"
        ),
        "Bootstrap evidence",
    )

    cards[1].metric(
        "Holm-significant tests",
        (
            f"{int(corrected_tests['holm_significant'].sum())}"
            f" / {len(corrected_tests)}"
        ),
        "Work + years",
    )

    best_work = tests[
        tests["field"].eq("work_arrangement")
    ].sort_values(
        "difference",
        ascending=False,
    ).iloc[0]

    cards[2].metric(
        "Largest work gain",
        f"{best_work['difference_pp']:+.1f} pp",
        best_work["method_display"],
    )

    st.info(
        "Skill comparisons use paired bootstrap confidence intervals "
        "without p-values. Work-arrangement and years comparisons use "
        "McNemar exact tests with Holm correction. These are separate "
        "forms of evidence and should not be described interchangeably."
    )

    selected_field = st.radio(
        "Evaluation field",
        options=[
            "Skills",
            "Work arrangement",
            "Years of experience",
        ],
        horizontal=True,
    )

    selected = tests[
        tests["field_display"].eq(selected_field)
    ].copy()

    method_order = {
        "Zero-shot": 0,
        "Few-shot": 1,
        "Schema-rules": 2,
        "Decomposed": 3,
    }

    selected["method_rank"] = (
        selected["method_display"]
        .map(method_order)
    )

    selected = selected.sort_values(
        ["metric_display", "method_rank"],
        ascending=[True, False],
    )

    selected["row_label"] = (
        selected["method_display"]
        + " — "
        + selected["metric_display"]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=selected["difference_pp"],
            y=selected["row_label"],
            mode="markers",
            marker={"size": 10},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (
                    selected["ci_high_pp"]
                    - selected["difference_pp"]
                ),
                "arrayminus": (
                    selected["difference_pp"]
                    - selected["ci_low_pp"]
                ),
                "thickness": 1.5,
                "width": 5,
            },
            customdata=selected[
                [
                    "ci_low_pp",
                    "ci_high_pp",
                    "p_raw",
                    "p_holm",
                    "test",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Difference: %{x:+.1f} pp<br>"
                "95% CI: [%{customdata[0]:+.1f}, "
                "%{customdata[1]:+.1f}] pp<br>"
                "Raw p: %{customdata[2]}<br>"
                "Holm p: %{customdata[3]}<br>"
                "Test: %{customdata[4]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    figure.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="No difference",
        annotation_position="top",
    )

    figure.update_layout(
        xaxis_title=(
            "Difference from rules "
            "(percentage points)"
        ),
        yaxis_title="",
        height=max(430, 55 * len(selected)),
        margin=dict(l=10, r=30, t=50, b=20),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    evidence_rows = selected.copy()

    def evidence_label(row: pd.Series) -> str:
        if row["field"] == "skills":
            return (
                "95% CI excludes zero"
                if row["ci_excludes_zero"]
                else "95% CI spans zero"
            )

        return (
            "Significant after Holm"
            if row["holm_significant"]
            else "Not significant after Holm"
        )

    evidence_rows["Evidence"] = evidence_rows.apply(
        evidence_label,
        axis=1,
    )

    evidence_rows["95% CI"] = evidence_rows.apply(
        lambda row: (
            f"[{row['ci_low_pp']:+.1f}, "
            f"{row['ci_high_pp']:+.1f}] pp"
        ),
        axis=1,
    )

    evidence_table = evidence_rows[
        [
            "method_display",
            "metric_display",
            "difference_pp",
            "95% CI",
            "test",
            "p_raw",
            "p_holm",
            "Evidence",
        ]
    ].rename(
        columns={
            "method_display": "Method",
            "metric_display": "Metric",
            "difference_pp": "Difference",
            "test": "Test",
            "p_raw": "Raw p",
            "p_holm": "Holm p",
        }
    )

    st.dataframe(
        evidence_table.style.format(
            {
                "Difference": "{:+.1f} pp",
                "Raw p": lambda value: (
                    "Not calculated"
                    if pd.isna(value)
                    else f"{value:.4f}"
                ),
                "Holm p": lambda value: (
                    "Not calculated"
                    if pd.isna(value)
                    else f"{value:.4f}"
                ),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Interpretation")

    if selected_field == "Skills":
        st.markdown(
            """
            All four LLM strategies underperform deterministic rules across
            required, preferred, and any-skill micro-F1. Every bootstrap
            confidence interval excludes zero.

            This is **robust confidence-interval evidence**, but it should not
            be described as Holm-significant because no p-values were
            calculated for the skill bootstrap comparisons.
            """
        )

    elif selected_field == "Work arrangement":
        st.markdown(
            """
            Every LLM strategy improves work-arrangement accuracy over the
            conservative rules baseline, and all four comparisons remain
            significant after Holm correction.

            The largest point estimate is **few-shot at +32.5 percentage
            points**. The rules baseline is gold-informed and diagnostic, so
            this result should not be presented as a fully independent
            held-out comparison.
            """
        )

    else:
        st.markdown(
            """
            Every LLM strategy improves both exact minimum-experience accuracy
            and accuracy within ±1 year. All eight comparisons remain
            significant after Holm correction.

            Zero-shot and few-shot achieve the strongest exact result, each
            improving accuracy by **20 percentage points** over rules.
            """
        )


def operations_tab(operations: pd.DataFrame) -> None:
    st.subheader("LLM reliability, latency, and cost")

    operations = operations.copy()
    operations["method_display"] = (
        operations["variant"].map(method_name)
    )
    operations["method_display"] = pd.Categorical(
        operations["method_display"],
        categories=METHOD_ORDER[1:],
        ordered=True,
    )
    operations = operations.sort_values(
        "method_display"
    )

    st.info(
        "Estimated model cost prices every prediction from its token "
        "usage, including cached responses. Recorded run cost is only "
        "the incremental amount charged during this particular run."
    )

    total_estimated_cost = (
        operations["estimated_model_cost_usd"].sum()
    )
    total_actual_calls = int(
        operations["actual_api_calls"].sum()
    )
    overall_cache_rate = (
        operations["cached_rows"].sum()
        / operations["n_predictions"].sum()
    )

    metrics = st.columns(3)
    metrics[0].metric(
        "Estimated fresh-run cost",
        f"${total_estimated_cost:.3f}",
    )
    metrics[1].metric(
        "Actual API calls recorded",
        f"{total_actual_calls}",
    )
    metrics[2].metric(
        "Cached predictions",
        f"{overall_cache_rate:.1%}",
    )

    left, right = st.columns(2)

    with left:
        cost_figure = px.bar(
            operations,
            x="method_display",
            y="estimated_cost_per_1k_predictions",
            text_auto="$.2f",
            labels={
                "method_display": "Prompt strategy",
                "estimated_cost_per_1k_predictions": (
                    "Estimated cost per 1,000 predictions"
                ),
            },
            title="Estimated fresh-run model cost",
        )
        cost_figure.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
            height=420,
        )
        st.plotly_chart(
            cost_figure,
            use_container_width=True,
        )

    with right:
        latency = operations.dropna(
            subset=["mean_uncached_latency_s"]
        )

        latency_figure = px.bar(
            latency,
            x="method_display",
            y="mean_uncached_latency_s",
            text_auto=".2f",
            labels={
                "method_display": "Prompt strategy",
                "mean_uncached_latency_s": (
                    "Mean uncached latency (seconds)"
                ),
            },
            title="Observed uncached inference latency",
        )
        latency_figure.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
            height=420,
        )
        st.plotly_chart(
            latency_figure,
            use_container_width=True,
        )

        st.caption(
            "Zero-shot is absent because all 80 zero-shot responses "
            "were served from cache, leaving no uncached latency sample."
        )

    reliability = operations[
        [
            "method_display",
            "n_predictions",
            "actual_api_calls",
            "cache_rate",
            "valid_json_rate",
            "mean_uncached_latency_s",
            "median_uncached_latency_s",
            "total_input_tokens",
            "total_output_tokens",
            "estimated_model_cost_usd",
            "estimated_cost_per_1k_predictions",
            "recorded_run_cost_usd",
        ]
    ].rename(
        columns={
            "method_display": "Prompt",
            "n_predictions": "Predictions",
            "actual_api_calls": "API calls",
            "cache_rate": "Cache rate",
            "valid_json_rate": "Valid JSON",
            "mean_uncached_latency_s": "Mean uncached latency",
            "median_uncached_latency_s": "Median uncached latency",
            "total_input_tokens": "Input tokens",
            "total_output_tokens": "Output tokens",
            "estimated_model_cost_usd": "Estimated model cost",
            "estimated_cost_per_1k_predictions": "Estimated cost / 1K",
            "recorded_run_cost_usd": "Recorded run cost",
        }
    )

    st.dataframe(
        reliability.style.format(
            {
                "Cache rate": "{:.1%}",
                "Valid JSON": "{:.1%}",
                "Mean uncached latency": (
                    lambda value: (
                        "No uncached sample"
                        if pd.isna(value)
                        else f"{value:.2f}s"
                    )
                ),
                "Median uncached latency": (
                    lambda value: (
                        "No uncached sample"
                        if pd.isna(value)
                        else f"{value:.2f}s"
                    )
                ),
                "Input tokens": "{:,.0f}",
                "Output tokens": "{:,.0f}",
                "Estimated model cost": "${:.3f}",
                "Estimated cost / 1K": "${:.2f}",
                "Recorded run cost": "${:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Token pricing uses the project client constants: "
        "$1.00 per million input tokens and $5.00 per million "
        "output tokens. Cached responses avoid incremental API charges "
        "but still represent model work that would cost money on a "
        "fresh uncached run."
    )



def sidebar(
    metadata: dict,
    market_metadata: dict,
) -> None:
    with st.sidebar:
        st.title("Job Market Intelligence")
        st.markdown(
            "Historical hiring trends and controlled extraction evaluation."
        )

        st.divider()

        st.markdown("**Historical market snapshot**")

        st.metric(
            "Market postings",
            f"{market_metadata['total_postings']:,}",
        )
        st.metric(
            "Data-role postings",
            f"{market_metadata['data_role_postings']:,}",
        )

        st.caption(
            f"{market_metadata['earliest_posting_date']} through "
            f"{market_metadata['latest_posting_date']}"
        )

        st.divider()

        st.markdown("**Extraction benchmark**")

        st.metric(
            "Manual gold set",
            f"{metadata['gold_postings']} postings",
        )
        st.metric(
            "Evaluation rows",
            f"{metadata['item_score_rows']}",
        )
        st.metric(
            "Paired tests",
            f"{metadata['pairwise_test_rows']}",
        )

        st.divider()

        st.markdown(
            """
            **Dashboard scope**

            - Historical market activity
            - Companies and industries
            - Data-role salary comparisons
            - Technology-demand indicators
            - Gold-set method performance
            - Statistical evidence
            - Error analysis
            - LLM cost and reliability
            """
        )


def main() -> None:
    configure_page()

    try:
        data = load_dashboard_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metadata = data["metadata"]
    market_metadata = data["market_metadata"]
    summary = prepare_summary(data["method_summary"])
    operations = data["llm_operations"]

    sidebar(
        metadata,
        market_metadata,
    )

    st.title("Job Market Intelligence")
    st.markdown(
        """
        ### Historical hiring trends and extraction-method evaluation

        Explore a 123,849-posting market snapshot, examine data-role demand,
        and compare deterministic extraction rules with four LLM prompt
        strategies on a manually reviewed gold benchmark.
        """
    )

    st.caption("Extraction benchmark at a glance")

    executive_metrics(metadata, summary)

    st.divider()

    (
        overview,
        market_view,
        data_role_view,
        comparison,
        statistical,
        error_analysis,
        operations_tab_view,
    ) = st.tabs(
        [
            "Executive overview",
            "Market intelligence",
            "Data-role deep dive",
            "Method comparison",
            "Statistical evidence",
            "Error analysis",
            "LLM operations",
        ]
    )

    with overview:
        overview_tab(metadata, summary)

    with market_view:
        market_intelligence_tab(data)

    with data_role_view:
        data_role_deep_dive_tab(data)

    with comparison:
        method_comparison_tab(summary)

    with statistical:
        statistical_evidence_tab(
            data["pairwise_tests"]
        )

    with error_analysis:
        error_analysis_tab(
            data["item_scores"]
        )

    with operations_tab_view:
        operations_tab(operations)


if __name__ == "__main__":
    main()
