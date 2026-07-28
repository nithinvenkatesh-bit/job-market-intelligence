"""Streamlit views for historical market and data-role insights."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def money(value: Any) -> str:
    """Format a numeric value as whole-dollar USD."""

    if value is None or pd.isna(value):
        return "—"

    return f"${float(value):,.0f}"


def percentage(value: Any) -> str:
    """Format a percentage value already expressed from 0 to 100."""

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.1f}%"


def integer(value: Any) -> str:
    """Format a numeric value as a whole number."""

    if value is None or pd.isna(value):
        return "—"

    return f"{int(value):,}"


def prepare_dates(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            )

    return result


def market_intelligence_tab(data: dict[str, Any]) -> None:
    """Render the full-market historical snapshot."""

    metadata = data["market_metadata"]
    overview = data["market_overview"].copy()
    daily = prepare_dates(
        data["market_daily"],
        ["posting_date"],
    )
    companies = data["market_companies"].copy()
    industries = data["market_industries"].copy()
    categories = data["market_skill_categories"].copy()

    row = overview.iloc[0]

    st.subheader("Historical market intelligence")

    st.markdown(
        """
        This section summarizes the broader LinkedIn job-posting snapshot.
        It is descriptive market analysis and is separate from the manually
        reviewed extraction benchmark.
        """
    )

    st.markdown(
        f"""
        <div class="dashboard-note">
            <strong>Snapshot:</strong>
            {metadata["earliest_posting_date"]} through
            {metadata["latest_posting_date"]}.
            Initial posting dates are based on
            <code>original_listed_time</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Market postings",
        integer(row["total_postings"]),
    )
    metric_columns[1].metric(
        "Companies with ID",
        integer(row["companies_with_id"]),
    )
    metric_columns[2].metric(
        "Salary coverage",
        percentage(row["salary_coverage_pct"]),
        help=(
            "Valid annualized USD salaries between $10,000 and "
            "$500,000 from yearly, hourly, monthly, or weekly listings."
        ),
    )
    metric_columns[3].metric(
        "Median salary",
        money(row["median_salary_usd"]),
        help="Calculated only from postings passing the salary filters.",
    )
    metric_columns[4].metric(
        "Remote tagged",
        percentage(row["remote_tagged_pct"]),
        help=(
            "The source provides Remote tagged or Unknown / not supplied. "
            "Unknown is not classified as onsite."
        ),
    )

    st.divider()

    st.markdown("### Posting activity")

    trend = daily.sort_values("posting_date")

    trend_figure = go.Figure()

    trend_figure.add_trace(
        go.Bar(
            x=trend["posting_date"],
            y=trend["postings"],
            name="Daily postings",
            opacity=0.35,
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "Daily postings: %{y:,.0f}<extra></extra>"
            ),
        )
    )

    trend_figure.add_trace(
        go.Scatter(
            x=trend["posting_date"],
            y=trend["postings_7d_average"],
            name="7-day average",
            mode="lines",
            line={"width": 3},
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "7-day average: %{y:,.1f}<extra></extra>"
            ),
        )
    )

    trend_figure.update_layout(
        xaxis_title=None,
        yaxis_title="Postings",
        legend_title=None,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )

    st.plotly_chart(
        trend_figure,
        use_container_width=True,
    )

    st.caption(
        "Zero-posting dates are retained, so the rolling average represents "
        "seven calendar days rather than seven observed posting dates. "
        "The sharp April concentration reflects the source snapshot and "
        "should not be interpreted as a complete long-term hiring cycle."
    )

    st.divider()

    industry_column, salary_column = st.columns(2)

    with industry_column:
        st.markdown("### Largest industries")

        top_industries = (
            industries
            .nlargest(15, "postings")
            .sort_values("postings")
        )

        industry_figure = px.bar(
            top_industries,
            x="postings",
            y="industry_name",
            orientation="h",
            text="postings",
            labels={
                "postings": "Postings",
                "industry_name": "",
            },
        )

        industry_figure.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )

        industry_figure.update_layout(
            showlegend=False,
            margin={"l": 10, "r": 45, "t": 10, "b": 10},
        )

        st.plotly_chart(
            industry_figure,
            use_container_width=True,
        )

    with salary_column:
        st.markdown("### Industry salary comparison")

        salary_industries = industries.loc[
            industries["salary_postings"].ge(30)
            & industries["median_salary_usd"].notna()
        ].copy()

        salary_industries["salary_coverage_pct"] = (
            salary_industries["salary_postings"]
            / salary_industries["postings"]
            * 100
        )

        salary_figure = px.scatter(
            salary_industries,
            x="postings",
            y="median_salary_usd",
            size="salary_postings",
            hover_name="industry_name",
            hover_data={
                "postings": ":,",
                "salary_postings": ":,",
                "salary_coverage_pct": ":.1f",
                "median_salary_usd": ":$,.0f",
            },
            labels={
                "postings": "Industry postings",
                "median_salary_usd": "Median annualized salary",
                "salary_postings": "Salary postings",
                "salary_coverage_pct": "Salary coverage",
            },
        )

        salary_figure.update_layout(
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
        )

        st.plotly_chart(
            salary_figure,
            use_container_width=True,
        )

        st.caption(
            "Bubble size represents the number of postings with a usable "
            "salary. Industries with fewer than 30 salary observations are "
            "excluded from this comparison."
        )

    st.divider()

    st.markdown("### Company activity")

    control_one, control_two = st.columns(2)

    with control_one:
        minimum_postings = st.select_slider(
            "Minimum company postings",
            options=[
                10,
                25,
                50,
                100,
                200,
                300,
                500,
                750,
                1000,
            ],
            value=100,
            key="market_company_minimum",
            help=(
                "Only companies meeting this posting-volume "
                "threshold are eligible for the chart."
            ),
        )

    with control_two:
        company_count = st.slider(
            "Maximum companies to display",
            min_value=5,
            max_value=40,
            value=20,
            step=5,
            key="market_company_count",
        )

    eligible_companies = companies.loc[
        companies["postings"].ge(minimum_postings)
    ].copy()

    company_view = eligible_companies.nlargest(
        company_count,
        "postings",
    )

    st.caption(
        f"{len(eligible_companies):,} companies meet the "
        f"{minimum_postings:,}-posting threshold; "
        f"showing {len(company_view):,}."
    )

    company_figure = px.bar(
        company_view.sort_values("postings"),
        x="postings",
        y="company_name",
        orientation="h",
        text="postings",
        hover_data={
            "salary_coverage_pct": ":.1f",
            "median_salary_usd": ":$,.0f",
            "remote_tagged_pct": ":.1f",
            "average_views": ":.1f",
            "average_applies": ":.1f",
        },
        labels={
            "company_name": "",
            "postings": "Postings",
            "salary_coverage_pct": "Salary coverage",
            "median_salary_usd": "Median salary",
            "remote_tagged_pct": "Remote tagged",
            "average_views": "Average views",
            "average_applies": "Average applies",
        },
    )

    company_figure.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
    )

    company_chart_height = max(
        320,
        min(850, 120 + len(company_view) * 30),
    )

    company_figure.update_layout(
        showlegend=False,
        height=company_chart_height,
        margin={"l": 10, "r": 45, "t": 10, "b": 10},
    )

    st.plotly_chart(
        company_figure,
        use_container_width=True,
    )

    company_table = company_view[
        [
            "company_name",
            "postings",
            "salary_postings",
            "salary_coverage_pct",
            "median_salary_usd",
            "remote_tagged_pct",
            "company_city",
            "company_state",
        ]
    ].copy()

    company_table.columns = [
        "Company",
        "Postings",
        "Salary postings",
        "Salary coverage",
        "Median salary",
        "Remote tagged",
        "City",
        "State",
    ]

    company_table["Salary coverage"] = company_table[
        "Salary coverage"
    ].map(percentage)

    company_table["Median salary"] = company_table[
        "Median salary"
    ].map(money)

    company_table["Remote tagged"] = company_table[
        "Remote tagged"
    ].map(percentage)

    st.dataframe(
        company_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "The deployed Streamlit export contains the 1,000 companies with "
        "the most postings. The Looker Studio export retains all 24,474 "
        "company-level rows."
    )

    st.divider()

    st.markdown("### Broad LinkedIn job-function categories")

    st.markdown(
        """
        These categories come from the supplied LinkedIn job-function mapping.
        They are broad functions such as Information Technology, Sales, and
        Finance—not detailed tools such as SQL or Python.
        """
    )

    category_view = (
        categories
        .nlargest(20, "postings")
        .sort_values("postings")
    )

    category_figure = px.bar(
        category_view,
        x="postings",
        y="skill_name",
        orientation="h",
        text="market_share_pct",
        hover_data={
            "postings": ":,",
            "market_share_pct": ":.2f",
            "salary_postings": ":,",
            "median_salary_usd": ":$,.0f",
        },
        labels={
            "skill_name": "",
            "postings": "Postings",
            "market_share_pct": "Market share",
            "salary_postings": "Salary postings",
            "median_salary_usd": "Median salary",
        },
    )

    category_figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
    )

    category_figure.update_layout(
        showlegend=False,
        margin={"l": 10, "r": 45, "t": 10, "b": 10},
    )

    st.plotly_chart(
        category_figure,
        use_container_width=True,
    )

    with st.expander("Definitions and limitations"):
        st.markdown(
            """
            - The source is a historical LinkedIn posting snapshot, not a
              complete census of all jobs.
            - Initial posting date uses `original_listed_time`.
            - Salary metrics include annualized USD values from yearly,
              hourly, monthly, and weekly postings between $10,000 and
              $500,000.
            - `remote_allowed` contains a positive remote tag or a missing
              value. Missing values are shown as **Unknown / not supplied**,
              never as onsite.
            - A posting may belong to more than one industry or broad
              job-function category, so category shares can sum to more
              than 100%.
            - Salary relationships are descriptive and do not establish
              causation.
            """
        )


def data_role_deep_dive_tab(data: dict[str, Any]) -> None:
    """Render the data and analytics role analysis."""

    metadata = data["market_metadata"]
    role_family = data["data_role_family"].copy()
    role_daily = prepare_dates(
        data["data_role_daily"],
        ["posting_date"],
    )
    role_skills = data["data_role_skills"].copy()

    total_postings = int(role_family["postings"].sum())
    total_salary_postings = int(
        role_family["salary_postings"].sum()
    )
    total_remote = int(
        role_family["remote_tagged_postings"].sum()
    )

    st.subheader("Data-role deep dive")

    st.markdown(
        """
        This section focuses on Data Analyst, Data Engineer, Data Scientist,
        Business Analyst, BI Analyst, Product Analyst, and Analytics Engineer
        postings.
        """
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Data-role postings",
        f"{total_postings:,}",
    )
    metric_columns[1].metric(
        "Role families",
        f"{role_family['role_family'].nunique()}",
    )
    metric_columns[2].metric(
        "Salary coverage",
        f"{total_salary_postings / total_postings:.1%}",
    )
    metric_columns[3].metric(
        "Remote tagged",
        f"{total_remote / total_postings:.1%}",
    )
    metric_columns[4].metric(
        "Technology patterns",
        f"{metadata['technology_patterns']}",
    )

    st.divider()

    volume_column, salary_column = st.columns(2)

    with volume_column:
        st.markdown("### Posting volume by role")

        role_volume = role_family.sort_values("postings")

        volume_figure = px.bar(
            role_volume,
            x="postings",
            y="role_family",
            orientation="h",
            text="postings",
            hover_data={
                "companies": ":,",
                "salary_coverage_pct": ":.1f",
                "remote_tagged_pct": ":.1f",
            },
            labels={
                "role_family": "",
                "postings": "Postings",
                "companies": "Companies",
                "salary_coverage_pct": "Salary coverage",
                "remote_tagged_pct": "Remote tagged",
            },
        )

        volume_figure.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )

        volume_figure.update_layout(
            showlegend=False,
            margin={"l": 10, "r": 45, "t": 10, "b": 10},
        )

        st.plotly_chart(
            volume_figure,
            use_container_width=True,
        )

    with salary_column:
        st.markdown("### Salary range by role")

        salary_roles = role_family.loc[
            role_family["median_salary_usd"].notna()
        ].sort_values("median_salary_usd")

        salary_figure = go.Figure(
            go.Bar(
                x=salary_roles["median_salary_usd"],
                y=salary_roles["role_family"],
                orientation="h",
                text=salary_roles["median_salary_usd"],
                texttemplate="$%{text:,.0f}",
                textposition="outside",
                cliponaxis=False,
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (
                        salary_roles["salary_p75_usd"]
                        - salary_roles["median_salary_usd"]
                    ),
                    "arrayminus": (
                        salary_roles["median_salary_usd"]
                        - salary_roles["salary_p25_usd"]
                    ),
                },
                customdata=salary_roles[
                    [
                        "salary_postings",
                        "salary_coverage_pct",
                    ]
                ],
                hovertemplate=(
                    "%{y}<br>"
                    "Median: $%{x:,.0f}<br>"
                    "Salary postings: %{customdata[0]:,.0f}<br>"
                    "Coverage: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        salary_figure.update_layout(
            xaxis_title="Annualized USD salary",
            yaxis_title=None,
            xaxis_tickprefix="$",
            xaxis_tickformat=",",
            margin={"l": 10, "r": 45, "t": 10, "b": 10},
        )

        st.plotly_chart(
            salary_figure,
            use_container_width=True,
        )

        st.caption(
            "Bars show median annualized salary. Error bars span the "
            "25th–75th percentile range. Small role families such as "
            "Analytics Engineer and Product Analyst should be interpreted "
            "with additional caution."
        )

    st.divider()

    st.markdown("### Daily role activity")

    ordered_roles = (
        role_family
        .sort_values("postings", ascending=False)
        ["role_family"]
        .tolist()
    )

    default_roles = ordered_roles[:4]

    selected_roles = st.multiselect(
        "Role families",
        options=ordered_roles,
        default=default_roles,
        key="data_role_daily_selection",
    )

    if not selected_roles:
        st.info("Select at least one role family to display the trend.")
    else:
        role_trend = role_daily.loc[
            role_daily["role_family"].isin(selected_roles)
        ]

        role_trend_figure = px.line(
            role_trend,
            x="posting_date",
            y="postings",
            color="role_family",
            labels={
                "posting_date": "",
                "postings": "Postings",
                "role_family": "Role family",
            },
        )

        role_trend_figure.update_layout(
            hovermode="x unified",
            legend_title=None,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )

        st.plotly_chart(
            role_trend_figure,
            use_container_width=True,
        )

        st.caption(
            "The date-by-role grid includes zero-posting dates, preventing "
            "the chart from connecting isolated observations as though jobs "
            "were posted continuously."
        )

    st.divider()

    st.markdown("### Technology-demand indicators")

    st.markdown(
        """
        Technologies are matched deterministically from posting descriptions.
        They indicate that a technology was mentioned; they are not manually
        validated required-skill labels.
        """
    )

    role_options = ["All data roles", *ordered_roles]

    selected_role = st.selectbox(
        "Analyze technology demand for",
        options=role_options,
        index=0,
        key="technology_role_selection",
    )

    selected_skills = role_skills.loc[
        role_skills["role_family"].eq(selected_role)
    ].copy()

    maximum_skills = min(25, len(selected_skills))

    skill_count = st.slider(
        "Technologies to display",
        min_value=10,
        max_value=maximum_skills,
        value=min(20, maximum_skills),
        key="technology_count",
    )

    top_skills = (
        selected_skills
        .nlargest(skill_count, "postings")
        .sort_values("role_share_pct")
    )

    skill_chart_column, skill_salary_column = st.columns(2)

    with skill_chart_column:
        st.markdown("#### Mention prevalence")

        skill_figure = px.bar(
            top_skills,
            x="role_share_pct",
            y="skill_name",
            orientation="h",
            color="skill_category",
            text="role_share_pct",
            hover_data={
                "postings": ":,",
                "salary_postings": ":,",
                "median_salary_usd": ":$,.0f",
                "skill_category": True,
            },
            labels={
                "role_share_pct": "Share of role postings",
                "skill_name": "",
                "skill_category": "Category",
                "postings": "Postings",
                "salary_postings": "Salary postings",
                "median_salary_usd": "Median salary",
            },
        )

        skill_figure.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )

        skill_figure.update_layout(
            legend_title=None,
            margin={"l": 10, "r": 45, "t": 10, "b": 10},
        )

        st.plotly_chart(
            skill_figure,
            use_container_width=True,
        )

    with skill_salary_column:
        st.markdown("#### Prevalence and salary")

        salary_skills = selected_skills.loc[
            selected_skills["salary_postings"].ge(10)
            & selected_skills["median_salary_usd"].notna()
        ].copy()

        skill_salary_figure = px.scatter(
            salary_skills,
            x="role_share_pct",
            y="median_salary_usd",
            size="salary_postings",
            color="skill_category",
            hover_name="skill_name",
            hover_data={
                "postings": ":,",
                "salary_postings": ":,",
                "role_share_pct": ":.1f",
                "median_salary_usd": ":$,.0f",
            },
            labels={
                "role_share_pct": "Share of role postings",
                "median_salary_usd": "Median annualized salary",
                "salary_postings": "Salary postings",
                "skill_category": "Category",
            },
        )

        skill_salary_figure.update_layout(
            legend_title=None,
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )

        st.plotly_chart(
            skill_salary_figure,
            use_container_width=True,
        )

        st.caption(
            "Only technologies with at least 10 salary-bearing postings in "
            "the selected role are shown. Salary associations are descriptive "
            "and do not imply that a technology causes higher compensation."
        )

    detail_table = (
        selected_skills
        .sort_values(
            ["postings", "display_order"],
            ascending=[False, True],
        )
        [
            [
                "skill_name",
                "skill_category",
                "postings",
                "role_share_pct",
                "salary_postings",
                "median_salary_usd",
            ]
        ]
        .copy()
    )

    detail_table.columns = [
        "Technology",
        "Category",
        "Postings",
        "Posting share",
        "Salary postings",
        "Median salary",
    ]

    detail_table["Posting share"] = detail_table[
        "Posting share"
    ].map(percentage)

    detail_table["Median salary"] = detail_table[
        "Median salary"
    ].map(money)

    st.dataframe(
        detail_table,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Role-family detail table"):
        role_table = role_family[
            [
                "role_family",
                "postings",
                "companies",
                "salary_postings",
                "salary_coverage_pct",
                "median_salary_usd",
                "salary_p25_usd",
                "salary_p75_usd",
                "remote_tagged_pct",
            ]
        ].copy()

        role_table.columns = [
            "Role family",
            "Postings",
            "Companies",
            "Salary postings",
            "Salary coverage",
            "Median salary",
            "Salary P25",
            "Salary P75",
            "Remote tagged",
        ]

        for column in [
            "Salary coverage",
            "Remote tagged",
        ]:
            role_table[column] = role_table[column].map(
                percentage
            )

        for column in [
            "Median salary",
            "Salary P25",
            "Salary P75",
        ]:
            role_table[column] = role_table[column].map(money)

        st.dataframe(
            role_table.sort_values(
                "Postings",
                ascending=False,
            ),
            hide_index=True,
            use_container_width=True,
        )
