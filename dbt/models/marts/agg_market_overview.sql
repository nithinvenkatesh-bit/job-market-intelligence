{{
    config(
        materialized='table'
    )
}}

select
    count(*)                                            as total_postings,
    count(distinct company_id)                          as companies_with_id,
    count(distinct company_name)                        as company_names,

    min(posting_date)                                   as earliest_posting_date,
    max(posting_date)                                   as latest_posting_date,

    count(*) filter (
        where has_valid_salary
    )                                                   as valid_salary_postings,

    round(
        count(*) filter (
            where has_valid_salary
        ) * 100.0 / count(*),
        2
    )                                                   as salary_coverage_pct,

    round(
        median(annualized_salary_usd)
            filter (where has_valid_salary),
        0
    )                                                   as median_salary_usd,

    round(
        quantile_cont(annualized_salary_usd, 0.25)
            filter (where has_valid_salary),
        0
    )                                                   as salary_p25_usd,

    round(
        quantile_cont(annualized_salary_usd, 0.75)
            filter (where has_valid_salary),
        0
    )                                                   as salary_p75_usd,

    count(*) filter (
        where remote_status = 'Remote tagged'
    )                                                   as remote_tagged_postings,

    round(
        count(*) filter (
            where remote_status = 'Remote tagged'
        ) * 100.0 / count(*),
        2
    )                                                   as remote_tagged_pct,

    count(*) filter (
        where is_sponsored
    )                                                   as sponsored_postings,

    round(
        avg(views) filter (
            where views is not null
        ),
        1
    )                                                   as average_views,

    round(
        avg(applies) filter (
            where applies is not null
        ),
        1
    )                                                   as average_applies

from {{ ref('fct_market_postings') }}
