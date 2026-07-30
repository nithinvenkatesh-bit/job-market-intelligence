{{
    config(
        materialized='table'
    )
}}

select
    role_family,

    count(*)                                            as postings,
    count(distinct company_id)                          as companies,

    min(posting_date)                                   as earliest_posting_date,
    max(posting_date)                                   as latest_posting_date,

    count(*) filter (
        where has_valid_salary
    )                                                   as salary_postings,

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
    )                                                   as remote_tagged_pct

from {{ ref('stg_data_roles') }}

group by role_family
