{{
    config(
        materialized='table'
    )
}}

select
    company_id,
    company_name,
    company_size,
    company_city,
    company_state,
    company_country,

    count(*)                                            as postings,

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

    count(*) filter (
        where remote_status = 'Remote tagged'
    )                                                   as remote_tagged_postings,

    round(
        count(*) filter (
            where remote_status = 'Remote tagged'
        ) * 100.0 / count(*),
        2
    )                                                   as remote_tagged_pct,

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

where company_id is not null

group by
    company_id,
    company_name,
    company_size,
    company_city,
    company_state,
    company_country
