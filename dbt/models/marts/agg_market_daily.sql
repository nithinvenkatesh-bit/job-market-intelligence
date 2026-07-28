{{
    config(
        materialized='table'
    )
}}

with daily as (

    select
        posting_date,
        count(*)                                        as postings,
        count(distinct company_id)                      as companies,

        count(*) filter (
            where has_valid_salary
        )                                               as salary_postings,

        round(
            count(*) filter (
                where has_valid_salary
            ) * 100.0 / count(*),
            2
        )                                               as salary_coverage_pct,

        round(
            median(annualized_salary_usd)
                filter (where has_valid_salary),
            0
        )                                               as median_salary_usd,

        count(*) filter (
            where remote_status = 'Remote tagged'
        )                                               as remote_tagged_postings,

        round(
            count(*) filter (
                where remote_status = 'Remote tagged'
            ) * 100.0 / count(*),
            2
        )                                               as remote_tagged_pct,

        count(*) filter (
            where is_sponsored
        )                                               as sponsored_postings

    from {{ ref('fct_market_postings') }}
    group by posting_date

)

select
    *,

    round(
        avg(postings) over (
            order by posting_date
            rows between 6 preceding and current row
        ),
        2
    )                                                   as postings_7d_average

from daily
