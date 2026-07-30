{{
    config(
        materialized='table'
    )
}}

/*
    Complete calendar-day series.

    Days without postings are retained with zero counts so the rolling
    average represents seven calendar days rather than seven observed dates.
*/

with bounds as (

    select
        min(posting_date) as minimum_date,
        max(posting_date) as maximum_date
    from {{ ref('fct_market_postings') }}

),

date_spine as (

    select
        cast(calendar_date as date) as posting_date
    from bounds,
    generate_series(
        minimum_date,
        maximum_date,
        interval 1 day
    ) as dates(calendar_date)

),

daily as (

    select
        posting_date,
        count(*)                                        as postings,
        count(distinct company_id)                      as companies,

        count(*) filter (
            where has_valid_salary
        )                                               as salary_postings,

        round(
            median(annualized_salary_usd)
                filter (where has_valid_salary),
            0
        )                                               as median_salary_usd,

        count(*) filter (
            where remote_status = 'Remote tagged'
        )                                               as remote_tagged_postings,

        count(*) filter (
            where is_sponsored
        )                                               as sponsored_postings

    from {{ ref('fct_market_postings') }}
    group by posting_date

),

complete as (

    select
        s.posting_date,
        coalesce(d.postings, 0)                         as postings,
        coalesce(d.companies, 0)                        as companies,
        coalesce(d.salary_postings, 0)                  as salary_postings,

        case
            when coalesce(d.postings, 0) = 0 then 0.0
            else round(
                d.salary_postings * 100.0 / d.postings,
                2
            )
        end                                             as salary_coverage_pct,

        d.median_salary_usd,

        coalesce(
            d.remote_tagged_postings,
            0
        )                                               as remote_tagged_postings,

        case
            when coalesce(d.postings, 0) = 0 then 0.0
            else round(
                d.remote_tagged_postings
                * 100.0 / d.postings,
                2
            )
        end                                             as remote_tagged_pct,

        coalesce(
            d.sponsored_postings,
            0
        )                                               as sponsored_postings

    from date_spine s
    left join daily d
        on s.posting_date = d.posting_date

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

from complete
