{{
    config(
        materialized='table'
    )
}}

/*
    Complete date-by-role grid so charts correctly display zero-posting
    days rather than connecting separated observations.
*/

with bounds as (

    select
        min(posting_date) as minimum_date,
        max(posting_date) as maximum_date
    from {{ ref('stg_data_roles') }}

),

dates as (

    select
        cast(calendar_date as date) as posting_date
    from bounds,
    generate_series(
        minimum_date,
        maximum_date,
        interval 1 day
    ) as generated(calendar_date)

),

roles as (

    select distinct role_family
    from {{ ref('stg_data_roles') }}

),

date_role_spine as (

    select
        d.posting_date,
        r.role_family
    from dates d
    cross join roles r

),

daily as (

    select
        posting_date,
        role_family,

        count(*)                                        as postings,

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
        )                                               as remote_tagged_postings

    from {{ ref('stg_data_roles') }}

    group by
        posting_date,
        role_family

)

select
    s.posting_date,
    s.role_family,
    coalesce(d.postings, 0)                             as postings,
    coalesce(d.salary_postings, 0)                      as salary_postings,
    d.median_salary_usd,
    coalesce(
        d.remote_tagged_postings,
        0
    )                                                   as remote_tagged_postings

from date_role_spine s
left join daily d
    on s.posting_date = d.posting_date
   and s.role_family = d.role_family
