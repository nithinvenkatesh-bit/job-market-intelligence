{{
    config(
        materialized='table'
    )
}}

/*
    One row per historical posting.

    original_listed_time is the initial posting timestamp and is used for
    trends. listed_time behaves more like a refresh/scrape timestamp.

    remote_allowed contains 1 or NULL. NULL is unknown, not onsite.
*/

with source as (

    select *
    from {{ source('market_raw', 'postings') }}

),

renamed as (

    select
        job_id,
        try_cast(company_id as bigint)                    as company_id,
        company_name,
        title,
        location,

        cast(
            to_timestamp(original_listed_time / 1000.0)
            as date
        )                                                 as posting_date,

        cast(
            date_trunc(
                'week',
                to_timestamp(original_listed_time / 1000.0)
            )
            as date
        )                                                 as posting_week,

        cast(
            date_trunc(
                'month',
                to_timestamp(original_listed_time / 1000.0)
            )
            as date
        )                                                 as posting_month,

        formatted_work_type                               as employment_type,
        formatted_experience_level                        as experience_level,

        case
            when remote_allowed = 1
                then 'Remote tagged'
            else 'Unknown / not supplied'
        end                                               as remote_status,

        sponsored = 1                                     as is_sponsored,
        views,
        applies,

        currency,
        pay_period,
        min_salary,
        med_salary,
        max_salary,
        normalized_salary,

        case
            when currency = 'USD'
             and pay_period in (
                 'YEARLY',
                 'HOURLY',
                 'MONTHLY',
                 'WEEKLY'
             )
             and normalized_salary between 10000 and 500000
                then normalized_salary
            else null
        end                                               as annualized_salary_usd,

        case
            when currency = 'USD'
             and pay_period in (
                 'YEARLY',
                 'HOURLY',
                 'MONTHLY',
                 'WEEKLY'
             )
             and normalized_salary between 10000 and 500000
                then true
            else false
        end                                               as has_valid_salary

    from source

)

select *
from renamed
