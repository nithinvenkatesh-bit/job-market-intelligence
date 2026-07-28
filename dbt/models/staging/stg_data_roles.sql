{{
    config(
        materialized='table'
    )
}}

with roles as (

    select *
    from {{ source('market_raw', 'data_roles') }}

)

select
    d.job_id,
    p.company_id,
    d.company_name,
    d.title,
    d.description,
    d.location,
    d.role_family,
    d.listed_date                              as posting_date,

    cast(
        date_trunc('week', d.listed_date)
        as date
    )                                          as posting_week,

    cast(
        date_trunc('month', d.listed_date)
        as date
    )                                          as posting_month,

    d.work_type                                as employment_type,
    d.exp_level                                as experience_level,

    case
        when d.remote_allowed = 1
            then 'Remote tagged'
        else 'Unknown / not supplied'
    end                                        as remote_status,

    d.currency,
    d.pay_period,
    d.min_salary,
    d.med_salary,
    d.max_salary,
    d.normalized_salary,

    case
        when d.currency = 'USD'
         and d.pay_period in (
             'YEARLY',
             'HOURLY',
             'MONTHLY',
             'WEEKLY'
         )
         and d.normalized_salary between 10000 and 500000
            then d.normalized_salary
        else null
    end                                        as annualized_salary_usd,

    case
        when d.currency = 'USD'
         and d.pay_period in (
             'YEARLY',
             'HOURLY',
             'MONTHLY',
             'WEEKLY'
         )
         and d.normalized_salary between 10000 and 500000
            then true
        else false
    end                                        as has_valid_salary

from roles d
join {{ ref('stg_market_postings') }} p
    on d.job_id = p.job_id
