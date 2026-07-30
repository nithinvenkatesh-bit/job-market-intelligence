{{
    config(
        materialized='table'
    )
}}

/*
    Enriched historical posting fact table.

    One row per job posting. This descriptive market dataset is separate
    from the controlled benchmark used to evaluate extraction methods.
*/

select
    p.job_id,
    p.company_id,

    coalesce(
        p.company_name,
        c.company_name,
        'Unknown company'
    )                                                   as company_name,

    p.title,
    p.location,
    p.posting_date,
    p.posting_week,
    p.posting_month,
    p.employment_type,
    p.experience_level,
    p.remote_status,
    p.is_sponsored,
    p.views,
    p.applies,

    p.currency,
    p.pay_period,
    p.min_salary,
    p.med_salary,
    p.max_salary,
    p.normalized_salary,
    p.annualized_salary_usd,
    p.has_valid_salary,

    c.company_size,
    c.city                                              as company_city,
    c.state                                             as company_state,
    c.country                                           as company_country

from {{ ref('stg_market_postings') }} p
left join {{ ref('stg_market_companies') }} c
    on p.company_id = c.company_id
