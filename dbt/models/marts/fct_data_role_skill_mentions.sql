{{
    config(
        materialized='table'
    )
}}

/*
    Deterministic technology mentions from data-role descriptions.

    One row per job and matched technology. These are description mentions,
    not manually validated requirements, so the dashboard will label them
    as technology-demand indicators.
*/

select
    d.job_id,
    d.role_family,
    d.posting_date,
    d.company_id,
    d.company_name,
    d.location,
    d.experience_level,
    d.remote_status,
    d.annualized_salary_usd,
    d.has_valid_salary,

    s.skill_name,
    s.skill_category,
    s.display_order

from {{ ref('stg_data_roles') }} d
cross join {{ ref('data_role_skill_patterns') }} s

where regexp_matches(
    coalesce(d.description, ''),
    s.regex_pattern
)
