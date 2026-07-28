{{
    config(
        materialized='table'
    )
}}

/*
    LinkedIn broad job-function categories.

    These are not detailed technology skills. Detailed SQL, Python,
    Tableau, cloud, and dbt demand will be added separately for data roles.
*/

with totals as (

    select count(*) as total_postings
    from {{ ref('fct_market_postings') }}

)

select
    s.skill_abr,
    s.skill_name,

    count(distinct p.job_id)                             as postings,

    round(
        count(distinct p.job_id) * 100.0
        / max(t.total_postings),
        2
    )                                                    as market_share_pct,

    count(
        distinct case
            when p.has_valid_salary then p.job_id
        end
    )                                                    as salary_postings,

    round(
        median(p.annualized_salary_usd)
            filter (where p.has_valid_salary),
        0
    )                                                    as median_salary_usd

from {{ ref('stg_market_job_skills') }} s
join {{ ref('fct_market_postings') }} p
    on s.job_id = p.job_id
cross join totals t

where s.skill_name is not null

group by
    s.skill_abr,
    s.skill_name
