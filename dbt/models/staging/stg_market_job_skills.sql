{{
    config(
        materialized='table'
    )
}}

/*
    These are LinkedIn's broad job-function categories, not detailed
    technologies such as SQL, Python, Tableau, or dbt.
*/

select distinct
    js.job_id,
    js.skill_abr,
    sm.skill_name
from {{ source('market_raw', 'job_skills') }} js
join {{ ref('stg_market_postings') }} p
    on js.job_id = p.job_id
left join {{ source('market_raw', 'skill_mapping') }} sm
    on js.skill_abr = sm.skill_abr
