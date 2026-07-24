{{
    config(
        materialized='view'
    )
}}

/*
    Every extraction from every method, in one long table.

    Unioning rules and LLM output into a single shape is what lets one set
    of downstream models score all methods without branching. Adding a
    fifth method later means adding one CTE here, and nothing downstream
    changes.
*/

with rules as (

    select
        job_id,
        'rules'                 as method,
        salary_min,
        salary_max,
        pay_period,
        seniority,
        years_experience_min,
        n_required              as n_required_skills,
        n_preferred             as n_preferred_skills,
        true                    as valid_json,
        0                       as input_tokens,
        0                       as output_tokens
    from {{ source('raw', 'baseline_benchmark') }}

),

llm as (

    select
        job_id,
        variant                 as method,
        salary_min,
        salary_max,
        pay_period,
        seniority,
        years_experience_min,
        n_required              as n_required_skills,
        n_preferred             as n_preferred_skills,
        valid_json,
        input_tokens,
        output_tokens
    from {{ source('raw', 'llm_extractions_benchmark') }}

),

combined as (

    select * from rules
    union all
    select * from llm

)

select
    *,
    {{ annualize_salary('salary_min', 'pay_period') }} as salary_min_annual,
    {{ annualize_salary('salary_max', 'pay_period') }} as salary_max_annual,

    -- Measured Haiku 4.5 rates: $1 / $5 per million tokens.
    (input_tokens / 1000000.0 * 1.00)
      + (output_tokens / 1000000.0 * 5.00)                as cost_usd

from combined