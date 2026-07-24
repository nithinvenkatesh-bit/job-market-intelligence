/*
    Join every extraction to ground truth and flag correctness per field.

    Three deliberate choices:

    1. Restricted to job_ids the LLM actually processed. The rules ran on
       all 2,000 postings but the LLM on a 400 subsample; comparing across
       different row sets would confound method quality with sampling.

    2. Salary correctness is judged AFTER annualisation. "$60" read as
       yearly rather than hourly is a 2,080x error, and comparing raw
       figures would score it as perfect.

    3. Abstention is judged on the RAW salary figure, not the annualised
       one. See the comment on abstained_correctly -- this was a real bug,
       caught by cross-checking against the Python implementation.
*/

with llm_scope as (

    -- The paired sample: postings every method saw.
    select distinct job_id
    from {{ source('raw', 'llm_extractions_benchmark') }}

),

joined as (

    select
        e.job_id,
        e.method,
        p.stratum,
        p.gt_label_shape,
        p.role_family,

        -- salary
        p.gt_salary_min_annual,
        e.salary_min,
        e.salary_min_annual,
        p.gt_pay_period,
        e.pay_period,

        -- seniority
        p.gt_seniority,
        e.seniority,

        -- telemetry
        e.valid_json,
        e.cost_usd

    from {{ ref('stg_extractions') }} e
    inner join {{ ref('stg_postings') }} p using (job_id)
    inner join llm_scope           using (job_id)

),

scored as (

    select
        *,

        -- Percentage error on the annualised figure.
        case
            when gt_salary_min_annual is null or gt_salary_min_annual = 0 then null
            when salary_min_annual is null then null
            else abs(salary_min_annual - gt_salary_min_annual)
                 / gt_salary_min_annual
        end                                                     as salary_pct_error,

        -- Correct only where extraction was possible in the first place.
        case
            when stratum != 'labeled_stated' then null
            when salary_min_annual is null then false
            when gt_salary_min_annual is null or gt_salary_min_annual = 0 then null
            else abs(salary_min_annual - gt_salary_min_annual)
                 / gt_salary_min_annual <= 0.10
        end                                                     as salary_within_10pct,

        -- The hallucination test: silence is correct when pay is unstated.
        --
        -- Tests salary_min, NOT salary_min_annual. A figure extracted with
        -- an unparseable pay period annualises to null, which would score a
        -- hallucinated number as a correct abstention. Found by cross-
        -- checking this model against the Python implementation: the two
        -- disagreed on exactly one posting per variant, and Python was right.
        case
            when stratum != 'labeled_not_stated' then null
            else salary_min is null
        end                                                     as abstained_correctly,

        case
            when stratum != 'labeled_stated' then null
            else pay_period is not distinct from gt_pay_period
        end                                                     as pay_period_correct,

        -- Strict: a null prediction is a miss, not an exclusion.
        case
            when gt_seniority is null then null
            else seniority is not distinct from gt_seniority
        end                                                     as seniority_correct,

        seniority is not null                                   as seniority_answered

    from joined

)

select * from scored