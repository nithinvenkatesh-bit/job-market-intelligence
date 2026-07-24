{{
    config(
        materialized='view'
    )
}}

/*
    Ground truth for the benchmark: LinkedIn's own structured fields.

    Note the label-quality tiering carried through from the analysis:
      strong  -- pay_period, min/med salary
      weak    -- exp_level (LinkedIn's tagging is inconsistent)
      absent  -- remote/hybrid, years of experience, skill detail

    remote_allowed is deliberately NOT exposed as a boolean. It only ever
    contains 1 or NULL, so null means "unknown", not "not remote". Casting
    it to a boolean would manufacture 88% false negatives.
*/

with source as (

    select * from {{ source('raw', 'benchmark') }}

),

renamed as (

    select
        job_id,
        company_name,
        title,
        description,
        location,
        desc_len                                as description_length,

        -- Ground truth (strong labels)
        pay_period                              as gt_pay_period,
        currency                                as gt_currency,
        coalesce(min_salary, med_salary)        as gt_salary_min,
        coalesce(max_salary, med_salary)        as gt_salary_max,
        case
            when min_salary is not null then 'range'
            when med_salary is not null then 'point'
            else 'none'
        end                                     as gt_label_shape,

        -- Ground truth (weak label -- treat with suspicion)
        exp_level                               as gt_seniority,

        work_type                               as gt_work_type,
        remote_allowed                          as remote_flag_raw,

        -- Experiment design
        stratum,
        pay_bucket,
        salary_in_text,
        has_salary_label,
        role_family

    from source

)

select
    *,
    {{ annualize_salary('gt_salary_min', 'gt_pay_period') }} as gt_salary_min_annual,
    {{ annualize_salary('gt_salary_max', 'gt_pay_period') }} as gt_salary_max_annual
from renamed