{{
    config(
        materialized='table'
    )
}}

with totals as (

    select count(*) as total_postings
    from {{ ref('fct_market_postings') }}

)

select
    i.industry_id,
    i.industry_name,

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
    )                                                    as median_salary_usd,

    count(
        distinct case
            when p.remote_status = 'Remote tagged'
                then p.job_id
        end
    )                                                    as remote_tagged_postings

from {{ ref('stg_market_job_industries') }} i
join {{ ref('fct_market_postings') }} p
    on i.job_id = p.job_id
cross join totals t

where i.industry_name is not null

group by
    i.industry_id,
    i.industry_name
