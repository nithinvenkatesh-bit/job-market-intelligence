{{
    config(
        materialized='table'
    )
}}

with totals as (

    select
        role_family,
        count(*) as role_postings
    from {{ ref('stg_data_roles') }}
    group by role_family

),

overall_total as (

    select count(*) as all_postings
    from {{ ref('stg_data_roles') }}

),

by_role as (

    select
        m.skill_name,
        m.skill_category,
        m.display_order,
        m.role_family,

        count(distinct m.job_id)                         as postings,

        round(
            count(distinct m.job_id) * 100.0
            / max(t.role_postings),
            2
        )                                                as role_share_pct,

        count(
            distinct case
                when m.has_valid_salary then m.job_id
            end
        )                                                as salary_postings,

        round(
            median(m.annualized_salary_usd)
                filter (where m.has_valid_salary),
            0
        )                                                as median_salary_usd

    from {{ ref('fct_data_role_skill_mentions') }} m
    join totals t
        on m.role_family = t.role_family

    group by
        m.skill_name,
        m.skill_category,
        m.display_order,
        m.role_family

),

overall as (

    select
        m.skill_name,
        m.skill_category,
        m.display_order,
        'All data roles'                                 as role_family,

        count(distinct m.job_id)                         as postings,

        round(
            count(distinct m.job_id) * 100.0
            / max(o.all_postings),
            2
        )                                                as role_share_pct,

        count(
            distinct case
                when m.has_valid_salary then m.job_id
            end
        )                                                as salary_postings,

        round(
            median(m.annualized_salary_usd)
                filter (where m.has_valid_salary),
            0
        )                                                as median_salary_usd

    from {{ ref('fct_data_role_skill_mentions') }} m
    cross join overall_total o

    group by
        m.skill_name,
        m.skill_category,
        m.display_order

)

select * from overall
union all
select * from by_role
