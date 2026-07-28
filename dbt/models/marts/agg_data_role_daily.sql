{{
    config(
        materialized='table'
    )
}}

select
    posting_date,
    role_family,

    count(*)                                            as postings,

    count(*) filter (
        where has_valid_salary
    )                                                   as salary_postings,

    round(
        median(annualized_salary_usd)
            filter (where has_valid_salary),
        0
    )                                                   as median_salary_usd,

    count(*) filter (
        where remote_status = 'Remote tagged'
    )                                                   as remote_tagged_postings

from {{ ref('stg_data_roles') }}

group by
    posting_date,
    role_family
