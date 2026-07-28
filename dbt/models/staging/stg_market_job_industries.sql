{{
    config(
        materialized='table'
    )
}}

select distinct
    ji.job_id,
    ji.industry_id,
    im.industry_name
from {{ source('market_raw', 'job_industries') }} ji
join {{ ref('stg_market_postings') }} p
    on ji.job_id = p.job_id
left join {{ source('market_raw', 'industry_mapping') }} im
    on ji.industry_id = im.industry_id
