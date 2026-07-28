{{
    config(
        materialized='table'
    )
}}

select
    company_id,
    name                                      as company_name,
    company_size,
    city,
    state,
    country
from {{ source('market_raw', 'companies') }}
