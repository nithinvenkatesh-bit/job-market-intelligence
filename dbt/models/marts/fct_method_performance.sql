/*
    One row per method: the headline scorecard.

    Every metric here is STRICT -- abstentions count as misses. A method
    that answers 36% of the time and is right when it answers is not an
    accurate method, and reporting only conditional accuracy would hide
    that.
*/

with scored as (

    select * from {{ ref('int_extractions_scored') }}

)

select
    method,

    count(*)                                                    as n_postings,

    -- Salary (only where pay is actually stated in the text)
    count(*) filter (where stratum = 'labeled_stated')           as n_salary_eligible,
    round(avg(case when salary_within_10pct then 1.0 else 0.0 end)
          filter (where stratum = 'labeled_stated'), 4)          as salary_within_10pct,
    round(median(salary_pct_error)
          filter (where stratum = 'labeled_stated'), 4)          as salary_median_error,

    -- Abstention (only where pay is NOT stated)
    count(*) filter (where stratum = 'labeled_not_stated')       as n_abstention_eligible,
    round(avg(case when abstained_correctly then 1.0 else 0.0 end)
          filter (where stratum = 'labeled_not_stated'), 4)      as abstention_rate,

    -- Pay period
    round(avg(case when pay_period_correct then 1.0 else 0.0 end)
          filter (where stratum = 'labeled_stated'), 4)          as pay_period_accuracy,

    -- Seniority (strict, plus the answer rate that explains it)
    count(*) filter (where gt_seniority is not null)             as n_seniority_labeled,
    round(avg(case when seniority_correct then 1.0 else 0.0 end)
          filter (where gt_seniority is not null), 4)            as seniority_accuracy,
    round(avg(case when seniority_answered then 1.0 else 0.0 end)
          filter (where gt_seniority is not null), 4)            as seniority_answer_rate,

    -- Reliability and cost
    round(avg(case when valid_json then 1.0 else 0.0 end), 4)    as valid_json_rate,
    round(sum(cost_usd), 4)                                      as total_cost_usd,
    round(sum(cost_usd) / count(*) * 1000, 2)                    as cost_per_1k_postings

from scored
group by method
order by seniority_accuracy desc