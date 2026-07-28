# Gold-set scoring integration

## Inputs used by this repository

The scorer auto-detects the current project files:

- `data/processed/gold_labels.json` — annotation export with records under `labels`
- `data/processed/llm_extractions_gold.parquet` — 80 jobs × 4 variants
- `data/processed/baseline_gold_seed.parquet` — 80 rules predictions

The shared scored fields are:

- `required_skills`
- `preferred_skills`
- `years_experience_min`

The gold file also contains `work_arrangement`, but the current LLM and rules
prediction files do not. Work metrics are therefore left blank rather than
counted as errors. Add a versioned extraction contract later to score that
field without changing the historical benchmark prompts.

## Run

```bash
python -m pytest -q tests/test_score_gold.py
python src/evaluation/score_gold.py --no-tests
python src/evaluation/score_gold.py
```

## Outputs

- `gold_method_summary.csv`
- `gold_metrics_long.csv`
- `gold_item_scores.parquet`
- `gold_skill_errors.csv`
- `gold_pairwise_tests.csv`
- `gold_evaluation.md`

## Metrics

Skills use entity-level micro precision, recall and F1, plus exact set match,
any-skill F1 and required/preferred bucket accuracy. Years use exact accuracy,
accuracy within one year, MAE, answer rate, correct abstention and false-positive
rate. Skill F1 uncertainty uses paired job-level bootstrap; years comparisons
use McNemar's exact test with Holm correction.
