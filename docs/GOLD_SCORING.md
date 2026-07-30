# Gold-set scoring integration

## Current benchmark

The primary extraction benchmark contains **80 manually reviewed,
difficulty-enriched job postings**.

Five methods are evaluated on the same job IDs:

- `rules`
- `zero_shot`
- `few_shot`
- `schema_rules`
- `decomposed`

The four LLM variants contribute 320 predictions, and the deterministic
baseline contributes 80 predictions, for 400 method-posting evaluations.

## Inputs

The scorer uses these files under `data/processed/`:

- `gold_labels.json` — 80 manually reviewed records
- `llm_extractions_gold.parquet` — 80 jobs × 4 LLM variants
- `baseline_gold_seed.parquet` — 80 deterministic rules predictions

The scored fields are:

- `required_skills`
- `preferred_skills`
- `work_arrangement`
- `years_experience_min`

## Work-arrangement labels

Work arrangement is normalized to four gold classes:

- `Remote`
- `Hybrid`
- `Onsite`
- `Unclear`

Missing or malformed output is represented as `InvalidPrediction`. It counts
as incorrect but does not create an artificial fifth class when macro F1 is
calculated.

## Run

Run the following commands:

    python -m pytest -q tests/test_score_gold.py
    python src/evaluation/score_gold.py --no-tests
    python src/evaluation/score_gold.py

Every method must cover all 80 gold job IDs. Prediction rows are never silently
dropped.

## Outputs

- `gold_method_summary.csv`
- `gold_metrics_long.csv`
- `gold_item_scores.parquet`
- `gold_skill_errors.csv`
- `gold_pairwise_tests.csv`
- `gold_evaluation.md`

## Metrics

### Skills

Skills use entity-level micro precision, recall, and F1 for required,
preferred, and combined skills.

The scorer also reports exact set match, per-posting macro F1, predicted and
gold skill counts, classification accuracy, missed skills, and extra skills.

### Work arrangement

Work arrangement uses strict accuracy and macro F1 over the gold label space.

### Minimum years of experience

Years of experience uses exact accuracy, accuracy within one year, MAE, answer
rate, correct-abstention rate, and false-positive rate.

## Statistical comparisons

All comparisons are paired against the deterministic rules baseline.

- Skill F1 differences use paired job-level bootstrap 95% confidence intervals.
- Skill comparisons do not report p-values.
- Work accuracy, years-exact accuracy, and years-within-one accuracy use
  McNemar's exact test.
- Holm correction is applied across the work and years hypothesis tests.

## Interpretation guardrails

- The gold sample deliberately overrepresents difficult postings.
- The set was reviewed by one annotator.
- Missing or malformed predictions count as errors.
- The manual gold benchmark is separate from the earlier 400-posting
  provider-reference routing experiment.
