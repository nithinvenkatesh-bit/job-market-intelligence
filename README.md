# Job Market Intelligence

[![CI](https://github.com/nithinvenkatesh-bit/job-market-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/nithinvenkatesh-bit/job-market-intelligence/actions/workflows/ci.yml)

**Does an LLM beat regex at reading job postings? Sometimes. I measured where.**

An extraction pipeline over 124K job postings that compares four LLM prompt
strategies against a deterministic rules baseline, field by field, using a
controlled paired offline benchmark with bootstrap confidence intervals and
McNemar's exact tests — then routes each field to whichever method wins.

**The robust finding: the LLM beat the rules on seniority by 23–31 percentage
points**, and a **hybrid policy — rules first, LLM only when the rules abstain —
beat LLM-only by 5.4pp while making 36% fewer API calls.** Better *and* cheaper,
because the two methods fail on different postings.

---

## Results

All comparisons are paired: every method processed the same 400 postings.
~24 tests were run, so Holm-Bonferroni corrections are reported alongside raw
p-values. Effect sizes and confidence intervals are the primary evidence;
p-values are secondary.

| Field | Direction | Effect (95% CI) | Survives correction? |
|---|---|---|---|
| **Seniority** | **LLM better** | +23.1 to +30.6pp | **Yes — all framings** |
| Abstention | Rules better | −10.1 to −11.4pp | Within-family only |
| Salary (±10%, annualised) | Rules better | −4.0 to −5.0pp | **No** |
| Pay period | No difference | CI spans zero | — |

### What that means, stated carefully

**Seniority is the solid result.** All four prompt variants beat the rules
independently, at Holm-adjusted p < 0.0001 under both the permissive and the
conservative correction. The reason is concrete: 444 postings labelled Entry
level carry no seniority keyword in the title at all — *File Clerk*, *Pharmacy
Technician*, *Dental Assistant*. Unreachable by regex, trivial for a model that
reads the description.

**On salary, the rules performed better, but the difference did not survive
correction.** The effect was consistent — −4 to −5pp against every variant, all
four confidence intervals excluding zero — but Holm-adjusted p = 0.097 within
metric. Reported as suggestive, not established. It remains notable that a free
deterministic method was *not outperformed* by a $2.39-per-1,000 model on this
field.

**On abstention the rules were better, and this survives within-metric
correction but not the conservative whole-family one** (Holm p = 0.016 vs
0.066). Worth a caveat: the test rests on few discordant pairs. Within this
400-posting experiment, there was **no observed case where an LLM variant
correctly returned null and the rules did not** — the discordant count is
literally zero in that direction for all four variants. That is a strong
pattern in this sample, not a proven property of the model.

---

## Hybrid routing

Conditional accuracy exposed the optimisation. On seniority the rules are *more*
accurate than the LLM when they fire (77.6% vs 62.6%) — they simply abstain on
64% of postings. So the policy is not "use the LLM," it is **"use the rules when
they speak, and the LLM only when they are silent."**

| Policy | Accuracy | Macro F1 | LLM calls | Cost / 1k |
|---|---|---|---|---|
| Rules only | 28.2% | 0.479 | 0 | $0.00 |
| LLM only | 58.8% | 0.613 | 294 | $2.39 |
| **Hybrid (rules first)** | **64.3%** | **0.653** | **187** | **$1.52** |

On 294 postings carrying a seniority reference label, hybrid routing improved
accuracy by **5.4 percentage points over LLM-only** — 95% paired bootstrap CI
**+2.0 to +9.2pp**, McNemar exact **p = 0.0052** — while reducing LLM calls by
**36%**.

Two qualifications belong next to that number. The 294 rows carry *reference
labels* from the dataset, not manually verified gold labels. And the result
applies to this dataset, taxonomy and model — not to extraction tasks generally.

The rules' *abstention* turns out to be an informative signal. Routing on it is
what produces the gain.

---

## Prompt experiment

Four strategies, same postings, same model (Claude Haiku 4.5), temperature 0.

| Variant | Seniority F1 | Salary ±10% | Valid JSON | Evidence inconsistency | Cost/1k |
|---|---|---|---|---|---|
| zero_shot | 0.602 | 84.1% | 99.0% | 7.4% | $2.09 |
| few_shot | 0.561 | 83.5% | 99.0% | 7.1% | $2.50 |
| **schema_rules** | **0.613** | 83.2% | 98.5% | **4.2%** | $2.39 |
| decomposed | 0.515 | 83.8% | 99.8% | 12.5% | $3.20 |

`schema_rules` scored highest. Only one pairwise difference survives
correction: **schema_rules over decomposed on seniority, +7.5pp
[+3.4, +11.6], Holm p = 0.012.** Its apparent +4.8pp edge over `zero_shot`
(raw p = 0.029) does **not** survive — a good illustration of why the
correction matters.

That prompt was written *from the baseline error analysis*: every rule in it
corresponds to an observed regex failure on a real posting. Doing the error
analysis before touching the LLM paid for itself twice — once fixing the regex,
once writing the best-performing prompt.

**A designed intervention that failed.** `decomposed` asked the model to locate
evidence spans before extracting, expecting better grounding. It finished
*last* on seniority, *worst* on evidence-value consistency (12.5% — it quotes a
salary then returns null three times as often as schema_rules), and cost the
most. Reported because a negative result from a deliberate hypothesis is worth
more than four variants that all behaved as predicted.

**Evidence-value consistency** — cases where the model quotes supporting text
then returns null for the value — is measured rather than assumed. It appeared
on the very first test call, which is why it became a metric.

---

## Data and label quality

[LinkedIn Job Postings 2023–2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
— 123,849 postings. Check the dataset page for current licence terms before reuse.

The evaluation design turns on one property: **the dataset supplies *reference
labels*, not verified gold labels**, and their quality varies by field.
Treating them alike would have quietly corrupted the metrics.

| Field | Label quality | Source |
|---|---|---|
| Salary, pay period | **Strong reference** | Structured fields, 29% coverage |
| Seniority | **Weak reference** | `formatted_experience_level` — demonstrably inconsistent |
| Remote / hybrid / onsite | **None** | `remote_allowed` is 1-or-null; null means *unknown* |
| Years of experience, skills | **None** | Requires hand-labelling |

**The seniority label is demonstrably unreliable.** Error analysis surfaced
postings tagged `YEARLY` whose text reads `$20 - $25 an hour`. A $20/year salary
does not exist. In those cases the extractor was right and the *label* was
wrong — which caps how high any method can score, and is stated plainly rather
than absorbed into the numbers.

**Salary is only scored where extraction is possible.** 71% of labelled postings
actually mention pay in the prose. The rest form a separate stratum where the
correct behaviour is silence — that's what makes the abstention metric
meaningful instead of rewarding a system that always guesses.

**A finding that justifies the project:** 22,417 postings state pay in the text
but carry no structured label. The dataset field captures 36K; extraction from
prose recovers another 22K — a 62% increase in salary coverage.

---

## Architecture

```
Kaggle CSV (124K postings)
      │
      ├─► build_datasets.py ──► benchmark (2,000, stratified)
      │                         data_roles (1,977)
      │                         gold_seed (150)
      └─► build_holdout.py ───► holdout (2,000, zero overlap)
                                       │
        ┌──────────────────────────────┴───────────────────┐
        ▼                                                  ▼
  baselines.py                                    llm/run_experiment.py
  regex + keyword rules                           4 prompts × 400 postings
  (free, deterministic)                           (cached, provider-agnostic)
        └──────────────────────────────┬───────────────────┘
                                       ▼
                     evaluation/compare_methods.py
         paired McNemar + bootstrap CIs + Holm correction
                                       ▼
                     evaluation/hybrid.py  ──► routing policy
                                       ▼
                     dbt: staging → intermediate → marts
                          4 models, 28 data tests
                                       ▼
                     Airflow DAG (10 tasks, weekly)
```

![Airflow DAG — all tasks green on a full run](docs/airflow_dag.png)

Stack: Python, DuckDB, dbt, Airflow 3, Anthropic API, pandas, scipy.

---

## Engineering decisions worth defending

**A holdout set exists because the baseline was tuned twice.** Error analysis
produced five regex fixes; any further gain on the same file would be tuning,
not capability. On 2,000 unseen postings the numbers held (85.4% vs 85.2%), so
the fixes generalise. The two 2,000-posting samples showed roughly **1.3
percentage points of observed between-sample variation**, which is why smaller
differences are not treated as real. (Two draws cannot establish a formal noise
floor; this is an observation, not an estimate.)

**dbt reimplements the Python metrics, and the disagreement found a bug.** The
two implementations differed on exactly one posting per variant. dbt was testing
the *annualised* salary for abstention, so a figure with an unparseable pay
period scored as a correct abstention when the model had in fact hallucinated a
number. Python was right. Cross-validating in a second language is how it
surfaced.

**Multiple comparisons are corrected, and the correction changed conclusions.**
Running ~24 tests at α=0.05 means roughly one false positive by chance. Both a
within-metric and a conservative whole-family Holm correction are reported. Two
claims that looked significant on raw p-values do not survive, and they are
labelled accordingly rather than quietly retained.

**Reproducibility bug that cost real money.** DuckDB's `setseed()` does *not*
make `ORDER BY random()` stable across sessions. Every pipeline run drew a
different sample, silently invalidating the LLM cache and re-billing for
extractions already paid for. Fixed three ways: order by `hash(job_id)` instead;
pin the exact sample in `config/experiment_sample.json`; and refuse to overwrite
a results file with fewer rows than it already has.

**Two conda environments.** Airflow pins dozens of dependencies and would
downgrade pandas and duckdb in the project environment, so it lives separately
and tasks invoke the project interpreter directly. This mirrors production,
where the scheduler's environment is not the task's environment.

**Cost is a first-class metric.** Tokens, latency, and dollars are recorded per
call. The conclusion of this project is a cost/quality tradeoff, so cost cannot
be an afterthought. Total spend for the full experiment: **$3.99**.

---

## Limitations

- **This is an offline benchmark, not an A/B test.** No users were randomised;
  no production traffic was involved. It demonstrates experimental design and
  statistical evaluation, not online experimentation.
- **The data is a 2023–24 snapshot, not the current market.** It predates most
  of the GenAI hiring wave. It was chosen because it is the only public dataset
  carrying both raw description text *and* structured fields — which the
  evaluation requires. Any market claim here is historical.
- **Labels are provider-supplied references, not verified gold labels.** For
  seniority in particular, known-incorrect labels mean the achievable ceiling is
  unknown.
- **Skills are not evaluated.** The dataset's skill taxonomy is coarse job
  *functions* (Information Technology, Analyst, Finance), not technologies, so
  it cannot serve as a reference for SQL/Python/dbt extraction.
- **Years of experience and work arrangement are unmeasured** — no reference
  labels exist. A hand-labelled gold set is scaffolded but not built.
- **Rare-class seniority estimates are volatile.** Executive F1 ranged
  0.390–0.750 across sample draws on ~20 postings. Per-class figures are
  reported with support counts and should not be read as precise.
- **The main findings replicated on a second deterministic sample draw** — one
  robustness check, not a guarantee of sample independence.
- Hourly pay is annualised at 2,080 h/yr. Part-time roles are overstated by that
  assumption.

---

## Running it

```bash
conda create -n jmi python=3.12 -y && conda activate jmi
pip install -r requirements.txt

# Download the Kaggle dataset into data/raw/
python src/build_datasets.py
python src/build_holdout.py
python src/baselines.py --dataset holdout
python src/evaluate_baselines.py --dataset holdout

# LLM steps need ANTHROPIC_API_KEY in .env.
# Responses are cached and the sample is pinned, so re-runs are free.
python src/llm/run_experiment.py
python src/evaluation/compare_methods.py
python src/evaluation/hybrid.py

cd dbt && dbt deps && dbt run && dbt test
pytest -q
```

---

## Layout

```
src/
  build_datasets.py        stratified benchmark, data-role slice, gold seed
  build_holdout.py         unseen validation set
  baselines.py             regex + keyword extractors (the bar to beat)
  evaluate_baselines.py    strict metrics, annualised salary, abstention
  error_analysis.py        failure inspection — this wrote the best prompt
  llm/
    client.py              cached, retrying, cost-tracking API client
    prompts.py             four prompt strategies
    run_experiment.py      paired runner over a pinned sample
  evaluation/
    compare_methods.py     McNemar + bootstrap + Holm, per field
    hybrid.py              routing policy comparison
tests/                     41 regression tests, one per bug found
dbt/                       staging → intermediate → marts, 28 tests
dags/                      Airflow DAG
config/                    pinned experiment sample
```
