# Evaluation SDK -- Metrics Reference

This document describes every metric used in the Knowledge Evaluation SDK,
including formulas, implementation details, and statistical testing procedures.

---

## 1. IR Retrieval Metrics

All metrics below are implemented in
`src/lasagna/knowledge_evaluation_sdk/ir_metrics.py`.

Every function follows the same calling convention:

- `relevance`: a `Dict[str, int | float]` mapping document IDs to graded
  relevance scores (binary `{0, 1}` or graded `{0, 1, 2, 3}`).
- `retrieved`: an ordered `Sequence[str]` of document IDs as returned by the
  retrieval system.
- `k`: cut-off rank (default 10 for most functions).

Binary relevance is determined by `rel > 0` (any positive score counts as
relevant).

### NDCG@k (Normalised Discounted Cumulative Gain)

Primary ranking metric. Accounts for both relevance grades and rank position.

```
DCG_k  = sum_{i=1}^{k}  (2^{rel_i} - 1) / log2(i + 1)
IDCG_k = DCG_k computed on the ideal (descending-relevance) ordering
NDCG_k = DCG_k / IDCG_k
```

Returns a value in `[0, 1]`. Returns `0.0` when there are no relevant
documents (IDCG is zero).

Function: `ndcg_at_k(relevance, retrieved, k=10)`

### Precision@k

Fraction of relevant documents in the top-k results.

```
P@k = |{relevant docs in top-k}| / k
```

Returns a value in `[0, 1]`.

Function: `precision_at_k(relevance, retrieved, k=10)`

### Recall@k

Fraction of all relevant documents that appear in the top-k results.

```
R@k = |{relevant docs in top-k}| / |{total relevant docs}|
```

Returns `0.0` when there are no relevant documents in the ground truth.

Function: `recall_at_k(relevance, retrieved, k=10)`

### F1@k

Harmonic mean of Precision@k and Recall@k.

```
F1@k = 2 * P@k * R@k / (P@k + R@k)
```

Returns `0.0` when both P@k and R@k are zero.

Function: `f1_at_k(relevance, retrieved, k=10)`

### MRR (Mean Reciprocal Rank)

Reciprocal of the rank of the first relevant document. For corpus-level MRR,
average the per-query values.

```
MRR = 1 / rank_of_first_relevant_document
```

Returns `0.0` if no relevant document is found within the evaluated list.
Accepts an optional `k` parameter to limit the search depth.

Function: `mrr(relevance, retrieved, k=None)`

### MAP (Mean Average Precision)

Average Precision for a single query (corpus-level MAP is the mean of per-query
AP values). The denominator uses total R (the count of all relevant documents),
making this a recall-aware convention: a system that retrieves only some
relevant documents is penalised even if the retrieved ones are ranked perfectly.

```
AP = (1/R) * sum_{k=1}^{n} P(k) * rel(k)
```

where `R` is the total number of relevant documents, `P(k)` is precision at
rank `k`, and `rel(k)` is 1 if the document at rank `k` is relevant and 0
otherwise. The sum is accumulated only at ranks where a relevant document
appears.

Accepts an optional `k` parameter to limit evaluation depth.

Function: `map_score(relevance, retrieved, k=None)`

### Hit Rate@k (Success@k)

Binary indicator: `1.0` if at least one relevant document appears in the top-k
results, `0.0` otherwise.

```
Hit@k = 1.0  if any relevant doc in top-k
        0.0  otherwise
```

Function: `hit_rate_at_k(relevance, retrieved, k=10)`

---

## 2. LLM Judge Scoring

Implemented in `src/lasagna/knowledge_evaluation_sdk/judge.py`
with data models in `models.py`.

### Score Scale (1--5)

| Score | Level       | Description                                             |
|-------|-------------|---------------------------------------------------------|
| 5     | Excellent   | Perfect or near-perfect answer, fully addresses query   |
| 4     | Good        | Correct answer with minor omissions or less detail      |
| 3     | Partial     | Partially correct, captures some key points but misses important info |
| 2     | Poor        | Mostly incorrect or largely irrelevant                  |
| 1     | Fail        | Completely wrong, no relevant information, or no answer |

A case is considered "passed" when `score >= 3`.

### Sub-Criteria

Each evaluation also produces three continuous sub-scores, each in `[0.0, 1.0]`:

| Criterion            | Description                                        |
|----------------------|----------------------------------------------------|
| `relevance`          | Does the answer directly address the question?     |
| `completeness`       | Does it cover all key points from the expected answer? |
| `factual_accuracy`   | Is the information accurate based on context and expected answer? |

### Output Parsing

The judge uses `with_structured_output(JudgeScore)` when the underlying model
supports it (Pydantic schema-based extraction). When structured output is not
available, the judge falls back to prompting the model for a JSON object and
parsing it manually, handling both raw JSON and markdown-fenced code blocks.

### Batch Evaluation

`LLMJudge.batch_evaluate(cases, max_concurrent=3)` evaluates multiple cases
with a semaphore-controlled concurrency limit.

---

## 3. RAGAS Metrics (Optional)

Implemented in
`src/lasagna/knowledge_evaluation_sdk/ragas_evaluator.py`.

Requires the `ragas` package: `pip install lasagna-rag[eval]`.

### Available Metrics

| SDK Name              | RAGAS Class                            | Description                              |
|-----------------------|----------------------------------------|------------------------------------------|
| `faithfulness`        | `Faithfulness`                         | Whether the answer is grounded in the retrieved context |
| `answer_relevancy`    | `AnswerRelevancy`                      | Whether the answer is relevant to the question |
| `context_precision`   | `LLMContextPrecisionWithReference`     | Precision of retrieved context against the reference |
| `context_recall`      | `LLMContextRecall`                     | Whether all relevant information from the reference is retrieved |
| `factual_correctness` | `FactualCorrectness`                   | Factual accuracy of the generated answer |

Default metrics (when none are specified):
`["faithfulness", "context_recall", "context_precision"]`.

### Evaluation Modes

- **`evaluate_from_samples()`** -- Aggregate scores only. Accepts raw lists of
  questions, answers, contexts, and ground truths.
- **`evaluate_detailed()`** -- Returns a `DetailedEvalResult` with both
  `.aggregate` (dict of corpus-level scores) and `.per_sample` (list of
  per-query score dicts). Per-sample scores are essential for paired
  significance testing via `BenchmarkResults.significance_test()`.

---

## 4. Statistical Testing

Implemented in
`src/lasagna/knowledge_evaluation_sdk/results.py`.

### Paired Bootstrap Significance Test

`BenchmarkResults.significance_test(config_a, config_b, metric, benchmark)`

Compares two pipeline configurations on the same benchmark using per-query
paired scores.

**Procedure:**

1. Compute per-query differences: `d_i = score_a_i - score_b_i`.
2. Compute the observed mean difference: `observed_diff = mean(d)`.
3. For each of `n_bootstrap` resamples (default 10,000):
   - Draw `n` indices with replacement.
   - Compute the bootstrap mean difference `boot_diff`.
4. Compute the two-tailed p-value using null-centered resampling:

```
p = proportion of |boot_diff - observed_diff| >= |observed_diff|
```

This shifts the bootstrap distribution to center on the null hypothesis (zero
difference), then counts how often the shifted values exceed the observed
effect in magnitude.

5. Compute the 95% confidence interval from bootstrap percentiles:

```
CI_95 = [percentile(boot_diffs, 2.5), percentile(boot_diffs, 97.5)]
```

6. Compute Cohen's d_z for paired design:

```
d_z = mean(diffs) / std(diffs, ddof=1)
```

**Parameters:**

| Parameter     | Default | Description                          |
|---------------|---------|--------------------------------------|
| `n_bootstrap` | 10,000  | Number of bootstrap resamples        |
| `alpha`       | 0.05    | Significance level                   |
| `seed`        | 42      | Random seed for reproducibility      |

**Returns** a dict with keys: `p_value`, `significant`, `confidence_interval_95`,
`effect_size` (Cohen's d_z), `mean_a`, `mean_b`, `difference`, `n_samples`.

**Requirements:**

- Both configs must be evaluated on the same benchmark (same query set).
- Per-sample score arrays must have equal length.
- Samples with missing metrics are excluded (not defaulted to 0.0) to avoid
  biasing the test.

### Holm-Bonferroni Multiple Comparison Correction

`BenchmarkResults.pairwise_significance(metric, benchmark, correction="holm")`

When comparing more than two configurations, the family-wise error rate
inflates. The Holm-Bonferroni step-down procedure controls it.

**Procedure:**

1. Run pairwise bootstrap significance tests for all config pairs.
2. Sort the `m` resulting p-values in ascending order: `p_1 <= p_2 <= ... <= p_m`.
3. For each test `i` (1-indexed), compute the corrected threshold:

```
alpha_i = alpha / (m - i + 1)
```

Note: the implementation uses 0-indexed iteration, so test `i` has threshold
`alpha / (m - i)` where `i` starts at 0.

4. Starting from the smallest p-value, reject if `p_i <= alpha_i`. Once a test
   fails to reject, all subsequent tests also fail to reject.

This controls the family-wise error rate (FWER) while being less conservative
than Bonferroni correction.

**Returns** a list of dicts with keys: `config_a`, `config_b`, `p_value`,
`effect_size`, `difference`, `corrected_alpha`, `significant`.

---

## 5. Corpus-Level Aggregation

### IR Metric Aggregation

`corpus_metrics(relevances, retrieved_lists, k=10)`

Computes all seven IR metrics (NDCG@k, Precision@k, Recall@k, F1@k, MRR@k,
MAP@k, Hit Rate@k) for each query, then averages across queries.

```
corpus_metric = (1/N) * sum_{q=1}^{N} metric(q)
```

Returns a dict with keys like `"ndcg@10"`, `"precision@10"`, etc.

Raises `ValueError` if the lengths of `relevances` and `retrieved_lists`
differ.

### Benchmark Results Aggregation

`BenchmarkResults.aggregate_by_benchmark(metric)` returns a nested dict:

```
{benchmark_name: {config_name: AggregatedMetrics}}
```

`BenchmarkResults.aggregate_by_pipeline(metric)` returns:

```
{config_name: AggregatedMetrics}
```

`AggregatedMetrics` contains:

| Field    | Description                              |
|----------|------------------------------------------|
| `mean`   | Arithmetic mean across runs              |
| `std`    | Sample standard deviation (ddof=1)       |
| `n_runs` | Number of runs aggregated                |
| `values` | Raw list of per-run values               |

Standard deviation uses Bessel's correction (`n-1` denominator) for `n > 1`
and is `0.0` for a single run.

### Eval-Level Aggregation

`EvalMetrics.from_results(results)` computes aggregate statistics across all
evaluation cases from a single run:

- `avg_score` -- mean judge score (1--5)
- `avg_relevance`, `avg_completeness`, `avg_factual_accuracy` -- mean sub-criteria
- `pass_rate` -- percentage of cases with `score >= 3`
- `score_distribution` -- count of cases per score level (1 through 5)
- `avg_latency_ms`, `avg_rag_latency_ms` -- mean latency statistics
- `rag_usage_rate` -- percentage of cases where RAG context was used
