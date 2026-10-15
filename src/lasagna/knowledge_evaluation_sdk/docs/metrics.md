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

A document ID that repeats in `retrieved` (several chunks of one document) is
counted once, at its first rank, as `trec_eval` does. Every definition below
agrees with `trec_eval` 9.0.8 (`ndcg_cut`, `P`, `recall`, `map_cut`,
`recip_rank`, `success`, `Rprec`) on the hand-computed cases in
`tests/test_retrieval_metrics_match_trec_eval.py`.

In live mode the harness scores each query's ranked list of `metric_k`
distinct documents (default 10), and every per-query key carries that cutoff:
`ndcg@10`, `map@10`, `mrr@10`, `r_precision@10` and so on.

### NDCG@k (Normalised Discounted Cumulative Gain)

Primary ranking metric. Accounts for both relevance grades and rank position.

```
DCG_k  = sum_{i=1}^{k}  gain(rel_i) / log2(i + 1)
gain(rel) = rel  if rel > 0,  else 0
IDCG_k = DCG_k of the ideal ordering: every judged grade, sorted descending
NDCG_k = DCG_k / IDCG_k
```

Linear gain, as in `trec_eval` and therefore BEIR. A negative grade gains
nothing, as in `trec_eval`. Pass `exponential_gain=True` for the Burges
`2^rel - 1` convention when comparing with work that uses it; on binary
judgements the two agree.

Returns a value in `[0, 1]`. Returns `0.0` when there are no relevant
documents (IDCG is zero).

Function: `ndcg_at_k(relevance, retrieved, k=10, exponential_gain=False)`

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

### R-Precision

Precision at rank `R`, where `R` is the number of relevant documents
(`trec_eval`'s `Rprec`; KILT's page-level retrieval metric). The cutoff adapts
to the query, so a one-document query is scored at rank 1 and a four-hop query
at rank 4. At rank `R`, precision equals recall.

```
R-Prec = |{relevant docs in top-R}| / R
```

Returns `0.0` when there are no relevant documents. The harness reports it as
`r_precision@k` because it scores the list cut at `k`: when `R > k` the ranks
past `k` count as misses, so the value is a lower bound (at most `k / R`) on
the uncut `Rprec`.

Function: `r_precision(relevance, retrieved)`

### Complete Recall@k (all-gold recall)

`1.0` only if every relevant document is in the top-k. This is the retrieval
measure multi-hop QA needs (HotpotQA, MuSiQue, 2WikiMultihop, MultiHop-RAG): a
two-hop question cannot be answered from one of its two supporting documents,
yet Recall@k gives that query 0.5 and Hit@k gives it 1.0. It is the
document-level form of the "paragraph EM" that multi-hop retrievers report.

```
CompleteRecall@k = 1.0  if R@k == 1
                   0.0  otherwise
```

With one relevant document it equals Hit@k; with more than `k` it is always
`0.0`.

Function: `complete_recall_at_k(relevance, retrieved, k=10)`

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
| `factual_correctness` | `FactualCorrectness`                   | Claim-level F1 of the answer against the reference answer: the answer's and the reference's claims are each checked against the other |
| `noise_sensitivity`   | `NoiseSensitivity` (relevant mode)     | Share of the answer's claims that are wrong (not supported by the reference) yet supported by some retrieved context. **Lower is better** |
| `context_relevance`   | `ContextRelevance`                     | Relevance of the retrieved contexts to the question, judged without a reference (the first leg of the TruLens RAG triad) |

`RAGASEvaluator`'s default metrics (when none are specified):
`["faithfulness", "context_recall", "context_precision"]`.

### What the Benchmark Harness Requests

Live and offline benchmark runs request
`["faithfulness", "context_recall", "context_precision", "factual_correctness"]`
(`ragas_evaluator.HARNESS_METRICS`). `factual_correctness` is the only judged
metric that compares the answer with the reference answer; the other three
measure groundedness and context quality. It costs about four judge calls per
answer. Offline samples without a gold context get only the metrics that do not
judge one (`factual_correctness`, `answer_relevancy`).

`RAG_EVAL_JUDGE_METRICS` (comma-separated names from the table) replaces that
set, for example `faithfulness,factual_correctness,noise_sensitivity`. An
unknown name fails the task rather than dropping the metric. `noise_sensitivity`
and `context_relevance` are opt-in because of their judge cost (noise
sensitivity checks every claim against every context: about 13 calls per
answer over five contexts). `answer_relevancy` needs an embedding endpoint for
the judge, which the harness does not configure.

A judged metric scores only answerable questions (it needs a reference). A
metric whose every judgement failed is reported in `failed_metrics`, fails the
task, and is never recorded as 0.0. For agentic configurations the per-query
keys are suffixed `_regen` and `_agentic`, and an unsuffixed canonical key
(agentic score where present) is emitted for every requested metric.

### Evaluation Modes

- **`evaluate_from_samples()`** -- Aggregate scores only. Accepts raw lists of
  questions, answers, contexts, and ground truths.
- **`evaluate_detailed()`** -- Returns a `DetailedEvalResult` with both
  `.aggregate` (dict of corpus-level scores) and `.per_sample` (list of
  per-query score dicts). Per-sample scores are essential for paired
  significance testing via `BenchmarkResults.significance_test()`.

---

## 4. Answer Metrics Without a Judge

Implemented in `src/lasagna/knowledge_evaluation_sdk/answer_metrics.py` and
scored per query by the harness in live and offline runs. They cost no LLM
call. All are computed on the answer with citation markers (`[1]`) removed, as
ALCE does: the markers are not part of the answer.

Each takes the reference answer and its aliases and reports the best match.
They share the SQuAD v2 no-answer rule: on a question the benchmark marks
unanswerable (SQuAD v2), a value is 1.0 if the system abstained and 0.0
otherwise; an answerable question answered with an abstention scores 0.0; an
empty answer (generation failed) scores 0.0 and is also counted in
`generations_failed`.

### Exact Match and Token F1

The official SQuAD v2.0 metrics: lowercase, strip punctuation, strip articles,
collapse whitespace; EM is string equality, token F1 the harmonic mean of
token precision and recall (shared tokens counted with multiplicity).
Functions: `exact_match`, `token_f1`.

### ROUGE-L

Longest-common-subsequence F1 (Lin, 2004) over `rouge_score`'s default tokens
(lowercase alphanumeric runs, no stemming), so values equal
`RougeScorer(["rougeL"]).score(reference, answer).fmeasure`.

```
P = LCS(answer, reference) / |answer|
R = LCS(answer, reference) / |reference|
ROUGE-L = 2PR / (P + R)
```

Token F1 counts shared words in any order; ROUGE-L credits them only in the
reference's order. Function: `rouge_l`. Key: `rouge_l`.

### Answer Containment

`1.0` if a reference answer appears in the answer as a contiguous run of
whole tokens, after SQuAD normalization of both. This is the "accuracy" of
PopQA (Mallen et al., 2023) and Self-RAG, FlashRAG's Sub-EM, and ALCE's STR-EM
for a single required answer: "The Eiffel Tower is in Paris." scores 0 on
exact match against "Paris" and 1 here. It is stricter than the raw substring
test those papers use: "no" is not found in "I do not know" and "1" is not
found in "1990". Function: `contains_answer`. Key: `contains_answer`.

### Abstention Rate

`1.0` if the answer is an abstention ("I don't know", "The context does not
...", or the exact reply `unanswerable` that the SQuAD v2 prompt asks for),
`0.0` otherwise; absent for an empty answer, which is a failed generation and
not a choice. The aggregate is the share of answers that abstain. On a
benchmark with unanswerable questions the harness also reports
`abstention_rate_has_ans` (over-refusal on answerable questions) and
`abstention_rate_no_ans` (RGB's negative rejection rate). Functions:
`is_abstention`, `abstention_rate`. Key: `abstention_rate`.

### Answer Similarity

Cosine similarity between embeddings of the answer and of the closest
reference answer: RAGAS's `answer_similarity` and the bi-encoder form of SAS
(Risch et al., 2021). It credits paraphrase ("she won the Nobel twice" against
"two Nobel Prizes") that token overlap misses.

```
answer_similarity = max over references r of cos(E(answer), E(r))
```

`E` is `sentence-transformers/all-MiniLM-L6-v2` on CPU, separate from the
pipeline's own embedder. `RAG_EVAL_SIMILARITY_MODEL` names another model, or
`none` to skip the metric. A different model gives different numbers, so hold
it fixed across a comparison; the comparability key does not record it. The
value is in `[-1, 1]` (in practice `[0, 1]`); `0.0` for an empty answer; absent
for a question with no reference (unanswerable), and absent with a warning when
the model cannot be loaded. Functions: `answer_similarity`,
`load_similarity_encoder`. Key: `answer_similarity`.

---

## 5. Citation Quality (ALCE)

Citation recall and precision as defined by ALCE (Gao et al., 2023), for
answers that cite passages with `[n]` markers (`[1]`, `[1][2]`, `[1, 2]`).
`[n]` is the n-th context the generator was shown (1-based), the same list
RAGAS judges. The answer is split into sentences at `.`, `!` or `?` followed by
whitespace; markers that open a sentence belong to the one before.

- **Citation recall**: the share of sentences whose cited passages,
  concatenated, entail the sentence. A sentence with no citation, or citing a
  passage number that was not shown, scores 0.
- **Citation precision**: the share of citations that are not irrelevant. On
  a single-citation sentence the citation counts if it entails the sentence.
  On a multi-citation sentence that its citations jointly entail, a citation
  is irrelevant when it does not entail the sentence alone and the remaining
  citations still do; on one they do not jointly entail, none counts.

Entailment is judged by the evaluation judge (`judge.entailment_judge`, the
model `RAG_EVAL_JUDGE_MODEL` names) where ALCE uses the TRUE T5-11B NLI model,
so absolute values are not comparable with ALCE's tables.

Absent, never 0.0, when the answer cites nothing (a generator that does not
cite), when no judge model is configured, or when the judge could not decide
one of the answer's entailments. If answers cite and the judge scored none of
them, the task fails, as for a RAGAS metric. Function:
`citation_precision_recall`. Keys: `citation_recall`, `citation_precision`.

---

## 6. Operational Metrics

| Key | Where | Definition |
|---|---|---|
| `latency_ms` | per query row | Live retrieval row: wall-clock time of the pipeline query (retrieval, plus the agentic graph when one runs). Live answer row: that plus the successful answer-writing call, end to end; `retrieval_ms` and `generation_ms` hold the two parts, and back-off sleeps between retries are not counted. Offline: the answer-writing call |
| `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms` | task | Over queries, with a query's retrieval and generation rows summed first |
| `input_tokens`, `output_tokens`, `llm_calls` | per query row | What the system under test spent, counted at the provider |
| `input_tokens_per_query`, `output_tokens_per_query`, `llm_calls_per_query` | task metrics | The same, summed over a query's rows, averaged over queries |
| `system_*`, `eval_*` tokens and calls | task | Totals, with the judge's spend (`eval_*`) kept apart from the system's |
| `generations_failed`, `short_lists` | task metrics | Answers that could not be generated; ranked lists shorter than k |
| `llm_cache_hits`, `llm_cache_misses` | task metrics | LLM calls served from the response cache, and calls it looked up and did not have. Present only on a task that looked something up. They describe the run, not the system: a replayed call is counted in the token totals at its cold-run usage, so tokens and calls read the same cold or warm. The run's hit rate is `experiment_config.llm_cache` |

`llm_calls == 0` means no call was seen, which for a retrieval-only arm is the
truth and for an uninstrumented client an undercount. Tokens are reported, not
dollars: a price is not a measurement.

Lower is better for these (and for `noise_sensitivity`);
`results.lower_is_better(metric)` says so, and the text leaderboard ranks them
ascending.

---

## 7. Metrics Considered and Left Out

- **bpref, infAP, Judged@k**: for incomplete judgements. The BEIR loader keeps
  only positive qrels, so Judged@k would equal P@k.
- **Capped recall (R_cap@k)**: equals Recall@k whenever R <= k, which holds for
  every registered QA benchmark; for many-relevant sets use a larger
  `metric_k`.
- **KILT-EM / KILT-F1** (answer credit only when R-Prec = 1): derivable from
  `results per-query`, which joins each query's retrieval and generation rows.
- **BLEU, BERTScore**: BLEU is precision-only n-gram overlap, poor on short
  answers; BERTScore needs a large model and answer similarity covers semantic
  overlap.
- **ARES**: needs judges fine-tuned on the target domain and a human-labelled
  validation set.
- **RAGChecker's claim-level diagnostics** (context utilization, self-knowledge,
  hallucination): `noise_sensitivity` is the RAGAS form of the closest one.
- **MAUVE** (ALCE fluency): for long-form generation.
- **Throughput (queries per second)**: the harness issues queries one at a
  time, so it would be `1000 / avg_latency_ms` and say nothing about concurrent
  capacity.
- **Index build time and index size**: the corpus is ingested once and reused
  across tasks and runs, so a task has no build of its own to time.

---

## 8. Statistical Testing

Implemented in
`src/lasagna/knowledge_evaluation_sdk/results.py`.

### Paired Sign-Flip Significance Test

`BenchmarkResults.significance_test(config_a, config_b, metric, benchmark)`

Compares two pipeline configurations on the same benchmark using per-query
paired scores. It is a sign-flip randomization test with a t interval, because
both hold their stated level at the 10 to 50 queries RAG evaluations use. The
paired bootstrap it replaced did not: at 12 to 20 queries it rejected a true
null 8 to 9 percent of the time at alpha 0.05.

**Procedure:**

1. Average each query's score across every run of each config, then compute
   the per-query differences: `d_i = score_a_i - score_b_i`.
2. Compute the observed mean difference: `observed_diff = mean(d)`.
3. Compute the two-sided p-value. If the two configs were equivalent, each
   difference would be as likely to come out with either sign, so all `2^n`
   sign patterns are equally likely:

```
p = share of sign patterns s with |sum(s_i * d_i)| >= |sum(d_i)|
```

   Differences of exactly zero cannot change sign and drop out. With at most 30
   non-zero differences every pattern is enumerated, so the p-value is exact.
   Above that, `n_bootstrap` random patterns are drawn (default 10,000) and
   `p = (count + 1) / (n_bootstrap + 1)`, which is never exactly 0. The `test`
   key says which was used.

4. Compute the `1 - alpha` confidence interval for the mean difference (95% at
   the default alpha):

```
CI = mean(d) +/- t(1 - alpha/2, n - 1) * std(d, ddof=1) / sqrt(n)
```

5. Compute Cohen's d_z for paired design:

```
d_z = mean(diffs) / std(diffs, ddof=1)
```

**Parameters:**

| Parameter     | Default | Description                                                    |
|---------------|---------|----------------------------------------------------------------|
| `n_bootstrap` | 10,000  | Random sign patterns drawn above 30 non-zero differences (the name is kept for compatibility) |
| `alpha`       | 0.05    | Significance level; the interval is at `1 - alpha`             |
| `seed`        | 42      | Random seed for the sampled patterns                           |

**Returns** a dict with keys: `p_value`, `significant`, `confidence_interval`
(at `confidence_level`; also as `confidence_interval_95`), `ci_method` (`"t"`),
`test`, `effect_size` (Cohen's d_z), `mean_a`, `mean_b`, `difference`,
`n_samples`, `n_paired`, `n_dropped_a`, `n_dropped_b`.

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

1. Run the paired sign-flip test above for every config pair.
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

## 9. Corpus-Level Aggregation

### IR Metric Aggregation

`corpus_metrics(relevances, retrieved_lists, k=10)`

Computes all nine IR metrics (NDCG@k, Precision@k, Recall@k, F1@k, MRR@k,
MAP@k, Hit Rate@k, R-Precision over the top-k list, Complete Recall@k) for
each query, then averages across queries.

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
