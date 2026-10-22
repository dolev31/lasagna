# Metrics

This page lists the metrics a run reports and what each one costs. The
[full definitions](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/docs/metrics.md)
give formulas and edge cases. The metrics version is 1.7. It is part of the
[comparability key](../concepts/comparability.md), so numbers from a different
version never share a leaderboard group.

## Retrieval metrics

The retrieval track scores each query's ranked list of `metric_k` distinct
documents (10 by default). Every key carries the cutoff, for example `ndcg@10`.
A document whose chunks appear more than once counts once, at its first rank. The
definitions agree with `trec_eval` 9.0.8.

| Metric | Key | What it measures |
|---|---|---|
| NDCG@k | `ndcg@10` | Rank-weighted gain, normalized by the best possible order. Linear gain, as in BEIR. |
| Precision@k | `precision@10` | The share of the top k that is relevant. |
| Recall@k | `recall@10` | The share of relevant documents found in the top k. |
| F1@k | `f1@10` | The harmonic mean of precision and recall. |
| MRR | `mrr@10` | The reciprocal rank of the first relevant document. |
| MAP | `map@10` | The mean of precision at each relevant rank. |
| Hit rate@k | `hit_rate@10` | Whether any relevant document is in the top k. |
| R-precision | `r_precision@10` | Precision at the number of relevant documents. |
| Complete recall@k | `complete_recall@10` | Whether every relevant document is in the top k. |

The ranked list is the retriever's order, or the reranker's when the pipeline
reranks. Context reordering is off for this track, because it is a choice about
what the generator reads, not a ranking.

## Answer metrics without a judge

These cost no LLM call.

| Metric | Key | What it measures |
|---|---|---|
| Exact match | `exact_match` | SQuAD v2 string equality after normalization. |
| Token F1 | `token_f1` | Shared tokens, as in SQuAD v2. |
| ROUGE-L | `rouge_l` | Longest common subsequence F1. |
| Answer containment | `contains_answer` | Whether a reference answer appears in the answer as whole tokens. |
| Abstention rate | `abstention_rate` | The share of answers that decline to answer. |
| Answer similarity | `answer_similarity` | Cosine similarity to the closest reference answer. Uses `all-MiniLM-L6-v2` on CPU by default. |

## Judged metrics

A judge LLM scores these. A judged metric scores only answerable questions. If
every judgment fails, the task fails. The score is never recorded as 0.0.

| Metric | Key | What it measures |
|---|---|---|
| Faithfulness | `faithfulness` | Whether the answer is grounded in the retrieved context. |
| Context precision | `context_precision` | Precision of the retrieved context against the reference. |
| Context recall | `context_recall` | Whether the context holds what the reference needs. |
| Factual correctness | `factual_correctness` | Claim-level F1 of the answer against the reference. |
| Noise sensitivity | `noise_sensitivity` | The share of the answer's claims that are wrong yet supported by a retrieved context. Lower is better. Opt in. |
| Context relevance | `context_relevance` | Relevance of the context to the question, with no reference. Opt in. |
| Citation recall | `citation_recall` | The share of sentences that their citations entail (ALCE). |
| Citation precision | `citation_precision` | The share of citations that are not irrelevant (ALCE). |

The harness asks for `faithfulness`, `context_recall`, `context_precision` and
`factual_correctness`. Set `RAG_EVAL_JUDGE_METRICS` to a comma-separated list to
change the set. `noise_sensitivity` and `context_relevance` are opt-in because they
cost many judge calls: about 13 per answer for noise sensitivity over five
contexts.

Citation metrics are absent, not zero, when the answer cites nothing or no judge
is available.

## Operational metrics

Lower is better for all of these.

| Metric | Key | What it measures |
|---|---|---|
| Latency | `latency_ms`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms` | Wall-clock time of a query, and its distribution. |
| Tokens and calls | `input_tokens`, `output_tokens`, `llm_calls` | What the system under test spent. The judge's spend is kept apart as `eval_*`. |
| Cost estimate | `cost_usd_estimate` | Present only when the sweep sets `prices:`. An estimate, from rates that change. |
| Cache counts | `llm_cache_hits`, `llm_cache_misses` | LLM calls served from the [response cache](../guides/llm-cache.md). They describe the run and rank nothing. |

`llm_calls == 0` means no call was seen. For a retrieval-only arm that is the
truth.

## Statistical tests

The significance, power and interaction tests are described in
[significance testing](../concepts/significance.md).
