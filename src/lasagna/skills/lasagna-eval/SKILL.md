---
name: lasagna-eval
description: Use when measuring whether a RAG change actually helps — running benchmarks, comparing configurations, testing significance, or interpreting retrieval and generation metrics. Covers the traps that produce confident but meaningless numbers.
---

# Measure whether a RAG change helps

The hard part is not running the benchmark. It is making sure the number means
what you think it means.

## Start here

```bash
lasagna benchmark list                 # what is available, and what supports live mode
lasagna benchmark run smoke            # 24 docs, no download, no API key
```

`smoke` is a wiring check. A green run means the stack works end to end; it is
deliberately easy, so the score never means "this is good".

## Run a real comparison

```bash
lasagna benchmark run beir/scifact \
  --rerankers cross_encoder,my_reranker \
  --n-samples 200 --n-runs 3
```

Or in Python, where sweeps are clearer:

```python
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

config = ExperimentConfig(
    name="does my reranker help",
    benchmarks=[BenchmarkSpec(name="beir/scifact", n_samples=200)],
    pipeline=PipelineSweepSpec(rerankers=["cross_encoder", "my_reranker"]),
    n_runs=3,
)
results = await Experiment(config).run()
```

## The four traps

These produce numbers that look fine and are not.

**1. Wrong mode.** `generation_only` never constructs the pipeline — it scores
gold context with an LLM judge. A component override there is now a hard error
rather than a plausible number. Leave `mode="auto"`: it picks live when the
benchmark ships a corpus, and logs which it chose and why.

**2. One run.** Retrieval metrics move between runs. `n_runs=1` cannot tell a
real gap from noise. Use 3 or more, and check `variance_source="run"` so the
spread reflects run-to-run nondeterminism rather than a resampled query subset.

**3. Comparing incomparable runs.** Different `n_samples`, split, embedding
model or judge means the numbers are not commensurable:

```bash
lasagna results compare theirs.json yours.json
```

It reports the field-level difference rather than refusing, so you can see which
knob to turn.

**4. Reading a mean without a test.** Per-sample scores are carried through and
paired by sample identity, so significance testing is available — use it. A
0.02 NDCG gap on 100 samples is usually nothing.

**5. Reading "not significant" as "no difference."** They are different claims,
and the second one needs evidence the first does not provide:

```bash
lasagna results power results.json baseline candidate --metric ndcg@10
```

If the observed difference is below the minimum detectable effect, the run could
not have found an effect of that size even had it been real. Report it as
inconclusive and say how many queries would be needed.

**6. Adding two published gains together.** Technique A helps by 3%, technique B
helps by 4%, so together 7% — this does not follow. They may be fixing the same
queries. Only a 2×2 can tell you:

```yaml
pipeline:
  factorial:
    factor_a: enable_hyde
    factor_b: enable_reranking
```

```bash
lasagna results interaction results.json --metric ndcg@10
```

**7. Knowing a config lost without knowing why.** Retrieval failure and
generation failure need opposite fixes:

```bash
lasagna results per-query results.json --sort-by ndcg@10 -o rows.csv
```

One row per query with retrieval and generation metrics side by side.

## Which metric to trust

- **Retrieval** — `ndcg@10`, `mrr@10`, `recall@k`. Computed against qrels; no
  LLM involved; trustworthy.
- **Generation** — RAGAS faithfulness, answer relevancy, context precision.
  These are LLM-judged, so they move with the judge model. Pin a dated judge
  snapshot, never a floating alias like `gpt-4o-mini`.
- `context_recall` is known to be unreliable with some judges. If it reads 0.0
  across the board, suspect the judge, not the pipeline.

## Publish the result

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json --strict
lasagna results leaderboard runs/ --metric ndcg@10 --output leaderboards/
```

Leaderboards group by `(benchmark, comparability_key)`. Rows in one table held
the same configuration fixed; rows in different tables are not comparable and
are never ranked against each other.

## Rules

- State `n_samples`, split, and `n_runs` next to any number you report. A metric
  without them is not interpretable.
- Never tune on the test split. Use `eval` for iteration and report `test` once.
- If a benchmark cannot run live, retrieval metrics are not being measured —
  check `lasagna benchmark list` before drawing a retrieval conclusion.
- SQuAD v2, HotpotQA, MuSiQue and RAGBench run live against a corpus pooled
  from their own gold context. That is a real retrieval task but an easier one —
  no hard negatives — and the run is stamped `corpus_provenance: pooled_gold`.
  Never compare those numbers to BEIR's. Natural Questions and TriviaQA carry
  no context and stay generation-only.
- Report tokens, not dollars. `system_*` is the pipeline, `eval_*` is the judge;
  a total that folds them together makes an expensive judge look like an
  expensive pipeline.
- Check dataset terms before publishing: `lasagna benchmark licenses`.
