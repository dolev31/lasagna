---
name: lasagna-eval
description: Use when measuring whether a RAG change actually helps: running benchmarks, comparing configurations, testing significance, or interpreting retrieval and generation metrics. Covers the traps that produce confident but meaningless numbers.
---

# Measure whether a RAG change helps

The hard part is not running the benchmark. It is making sure the number means
what you think it means.

## Start here

```bash
lasagna benchmark list                       # what is available, and what supports live mode
lasagna benchmark run smoke --profile speed  # 24 docs, no download beyond a 91 MB model, no API key
```

`smoke` is a wiring check. A green run means the stack works end to end; it is
deliberately easy, so the score never means "this is good". Without
`--profile speed`, `benchmark run` starts from the `standard` profile, which
downloads about 3.1 GB of models.

## Run a real comparison

A comparison is a sweep file: the same benchmark, the same queries and the same
everything-else, through two or more arms.

```yaml
# sweep.yaml
name: does my reranker help
benchmarks:
  - name: beir/scifact
    n_samples: 200
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_ideas:my_reranker"]   # a control and the candidate
n_runs: 1
```

```bash
lasagna benchmark experiment sweep.yaml --dry-run   # arms, tasks, models, token estimate
lasagna benchmark experiment sweep.yaml
lasagna results significance <output>/results.json -m ndcg@10 --all
```

`my_ideas:my_reranker` is a function or class in `my_ideas.py` in the working
directory; a registered or installed component is named by its name
(`lasagna component list reranker`). A single
`lasagna benchmark run beir/scifact --reranker my_reranker` runs one arm, which
compares nothing by itself.

Or in Python:

```python
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

config = ExperimentConfig(
    name="does my reranker help",
    benchmarks=[BenchmarkSpec(name="beir/scifact", n_samples=200)],
    tracks=["retrieval"],
    pipeline=PipelineSweepSpec(base_profile="speed", rerankers=["cross_encoder", "my_reranker"]),
    n_runs=1,
)
results = await Experiment(config).run()
```

## Seven traps

These produce numbers that look fine and are not.

**1. Wrong mode.** `generation_only` never constructs the pipeline; it scores
gold context with an LLM judge. A component override there is a hard error
rather than a plausible number. Leave `mode: auto`: it picks live when the
benchmark ships a corpus, and logs which it chose and why.

**2. Repeating what does not vary, or not repeating what does.** Retrieval with
fixed models returns the same lists on every run, and significance tests pair
by query, so a retrieval-only sweep needs `n_runs: 1` (the sweep says so when
every run of every arm was identical). Repeat runs when an LLM stage is in the
loop (answer generation, an LLM reranker, HyDE, an agentic pipeline):
`n_runs: 3` or more, with `variance_source: run` so the spread is run to run
rather than a resampled query subset. A model that takes no temperature
(Claude 4.7 and later) varies whatever you ask; `lasagna doctor` names it.

**3. Comparing incomparable runs.** Different `n_samples`, split, embedding
model or judge means the numbers are not commensurable:

```bash
lasagna results compare theirs.json yours.json
```

It reports the field-level difference rather than refusing, so you can see which
knob to turn.

**4. Reading a mean without a test.** Per-sample scores are carried through and
paired by sample identity, so significance testing is available; use it. A
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
helps by 4%, so together 7%: this does not follow. They may be fixing the same
queries. Only a 2x2 can tell you:

```yaml
pipeline:
  base_profile: speed
  factorial:
    factor_a: enable_hybrid_search
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

## Baselines a reviewer will ask for

An arm `{name: none, no_rag: true}` under `pipeline.config_overrides` answers
from the question alone (the `closed_book` generator); `generator: long_context`
reads all ten retrieved passages instead of five. Both need
`tracks: [generation]` and an LLM key. Arms that read no passages are not
scored on faithfulness or the context metrics; compare exact match, token F1
and factual correctness instead.

## Which metric to trust

- **Retrieval**: `ndcg@10`, `mrr@10`, `recall@k`. Computed against qrels; no
  LLM involved; trustworthy.
- **Generation**: RAGAS faithfulness, context precision, context recall and
  factual correctness (answer relevancy is opt-in). These are LLM-judged, so
  they move with the judge model. Pin a judge snapshot: a dated id, or a
  dateless id that is its own snapshot (`gpt-6-luna`, `claude-sonnet-5-5`),
  never a floating alias like `gpt-5.5`. The judge defaults to `gpt-6-luna`;
  `RAG_EVAL_JUDGE_MODEL` or `judge_model:` sets another.
- `context_recall` is known to be unreliable with some judges. If it reads 0.0
  across the board, suspect the judge, not the pipeline.

## Publish the result

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json                       # --strict also fails on warnings
lasagna results view results.json -o view.html          # one offline HTML page
lasagna results leaderboard runs/ --metric ndcg@10 --format latex
```

Leaderboards group by `(benchmark, comparability_key)`. Rows in one table held
the same configuration fixed; rows in different tables are not comparable and
are never ranked against each other.

## Rules

- State `n_samples`, split, and `n_runs` next to any number you report. A metric
  without them is not interpretable.
- Never tune on the split you report. Iterate on another split or on a
  synthetic set (`lasagna benchmark synthesize`), and report the test split once.
- A synthetic set written by an LLM is for development only; its questions
  favor systems like the model that wrote them.
- If a benchmark cannot run live, retrieval metrics are not being measured:
  check `lasagna benchmark list` before drawing a retrieval conclusion.
- SQuAD v2, HotpotQA, MuSiQue and RAGBench run live against a corpus pooled
  from their own gold context. That is a real retrieval task but an easier one
  (no hard negatives), and the run is stamped `corpus_provenance: pooled_gold`.
  Never compare those numbers to BEIR's. Natural Questions and TriviaQA carry
  no context and stay generation-only.
- Report tokens. `system_*` is the pipeline, `eval_*` is the judge; a total
  that folds them together makes an expensive judge look like an expensive
  pipeline. A dollar figure exists only when the sweep sets `prices:`, and is
  labeled `cost_usd_estimate`.
- A re-run replays temperature-0 LLM calls from `~/.lasagna/cache/llm.sqlite`
  and reports the hit rate; `--no-cache` turns that off.
- Check dataset terms before publishing: `lasagna benchmark licenses`.
