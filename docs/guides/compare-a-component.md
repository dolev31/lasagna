# Compare a component

A sweep file names the arms, and Lasagna runs each arm through the same
pipeline with everything else held fixed. This page covers the sweep file, the
stages you can sweep, and how to read the verdict.

## Check the setup first

```bash
lasagna benchmark run smoke --profile speed
```

`smoke` is 24 documents and 12 queries held inside the package. There is no
dataset download and no API key, and on `speed` the only model is a 92 MB
embedder. A green run means the stack is wired up. The command measures
retrieval unless asked for more. Add `--tracks retrieval,generation` once an
LLM key and the `eval` extra are set up. The scores mean "this works", never
"this is good".

`lasagna benchmark list` names every benchmark and `lasagna benchmark info`
describes one. The [benchmark catalog](../catalog/benchmarks.md) lists them
all.

## Write the sweep

```yaml
name: does my reranker help
benchmarks:
  - name: beir/scifact
    n_samples: 200
tracks: [retrieval]            # NDCG, recall and the rest; no LLM, no judge
pipeline:
  base_profile: speed          # a 92 MB embedder and no API key
  rerankers: [cross_encoder, "my_ideas:keyword_overlap"]
n_runs: 1
```

```bash
lasagna benchmark experiment sweep.yaml
```

Name `base_profile` in every sweep. Without it, each arm starts from `standard`
(a 3.2 GB download), while `lasagna benchmark run` starts from `speed`. The run
header and `--dry-run` print the profile each arm starts from.

The same sweep in Python:

```python
import asyncio
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

config = ExperimentConfig(
    name="does my reranker help",
    benchmarks=[BenchmarkSpec(name="beir/scifact", n_samples=200)],
    tracks=["retrieval"],
    pipeline=PipelineSweepSpec(
        base_profile="speed",
        rerankers=["cross_encoder", "my_ideas:keyword_overlap"],
    ),
    n_runs=1,
)
results = asyncio.run(Experiment(config).run())
for task in results.failed_tasks:
    print("failed:", task.task_key, task.error)
print(results.to_markdown("ndcg@10"))
```

Look at the failed tasks before the scores. A task that failed is listed there
and not in the table.

## Sweep any stage

Five stages have a list key. The others are named in a `config_overrides` entry.

| Stage | How to sweep it |
|---|---|
| reranker | `rerankers: [...]` |
| query enhancer | `query_enhancers: [...]` |
| chunker | `chunkers: [...]` |
| embedder | `embedders: [...]` |
| generator | `generators: [...]` |
| retriever | `config_overrides: [{name: bm25, retriever: bm25}]` |
| agentic pipeline | `config_overrides: [{name: ircot, agentic_pipeline: ircot}]` |
| document processor | does not sweep; check it on [your own files](../own-data.md) |

An agentic pipeline arm changes generation metrics only, so sweep it with the
generation track. A chunker or embedder arm gets a corpus of its own, chunked or embedded by its
component, so arms never share an index. Every chunking or embedding setting
(`chunk_size`, `chunk_unit`, `embedding_model` and so on) gets its own corpus in
the same way.

### A retriever

Put `dense`, the default vector search as a component, in as the control. Then
every arm names its first stage.

```yaml
name: which first stage
benchmarks:
  - name: beir/scifact
    n_samples: 200
tracks: [retrieval]
pipeline:
  base_profile: speed
  config_overrides:
    - {name: dense, retriever: dense}           # the control
    - {name: bm25, retriever: bm25}
    - {name: hybrid_rrf, retriever: hybrid_rrf}
n_runs: 1
```

This one is keyless: BM25 needs no model, and `speed` embeds with the 92 MB one.

### A generator

List the answer strategies under `generators:`. Every arm retrieves the same
way, so a difference comes from how the answer was written. `concise` is the
harness's own prompt, word for word, so it is the control.

```yaml
name: how the answer is written
benchmarks:
  - name: hotpotqa
    n_samples: 100
tracks: [generation]
pipeline:
  base_profile: speed
  generators: [concise, cited, chain_of_note]
n_runs: 3          # repeat runs when an LLM stage is in the loop
```

This one needs an LLM API key and the `eval` extra (see
[an LLM for the generation track](../get-started/installation.md#an-llm-for-the-generation-track)).
`cited` answers carry `[n]` markers, and the citation metrics score them. From
the shell, `lasagna benchmark run <name> --generator cited` runs one generator.

### An ensemble of rerankers

`ensemble` averages the scores of the cross-encoders an arm names in
`ensemble_reranker_models`. Weights go in `ensemble_reranker_weights` and are
equal when left out. An arm with fewer than two models, or with one that cannot
load, fails instead of scoring as one model.

```yaml
pipeline:
  base_profile: speed
  config_overrides:
    - {name: minilm, reranker: cross_encoder}       # the control
    - name: ensemble
      reranker: ensemble
      ensemble_reranker_models: [cross-encoder/ms-marco-MiniLM-L-6-v2, cross-encoder/ms-marco-TinyBERT-L-2-v2]
```

`mmr` and `combined` (the cross-encoder followed by MMR) run MMR whatever
`enable_mmr` says, because naming them is what turns it on.

## Live mode and pooled corpora

A component override is only measured in live mode. `generation_only` never
constructs the pipeline, so passing `reranker=` there is a hard error rather than
a plausible number scored from gold context. The default `mode: auto` picks the
right mode and logs which it chose and why.

SQuAD v2, HotpotQA, MuSiQue and RAGBench ship each question's supporting
passages inline and no corpus to search. In `auto` mode Lasagna pools those
passages across every sample in the run, so each query has to find its own among
all the others'. That is a real retrieval task, but an easier one: the pool holds
no hard negatives, so recall reads higher than it would against a corpus of the
same size. These runs are stamped `corpus_provenance: pooled_gold`, which is part
of the comparability key, so a pooled number never shares a leaderboard row with
a BEIR one. For the same questions against a real corpus, use `beir/hotpotqa`.

Natural Questions and TriviaQA carry no context at all and stay
`generation_only`.

## How many runs

Retrieval with fixed models is deterministic. The same queries return the same
ranked lists on every run, and the significance tests pair arms query by query,
so `n_runs: 1` gives the same answer as three runs, three times faster. A sweep
that repeats identical runs says so when it ends.

Repeat runs when an LLM stage is in the loop: an answer generator, an LLM
reranker, HyDE or an agentic pipeline. A model that takes no temperature (see
[temperature and repeatable runs](../configuration.md#temperature-and-repeatable-runs))
varies from run to run whatever you ask for. There, `n_runs: 3` or more measures
the spread, and `variance_source: run` keeps the queries fixed so the spread is
run to run rather than a resampled subset.

## When corpus documents fail to ingest

A document that cannot be chunked (an error, or more than 120 s) is left out of
the corpus, and the log names it. A partial corpus scores lower than the
benchmark would, so every task on it fails, naming the count, the first
documents and the setting that accepts it:

```yaml
ingest_failure_tolerance: 0.02   # up to 2% of the corpus may be missing (at most 0.1)
```

A run that accepts a partial corpus records `corpus_ingest_gap` in each result.
That is part of the comparability key, so the run never shares a leaderboard
group with one on the complete corpus, and the bundle carries a `PARTIAL_CORPUS`
warning.

## Read the verdict

```bash
lasagna results significance "<output>/results.json" -m ndcg@10 --all
```

The command prints each pair of arms with the difference, the effect size and
both the raw and the Holm-adjusted p-value. It ends in a box. Green names each
pair that survives the correction, which side won and by how much. Yellow says
nothing survived and points at `lasagna results power`, which tells "the arms
perform alike" apart from "too few queries to tell".

| Question | Command |
|---|---|
| Is the gap real, across every pair of arms? | `lasagna results significance results.json --all` |
| Could this run have detected the effect I care about? | `lasagna results power results.json baseline candidate` |
| Do two techniques compose, or overlap? | `lasagna results interaction results.json` (a 2x2 run) |
| Which queries did an arm lose? | `lasagna results per-query results.json --sort-by ndcg@10` |
| Are two runs comparable at all? | `lasagna results compare a.json b.json` |

[Significance testing](../concepts/significance.md) explains what the test does.
[Results, bundles and leaderboards](results.md) covers sharing the run.

## Three more checks

[Dry run and cost estimates](dry-run.md) tells you what a sweep costs before it
starts. [Baselines](baselines.md) adds the arms a reviewer asks for. The
[LLM response cache](llm-cache.md) makes a second run of a sweep spend nothing
on what it already asked.
