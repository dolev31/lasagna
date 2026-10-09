# Working with Lasagna

A guide for the thing this package exists to do: **find out whether a retrieval
component actually helps.**

Every Python block below is executed by the release gate on each run
(`e2e/phases/p3_docs.py`). If one stops working, the build fails. Blocks that
cannot run hermetically — anything needing a model download or a populated
store — are marked and say why.

**Contents**

1. [The question this answers](#1-the-question-this-answers)
2. [Install](#2-install)
3. [The stages you can swap](#3-the-stages-you-can-swap)
4. [Write a component](#4-write-a-component)
5. [Register it, and see it listed](#5-register-it-and-see-it-listed)
6. [Compare it against a baseline](#6-compare-it-against-a-baseline)
7. [Swap it into a pipeline](#7-swap-it-into-a-pipeline)
8. [Measure it on a benchmark](#8-measure-it-on-a-benchmark)
9. [Where to go next](#9-where-to-go-next)

---

## 1. The question this answers

You have an idea for a reranker — or a chunker, or a query rewriter. The
literature says the technique gives +4 NDCG. You want to know whether it helps
*your* corpus, and you want a number you would be willing to publish.

Doing that honestly is harder than it sounds, because the usual approach —
fork a RAG pipeline, patch your component in, run it — changes more than one
thing at once. Different chunking, a different embedding model, a different
top-k, and the +4 you measured belongs to something other than your idea.

Lasagna's answer is that the seven stages are isolated behind typed contracts.
You swap exactly one and everything else stays fixed, so a difference in the
result is attributable to the swap. That property is enforced by a conformance
suite, not by convention.

The rest of this guide walks that workflow end to end.

---

## 2. Install

```bash
pip install lasagna-rag
```

**First run downloads models.** The default profile pulls an embedding model
*and* a reranker: about **5.4 GB**, and between 7 and 35 minutes to your first
search result depending on your connection, as measured on every release-gate
run to date. After that, measured warm on two CPU threads, a search takes about
0.3 s on the speed profile and 7–9 s on the default profile, whose large
reranker scores every query; more cores or a GPU cut the second.

To try it faster, start on the `speed` profile: pure vector search on an 80 MB
model, no reranker.

```bash
lasagna configure --profile speed
```

Nothing in sections 3 to 7 needs a model or a network — components are plain
Python objects, and you can exercise them directly. Only section 8 needs a
corpus.

---

## 3. The stages you can swap

```python
from lasagna.core.registry import registry

# list_all() maps each stage to the names registered for it.
for stage, names in sorted(registry.list_all().items()):
    print(f"{stage}: {len(names)} registered -> {', '.join(sorted(names)[:3])}")
```

Each stage has a typed abstract base class. Implementing it is the whole
contract — there is no plugin manifest and no registration ceremony beyond one
decorator.

| Stage | Base class | Swap it to change |
|---|---|---|
| Chunker | `BaseChunker` | how documents are split |
| Embedder | `BaseEmbedder` | what vectors are stored |
| Query enhancer | `BaseQueryEnhancer` | how the query is rewritten before retrieval |
| Retriever | `BaseRetriever` | how candidates are fetched |
| Reranker | `BaseReranker` | the ordering of retrieved candidates |
| Document processor | `BaseDocumentProcessor` | how files are parsed |
| Agentic pipeline | `BaseAgenticPipeline` | the multi-step retrieval loop |

---

## 4. Write a component

A reranker takes a query and a list of candidates, and returns them reordered.
That is the entire interface:

```python
from lasagna.core.base_components import BaseReranker, ScoredDocument


class LengthReranker(BaseReranker):
    """Ranks by word count. Deliberately unlike retrieval-score order.

    A real component would score query-document relevance. This one is
    obviously wrong on purpose, so that when the ordering changes below you
    know the change came from the swap and not from chance.
    """

    def rerank(self, query, documents, top_k=5, **kwargs):
        import copy

        # Copy first. A stage must not mutate what it was handed -- the
        # caller may still be holding those objects, and a component that
        # edits them in place makes results depend on execution order.
        out = [copy.deepcopy(d) for d in documents]
        for d in out:
            d.score = float(len(d.content.split()))
        out.sort(key=lambda d: d.score, reverse=True)
        return out[:top_k]


print("defined:", LengthReranker.__name__)
```

Two things that matter more than they look:

**Deep-copy before you touch anything.** The pipeline may hand the same objects
to more than one component. A stage that mutates its input makes the result
depend on the order stages ran in, which is exactly the kind of hidden coupling
this design exists to prevent.

**Return at most `top_k`.** The caller decides how many results it wants.

---

## 5. Register it, and see it listed

Registration is one decorator. The `paper` and `claims` arguments are optional
but worth filling in: they are carried through to the listing API, so a results
table can show a claimed number next to the one you reproduced.

```python
from lasagna.core.base_components import BaseReranker
from lasagna.core.registry import registry


@registry.register(
    "reranker",
    "length_demo",
    paper="Nobody et al. 2026",
    claims={"ndcg@10": 0.42},
)
class LengthDemoReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        import copy

        out = [copy.deepcopy(d) for d in documents]
        out.sort(key=lambda d: len(d.content.split()), reverse=True)
        return out[:top_k]


entry = next(e for e in registry.list_rerankers() if e["name"] == "length_demo")
print("implements_contract:", entry["implements_contract"])
print("paper:", entry["paper"])
print("claims:", entry["claims"])
```

`implements_contract` is the field to look at. It is `True` when the class
genuinely satisfies the abstract base class — not merely that it was
registered. A component that registers but does not conform is listed with
`implements_contract: False`, which is how you find out before your results
depend on it.

---

## 6. Compare it against a baseline

This is the part that makes a number defensible. Run the same input through two
components and compare the outputs — nothing else varies, so any difference is
the component.

```python
import copy

from lasagna.core.base_components import BaseReranker, ScoredDocument


class ByLength(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        out = [copy.deepcopy(d) for d in documents]
        out.sort(key=lambda d: len(d.content.split()), reverse=True)
        return out[:top_k]


class Identity(BaseReranker):
    """Returns candidates untouched, in retrieval order.

    The control. Without it you cannot tell "my component reordered things"
    from "the pipeline never called my component at all" -- the two look
    identical from the outside.
    """

    def rerank(self, query, documents, top_k=5, **kwargs):
        return [copy.deepcopy(d) for d in documents][:top_k]


candidates = [
    ScoredDocument(content="short one", score=0.9, document_id="d1", chunk_id="c1"),
    ScoredDocument(content="a rather longer passage with more words in it", score=0.5,
                   document_id="d2", chunk_id="c2"),
    ScoredDocument(content="middling length text here", score=0.7,
                   document_id="d3", chunk_id="c3"),
]

baseline = [d.document_id for d in Identity().rerank("q", candidates, top_k=3)]
mine = [d.document_id for d in ByLength().rerank("q", candidates, top_k=3)]

print("baseline order:", baseline)
print("mine:          ", mine)
print("changed:       ", baseline != mine)

# The inputs were not mutated -- verify, do not assume.
assert [d.score for d in candidates] == [0.9, 0.5, 0.7]
print("inputs intact:  True")
```

Always run the control. `changed: False` has two very different causes — your
component agreed with retrieval order, or it never ran — and only the control
tells them apart.

### Picking a baseline that means something

Comparing against *any* baseline is easy; comparing against a fair one is the
work. `implements_contract` tells you which registered components are
interchangeable with yours, which is the set worth comparing against:

```python
from lasagna.core.registry import registry

conforming = [e["name"] for e in registry.list_rerankers() if e["implements_contract"]]
print(f"{len(conforming)} interchangeable rerankers available as baselines")
```

---

## 7. Swap it into a pipeline

Outside a test you want the component running in a real pipeline. `replace()`
swaps one stage and leaves the rest alone.

```python
import copy

from lasagna.core.base_components import BaseReranker
from lasagna.core.pipeline import KnowledgePipeline
from lasagna.core.registry import registry


@registry.register("reranker", "guide_swap_demo")
class SwapDemo(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        return [copy.deepcopy(d) for d in documents][:top_k]


pipeline = KnowledgePipeline(reranker="cross_encoder")
print("before:", type(pipeline.get_component("reranker")).__name__)

pipeline.replace("reranker", "guide_swap_demo")
print("after: ", type(pipeline.get_component("reranker")).__name__)
```

Check `get_component()` after replacing, as above. It confirms the swap reached
execution rather than only updating a name — a pipeline that kept serving a
cached instance would leave you measuring the old component while believing you
had changed it.

---

## 8. Measure it on a benchmark

Sections 4 to 7 prove your component behaves. This section is about whether it
*helps*, which needs a corpus and real metrics.

### Check the setup first

```bash
lasagna benchmark run smoke --profile speed --tracks retrieval
```

`smoke` is 24 documents and 12 queries held inside the package — no dataset
download and no API key; on `speed` the only model is an 80 MB embedder. It runs
the real retrieval pipeline end to end, so a green run means the stack is wired
up. Drop `--tracks retrieval` once an LLM and the `[eval]` extra are set up, to
check answer generation too. It is deliberately easy, so the scores mean "this works", never "this
is good".

### Then the real thing

```python
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

config = ExperimentConfig(
    name="does my reranker help",
    benchmarks=[BenchmarkSpec(name="beir/scifact", n_samples=200)],
    # Sweep across rerankers with everything else fixed. The comparison is
    # only meaningful because the other six stages do not move.
    pipeline=PipelineSweepSpec(rerankers=["cross_encoder", "length_demo"]),
    # `mode` defaults to "auto": live when the benchmark ships a corpus, so
    # retrieval is actually measured. Say mode="generation_only" only when you
    # want answer quality scored against gold context, with no retrieval.
    n_runs=3,
)
print(config.mode, "->", len(config.benchmarks), "benchmark(s)")
```

As written, this is not the small path. It runs the `standard` profile on both
tracks, so it needs an LLM API key (to write answers and to judge them), the
`[eval]` extra, and `BAAI/bge-m3` plus a large reranker (about 5.4 GB on first
use). For a keyless run on small models, pass `tracks=["retrieval"]` to
`ExperimentConfig` and `base_profile="speed"` to `PipelineSweepSpec`. Run it,
and look at the failed tasks before the scores: a task that failed is listed
there, not in the table.

```python
import asyncio

results = asyncio.run(Experiment(config).run())
for task in results.failed_tasks:
    print("failed:", task.task_key, task.error)
print(results.to_markdown("ndcg@10"))
```

**A component override is only measured in live mode.** `generation_only` never
constructs the pipeline, so passing `reranker=` there is now a hard error rather
than a plausible number scored from gold context. `auto` picks the right mode
and logs which it chose and why.

**Benchmarks with no corpus can still measure retrieval — against a pooled
corpus, which is easier.** SQuAD v2, HotpotQA, MuSiQue and RAGBench ship each
question's supporting passages inline and no corpus to search. `auto` assembles
one by pooling those passages across every sample in the run, so each query has
to find its own among all the others'. This is a real retrieval task, but a
gentler one: HotpotQA and MuSiQue drop their distractor paragraphs at
conversion, so the pool holds no hard negatives and recall reads higher than it
would against a corpus of the same size. Runs are stamped
`corpus_provenance: pooled_gold`, which is part of the comparability key — a
pooled number can never share a leaderboard row with a BEIR one. For the same
questions against a real corpus, use `beir/hotpotqa`.

Natural Questions and TriviaQA carry no context at all and stay
`generation_only`. There is nothing to pool.

`n_runs=3` is not decoration. Retrieval metrics move between runs; a single run
tells you nothing about whether a gap is real. The evaluation SDK carries
per-sample scores through so significance testing is possible — and pairs them
by sample identity, so the comparison is between the same queries on both sides.

For what the metrics mean and which are trustworthy, see
[metrics](../src/lasagna/knowledge_evaluation_sdk/docs/metrics.md); for the
available datasets, see
[benchmarks](../src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md).

---

## 9. Where to go next

| I want to… | Read |
|---|---|
| **Publish my component as a package** | [Plugin template](../examples/lasagna-plugin-template/) |
| **Check my component satisfies its contract** | `lasagna component validate <module:Class>` |
| **See which plugins are installed (and which failed)** | `lasagna plugins list` |
| **Publish a result someone else can check** | `lasagna results bundle` then `validate` |
| **Ask whether two techniques compose or overlap** | `lasagna results interaction` |
| **See which queries a config lost, and at which stage** | `lasagna results per-query` |
| **Find out whether a null result was inconclusive** | `lasagna results power` |
| **Run a leaderboard for those results** | [Results hub](../hub/README.md) |
| **Know a dataset's terms before publishing** | [Dataset licenses](../src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md) |
| Use the Python SDK in an application | [SDK user guide](../src/lasagna/sdk/docs/user_guide.md) |
| Add a component type, not just an implementation | [Extending](../src/lasagna/sdk/docs/extending.md) |
| Configure retrieval behaviour | [Configuration](../src/lasagna/sdk/docs/configuration.md) |
| Call it over HTTP | [REST endpoints](../src/lasagna/api/docs/endpoints.md) |
| Run a different vector database | [Vector backends](../src/lasagna/vector_backends/README.md) |
| Understand the retrieval internals | [Architecture](../src/lasagna/core/docs/architecture.md) |
| Deploy it | [Deployment](../src/lasagna/deploy/README.md) |
| Know what is *not* verified | [Honest limitations](../README.md#honest-limitations) |

### Two honest warnings

**Check the model licence before publishing.** Several strong rerankers are
non-commercial. `lasagna.core.model_licenses` warns once at load, and
[model-licenses](model-licenses.md) lists the terms per model.

**A component that conforms is not a component that helps.** Everything up to
section 7 proves your code is wired in correctly. Only section 8 tells you
whether it improves retrieval, and only with enough runs to distinguish a real
gap from noise.
