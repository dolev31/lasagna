<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/dolev31/lasagna/raw/main/docs/assets/hero-dark.png">
  <img src="https://github.com/dolev31/lasagna/raw/main/docs/assets/hero-light.png" width="100%" alt="Lasagna: swap one RAG stage, hold the rest fixed, and measure what actually changed.">
</picture>

**Test your RAG components against a real stack — not a toy baseline.**

[![Project page](https://img.shields.io/badge/project-page-1f6feb.svg)](https://dolev31.github.io/lasagna/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dolev31/lasagna/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-36%25-yellow.svg)](#testing)

[Guide](https://github.com/dolev31/lasagna/blob/main/docs/guide.md) · [Quickstart](#quickstart) · [Why](#the-problem) · [Compare components](#compare-a-component) · [What's inside](#whats-inside) · [Docs](#documentation)

</div>

> **For researchers** publishing a component and needing a defensible number.
> **For engineers** choosing between six vector backends and a dozen retrieval
> strategies without building the harness first.

---

## The paper

Lasagna is the framework behind **[Retrieval Augmented Generation Compositionality](https://dolev31.github.io/lasagna/)**
(Ido Levy, Segev Shlomov, Leshem Choshen, 2026). The study tests 450 configurations across eight
benchmarks and corpora from 5K to 200K documents, and asks whether gains measured one component at
a time survive when the components are combined.

<p align="center">
  <img src="https://github.com/dolev31/lasagna/raw/main/docs/assets/fig-overview.png" width="85%" alt="LASAGNA overview: (a) seven functional blocks, (b) eight dimensions in a 167-configuration D-optimal design, (c) faithfulness versus latency for four operational profiles.">
</p>

| What the study found | Predicted | Realized |
|---|---|---|
| Retrieval gains do not add up: the three strongest single techniques, combined (NDCG@10) | +47% | +17% |
| Faithfulness gains do carry over: one factor at a time versus the best full pipeline | +28% | +32% |

**The best pipeline also changes with corpus size.** At 200K documents, the design that leads at
small corpora trails by 13.7% NDCG@10, with the rank flip visible from 25K.

[Project page](https://dolev31.github.io/lasagna/) · [Explainer video](https://dolev31.github.io/lasagna/#explainer) · [Pre-registration](https://github.com/dolev31/Lasagna-Retrieval-Augmented-Generation-Compositionality)
<!-- After the arXiv listing: add the paper link to this row, and the arXiv id to the BibTeX below and to CITATION.cff. -->

---

## The problem

Most RAG results are reported one component at a time. A new reranker is
measured against a baseline with no query augmentation, no agentic loop, and a
chunker from two years ago — because building the rest of the stack just to run
the comparison is more work than the contribution itself.

That makes published gains hard to add up. If a reranker claims +4 NDCG and a
query expander claims +3, you cannot tell from the papers whether stacking them
gives you **+7, +4, or −1**.

Lasagna exists to make that question answerable.

```
                    ┌─────────── your component drops in here ───────────┐
                    ▼                                                     ▼
 ingest → chunk → embed → retrieve → rerank → augment → agentic loop → generate
                    ▲                                                     ▲
                    └──── everything else stays fixed, so the delta ──────┘
                              is attributable to what you changed
```

**All seven stages take effect.** Five at query time — `reranker`,
`query_enhancer`, `embedder`, `retriever`, `agentic_pipeline` — and two at
ingest: `chunker` and `document_processor`. Swap one with a single config key
and the rest hold still, so the difference you measure belongs to the thing you
changed.

A swapped stage *replaces* the built-in rather than layering on top of it. A
retriever owns how candidates are found; a chunker owns how text is split.
Applying the built-in's settings on top of someone else's algorithm would mean
the component being measured is not the component that ran.

---

## How Lasagna compares to LlamaIndex, LangChain and RAGAS

Short version: **it is not an alternative to most of them.** Lasagna is built on
LangChain and LangGraph — they are core dependencies — and it uses RAGAS for
generation metrics. It is a layer for measuring, not a replacement for building.

| | **Lasagna** | **LlamaIndex** | **LangChain** |
|---|---|---|---|
| The job it is for | Knowing *which* change moved the number | Turning messy documents into a usable index | Providers, agents, shipping to production |
| Reach for it when | You must defend a result in a paper or review | Your hard problem is PDFs, OCR, tables | Your hard problem is integrations and agents |
| Don't, when | You need breadth, stability or a small install | You need statistical rigour about *why* | You want an opinionated, attributable experiment |

**Typed swap seams are table stakes, not our differentiator.** LlamaIndex has
`BaseNodePostprocessor`, LangChain has `BaseDocumentCompressor` and `Embeddings`;
both are real ABCs enforced at instantiation. Anyone claiming otherwise has not
read their source. Our conformance is in one respect *weaker*: registration
warns rather than raises, deliberately, so duck-typed components still work
(`core/registry.py:92`) — and some of our own built-ins do not yet implement
their stage's ABC. The count is in [Honest limitations](#honest-limitations).

**What is genuinely different is the measurement apparatus.** Paired bootstrap
with confidence intervals, Holm–Bonferroni correction across a sweep, per-sample
scores joined by sample identity, comparability keys that record what was held
fixed, and tamper-evident result bundles. None of LangChain, LlamaIndex, Haystack
or RAGAS ships significance testing; RAGAS scores a run, and we depend on it to
do so. If your question is "is this gap real or is it noise", that gap in the
ecosystem is the one Lasagna exists to fill.

**Where the others plainly win.** Ecosystem — LangChain and LlamaIndex have
thousands of integrations and years of production hardening; Lasagna is a
pre-release with none of either. Install weight — ours is 1.5 GB because torch and sentence-transformers
are core dependencies, which is a real cost for an API-embeddings user and both
of them are far lighter. Document parsing — LlamaParse handles formats our
pypdf/pdfplumber stack does not. Production observability — LangSmith exists;
we have nothing comparable. Agent orchestration — LangGraph is the standard, and
we build on it.

---

## Quickstart

```bash
pip install lasagna-rag
```

```python
import asyncio
from lasagna.sdk import KnowledgeClient

async def main():
    client = KnowledgeClient()
    await client.upload("paper.pdf")

    results = await client.search_documents("What did the ablation show?")
    for r in results.results:
        print(f"[{r.score:.2f}] {r.chunk.content[:120]}")

    await client.close()

asyncio.run(main())
```

Or from the shell:

```bash
lasagna init                       # interactive setup — pick the speed profile
lasagna upload paper.pdf
lasagna search "what did the ablation show?"
lasagna serve                      # REST API + configuration UI
```

**Nothing to download at all?** The [component comparison](#compare-a-component)
below runs as-is — no model, no corpus, no key. It is the fastest way to see
what this library is actually for, and it is where we suggest you start.

> **First run is a real download, and you should know before you start.** The
> default profile fetches an embedding model *and* a reranker on first use.
> Measured on a cold cache on every release-gate run to date:
> **5.4 GB downloaded, and 7–35 minutes to the first search result** — the
> download size was identical every time, so the spread is your connection, not
> the software. Queries after that are sub-second. The install itself is 1.5 GB.
>
> To get the 80 MB model instead, run `lasagna init` and choose the speed
> profile. It writes a settings file and points `RAG_SETTINGS_FILE` at it, which
> is what the config loader actually reads.
>
> Or from Python, which now works for the embedding model too:
>
> ```python
> from lasagna.sdk import KnowledgeClient, KnowledgeConfig
>
> client = KnowledgeClient(config=KnowledgeConfig(
>     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
>     embedding_dim=384,
> ))
> ```
>
> The embedding manager is a process-wide singleton, so this is applied before
> the backend is constructed and is process-wide in effect — two clients in one
> process cannot hold different embedding models.
>
> These numbers come from the release gate, not from an estimate. Run
> `python -m e2e.run` to measure on your own machine; CI keeps each run's
> report as a build artifact.

Note that `upload()` returns as soon as the document is queued — processing is
asynchronous, so `result.chunk_count` is `0` at that moment. Await
`client.documents.wait_for_processing(result.document_id)` if you need the
final count.

---

## Compare a component

This is what Lasagna is for. Here is the whole loop — write a component,
register it with the provenance of the paper it came from, and run it against a
control. It executes as-is, with no model download and no corpus:

```python
import copy

from lasagna.core.base_components import BaseReranker, ScoredDocument
from lasagna.core.registry import registry


@registry.register(
    "reranker",
    "readme_demo",
    paper="arxiv:2XXX.XXXXX",     # optional provenance
    claims={"ndcg@10": 0.456},    # what the paper reported
)
class MyReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        out = [copy.deepcopy(d) for d in documents]   # never mutate the input
        out.sort(key=lambda d: len(d.content.split()), reverse=True)
        return out[:top_k]


class Control(BaseReranker):
    """Retrieval order, untouched. Without it you cannot tell "my component
    reordered things" from "my component never ran"."""

    def rerank(self, query, documents, top_k=5, **kwargs):
        return [copy.deepcopy(d) for d in documents][:top_k]


candidates = [
    ScoredDocument(content="short", score=0.9, document_id="d1", chunk_id="c1"),
    ScoredDocument(content="a considerably longer passage of text", score=0.5,
                   document_id="d2", chunk_id="c2"),
    ScoredDocument(content="middling length here", score=0.7,
                   document_id="d3", chunk_id="c3"),
]

control = [d.document_id for d in Control().rerank("q", candidates)]
mine = [d.document_id for d in MyReranker().rerank("q", candidates)]

entry = next(e for e in registry.list_rerankers() if e["name"] == "readme_demo")
print("control order:      ", control)
print("my reranker:        ", mine)
print("attributable change:", control != mine)
print("claims recorded:    ", entry["claims"])
print("interchangeable:    ", entry["implements_contract"])
```

```text
control order:       ['d1', 'd2', 'd3']
my reranker:         ['d2', 'd3', 'd1']
attributable change: True
claims recorded:     {'ndcg@10': 0.456}
interchangeable:     True
```

`paper` and `claims` travel with the component and come back out of the
registry, so a results table can put the claimed number beside the one you
reproduced. `implements_contract` tells you the component is genuinely
interchangeable with others — not merely that it registered.

Once it behaves, [measure whether it *helps*](https://github.com/dolev31/lasagna/blob/main/docs/guide.md#8-measure-it-on-a-benchmark)
on 31 benchmarks.

### Picking a baseline

`implements_contract` is the field that matters. A comparison is only meaningful
between components that exchange the same types — otherwise you are measuring
the interface difference along with the algorithm.

Some built-ins predate the typed contracts and take or return different shapes.
They work fine in the pipeline they were written for, but they are **not**
interchangeable with a component written against `BaseReranker`. Selecting one
says so:

```python
pipeline.replace("reranker", "cross_encoder")
# WARNING  reranker 'cross_encoder' does not implement the reranker contract,
#          so it is not interchangeable with components that do.
#          Contract-compliant rerankers: ['combined_adapted',
#          'cross_encoder_adapted', 'ensemble_adapted', 'mmr_adapted']
```

Filter to comparable baselines in one line:

```python
[r["name"] for r in registry.list_rerankers() if r["implements_contract"]]
```

### Why the stages stay isolated

Each stage exchanges only its declared types with the next, so toggling one
component cannot leak into another's execution and contaminate the measurement.
That is not a stylistic preference — it is what makes an interaction estimate
attributable rather than an artefact.

It is enforced, not asserted. A conformance suite checks that components
subclass their stage's contract, do not mutate the objects they are handed, and
do not hand out chunks that share one metadata dict:

```bash
pytest tests/test_abc_conformance.py tests/test_stage_isolation_behavioural.py
```

### Do two techniques compose, or do they overlap?

The question two papers cannot answer between them. HyDE is published against a
baseline. Reranking is published against a baseline. Whether HyDE still earns
its latency once reranking is on does not follow from those two numbers — both
may be fixing the same queries, in which case stacking them buys far less than
the sum of their headline gains.

Answering it needs all four cells measured on the same queries. Declare the 2×2
and the harness builds them:

```yaml
pipeline:
  base_profile: standard
  factorial:
    factor_a: enable_hyde
    factor_b: enable_reranking
```

```bash
lasagna results interaction results.json --metric ndcg@10
```

```
  expected if additive : +0.1000
  actually observed    : +0.0600
  interaction          : -0.0400  95% CI [-0.0490, -0.0311]  p=0.0001
  composition efficiency: 60.0% (of the additive prediction actually delivered)

  redundant (they overlap; the second buys less than it does alone)
```

The statistic is `(AB − neither) − (A − neither) − (B − neither)`, bootstrapped
over queries with all four cells of a query resampled together. Positive means
they compound; negative means they overlap.

Factors are boolean flags, not component names — a component slot has no *off*
level for the `neither` cell to use, and swapping a component does not enable
the feature. `lasagna` refuses a factorial over anything that is not a real
boolean setting, before the first corpus is ingested.

---

## What's inside

### Pluggable RAG pipeline stages

| Stage | Registered | Notes |
|---|---:|---|
| `reranker` | 9 | cross-encoder, ColBERT, MMR, LLM listwise, ensemble |
| `query_enhancer` | 5 | HyDE, expansion, decomposition, combined |
| `chunker` | 3 | semantic, hierarchical, structure-aware |
| `agentic_pipeline` | 2 | default graph, legacy ReAct |
| `embedder` | 1 | pluggable via the same interface |
| `document_processor` | 1 | pluggable via the same interface |
| `retriever` | 0 | interface defined; bring your own |

### Retrieval: hybrid search, reranking, MMR

Dense · BM25 · SPLADE learned-sparse · hybrid with RRF · HyDE · query expansion
and decomposition · RAG Fusion · MMR · adaptive-k · cross-encoder and ColBERT
reranking · lost-in-the-middle reordering · corrective RAG · self-RAG quality
gates.

### Agentic RAG: CRAG, self-RAG, quality gates

A LangGraph pipeline with document grading, query rewriting, multi-hop
sub-query planning, hallucination checks, and quality gates — swappable as a
unit or node by node.

### Vector databases: six backends, one interface

Milvus · pgvector · Qdrant · Weaviate · Pinecone · OpenSearch, behind one
interface with a shared conformance suite asserting the same behaviour against
each. Milvus and Qdrant run in-process and are verified on every commit; the
rest need their service running
([details](https://github.com/dolev31/lasagna/blob/main/tests/test_vector_backend_conformance.py)).

### RAG evaluation: 31 benchmarks, retrieval and generation metrics

31 benchmarks — BEIR (NFCorpus, FiQA, SciFact, ArguAna, HotpotQA), HotpotQA,
SQuAD v2, MuSiQue, TriviaQA, Natural Questions, MS MARCO, FEVER, PopQA,
Bamboogle, MultiHop-RAG, RAGBench, KILT, 2WikiMultihop.

| Family | Metrics |
|---|---|
| Retrieval | NDCG@k, precision@k, F1@k, hit-rate@k, MAP, MRR |
| Generation | faithfulness, answer relevancy, context precision, context recall, factual correctness (RAGAS) |
| Operational | p50/p95 latency, cost per query |

Plus multi-objective tradeoff analysis, significance testing, and bootstrap
confidence intervals.

### Interfaces

Python SDK · REST API · MCP server · CLI · schema-driven configuration UI.

### Operating profiles

Four built-in profiles, from `speed` (pure vector search) through `standard`
and `balanced` to `max_quality` (everything on). Each is genuinely distinct and
does what it documents — `speed` really does disable reranking, hybrid, MMR,
HyDE and the agentic loop.

---

## MCP server: give an AI agent RAG over your documents

Lasagna ships an MCP server with **31 tools**, so any MCP client — Claude Code,
Claude Desktop, Cursor — can search your documents as a tool call.

```bash
pip install lasagna-rag
lasagna configure --profile speed          # 80 MB model instead of 5.4 GB
for f in ./docs/*.md; do lasagna upload "$f" -s my-corpus; done
```

Then register the server. **Claude Code** — `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "lasagna": {
      "command": "lasagna-mcp",
      "args": ["--transport", "stdio"],
      "env": { "RAG_SESSION_ID": "my-corpus" }
    }
  }
}
```

**Claude Desktop** — the same block inside `claude_desktop_config.json`. If
`lasagna-mcp` is not on the inherited `PATH`, use the absolute path to it inside
your virtualenv.

The `speed` profile above matters for more than download size: the default
profile enables agentic RAG, which calls an LLM, so a keyless `search` fails.
On `speed` retrieval is fully keyless. Generation, LLM reranking and benchmark
judging always need a key.

### Skills, so the agent stops guessing

```bash
lasagna skills install        # writes .claude/skills/
```

Four skills — using retrieval, building a component, running an evaluation, and
filing a useful bug report. They encode the things an agent otherwise gets wrong:
which mode a benchmark must run in, why the first search is slow, and which
config actually applies.

```bash
lasagna doctor                # what is wrong with this install, and how to fix it
lasagna doctor --issue        # a redacted block to paste into a bug report
```

---

## Install

```bash
pip install lasagna-rag                    # core: ingest, embed, search, serve
```

| Extra | Adds |
|---|---|
| `eval` | RAGAS metrics, benchmark loaders, statistics, plots |
| `ingest` | OCR and semantic chunking |
| `docling` | structure-aware document conversion |
| `vector-backends` | Pinecone, Qdrant, Weaviate clients |
| `opensearch` / `pgvector` | those backends |
| `flag-embedding` | BGE-M3 dense+sparse+ColBERT |
| `graphrag` | entity-linked retrieval |
| `storage-all` | S3, GCS, Azure document storage |
| `local` | local LLM inference (HF, Ollama) |
| `all` | everything above |

> **`pip` cannot resolve `eval` together with `storage-all`**, and so cannot
> resolve `all` either. The requirement set is satisfiable — pip's resolver just
> fails to find the solution. Use `uv`, or hand pip the versions it misses:
>
> ```bash
> uv pip install "lasagna-rag[all]"
> pip install "lasagna-rag[eval,storage-all]" s3fs==2026.6.0 fsspec==2026.6.0
> ```
>
> `eval` and `storage-all` each resolve fine on their own — only the
> combination defeats pip. See CHANGELOG "Known issues".

`import lasagna` deliberately loads no torch, transformers, sentence-transformers
or pymilvus — those arrive only when something needs them, so a remote-backend
client stays light.

---

## Model licenses

Lasagna ships **no model weights**. They download from their upstream host on
first use, each under its own license.

Three selectable models are **non-commercial**, including
`naver/splade-cocondenser-ensembledistil`, the default SPLADE sparse encoder.
They are offered deliberately — comparing well-known models is the point of a
research framework — and selecting one logs a warning naming the license.

For research and benchmarking this is fine. Before taking a configuration into a
commercial product, read **[docs/model-licenses.md](https://github.com/dolev31/lasagna/blob/main/docs/model-licenses.md)**.

```python
from lasagna.core.model_licenses import is_commercial_use_restricted
is_commercial_use_restricted("naver/splade-cocondenser-ensembledistil")  # True
```

Benchmark corpora are likewise never redistributed — they are fetched from their
official sources and carry their own terms.

---

## Testing

```bash
pytest tests/ --ignore=tests/live --ignore=tests/stress
```

Hermetic by default: no network, no external services, and the suite never
touches your real `~/.rag_knowledge` store. `tests/live/` needs a running stack;
`tests/stress/` is long-running.

Branch coverage is measured and floored in CI — branches rather than statements,
because this package's bugs have lived overwhelmingly in error paths.

---

## Documentation

**Start here: [Working with Lasagna](https://github.com/dolev31/lasagna/blob/main/docs/guide.md)** — install, write a
component, compare it against a baseline, and measure it on a benchmark. Every
example in it is executed by the release gate on each run.

**Task guides**

| I want to… | Read |
|---|---|
| Use the Python SDK in an application | [SDK user guide](https://github.com/dolev31/lasagna/blob/main/src/lasagna/sdk/docs/user_guide.md) |
| **Contribute a component** | [Contributing components](https://github.com/dolev31/lasagna/blob/main/docs/contributing-components.md) |
| Add a whole component type | [Extending](https://github.com/dolev31/lasagna/blob/main/src/lasagna/sdk/docs/extending.md) |
| Tune retrieval behaviour | [Configuration](https://github.com/dolev31/lasagna/blob/main/src/lasagna/sdk/docs/configuration.md) |
| Call it over HTTP | [REST endpoints](https://github.com/dolev31/lasagna/blob/main/src/lasagna/api/docs/endpoints.md) |
| Run a different vector database | [Vector backends](https://github.com/dolev31/lasagna/blob/main/src/lasagna/vector_backends/README.md) |
| Understand the retrieval internals | [Architecture](https://github.com/dolev31/lasagna/blob/main/src/lasagna/core/docs/architecture.md) |
| Evaluate on benchmarks | [Benchmarks](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md) · [Metrics](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/docs/metrics.md) |
| Deploy the stack | [Deployment](https://github.com/dolev31/lasagna/blob/main/src/lasagna/deploy/README.md) |
| Give an agent retrieval | [MCP server](https://github.com/dolev31/lasagna/blob/main/src/lasagna/mcp/README.md) |

**Project**

| | |
|---|---|
| [Model licenses](https://github.com/dolev31/lasagna/blob/main/docs/model-licenses.md) | per-model terms and commercial-use status |
| [Contributing](https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md) | adding a component, the conformance contract |
| [Security](https://github.com/dolev31/lasagna/blob/main/SECURITY.md) | reporting a vulnerability, operational notes |
| [Changelog](https://github.com/dolev31/lasagna/blob/main/CHANGELOG.md) | releases and known issues |
| [Code of conduct](https://github.com/dolev31/lasagna/blob/main/CODE_OF_CONDUCT.md) | how we work together |
| [Citation](https://github.com/dolev31/lasagna/blob/main/CITATION.cff) | how to cite Lasagna |
| [Dataset licenses](https://github.com/dolev31/lasagna/blob/main/src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md) | per-benchmark terms; generated, never hand-edited |

Reference material for each subsystem sits in that subsystem's directory, next
to the code it describes. Only the guide, this README and `CONTRIBUTING.md` have
their examples executed by the gate — treat the rest as reference that may
drift, and report anything that does not work.

---

## Honest limitations

A framework that hides its gaps wastes your time. As of this release:

- **Three of six vector backends** (pgvector, Weaviate, OpenSearch) have a
  conformance suite but need their service running to execute it. Pinecone is
  excluded from conformance entirely — it is SaaS-only.
- **Nine of eighteen *listable* registered components** do not implement their
  stage's ABC. They predate the typed contracts, are flagged by
  `implements_contract` and warned about on selection, but re-typing them is
  outstanding. The denominator is worth stating precisely: 21 components are
  registered across seven stages, and the conformance suite enumerates four of
  them — `retriever`, `document_processor` and `agentic_pipeline` have no
  conformance signal at all. **`retriever` currently has zero registered
  components**, so the swap seam exists and ships nothing to swap in.
- **`list_document_ids` fails past ~16k chunks on Milvus** — the backend
  emulates offset as `limit + offset`, which exceeds Milvus's cap. Affects
  iteration speed, not correctness. See [CHANGELOG](https://github.com/dolev31/lasagna/blob/main/CHANGELOG.md).
- **Coverage is 34%.** That is coverage.py's combined statement+branch TOTAL,
  which is what `fail_under` gates on; branch coverage alone is lower.
  Measured, floored, and ratcheting up — but not yet
  where it should be. The evaluation SDK and MCP server are the thinnest.
- **Pooled-gold corpora are an easier task than the benchmarks they come from.**
  SQuAD v2, HotpotQA, MuSiQue and RAGBench ship no corpus, so one is assembled
  from the samples' own gold context. HotpotQA and MuSiQue drop their distractor
  paragraphs at conversion, so the pool has no hard negatives and recall reads
  higher than against a real corpus. Runs are stamped
  `corpus_provenance: pooled_gold`, which is in the comparability key — but the
  number itself is still easier than BEIR's. Natural Questions and TriviaQA
  cannot be pooled at all and stay generation-only.
- **Large BEIR corpora are capped unless you raise the limit.** `max_corpus_docs`
  defaults to 0, which now genuinely means unlimited — but 8 of the 13 BEIR
  corpora exceed 100,000 documents, and the largest is 8.8M, which is hours of
  embedding and tens of GB of index. If you set a cap instead, a run that leaves
  documents behind is stamped `corpus_truncated` and `lasagna results validate`
  blocks the bundle: a metric computed against part of a corpus is not that
  benchmark's metric, and it simply comes out lower with nothing else to show
  for it. Where more than half the judged documents fall outside the cap the run
  aborts rather than reporting a number.
- **No dataset revisions are pinned.** `pin_dataset_revision()` and
  `LASAGNA_DATASET_REVISIONS` exist and are recorded on every result, but the
  catalogue ships no SHAs — writing ones not resolved against the Hub would be
  a fabricated provenance record. Until you pin, `UNPINNED_DATASET` is warned
  and `--strict` will refuse to publish.
- **Token counts, not dollars.** Every LLM call is counted at the provider and
  split system-vs-judge. There is no price table, deliberately: a rate frozen
  into a published bundle is indistinguishable from a measurement to whoever
  reads it next year.

---

## Citing Lasagna

If you publish work that uses Lasagna, please cite it — see
[CITATION.cff](https://github.com/dolev31/lasagna/blob/main/CITATION.cff), which GitHub renders as a ready-made BibTeX entry.

To cite the paper:

```bibtex
@misc{levy2026compositionality,
  title  = {Retrieval Augmented Generation Compositionality},
  author = {Levy, Ido and Shlomov, Segev and Choshen, Leshem},
  year   = {2026},
  note   = {Preprint},
  url    = {https://dolev31.github.io/lasagna/}
}
```

If you also report a number from the evaluation harness, publish the results
bundle with it:

```bash
lasagna results bundle <results.json> -o run.json
```

It carries the library versions, hardware, resolved sweep, corpus fingerprint
and comparability key — everything a reader needs to decide whether your number
and theirs are measuring the same thing. `lasagna results validate` recomputes
the key and the run id, so an edited number no longer matches its own bundle.

Bundles are meant to be shared. [hub/](https://github.com/dolev31/lasagna/tree/main/hub) has everything needed to run a
public leaderboard for them — a generated JSON Schema, a PR validation workflow,
and a trust model that says out loud what the numbers are and are not.

---

## License

Apache 2.0 — see [LICENSE](https://github.com/dolev31/lasagna/blob/main/LICENSE) and [NOTICE](https://github.com/dolev31/lasagna/blob/main/NOTICE). Lasagna vendors no
third-party source code.
