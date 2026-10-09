<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/dolev31/lasagna/raw/main/docs/assets/hero-dark.png">
  <img src="https://github.com/dolev31/lasagna/raw/main/docs/assets/hero-light.png" width="100%" alt="Lasagna: swap one RAG stage, hold the rest fixed, and measure what actually changed.">
</picture>

**Test your RAG components against a real stack — not a toy baseline.**

[![Project page](https://img.shields.io/badge/project-page-1f6feb.svg)](https://dolev31.github.io/lasagna/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dolev31/lasagna/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/dolev31/lasagna/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/dolev31/lasagna/actions/workflows/tests.yml)

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

**All eight stages take effect.** Five at query time — `reranker`,
`query_enhancer`, `embedder`, `retriever`, `agentic_pipeline` — two at
ingest, `chunker` and `document_processor`, and `generator`, which writes the
answer from what retrieval returned. In a sweep, every stage but the document
processor swaps as a component with a single config key; a chunker or embedder
component gets a corpus of its own, chunked or embedded by it. Settings
(`chunking_strategy`, `chunk_size`, `embedding_model` and so on) swap the same
way and re-ingest per arm. Benchmark corpora are loaded as text, so a document
processor is checked on your own files instead. Either way the rest hold still,
so the difference you measure belongs to the thing you changed.

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
both are real ABCs enforced at instantiation. What we add on top is that a
function is enough (one decorator wraps it in the stage's interface) and that a
sweep checks, from what actually ran, that the configuration reaching each stage
differs between arms only in the stages they name, and flags an arm whose ranked
lists never moved as inert.

**What is genuinely different is the measurement apparatus.** A paired sign-flip
test with t intervals, Holm–Bonferroni correction across a sweep (with
adjusted p-values), a 2×2 interaction test with a t interval, per-sample
scores joined by sample identity, comparability keys that record what was held
fixed, and integrity-checked result bundles. None of LangChain, LlamaIndex,
Haystack or RAGAS ships significance testing; RAGAS scores a run, and we depend
on it to do so. Significance testing itself is not new: PyTerrier and ranx ship
it for IR runs, UltraRAG ships a paired permutation test, and some papers ship
their own scripts. What we did not find elsewhere is the combination inside a
RAG pipeline framework: paired tests with multiple-comparison correction,
interaction CIs and provenance bundles, all from one sweep file.

### Against research RAG tools

A researcher will more likely weigh Lasagna against these. "Not found" means we
found it in neither the docs nor a code search (October 2026), not that it is
proven absent.

| | **Lasagna** | **FlashRAG** | **BERGEN** | **UltraRAG** | **AutoRAG** | **PyTerrier / ranx** |
|---|---|---|---|---|---|---|
| Significance tests | Paired sign-flip test, Holm, power | Not found | Not found | Paired permutation, no correction | Not found | Yes, IR runs only |
| Interaction between components | 2×2 test with CI | Not found | Not found | Not found | Greedy search assumes none | Not found |
| Result provenance | Bundles with a validator | Saved configs and scores | Experiment folders | Not found | Trial folders | Not found |
| Published methods | 53 components | 23 methods | 20+ retrievers, 4 rerankers | Several baselines | Many modules | IR pipelines |
| Ready corpora and indexes | Benchmark loaders, no prebuilt index | Wikipedia + prebuilt index | KILT + prebuilt index | Not found | Not found | Via ir_datasets |
| License | Apache-2.0 | MIT | CC BY-NC-SA 4.0 | Apache-2.0 | MIT (badge) | MPL-2.0 / MIT |

Where they win: FlashRAG and BERGEN ship prebuilt indexes and result tables for
many published methods, so a baseline costs little to reproduce. AutoRAG has
Colab notebooks and a hosted demo; FlashRAG, UltraRAG and XRAG have visual UIs.
Lasagna ships none of its own paper's per-query results yet.

**Where the others plainly win.** Ecosystem — LangChain and LlamaIndex have
thousands of integrations and years of production hardening; Lasagna is a
pre-release with none of either. Install weight — torch and sentence-transformers
are core dependencies: about 1.3 GB installed on macOS arm64, and on Linux x86_64
the default PyPI torch pulls CUDA wheels, about 3.2 GB of downloads. That is a
real cost for an API-embeddings user, and both of them are far lighter. Document parsing — LlamaParse handles formats our
built-in pypdf/pdfplumber extraction does not. Production observability — LangSmith exists;
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
    client = KnowledgeClient(session_id="quickstart")   # same session on every run
    doc = await client.upload("paper.pdf")
    await client.documents.wait_for_processing(doc.document_id)  # upload returns once queued

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
> the software. After that, measured warm on two CPU threads, a search takes
> about 0.3 s on the speed profile and 7–9 s on the default profile, whose large
> reranker scores every query; more cores or a GPU cut the second. The install itself is
> about 1.3 GB on macOS; see [Install](#install) for Linux.
>
> To get the 80 MB model instead, run `lasagna init` and choose the speed
> profile. It writes `settings.lasagna.toml` and adds `RAG_SETTINGS_FILE` to
> `./.env`, which is what the config loader actually reads. The `lasagna`
> commands run in that directory read that one key from `./.env` (never API
> keys). Python does not read `.env`, so from your own code
> `export RAG_SETTINGS_FILE="$PWD/settings.lasagna.toml"` first.
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

This is what Lasagna is for. Adding a technique is one line: a decorator on a
plain function. The function becomes a component of its stage, interchangeable
with every built-in, and runs inside the complete pipeline. This executes as-is,
with no model download and no corpus:

```python
import lasagna
from lasagna.core.base_components import ScoredDocument
from lasagna.core.registry import registry


@lasagna.reranker("readme_demo", paper="arxiv:2XXX.XXXXX", claims={"ndcg@10": 0.456})
def longer_first(query, text):
    return len(text.split())  # your technique: score one passage


candidates = [
    ScoredDocument(content="short", score=0.9, document_id="d1", chunk_id="c1"),
    ScoredDocument(content="a considerably longer passage of text", score=0.5,
                   document_id="d2", chunk_id="c2"),
    ScoredDocument(content="middling length here", score=0.7,
                   document_id="d3", chunk_id="c3"),
]

mine = registry.get("reranker", "readme_demo").rerank("q", candidates)
entry = next(e for e in registry.list_rerankers() if e["name"] == "readme_demo")
print("retrieval order:", [d.document_id for d in candidates])
print("my reranker:    ", [d.document_id for d in mine])
print("claims recorded:", entry["claims"])
print("interchangeable:", entry["implements_contract"])
```

```text
retrieval order: ['d1', 'd2', 'd3']
my reranker:     ['d2', 'd3', 'd1']
claims recorded: {'ndcg@10': 0.456}
interchangeable: True
```

`paper` and `claims` travel with the component and come back out of the
registry, so a results table can put the claimed number beside the one you
reproduced. Every stage but the agentic pipeline takes a function the same way:
`@lasagna.query_enhancer` (`query -> [str]`), `@lasagna.chunker` (`text -> [str]`),
`@lasagna.embedder` (`texts -> vectors`), `@lasagna.generator`
(`query, contexts -> answer`), `@lasagna.retriever` and
`@lasagna.document_processor`. A reranker can also order the whole list
(`query, documents, top_k`). The function is handed copies, so it cannot change
what another stage sees. When a technique needs state or a model, or is an
agentic loop, subclass the stage's interface (`BaseReranker`,
`BaseAgenticPipeline` and so on) and register the class with
`@registry.register` instead.

Then measure it, again in one line. A sweep can name the function by its import
path, with no registration call and no package:

```yaml
# sweep.yaml, next to my_ideas.py
name: does longer_first help
benchmarks: [{name: smoke}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_ideas:longer_first"]
n_runs: 3
```

```bash
lasagna benchmark experiment sweep.yaml
lasagna results significance "<output>/results.json" -m ndcg@10 --all
```

Before any corpus is ingested, the run checks that each named component
implements its stage and passes its behavioural contract (return types, `top_k`,
empty input, no mutation of its input). It then runs both arms on the same
queries and ends by confirming that the configuration reaching each stage
differed only at the reranker, and that the reranker arm's ranked lists moved:
an arm whose lists never moved is flagged inert.
[The guide](https://github.com/dolev31/lasagna/blob/main/docs/guide.md#8-measure-it-on-a-benchmark)
covers the 31 benchmarks.

### Picking a baseline

Any listed built-in will do: every one implements its stage's interface, so a
comparison measures the algorithms rather than an interface difference. That
is what `implements_contract` in a listing reports:

```python
from lasagna.core.registry import registry

[r["name"] for r in registry.list_rerankers() if r["implements_contract"]]
# ['cross_encoder', 'combined', 'ensemble', 'mmr', 'colbert', 'llm_listwise',
#  'monot5', 'llm_pointwise', 'flashrank', 'cohere']
```

A component registered without its stage's interface (a duck-typed class) is
reported as not interchangeable when selected, and a sweep refuses it, naming
the fix.

### Why the stages stay isolated

A comparison is attributable only if the arms differ in the stage under test and
nowhere else. Lasagna checks that rather than assuming it.

- **A fixed interface per stage.** Components exchange only their stage's
  declared types, and function components get copies of what they are handed.
  A conformance suite checks that every registered component subclasses its
  stage's contract, does not mutate its input, and does not hand out chunks that
  share one metadata dict. From a source checkout (the tests are not in the
  wheel), run
  `pytest tests/test_abc_conformance.py tests/test_stage_isolation_behavioural.py`.
- **A stage manifest per task.** Each run records what reached each of the eight
  stages: every setting, from the configuration the search actually resolved,
  and the component in the slot.
- **A check across arms.** If two arms differ at a stage neither of them names,
  the run says so, the bundle is marked `UNDECLARED_STAGE_CHANGE`, and
  `lasagna results validate` refuses it, naming the stage and the settings that
  moved. An arm whose named change never reached its stage is flagged as inert,
  since its number is the control's under another name.

### Do two techniques compose, or do they overlap?

The question two papers cannot answer between them. HyDE is published against a
baseline. Reranking is published against a baseline. Whether HyDE still earns
its latency once reranking is on does not follow from those two numbers — both
may be fixing the same queries, in which case stacking them buys far less than
the sum of their headline gains.

Answering it needs all four cells measured on the same queries. Declare the 2×2
and the harness builds them:

```yaml
# compose.yaml
name: hyde with reranking
benchmarks:
  - name: beir/scifact
    n_samples: 100
tracks: [retrieval]
pipeline:
  base_profile: standard
  factorial:
    factor_a: enable_hyde
    factor_b: enable_reranking
```

```bash
lasagna benchmark experiment compose.yaml
lasagna results interaction "<output>/results.json" --metric ndcg@10
```

Unlike the sweep above, this one is neither keyless nor small. HyDE writes a
hypothetical answer with an LLM, so it needs an API key: without one the HyDE
cells fail rather than report the control's number. The `standard` profile
embeds with `BAAI/bge-m3` (about 2.3 GB on first use) and reranks with a large
cross-encoder.

```
  expected if additive : +0.1000
  actually observed    : +0.0600
  interaction          : -0.0400  95% CI [-0.0490, -0.0311]  p=0.0001
  composition efficiency: 60.0% (of the additive prediction actually delivered)

  redundant (they overlap; the second buys less than it does alone)
```

The statistic is `(AB − neither) − (A − neither) − (B − neither)`, tested by
sign-flipping each query's interaction contrast (all four cells of a query move
together), with a t interval. Positive means they compound; negative means they
overlap.

Factors are boolean flags, not component names — a component slot has no *off*
level for the `neither` cell to use, and swapping a component does not enable
the feature. `lasagna` refuses a factorial over anything that is not a real
boolean setting, before the first corpus is ingested.

---

## What's inside

### Pluggable RAG pipeline stages

| Stage | Registered | Notes |
|---|---:|---|
| `retriever` | 5 | dense, BM25, SPLADE, hybrid RRF, hybrid convex (normalized score fusion) |
| `reranker` | 10 | cross-encoder (any Hugging Face reranker, e.g. bge-reranker-v2-m3), ColBERT, monoT5, RankGPT listwise, LLM pointwise, FlashRank, Cohere Rerank, MMR, ensemble, combined |
| `query_enhancer` | 8 | HyDE, Query2doc, step-back, Rewrite-Retrieve-Read, RAG-Fusion multi-query, expansion, least-to-most decomposition, combined |
| `agentic_pipeline` | 7 | default graph (CRAG grading, Self-RAG-style gates, multi-hop planning), Self-Ask, IRCoT, FLARE, Adaptive-RAG, Self-RAG, legacy ReAct |
| `generator` | 4 | concise, cited (ALCE), chain-of-thought, chain-of-note |
| `chunker` | 9 | fixed (characters, words or tokens), recursive, sentence window, Markdown header, semantic (Kamradt breakpoints), hierarchical, structure-aware, propositions (Dense X), contextual (Contextual Retrieval) |
| `embedder` | 6 | any sentence-transformers model with its model card's prefixes, BGE-M3, OpenAI, Cohere, Voyage, and the configured default |
| `document_processor` | 4 | built-in (pypdf, pdfplumber, OCR), Docling, MarkItDown, PyMuPDF4LLM |

### Every stage, with the methods a paper compares against

Each method below is a registered component or a setting, and either swaps in
a sweep with one line.

- **Ingest and chunk:** built-in PDF extraction, Docling, MarkItDown,
  PyMuPDF4LLM; fixed (characters, words or tokens), recursive, sentence
  window, Markdown header, semantic breakpoints, hierarchical, propositions
  (Dense X Retrieval), Contextual Retrieval.
- **Embed:** any sentence-transformers model with its model card's query and
  document prefixes, BGE-M3, OpenAI, Cohere, Voyage, Matryoshka truncation,
  and late chunking (a setting that works with any chunker).
- **Retrieve:** dense, BM25, SPLADE, hybrid by reciprocal rank fusion, hybrid by
  normalized score fusion.
- **Rerank:** cross-encoders (bge-reranker, mxbai and any Hugging Face
  reranker), ColBERT, monoT5, RankGPT listwise, LLM pointwise grading,
  FlashRank, Cohere Rerank, MMR, ensembles.
- **Augment:** HyDE, Query2doc, step-back prompting, Rewrite-Retrieve-Read,
  RAG-Fusion, query expansion, least-to-most decomposition, lost-in-the-middle
  reordering, context compression.
- **Agentic loop:** a LangGraph pipeline with CRAG-style document grading,
  multi-hop planning, hallucination checks and quality gates (swappable as a
  unit or node by node), plus Self-Ask, IRCoT, FLARE, Adaptive-RAG and
  Self-RAG. Each hop searches with the arm's own components.
- **Generate:** a concise answer, answers with ALCE-style citations,
  chain-of-thought, chain-of-note, speculative RAG.

### Vector databases: six backends, one interface

Milvus · pgvector · Qdrant · Weaviate · Pinecone · OpenSearch, behind one
interface with a shared conformance suite asserting the same behaviour against
each. Milvus and Qdrant run in-process and are verified on every commit; the
rest need their service running
([details](https://github.com/dolev31/lasagna/blob/main/tests/test_vector_backend_conformance.py)).

### RAG evaluation: 31 benchmarks, retrieval and generation metrics

31 benchmark entries over about 22 datasets (BEIR, RAGBench and KILT subsets
count separately) — BEIR (NFCorpus, FiQA, SciFact, ArguAna, HotpotQA), HotpotQA,
SQuAD v2, MuSiQue, TriviaQA, Natural Questions, MS MARCO, FEVER, PopQA,
Bamboogle, MultiHop-RAG, RAGBench, KILT, 2WikiMultihop.

| Family | Metrics |
|---|---|
| Retrieval | NDCG@k, precision@k, recall@k, F1@k, hit rate (success)@k, MAP, MRR, R-precision, complete (all-gold) recall@k for multi-hop questions |
| Generation, no judge | exact match and token F1 (SQuAD-normalized; SQuAD v2 scored answerable and unanswerable separately), ROUGE-L, answer containment, answer similarity (embedding), abstention rate |
| Generation, judged | faithfulness, context precision, context recall, factual correctness (RAGAS); citation recall and precision (ALCE) when answers cite; opt-in: answer relevancy, noise sensitivity, context relevance |
| Operational | p50/p95/p99 latency (retrieval and generation, end to end), input and output tokens and LLM calls per query, with the judge's spend reported separately |

Plus multi-objective tradeoff analysis, paired significance testing, and t
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
lasagna configure --profile speed -s my-corpus   # 80 MB model instead of 5.4 GB
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
python -m nltk.downloader punkt_tab        # once: nltk's sentence tokenizer data
```

Lasagna never downloads the tokenizer data by itself. Without it, sentences are
split with a regular expression, which can move chunk boundaries, and run
bundles record that as the `sentence_split_without_punkt` fallback.

To reproduce a result exactly, install the versions the release was tested with.
The constraints file was resolved on Python 3.12 and does not install on 3.10 or
3.11 (its numpy needs 3.12 or later):

```bash
pip install "lasagna-rag[eval]==0.1.0a1" -c https://raw.githubusercontent.com/dolev31/lasagna/main/constraints/0.1.0a1.txt
```

| Extra | Adds |
|---|---|
| `eval` | RAGAS metrics, benchmark loaders, statistics, plots |
| `ingest` | EasyOCR for scanned documents |
| `docling` | structure-aware document conversion (the `docling` processor) |
| `markitdown` | Microsoft MarkItDown: PDF, Office and HTML to Markdown (the `markitdown` processor) |
| `pymupdf4llm` | PDF to Markdown (the `pymupdf4llm` processor); AGPL-3.0 |
| `vector-backends` | Pinecone, Qdrant, Weaviate clients |
| `opensearch` / `pgvector` | those backends |
| `flag-embedding` | BGE-M3 dense+sparse+ColBERT |
| `graphrag` | entity-linked retrieval |
| `storage-all` | S3, GCS, Azure document storage |
| `local` | local LLM inference (HF, Ollama) |
| `providers` | LLM providers beyond OpenAI: Anthropic, Google, Mistral, Cohere, NVIDIA |
| `ocr` | Tesseract and PaddleOCR engines (Tesseract also needs the `tesseract` program) |
| `web` | web search fallback for corrective RAG |
| `lifecycle` | scheduled document expiry and cleanup |
| `production` | Redis, Celery, Prometheus, OpenTelemetry and PostgreSQL: what `settings.lasagna_production.toml` turns on |
| `all` | everything above except `markitdown` and `pymupdf4llm` |

A few components need a package or an API key of their own: the `flashrank`
reranker (`pip install flashrank`), the `cohere` reranker and embedder
(`pip install cohere`, `COHERE_API_KEY`), the `voyage` embedder
(`pip install voyageai`, `VOYAGE_API_KEY`) and the `openai` embedder
(`OPENAI_API_KEY`). Each raises naming what is missing.

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

On Linux x86_64, `pip` installs the CUDA build of torch, about 3.2 GB of
downloads. For CPU only, install torch from the PyTorch CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lasagna-rag
```

On Apple Silicon, model calls on the GPU (MPS) run one at a time, because two at
once crash Metal; concurrent sweep arms wait for each other there. To run
everything on the CPU instead, set `RAG_USE_GPU=false`.

> **Name collision.** `pip install lasagna` installs a different project
> (`lasagna-ai`, an LLM-agent framework), which also provides a top-level
> `lasagna` import package. Installing both in one environment makes them
> overwrite each other. This package is `lasagna-rag`.

`import lasagna` deliberately loads no torch, transformers, sentence-transformers
or pymilvus — those arrive only when something needs them, so a remote-backend
client stays light.

**No usage telemetry.** Lasagna collects none. ragas, the library behind the
generation metrics, sends its authors a usage event (its version, the LLM
provider and model name, and a random id) each time a judge is built. Importing
`lasagna` turns that off by setting `RAGAS_DO_NOT_TRACK=true`, unless you have
already set it. To send ragas those events, set `RAGAS_DO_NOT_TRACK=false`
before importing `lasagna`. ragas still writes its random id to a file in your
user data folder, but with tracking off it sends nothing.

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

Hermetic by default: no network, no external services, and the suite runs in a
throwaway home directory, so it never touches your real `~/.rag_knowledge` store.
CI enforces it: sockets are blocked except loopback, and test order is shuffled
on every run (pytest-randomly prints the seed). `tests/live/` needs a running
stack; `tests/stress/` is long-running.

Coverage is measured with branches on and floored in CI. The floor is on the
combined statement and branch total, because this package's bugs have lived
overwhelmingly in error paths.

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
the key, the run id and every aggregate from its per-query scores, so a number
edited after the run no longer matches its bundle. That is an integrity check
against mistakes and naive edits, not a signature: anyone who edits a number can
recompute the id, which is why the hub's trust model calls results
self-reported.

Bundles are meant to be shared. [hub/](https://github.com/dolev31/lasagna/tree/main/hub) has everything needed to run a
public leaderboard for them — a generated JSON Schema, a PR validation workflow,
and a trust model that says out loud what the numbers are and are not.

---

## License

Apache 2.0 — see [LICENSE](https://github.com/dolev31/lasagna/blob/main/LICENSE) and [NOTICE](https://github.com/dolev31/lasagna/blob/main/NOTICE). Lasagna vendors no
third-party source code.
