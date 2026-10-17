# Working with Lasagna

A guide for the thing this package exists to do: **find out whether a retrieval
component actually helps.**

Every Python block below is executed by the release gate on each run
(`e2e/phases/p3_docs.py`). If one stops working, the build fails. Blocks that
cannot run hermetically (anything needing a model download or a populated
store) are marked and say why.

**Contents**

1. [The question this answers](#1-the-question-this-answers)
2. [Install](#2-install)
3. [The stages you can swap](#3-the-stages-you-can-swap)
4. [Write a component](#4-write-a-component)
5. [Register it, and see it listed](#5-register-it-and-see-it-listed)
6. [Compare it against a baseline](#6-compare-it-against-a-baseline)
7. [Swap it into a pipeline](#7-swap-it-into-a-pipeline)
8. [Measure it on a benchmark](#8-measure-it-on-a-benchmark)
9. [Read and publish the result](#9-read-and-publish-the-result)
10. [Where to go next](#10-where-to-go-next)

---

## 1. The question this answers

You have an idea for a reranker, or a chunker, or a query rewriter. The
literature says the technique gives +4 NDCG. You want to know whether it helps
*your* corpus, and you want a number you would be willing to publish.

Doing that honestly is harder than it sounds, because the usual approach (fork
a RAG pipeline, patch your component in, run it) changes more than one thing at
once. Different chunking, a different embedding model, a different top-k, and
the +4 you measured belongs to something other than your idea.

Lasagna's answer is that the eight stages are isolated behind typed contracts.
You swap exactly one and everything else stays fixed, so a difference in the
result is attributable to the swap. That property is enforced by a conformance
suite, not by convention.

The rest of this guide walks that workflow end to end.

---

## 2. Install

```bash
pip install lasagna-rag                    # core: ingest, embed, search, serve
python -m nltk.downloader punkt_tab        # once: nltk's sentence tokenizer data
```

Lasagna never downloads the tokenizer data by itself. Without it, sentences are
split with a regular expression, which can move chunk boundaries, and run
bundles record that as the `sentence_split_without_punkt` fallback.

**First run downloads models.** The default profile pulls an embedding model
on the first upload or search *and* a reranker on the first search that
reranks: about **3.2 GB** (BGE-M3, 2.3 GB, and the mxbai reranker, 881 MB;
2.3 GB more with the `flag-embedding` extra, which fetches BGE-M3's whole
repository, ONNX export included), and between 3 and 40 minutes to your first
search result on release-gate runs, depending on the connection. After that, a warm search on the speed profile takes tens of
milliseconds (median 32 ms over 18,442 chunks on Milvus Lite). The default
profile takes 7 to 9 s on two CPU threads, because its large reranker scores
every query; more cores or a GPU cut that.

To try it faster, start on the `speed` profile: dense vector search on a
92 MB model, no reranker and no API key.

```bash
lasagna configure --profile speed
```

[Configuration](configuration.md#profiles) lists the five profiles and what
each one downloads, and explains which settings win when a profile, the
settings file and the environment disagree.

Nothing in sections 3 to 7 needs a model or a network: components are plain
Python objects, and you can exercise them directly. Only section 8 needs a
corpus.

### Extras

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
| `llamaindex` | wrapping LlamaIndex components as stages |
| `storage-all` | S3, GCS, Azure document storage |
| `local` | local LLM inference (HF, Ollama) |
| `providers` | LLM providers beyond OpenAI: Anthropic, Google, Mistral, Cohere, NVIDIA |
| `ocr` | Tesseract and PaddleOCR engines (Tesseract also needs the `tesseract` program) |
| `web` | web search fallback for corrective RAG |
| `lifecycle` | scheduled document expiry and cleanup |
| `production` | Redis, Celery, Prometheus, OpenTelemetry and PostgreSQL: what `settings.lasagna_production.toml` turns on |
| `all` | everything above except `markitdown`, `pymupdf4llm` and `llamaindex` |

A few components need a package or an API key of their own: the `flashrank`
reranker (`pip install flashrank`), the `cohere` reranker and embedder
(`pip install cohere`, `COHERE_API_KEY`), the `voyage` embedder
(`pip install voyageai`, `VOYAGE_API_KEY`) and the `openai` embedder
(`OPENAI_API_KEY`). Each raises naming what is missing.

`all` is large: pip takes about two minutes to resolve it (measured with pip
26.2), and `uv pip install "lasagna-rag[all]"` a few seconds.

### Platform notes

On Linux x86_64, `pip` installs the CUDA build of torch, about 3.2 GB of
downloads. For CPU only, install torch from the PyTorch CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lasagna-rag
```

On Apple Silicon, model calls on the GPU (MPS) run one at a time, because two at
once crash Metal; concurrent sweep arms wait for each other there. To run
everything on the CPU instead, set `RAG_USE_GPU=false`.

**Name collision.** `pip install lasagna` installs a different project
(`lasagna-ai`, an LLM-agent framework), which also provides a top-level
`lasagna` import package. Installing both in one environment makes them
overwrite each other. This package is `lasagna-rag`.

`import lasagna` loads no torch, transformers, sentence-transformers or
pymilvus; those arrive only when something needs them, so a remote-backend
client stays light.

**No usage telemetry.** Lasagna collects none. ragas, the library behind the
generation metrics, sends its authors a usage event (its version, the LLM
provider and model name, and a random id) each time a judge is built. Importing
`lasagna` turns that off by setting `RAGAS_DO_NOT_TRACK=true`, unless you have
already set it. To send ragas those events, set `RAGAS_DO_NOT_TRACK=false`
before importing `lasagna`. ragas still writes its random id to a file in your
user data folder, but with tracking off it sends nothing.

---

## 3. The stages you can swap

```python
from lasagna.core.registry import registry

# list_all() maps each stage to the names registered for it.
for stage, names in sorted(registry.list_all().items()):
    print(f"{stage}: {len(names)} registered -> {', '.join(sorted(names)[:3])}")
```

From the shell, `lasagna component list` (or `lasagna component list reranker`)
prints the same, with each component's description and paper.

Each stage has a typed abstract base class. Implementing it is the whole
contract: there is no plugin manifest and no registration ceremony beyond one
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
| Generator | `BaseGenerator` | how the answer is written from the retrieved passages |

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

A function is enough for most techniques: `@lasagna.reranker("name")` on
`f(query, text) -> score` wraps it in this interface, and the other stages have
decorators of their own. The [README](../README.md#compare-a-component) shows
one, and [Contributing components](contributing-components.md) lists every
stage's function signature.

---

## 5. Register it, and see it listed

Registration is one decorator. The `paper` and `claims` arguments are optional
but worth filling in: they are carried through to the listing API and recorded
by every run, and the leaderboards and `results view` show a claimed number next
to the one you reproduced.

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
satisfies the abstract base class, not merely when it was registered. A
component that registers but does not conform is listed with
`implements_contract: False`, which is how you find out before your results
depend on it.

### Already built in LangChain or LlamaIndex?

`lasagna.from_langchain` and `lasagna.from_llamaindex` register an object you
already have. A document compressor or node postprocessor becomes a reranker,
embeddings become an embedder, and a retriever becomes a retriever. The listing
describes the entry as `wrapped: langchain.<Class>`.

```python
import lasagna
from langchain_core.embeddings import DeterministicFakeEmbedding
from lasagna.core.registry import registry

name = lasagna.from_langchain(DeterministicFakeEmbedding(size=64))
print(name)  # langchain.DeterministicFakeEmbedding
entry = next(e for e in registry.list_embedders() if e["name"] == name)
print(entry["description"])  # wrapped: langchain.DeterministicFakeEmbedding
```

A sweep run from Python names the returned name like any other component. A
sweep file run with `lasagna benchmark experiment` names the object where you
defined it instead: with `cohere = CohereRerank(top_n=10)` in
`my_rerankers.py`, the arm `"my_rerankers:cohere"` is wrapped the same way.

Three things to know:

- **A wrapped retriever brings its own index.** It searches whatever the
  object was built over, not the corpus Lasagna ingested. For a benchmark, its
  index must hold that benchmark's corpus with each corpus id in the metadata
  (`doc_id`, or the key you pass as `id_key=`). A document with no id gets a
  hash of its text as its id, which matches no relevance judgment.
- **The object's own limits apply.** A compressor's `top_n` or a retriever's
  `k` caps how many documents come back; set it at least as high as the
  pipeline's depth, or the arm is scored on shorter lists than the others.
- **LlamaIndex is an extra:** `pip install "lasagna-rag[llamaindex]"`.
  LangChain needs nothing more.

---

## 6. Compare it against a baseline

This is the part that makes a number defensible. Run the same input through two
components and compare the outputs. Nothing else varies, so any difference is
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

Always run the control. `changed: False` has two very different causes (your
component agreed with retrieval order, or it never ran), and only the control
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
execution rather than only updating a name: a pipeline that kept serving a
cached instance would leave you measuring the old component while believing you
had changed it.

---

## 8. Measure it on a benchmark

Sections 4 to 7 prove your component behaves. This section is about whether it
*helps*, which needs a corpus and real metrics.

### Check the setup first

```bash
lasagna benchmark run smoke --profile speed
```

`smoke` is 24 documents and 12 queries held inside the package: no dataset
download and no API key, and on `speed` the only model is a 92 MB embedder. It
runs the real retrieval pipeline end to end, so a green run means the stack is
wired up. `lasagna benchmark run` measures retrieval unless asked for more; add
`--tracks retrieval,generation` once an LLM key and the `[eval]` extra are set
up, to check answer generation too. It is deliberately easy, so the scores mean
"this works", never "this is good".

`lasagna benchmark list` names every benchmark and `lasagna benchmark info`
describes one. To run on your own questions, see
[Your own data](own-data.md): a JSONL or CSV file, or a BEIR folder, runs with
no adapter to write.

### Then the real thing

```python
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

config = ExperimentConfig(
    name="does my reranker help",
    benchmarks=[BenchmarkSpec(name="beir/scifact", n_samples=200)],
    tracks=["retrieval"],  # NDCG, recall and the rest; no LLM, no judge
    pipeline=PipelineSweepSpec(
        base_profile="speed",  # a 92 MB embedder and no API key
        # Sweep across rerankers with everything else fixed. The comparison is
        # only meaningful because the other seven stages do not move.
        rerankers=["cross_encoder", "length_demo"],
    ),
    # `mode` defaults to "auto": live when the benchmark ships a corpus, so
    # retrieval is actually measured. Say mode="generation_only" only when you
    # want answer quality scored against gold context, with no retrieval.
    n_runs=1,  # retrieval with fixed models gives the same lists every run
)
print(config.mode, "->", len(config.benchmarks), "benchmark(s)")
```

This one is keyless: `speed` embeds with a 92 MB model and the cross-encoder
control is 92 MB. The same file in YAML runs with
`lasagna benchmark experiment sweep.yaml`, which is what the
[README](../README.md#compare-a-component) and
[Contributing components](contributing-components.md#4-check-it-helps) show.
Run it, and look at the failed tasks before the scores: a task that failed is
listed there, not in the table.

```python
import asyncio

results = asyncio.run(Experiment(config).run())
for task in results.failed_tasks:
    print("failed:", task.task_key, task.error)
print(results.to_markdown("ndcg@10"))
```

**A component override is only measured in live mode.** `generation_only` never
constructs the pipeline, so passing `reranker=` there is a hard error rather
than a plausible number scored from gold context. `auto` picks the right mode
and logs which it chose and why.

**Benchmarks with no corpus can still measure retrieval, against a pooled
corpus, which is easier.** SQuAD v2, HotpotQA, MuSiQue and RAGBench ship each
question's supporting passages inline and no corpus to search. `auto` assembles
one by pooling those passages across every sample in the run, so each query has
to find its own among all the others'. This is a real retrieval task, but a
gentler one: HotpotQA and MuSiQue drop their distractor paragraphs at
conversion, so the pool holds no hard negatives and recall reads higher than it
would against a corpus of the same size. Runs are stamped
`corpus_provenance: pooled_gold`, which is part of the comparability key, so a
pooled number can never share a leaderboard row with a BEIR one. For the same
questions against a real corpus, use `beir/hotpotqa`.

Natural Questions and TriviaQA carry no context at all and stay
`generation_only`. There is nothing to pool.

### How many runs

Retrieval with fixed models is deterministic: the same queries return the same
ranked lists on every run. Significance tests pair arms query by query, so the
spread they need comes from the queries, and `n_runs: 1` gives the same answer
as three runs, three times faster. A sweep that repeats identical runs says so
when it ends.

Repeat runs when an LLM stage is in the loop. An answer generator, an LLM
reranker, HyDE or an agentic pipeline samples, and a model that takes no
temperature (see [Configuration](configuration.md#temperature-and-repeatable-runs))
varies from run to run whatever you ask for. There, `n_runs: 3` or more
measures that spread, and `variance_source: run` keeps the queries fixed so the
spread is run to run rather than a resampled subset.

### When corpus documents fail to ingest

A document that cannot be chunked (an error, or more than 120 s) is left out of
the corpus, and the log names it. A partial corpus scores lower than the
benchmark would, so every task on it fails, naming the count, the first
documents and the setting that accepts it:

```yaml
ingest_failure_tolerance: 0.02   # up to 2% of the corpus may be missing (at most 0.1)
```

A run that accepts a partial corpus records it in each result as
`corpus_ingest_gap` (how many documents were ingested, how many failed, and a
digest of the ingested ids). That is part of the comparability key, so the run
never shares a leaderboard group with one on the complete corpus, and the
bundle carries a `PARTIAL_CORPUS` warning. A later run that reuses the same
corpus records the same gap.

### Sweep the other stages

The same sweep file swaps any stage. Two you will often want:

**A retriever.** Name it in a `config_overrides` entry. Put `dense`, the default
vector search as a component, in as the control. Then every arm names its first
stage, and the comparison is between retrieval methods:

```yaml
# retrievers.yaml
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

**A generator.** List the answer strategies under `generators:`. Every arm
retrieves the same way, so a difference comes from how the answer was written.
`concise` is the harness's own prompt, word for word, so it is the control:

```yaml
# generators.yaml
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

This one needs an LLM API key and the `[eval]` extra. `cited` answers carry
`[n]` markers, and the citation metrics score them. From the shell,
`lasagna benchmark run <name> --generator cited` runs one generator.

**An ensemble of rerankers.** `ensemble` averages the scores of the
cross-encoders an arm names in `ensemble_reranker_models` (weights in
`ensemble_reranker_weights`, equal when left out). An arm with fewer than two
models, or with one that cannot load, fails instead of scoring as one model:

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
`enable_mmr` says, since naming them is what turns it on.

**Chunkers and embedders sweep as components too** (`chunkers:`,
`embedders:`). Each arm's corpus is chunked or embedded by its own component and
kept apart from the other arms' corpora; an embedder also gets its own
collection. Every chunking or embedding setting (`chunk_size`, `chunk_unit`,
`embedding_model` and so on) gets its own corpus in the same way. An agentic
loop is named like a retriever (`{name: ircot, agentic_pipeline: ircot}`), and
each hop it makes searches with the arm's components. Document processors do not
sweep: benchmark corpora are loaded as text, so check a processor on your own
files.

### The baselines a reviewer asks for

"Does retrieval help at all?" and "would a long context window do as well?" are
answered by arms, not by a separate tool:

```yaml
# baselines.yaml
name: is retrieval worth it
benchmarks:
  - name: hotpotqa
    n_samples: 100
tracks: [generation]
pipeline:
  base_profile: speed
  config_overrides:
    - {name: rag}                                # the pipeline as configured
    - {name: no_retrieval, no_rag: true}         # answers from the question alone
    - {name: long_context, generator: long_context}
n_runs: 3
```

- `no_rag: true` retrieves nothing: the arm's retrieval track is skipped and it
  answers with the `closed_book` generator. It is scored on generation only, so
  a sweep with `tracks: [retrieval]` refuses it before anything runs.
- `closed_book` is that generator on its own, if you name it under
  `generators:`.
- `long_context` reads all ten retrieved passages, in rank order, where the
  default reads five, and the judge sees the same ten.

An arm whose generator reads no passages is not scored on faithfulness or the
context metrics, which would be meaningless with no context; the report and the
leaderboard say so and point at exact match, token F1 and factual correctness
instead. The isolation check files a `no_rag` arm under the retriever stage.

### Before a long run: `--dry-run`

```bash
lasagna benchmark experiment sweep.yaml --dry-run              # with a 3-sample probe
lasagna benchmark experiment sweep.yaml --dry-run --no-probe   # nothing runs at all
```

A dry run prints the arms, the tasks, the models the run loads and whether each
is already downloaded, and a token estimate. Unless `--no-probe` is given it
runs three samples per benchmark and extrapolates tokens and time to the whole
sweep. Every number is labeled an estimate, and nothing is written under the
results folder. For the plugin template's sweep it printed:

```text
Arms (2): reranker=cross_encoder, reranker=template_length
Tasks (2): 1 benchmark x 2 arms x 1 run
  smoke  split=test  12 samples
Models:
  embedder           all-MiniLM-L6-v2  cached
  reranker           cross-encoder/ms-marco-MiniLM-L-6-v2  cached
  reranker           component template_length
Tokens: no generation track, so no answer is generated or judged; an arm still calls an LLM if its pipeline does (HyDE, an agentic pipeline, an LLM reranker).
Probe of smoke (estimate): arm 'reranker=cross_encoder' on 3 samples took 2.8 s including setup (loading models, ingesting), 0.12 s per query, 0 LLM calls
  Extrapolated to 24 queries (estimate, every arm assumed to spend what this one did): about 3 s of query time plus setup, 0 system tokens and 0 judge tokens
```

**Dollars only if you ask.** A run records tokens, which are observed. Add
`prices:` to the sweep file for an estimate in dollars:

```yaml
prices: builtin          # the dated table in the package (builtin-2026-10)
# or your own rates, in USD per million tokens, which win over the table:
# prices: {gpt-6-luna: {input_per_1m: 0.10, output_per_1m: 0.50}}
```

Each task whose tokens all have a rate gets `cost_usd_estimate`, and the run
metadata records the table and the rates used. A rate that holds only for
shorter prompts gives no estimate for a task whose calls average longer. With
no `prices:`, a run reports tokens only.

### Re-running a sweep replays LLM calls

Inside `lasagna benchmark experiment`, a request asked at temperature 0 that an
earlier run made with the same seed is answered from
`~/.lasagna/cache/llm.sqlite`. A second run of the same sweep, after a crash or
with one arm added, then spends no tokens on what it already asked. Token
counts read as in the cold run; `llm_cache_hits` and `llm_cache_misses` on each
task and a hit rate in `results.json` and the bundle say what was replayed.

What is never replayed:

- **Sampled calls.** A response at temperature 0.7 is one draw; replaying it
  would make repeated runs identical and hide the spread `n_runs` is meant to
  measure.
- **The RAGAS judge**, which asks for temperature 0.01, unless you set
  `llm_cache: {deterministic_only: false}`.
- **Models that take no temperature.** The cache replays a call only when
  temperature 0 was actually sent, so a model the request builder sends no
  temperature (Claude 4.7 and later, some OpenAI reasoning models) is always
  called.

A provider of your own, set with `set_llm_provider`, gets the cache too: the
chat model its `get_chat_model` returns is replayed when temperature 0 was asked
for and the model's `temperature` is 0. A returned object that is not a
LangChain chat model cannot carry the cache, and `results.json` then says
`"applied": false` with the reason under `llm_cache`.

Turn it off with `llm_cache: false` in the sweep, `--no-cache` on the command,
or `LASAGNA_LLM_CACHE=0`. `llm_cache: {path: ...}` puts the file elsewhere.
Nothing is cached outside an experiment, so `lasagna serve` and the MCP server
write no prompt to disk.

### A development set from your own documents

`lasagna benchmark synthesize ./docs --n 200` has an LLM write factoid,
multi-hop and unanswerable questions from your files, checks each one with a
second call, and writes a JSONL file that `lasagna benchmark run` takes as is.
It is for comparing configurations while you build, not for reported results:
questions a model wrote favor systems like that model.
[Your own data](own-data.md#synthetic-questions-from-your-documents) explains
why and shows the options.

For what the metrics mean and which are trustworthy, see
[metrics](../src/lasagna/knowledge_evaluation_sdk/docs/metrics.md); for the
available datasets, see
[benchmarks](../src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md).

---

## 9. Read and publish the result

Every run writes a `results.json` and prints its path. Each command below takes
it.

| Question | Command |
|---|---|
| Is the gap real, across every pair of arms? | `lasagna results significance results.json --all` |
| Could this run have detected the effect I care about? | `lasagna results power results.json baseline candidate` |
| Do two techniques compose, or overlap? | `lasagna results interaction results.json` (a 2x2 run) |
| Which queries did an arm lose, at retrieval or generation? | `lasagna results per-query results.json --sort-by ndcg@10` |
| Are two runs comparable at all? | `lasagna results compare a.json b.json` |

`significance` runs a paired sign-flip test on every pair of arms and corrects
the p-values with Holm-Bonferroni, then ends in a box that says what the table
supports: green names each pair that survives and which side won; yellow says
nothing survived and points at `power`.

**Hand someone the whole run as one page:**

```bash
lasagna results view results.json -o view.html
```

The page carries the versions and run id, the stage isolation check, and per
benchmark the leaderboard, the significance table with its verdict, every
metric per arm, and a per-query diff between two arms (`--diff A B`, default the
top two) with what each one retrieved. It is one file with no external
requests, so it opens offline and can be attached to a review.

**Put a leaderboard in a paper:**

```bash
mkdir -p bundles
lasagna results bundle results.json -o bundles/run.json
lasagna results leaderboard bundles/ --format latex
```

`--format latex` prints one booktabs table per comparability group (rows in
one table held the same configuration fixed; rows in different tables are
never ranked together), or writes one `.tex` file per benchmark with `-o`.
Arm names are escaped for LaTeX.

**Publish a number someone else can check:**

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json
```

The bundle carries the library versions, hardware, resolved sweep, corpus
fingerprint and comparability key, and `validate` recomputes the key, the run
id and every aggregate from the per-query scores. It also lists integrity
warnings that do not invalidate a bundle, such as `SINGLE_RUN` (one run of a
pipeline in which something samples) or `UNRESOLVED_JUDGE_ALIAS` (a judge named
by a floating alias); `--strict` turns them into failures. A run records
`nothing_samples` when only retrieval ran and no task called an LLM; one run of
it is complete and raises no `SINGLE_RUN`. The [results hub](../hub/README.md)
runs a public leaderboard from such bundles, and marks a row verified only when
someone else's bundle reproduces it: within its spread, which needs at least two
runs, or by equalling its mean when nothing in the run samples.

---

## 10. Where to go next

| I want to... | Read |
|---|---|
| **Run on my own questions or documents** | [Your own data](own-data.md) |
| **Publish my component as a package** | [Plugin template](../examples/lasagna-plugin-template/) |
| **Check my component satisfies its contract** | `lasagna component validate <module:Class>` |
| **See which plugins are installed (and which failed)** | `lasagna plugins list` |
| **Know a dataset's terms before publishing** | [Dataset licenses](../src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md) |
| Know which settings apply, and where data lives | [Configuration](configuration.md) |
| Look up any setting | [Settings reference](reference/settings.md) |
| Use the Python SDK in an application | [SDK user guide](../src/lasagna/sdk/docs/user_guide.md) |
| Add a component type, not just an implementation | [Extending](../src/lasagna/sdk/docs/extending.md) |
| Call it over HTTP | [REST endpoints](../src/lasagna/api/docs/endpoints.md) |
| Give an agent search over documents | [MCP server](../src/lasagna/mcp/README.md) |
| Run a different vector database | [Vector backends](../src/lasagna/vector_backends/README.md) |
| Understand the retrieval internals | [Architecture](../src/lasagna/core/docs/architecture.md) |
| Deploy it | [Deployment](../src/lasagna/deploy/README.md) |

### Two honest warnings

**Check the model license before publishing.** Several strong rerankers are
non-commercial. `lasagna.core.model_licenses` warns once at load, and
[model-licenses](model-licenses.md) lists the terms per model.

**A component that conforms is not a component that helps.** Everything up to
section 7 proves your code is wired in correctly. Only section 8 tells you
whether it improves retrieval, and only a significance test over enough queries
tells a real gap from noise.
