# Installation

Lasagna runs on Python 3.10 to 3.13.

```bash
pip install lasagna-rag                    # core: ingest, embed, search, serve
python -m nltk.downloader punkt_tab        # once: nltk's sentence tokenizer data
```

Lasagna never downloads the tokenizer data by itself. Without it, sentences are
split with a regular expression, which can move chunk boundaries. Run bundles
record that as the `sentence_split_without_punkt` fallback.

!!! warning "The package is `lasagna-rag`"
    `pip install lasagna` installs a different project (`lasagna-ai`, an
    LLM-agent framework). It also provides a top-level `lasagna` import
    package, so installing both in one environment makes them overwrite each
    other.

## What the first run downloads

The default profile pulls an embedding model on the first upload or search and
a reranker on the first search that reranks. That is about **3.2 GB**: BGE-M3
(2.3 GB) and the mxbai reranker (881 MB). The `flag-embedding` extra adds 2.3 GB
more, because it fetches BGE-M3's whole repository, ONNX export included.
Expect between 2 and 40 minutes to your first search result on release-gate
runs. The spread is your connection.

To try Lasagna faster, start on the `speed` profile: dense vector search on a
92 MB model, no reranker and no API key.

```bash
lasagna configure --profile speed
```

[Configuration](../configuration.md#profiles) lists the five profiles and what
each one downloads.

## Pin the versions the release was tested with

For a result you want to repeat later, install with the constraints file from
the release. It was resolved on Python 3.12 and needs 3.12 or later, because
numpy does not install on 3.10 or 3.11 with those pins.

```bash
pip install "lasagna-rag[eval]==0.1.0" -c https://raw.githubusercontent.com/dolev31/lasagna/v0.1.0/constraints/0.1.0.txt
```

## Extras

The core install covers ingest, embedding, search and serving. Add an extra for
anything else.

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
(`OPENAI_API_KEY`). Each raises an error that names what is missing.

`all` is large. pip takes about two minutes to resolve it (measured with pip
26.2), and `uv pip install "lasagna-rag[all]"` a few seconds.

## An LLM for the generation track

Retrieval comparisons need no LLM. The generation track, the judged metrics and
LLM-based components (HyDE, an LLM reranker, an agentic pipeline) do.

With no LLM configured, Lasagna uses the first API key it finds in the
environment, in the order of the
[provider table](../configuration.md#llm-providers-and-models), with that
provider's default model. Set `MODEL_NAME` to pick another model on that
provider.

- The core install talks to OpenAI, Azure OpenAI and OpenAI-compatible
  endpoints.
- Anthropic, Google, Mistral, Cohere and NVIDIA need the `providers` extra.
- A local Hugging Face model needs the `local` extra.
- Groq, watsonx and llama.cpp need their LangChain package.

In an experiment, the answers come from `RAG_EVAL_GENERATOR_MODEL`, else
`MODEL_NAME`, else the provider default. The judge is separate:
`RAG_EVAL_JUDGE_MODEL`, else `judge_model:` in the sweep file, else
`gpt-6-luna`. To write your own provider, see the
[providers package](https://github.com/dolev31/lasagna/blob/main/src/lasagna/providers/README.md).

## Platform notes

On Linux x86_64, `pip` installs the CUDA build of torch, about 3.2 GB of
downloads. For CPU only, install torch from the PyTorch CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lasagna-rag
```

On Apple Silicon, model calls on the GPU (MPS) run one at a time, because two at
once crash Metal. Concurrent sweep arms wait for each other there. To run
everything on the CPU instead, set `RAG_USE_GPU=false`. The install weighs about
1.3 GB on macOS arm64.

`import lasagna` loads no torch, transformers, sentence-transformers or
pymilvus. Those arrive only when something needs them, so a remote-backend
client stays light.

## Telemetry

Lasagna collects no usage data. ragas, the library behind the generation
metrics, sends its authors a usage event (its version, the LLM provider and
model name, and a random id) each time a judge is built. Importing `lasagna`
turns that off by setting `RAGAS_DO_NOT_TRACK=true`, unless you have already set
it. To send ragas those events, set `RAGAS_DO_NOT_TRACK=false` before importing
`lasagna`. ragas still writes its random id to a file in your user data folder,
but with tracking off it sends nothing.

## Check the install

```bash
lasagna doctor
```

`lasagna doctor` checks the install and names what is missing. From a source
checkout, `python -m e2e.run` measures the cold start (the `e2e` package is not
in the wheel).

## Next

[Quickstart](quickstart.md) uploads a file and searches it.
