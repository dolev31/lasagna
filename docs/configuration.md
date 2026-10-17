# Configuration

Which settings are in force, where Lasagna keeps its data, and which LLM it
calls. Every setting, with its type, default and environment variable, is in
[the settings reference](reference/settings.md), which is generated from the
code.

To see what applies to a session right now:

```bash
lasagna doctor -s my-corpus      # session, profile, settings file and data directory in force
```

## Where a setting comes from

A setting is looked up in this order, and the first place that has it wins:

| Order | Source | Set by | Scope |
|---|---|---|---|
| 1 | Per-query overrides | the evaluation harness, for each arm of a sweep | one query |
| 2 | Session overrides | `lasagna configure`, `lasagna import`, `lasagna init`, the REST config routes, the MCP config tools, the configurator page | one session |
| 3 | Environment variables | `LASAGNA_<NAME>` or `RAG_<NAME>` | the process |
| 4 | The settings file | the packaged `settings.lasagna.toml`, or the file `LASAGNA_SETTINGS_FILE` (or `RAG_SETTINGS_FILE`) names | the process |
| 5 | Built-in defaults | the code | the process |

So the settings file and the environment set the defaults for a process, and
the session overrides that `lasagna configure` writes win for that session.
`lasagna configure --profile speed -s A` changes session `A` only; another
session, and a new one, still start from the settings file and the
environment. A profile sets only the settings that define it (for `speed`, the
MiniLM model, no reranking and the rest of its list); the vector backend and
every other setting it does not name still come from the environment and the
settings file.

**What a session can set.** Session overrides hold retrieval and generation
settings: the ones that reach a pipeline stage (chunking, embedding model,
retrieval, query enhancement, reranking, agentic retrieval, generation) and how
results are cited. Where data lives and how the server reaches it (the vector
backend and its connection settings, the collection, index settings, paths,
URLs, credentials, limits) are the process's, from the environment and the
settings file. Every surface that writes session overrides refuses those keys,
and keys that are no setting, with an error naming each one; a value equal to
the one already in force is dropped instead, so an exported configuration
imports again. A model named by a path or URL is refused the same way.

**Where session overrides are kept.** In the metadata database (SQLite under
the data directory, or PostgreSQL when `database_backend` is `postgresql`), for
every surface: the `lasagna` command, `lasagna-mcp`, `lasagna serve` and Python
code. A profile set by one command applies in the next one, in a running
server and in a `KnowledgeClient` for that session; the compose stack's api
workers, Celery worker and MCP server read one session's settings alike.
`RAG_SESSION_BACKEND=memory` keeps a process's sessions in memory, for the life
of the process (tests, or a notebook that should leave saved sessions alone).

**From Python**, a `KnowledgeConfig` passed to `KnowledgeClient` splits the
same way: the fields that choose what the process connects to (vector backend,
database, cache, embedding model) are written where environment variables go,
before the backend starts, and apply to the whole process; the search and
chunking fields become session overrides for the client's session. The
[SDK configuration page](../src/lasagna/sdk/docs/configuration.md) lists which
is which.

**Environment variables.** Every setting reads from `LASAGNA_<NAME>` and from
`RAG_<NAME>`, in capitals (`LASAGNA_CHUNK_SIZE=512`). When both are set, the
`LASAGNA_` one wins. A value that does not parse, is out of range or is not one
of the allowed values stops the process with an error naming the variable, its
value and what it takes. A `RAG_` name that is no setting prints one warning
with the closest setting. An empty value counts as unset. `on`, `off`, `yes`,
`no`, `1` and `0` are booleans.

The log level is the `log_level` setting, `LASAGNA_LOG_LEVEL` or
`RAG_LOG_LEVEL`: the console level of the `lasagna` command (`WARNING` unless
set), the MCP server and the REST server. A value it does not take warns and
is ignored, so a typo never stops `lasagna doctor`.

A few `LASAGNA_` variables are not settings: `LASAGNA_DATA_DIR`,
`LASAGNA_DB_DIR` and `LASAGNA_UPLOAD_DIR` (below), `LASAGNA_SETTINGS_FILE`,
`LASAGNA_LOG_FORMAT=json`, `LASAGNA_MCP_TOOLS`,
`LASAGNA_LLM_CACHE=0` (turns the [experiment cache](guide.md#re-running-a-sweep-replays-llm-calls)
off) and `LASAGNA_STRICT_PLUGINS=1`.

**`.env` files.** Lasagna does not load `.env` into the environment. The one
exception: the `lasagna` command reads `RAG_SETTINGS_FILE` from `./.env` in
the working directory when the variable is not already set, because that is
where `lasagna init` records the settings file it wrote. Python code, other
directories and `lasagna-mcp` do not; there, `export RAG_SETTINGS_FILE=...`
first. API keys are never read from `.env`: export them in your shell.

## Profiles

A profile is a set of session overrides with a name. `lasagna configure` lists
them and `lasagna configure --profile NAME -s SESSION` applies one. There are
five: `default`, which is the settings file as it stands, and four presets.

| Profile | What it runs | Downloads on first use | Calls an LLM during search |
|---|---|---|---|
| `default` | the settings file: BGE-M3, hybrid search, reranking with mxbai-rerank-large-v1, MMR, query routing with agentic retrieval | 3.2 GB | for queries routed to agentic retrieval |
| `speed` | dense vector search only, all-MiniLM-L6-v2; no reranking, hybrid, MMR, HyDE or agentic loop | 92 MB | never |
| `standard` | hybrid search, reranking with mxbai-rerank-large-v1, MMR, query routing with agentic retrieval, BGE-M3 | 3.2 GB | for queries routed to agentic retrieval |
| `balanced` | `standard` plus hierarchical chunking, the structured agentic graph with a hallucination check, and adaptive routing | 3.2 GB | for queries routed to agentic retrieval |
| `max_quality` | `balanced` plus HyDE, query expansion, RAG fusion, CRAG, an answer quality check and contextual retrieval (an LLM call per chunk at ingest); every query goes to the agentic loop | 3.2 GB | on every search and every upload |

The sizes are the model weights, which `lasagna init` also prints. With query
routing on, each query is classified by rules: a short factual one runs vector
search only, most run the profile's pipeline, and one judged complex goes to the
agentic loop, which calls an LLM. Without an API key such a query runs the
profile's pipeline instead, and the result names `agentic` in
`degraded_stages`; the MCP `search` tool does the same for a session that sends
every query to the agentic loop. `speed` is the
profile that never calls an LLM and downloads no large model. A benchmark run
picks its profile with
`lasagna benchmark run --profile`, or `base_profile:` in a sweep file.
`lasagna benchmark run` starts from `speed` unless told otherwise, and a sweep
file without `base_profile:` starts from `standard`.

## Where data lives

Everything a user stores is under one data directory, `~/.lasagna`: the
metadata database and the Milvus Lite store in `dbs/`, uploaded files in
`uploads/`, experiment results in `research/results/` (for a sweep without
`output_dir:`), and the experiment LLM cache in `cache/llm.sqlite`.

| Variable | Default | What it places |
|---|---|---|
| `LASAGNA_DATA_DIR` (or `RAG_DATA_DIR`) | `~/.lasagna` | the whole data directory |
| `LASAGNA_DB_DIR` (or `RAG_DB_DIR`) | `<data dir>/dbs` | the metadata database and the Milvus Lite store |
| `LASAGNA_UPLOAD_DIR` (or `RAG_UPLOAD_DIR`) | `<data dir>/uploads` | uploaded files |

`LASAGNA_DATA_DIR` moves the results and the cache with the rest;
`llm_cache: {path: ...}` in a sweep puts the cache file elsewhere.

Milvus Lite admits one process per store. When another process holds the
default one (a running `lasagna serve`, an MCP server, or another benchmark),
`lasagna benchmark run` and `lasagna benchmark experiment` use a store of
their own in `<data dir>/bench/dbs` and say so. Benchmark corpora are ingested
there once and reused by later runs that find the default store held; a run
that finds it free uses `dbs/` as before. A store chosen with `LASAGNA_DB_DIR`
or a `db_path` is never moved: the run stops before it starts and says what to
do. An experiment started from Python stops the same way.

Before 0.1.0 the data directory was `~/.rag_knowledge`. The first time the
default is used and only `~/.rag_knowledge` exists, it is renamed to
`~/.lasagna`. When both exist nothing is moved or deleted, and a warning gives
the `mv` command, or `LASAGNA_DATA_DIR=~/.rag_knowledge` to keep using the old
one.

**The metadata store records its version.** SQLite and PostgreSQL stores keep
a store version, and a Lasagna that finds a store written by a newer version
refuses to open it, naming both versions. Upgrade the package, or point
`LASAGNA_DB_DIR` (or the PostgreSQL DSN) at another store. Downgrading over a
store is therefore not possible. The deployment guide covers
[upgrading a running stack](../src/lasagna/deploy/README.md).

## One collection per embedding model

Vectors from two embedding models cannot be compared, even at the same size
(`BAAI/bge-large-en-v1.5` and `BAAI/bge-m3` both produce 1024 dimensions). So
each model writes to a collection of its own, named
`rag_documents_v2__<model>__<dimensions>`, for example
`rag_documents_v2__all_minilm_l6_v2__384` for the `speed` profile.

- Setting `embedding_model` without `embedding_dim` (in the environment, a
  session or `KnowledgeConfig`) uses that model's own size, read from a table
  of common models or from the model's files in the Hugging Face cache. When
  neither has it, an upload is refused before anything is stored, naming the
  setting to add.
- Sessions on the same model share its collection, and each session sees only
  its own documents in it.
- Switching a session to a profile with another model makes it search that
  model's collection, so a session that holds documents is refused the switch,
  even at the same embedding size: none of its documents would be found.
  Use a new session for the new model, or switch and reindex in one step:
  `lasagna configure --profile <name> -s <session> --reindex`
  (`reindex=true` on `PUT /api/v1/config/profiles/{name}` and
  `PUT /api/config/knowledge/active`, and on the MCP tool
  `set_active_profile`). It first checks that the new profile's models load
  (downloaded, or downloadable here), then reindexes every document with the
  new profile while the session stays on the old one, and switches only once
  all of them are reindexed. If one fails, the documents already moved are
  moved back, the session keeps its profile, and the error names the document
  and why. `lasagna doctor -s <session>` reports documents that sit in another
  model's collection as blocking.
  To switch without moving them, use
  `lasagna configure --profile <name> -s <session> --force` (`force=true`
  over REST and MCP), which also checks that the new models load, and reindex
  the documents with the new model afterwards:
  `client.documents.reindex(document_id)` in the SDK,
  `POST /api/v1/documents/{id}/reindex` (or `POST /api/knowledge/reindex` for
  all of a session's documents) over REST, or the MCP tool
  `reindex_document`. A reindex moves each document to the new model's
  collection and keeps answering from the old one until the new chunks are
  stored; one that fails leaves the document as it was. Until then the
  documents stay where they were, and a session on the old model still finds
  them. Uploading a file again also indexes it for the new model: an upload
  is a duplicate only of a copy in the collection the session searches now.
- A `collection_name` you set yourself (in the settings file or as
  `RAG_COLLECTION_NAME`) is used as given. The evaluation
  harness names one per sweep arm this way, so sweeps are unaffected.
- Each collection records the model and size it was created with. Opening it
  with another model is refused with a message naming both models and the
  command that switches back, and nothing is changed.

A store from before 0.1.0 has one collection, `rag_documents_v2`, shared by
every model, and nothing recorded which model wrote it. Name that model with
`LASAGNA_LEGACY_COLLECTION_MODEL` (for example `BAAI/bge-m3`), and that model
keeps using the collection under its old name; any other model gets its own.
Until then a model of the same size logs a warning saying so, and
`lasagna status` lists the collection.

## LLM providers and models

With no LLM configured, Lasagna uses the first of these API keys it finds in
the environment, in this order, with that provider's default model:

| Provider | Key | Default model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| Mistral | `MISTRAL_API_KEY` | `mistral-large-latest` |
| Together | `TOGETHER_API_KEY` | `meta-models/Muse-Glimmer-30B` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` |
| Google | `GOOGLE_API_KEY` | `gemini-3.6-flash` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `gpt-5.6-luna` (a deployment name) |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-5.5` |
| Fireworks | `FIREWORKS_API_KEY` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| Cohere | `COHERE_API_KEY` | `command-a-03-2025` |
| NVIDIA | `NVIDIA_API_KEY` | `meta/llama-3.1-70b-instruct` |
| Perplexity | `PERPLEXITY_API_KEY` | `sonar` |
| watsonx | `WATSONX_APIKEY` (or `WATSONX_API_KEY`) | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |
| Ollama | `OLLAMA_BASE_URL` | `llama3.2` |
| vLLM | `VLLM_BASE_URL` | the model the server loaded |
| llama.cpp | `LLAMACPP_MODEL_PATH` | the file it names |
| Hugging Face (local) | `HF_MODEL_ID` | the model it names |

`MODEL_NAME` picks another model on that provider. The core install talks to
OpenAI, Azure OpenAI and every OpenAI-compatible endpoint (Together, DeepSeek,
OpenRouter, Fireworks, Perplexity, Ollama, vLLM). Anthropic, Google, Mistral,
Cohere and NVIDIA need the `providers` extra
(`pip install "lasagna-rag[providers]"`), a local Hugging Face model needs the
`local` extra, and Groq, watsonx and llama.cpp need their LangChain package
(`langchain-groq`, `langchain-ibm`, `langchain-community` with
`llama-cpp-python`), which no extra installs. Writing your own provider, or
configuring one in code, is covered in
[the providers package](../src/lasagna/providers/README.md).

**In an experiment**, the answers come from `RAG_EVAL_GENERATOR_MODEL`, else
`MODEL_NAME`, else the provider default above. The judge is separate:
`RAG_EVAL_JUDGE_MODEL`, else `judge_model:` in the sweep file, else
`gpt-6-luna`. The run metadata and the comparability key name the judge, so
results judged by different models never share a leaderboard group.

## Temperature and repeatable runs

Lasagna asks for temperature 0 wherever it wants a repeatable answer, and
builds each request for the model it goes to, because current models do not
all accept a temperature:

| Model | What is sent |
|---|---|
| Claude 4.7 and later (`claude-sonnet-5-5`, `claude-haiku-5-5`, ...) | no `temperature`, `top_p` or `top_k`; the API refuses them |
| Claude 4.6 and earlier | `temperature` as asked |
| OpenAI reasoning models that allow it (`gpt-6-luna`, `gpt-6-sol`, `gpt-5.6-luna`, ...) | `reasoning_effort: "none"` with the temperature |
| OpenAI reasoning models that do not (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-5`) | no `temperature` |
| Any other model | `temperature` as asked |

Setting `reasoning_effort` in the model settings to anything but `"none"`
means no temperature is sent. The RAGAS judge's requests are built the same
way, at the 0.01 RAGAS asks for.

A model that is sent no temperature answers differently from run to run, and
Lasagna says so rather than claim otherwise: `lasagna doctor` and the start of
an experiment print one line naming the model, the run metadata records
`sampling: provider-default` for the generator or the judge, and the
experiment cache never replays its answers. Repeat runs (`n_runs: 3` or more)
to measure that spread. Retrieval with fixed models does not sample, so a
retrieval-only sweep needs one run.
