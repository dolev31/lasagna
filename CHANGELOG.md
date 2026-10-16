# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

The first public release. Eight swappable stages and 55 built-in components,
each one that implements a published method naming its paper; an evaluation
harness with paired significance tests, interaction tests and checkable result
bundles; and the same pipeline served over MCP, REST and a Docker Compose stack.

Its metric definitions are `METRICS_VERSION` 1.5: NDCG with linear gain and
zero gain for a negative grade (as `trec_eval`), cutoff metrics over `k`
documents, exact match and token F1 normalized in the official SQuAD order with
`[n]` citation markers removed and abstention counted only when an answer opens
with one, and `factual_correctness` among the judged metrics
([metrics.md](src/lasagna/knowledge_evaluation_sdk/docs/metrics.md) defines
each). The version is part of every comparability key, so results from the
internal 0.1.0a1 pre-release (1.1) or from development builds since (1.2 to
1.4) never share a leaderboard group with results from this release.

### Added

Components, by stage:

- **Generator, an eighth stage.** How the answer is written from the
  retrieved passages can be swapped with everything else held fixed.
  Built-ins: `concise` (the harness's own prompt, word for word), `cited`
  (ALCE), `chain_of_thought` and `chain_of_note`. A function
  `f(query, contexts) -> str` becomes a generator with `@lasagna.generator`.
  Sweep it with `pipeline.generators`, run one with
  `lasagna benchmark run --generator`, or call `KnowledgePipeline.generate()`.
  A run that names no generator writes answers with `concise`.
- **Baseline arms.** The `closed_book` generator answers from the question
  alone; `long_context` reads all ten retrieved passages where the default
  reads five, and the judge sees the same ten. An arm with `no_rag: true`
  retrieves nothing: its retrieval track is skipped, it answers with
  `closed_book`, and the isolation check files it under the retriever stage.
  An arm whose generator reads no passages is not judged on faithfulness or
  the context metrics, and the report and leaderboard say so.
  `BaseGenerator.needs_context` and `context_k` let a component declare either.
- **Retriever:** `dense`, `bm25`, `splade`, `hybrid_rrf` and `hybrid_convex`.
  `dense` is the default vector search as a component, the control for a
  retriever sweep. Settings: `bm25_k1`, `bm25_b`, `splade_model`,
  `hybrid_fusion_depth` and `hybrid_convex_normalization`.
- **Reranker:** `monot5`, `llm_pointwise`, `flashrank` and `cohere`.
  `cross_encoder` loads current Hugging Face rerankers by name, and
  `reranker_trust_remote_code` allows checkpoints that ship their own code.
- **Query enhancer:** `step_back`, `query2doc`, `rewrite` and `multi_query`.
- **Agentic pipeline:** `self_ask`, `ircot`, `flare`, `adaptive` and `self_rag`,
  each with its paper's defaults as settings (`self_ask_*`, `ircot_*`,
  `flare_*`, `adaptive_rag_*`, `self_rag_*`). `adaptive` and `self_rag` prompt
  a general LLM where the papers train a model. What each loop did is recorded
  in `loop_trace`.
- **Chunker:** `fixed`, `recursive`, `sentence_window`, `markdown_header`,
  `proposition` and `contextual`. `chunk_unit` measures `fixed` and `recursive`
  in characters, words or tokens. `chunking_strategy: recursive` runs the
  recursive algorithm. `semantic` gains Kamradt's breakpoint methods
  (`semantic_breakpoint_method`); the default, `absolute`, is unchanged.
- **Embedder:** `sentence_transformers`, `bge_m3`, `openai`, `cohere` and
  `voyage`. Query and document prefixes follow each model family's model card.
- **Document processor:** `docling`, `markitdown` and `pymupdf4llm`, each with an
  extra of the same name. `pymupdf4llm` is AGPL-3.0 and is not part of `all`.
- **LangChain and LlamaIndex components as stages.** `lasagna.from_langchain`
  registers a document compressor as a reranker, embeddings as an embedder and
  a retriever as a retriever; `lasagna.from_llamaindex` does the same for a node
  postprocessor, an embedding and a retriever. A sweep file can also name such
  an object as `module:attr`. The listing describes the entry as
  `wrapped: langchain.<Class>`. A wrapped retriever searches its own index, not
  the ingested corpus. New extra: `llamaindex`.
- `lasagna component list [stage]` lists every component a sweep can name, with
  its stage, description and paper, without loading any of them.

Experiments:

- Chunker and embedder components sweep. Each arm's corpus is chunked or
  embedded by its own component, in a corpus of its own; an embedder also gets
  its own collection. Every chunking, embedding and document-processing setting
  gets its own corpus in a sweep. Document processors do not sweep, since
  benchmark corpora are loaded as text.
- **Your own data as a benchmark, with no adapter.** A JSONL or CSV file, or a
  BEIR folder, runs with `benchmarks: [{path: data/my.jsonl}]` in a sweep or
  `lasagna benchmark run ./my.jsonl`. It registers as
  `local/<file name>@<12 hex digits of its sha256>`, so an edited file never
  resumes the old file's checkpoints or shares its leaderboard group.
  `lasagna benchmark validate-data` names each bad row and field and says which
  tracks the data supports. See `docs/own-data.md`.
- **`lasagna benchmark synthesize`** writes factoid, multi-hop and unanswerable
  questions from your documents with the configured LLM, checks each one with a
  second call and drops those that fail. Each row records its prompt's sha256,
  the model and the seed; `--dry-run` prints the plan and a token estimate
  without calling the LLM. For development, not for reported results.
- **A re-run replays the LLM responses it already has.** Inside an experiment,
  a request at temperature 0 that an earlier run made with the same seed is
  answered from `~/.lasagna/cache/llm.sqlite`. Token counts read as in the cold
  run; `llm_cache_hits` and `llm_cache_misses` on each task and a hit rate in
  `results.json` and the bundle say what was replayed. Sampled calls are never
  replayed, which includes the RAGAS judge (it asks for temperature 0.01) unless
  `llm_cache: {deterministic_only: false}`, and a call to a model that is sent
  no temperature. Off with `llm_cache: false`, `--no-cache` or
  `LASAGNA_LLM_CACHE=0`. Nothing is cached outside an experiment, so
  `lasagna serve` writes no prompt to disk.
- **`lasagna benchmark experiment --dry-run`** prints the arms, the tasks, the
  models the run loads and whether each is downloaded, a static token estimate
  and, unless `--no-probe`, a 3-sample probe per benchmark extrapolated to the
  whole run in tokens, seconds and (from a price table) dollars. Every number
  is labeled an estimate, and nothing is written under the results folder.
- **A dollar estimate, if asked for.** `prices: builtin` (a table dated
  `builtin-2026-10`, with Claude Fable 5.1, Haiku 5.5, gpt-6-luna and
  gpt-6-sol among its rates) or `prices: {model: {input_per_1m: ...,
  output_per_1m: ...}}` adds `cost_usd_estimate` to each task whose tokens all
  have a rate; the run metadata records the table and rates. A rate that holds
  only for shorter prompts (Haiku 5.5 up to 100,000 tokens, gpt-6-luna and
  gpt-6-sol up to 272,000) gives no estimate for a task whose calls average
  longer. Without `prices:` a run reports tokens only.
- A sweep key the harness does not read is an error that names the likely
  intended key, at the top level, under `pipeline:`, in a factorial, in a
  benchmark entry and in an arm. An unknown profile is refused before anything
  is ingested.
- `--resume` resumes the newest run of the same experiment, says which, and
  refuses when there is nothing to resume.
- `lasagna component validate` imports `module:function` from the working
  directory, as a sweep does, and takes `--stage` for an undecorated function.
  It checks that a reranker, embedder or generator leaves its input unchanged.
- New metrics, each defined in `metrics.md`: R-precision, complete (all-gold)
  recall@k, ROUGE-L, answer containment, answer similarity, ALCE citation
  recall and precision, abstention rate, SQuAD v2 scores split into answerable
  (`has_ans`) and unanswerable (`no_ans`) questions, and input tokens, output
  tokens and LLM calls per query. The judge is also asked for
  `factual_correctness`.
- `lasagna --version` prints the installed version.

Results:

- **`lasagna results significance` ends in a verdict.** A green box names each
  pair that survives Holm correction, which side won (lower-is-better metrics
  included) and by how much; when nothing survives, a yellow box says so and
  points at `lasagna results power`.
- **`lasagna results view`** writes a run (a `results.json` or a bundle) as one
  HTML page: the header with versions and run id, the stage isolation check,
  and per benchmark the leaderboard, the significance table with the verdict
  `results significance` prints, every metric per arm, and a per-query diff
  between two arms (`--diff A B`, default the top two) with what each one
  retrieved. The page is a single file with no external requests and works
  offline, in light and dark mode.
- `lasagna results leaderboard --format latex` prints one booktabs table per
  comparability group, or writes one `.tex` file per benchmark with `-o`.

Servers and deployment:

- **`KnowledgeClient.from_url` failed against a server with auth on, and its
  `close()` raised `AttributeError`.** It never sent `X-Session-Token`, so
  `RAG_REQUIRE_AUTH=true` answered 401 to every call. With `api_key` (a key
  from `RAG_API_KEYS`) it mints a session token on the first 401;
  `session_token=` sends one minted already.
- **The Docker image had no Qdrant, Weaviate or Pinecone client**, so the
  stack built with those fragments never passed its readiness check. The
  Dockerfile's `VECTOR_BACKEND` build argument installs the client (and fails
  the build if it does not import), the three fragments set it, and `make
  validate` refuses an existing image built for another backend.
- **Without the `celery` profile, uploads waited for a worker that never
  ran.** Compose set `RAG_CELERY_ENABLED=true` whatever the profiles, and
  with Redis reachable the api queued every upload. `make` now derives the
  setting from `COMPOSE_PROFILES`, `make validate` refuses `true` without the
  profile, and the api queues an upload only when a worker answers on the
  broker, otherwise processing it in its own threads and logging why. GC is
  scheduled inside the api in the same case; it was left to a Beat that was
  not running.
- **Orphan repair deleted the vectors of documents it should keep.** Detection
  counted only `ready` rows, so a document still uploading, processing or
  indexing, one in `error` and an archived one were vector-only orphans, and a
  repair (the Celery scan with `dry_run=False`, `POST /admin/gc/repair`)
  deleted their vectors. A vector is an orphan only when no row in any status
  owns it, and repair checks again before deleting.
- **Search responses name the stages that fell back.** `degraded_stages` lists
  `hyde`, `graphrag`, `hybrid`, `query_expansion` or `agentic` when that stage
  failed and the search carried on with less; each fallback is also counted in
  `lasagna_degraded_fallbacks_total{stage}`.
- Prometheus counters `lasagna_llm_calls_total{outcome}` and
  `lasagna_ingests_total{outcome}`; the request histogram gains a `status`
  label.
- Every log line written during a request carries its request id, and
  `LASAGNA_LOG_FORMAT=json` writes one JSON object per line.
- `GET /api/v1/health/ready` pings the vector store, the database and Redis,
  each with a 2 s timeout, and lists what did not answer in `failing`.
- `/api/v1` and the MCP tool schemas are under the deprecation policy in
  `GOVERNANCE.md`, recorded in `tests/snapshots/openapi.json` and
  `tests/snapshots/mcp_tools.json`. Errors are documented as `ErrorResponse`.
- CI builds the image and runs the compose stack (Postgres, Redis, Qdrant,
  Celery worker and beat, MCP) with MiniLM embeddings, then runs
  `tests/live/test_deployment.py` against it. It also runs `pip-audit` on the
  core dependencies, `mypy` on the SDK, `components.py` and
  `core/base_components.py`, and prints the reason for every skipped test.

Documentation and examples:

- `docs/configuration.md`: which of the session, the environment and the
  settings file wins, the profiles, where data lives, one collection per
  embedding model, LLM providers and their default models, and which requests
  can be pinned to temperature 0.
- `docs/reference/settings.md` lists every setting with its type, allowed
  values, default, environment variables and description. It is generated by
  `scripts/gen_settings_doc.py`, and a test fails when it falls behind.
- **A Colab notebook**, `examples/notebooks/compare_your_reranker.ipynb`: a
  one-function reranker against the built-in cross-encoder on 100 BEIR SciFact
  queries over the full corpus, ending in the significance test, the isolation
  check and a validated bundle. No API key.
- **A BEIR reproduction**, `examples/reproduction/`: BM25, two dense models,
  hybrid RRF and BM25 with a cross-encoder on the full SciFact and NFCorpus test
  sets. The dense models land within 0.002 NDCG@10 of MTEB, BM25 with the
  cross-encoder within 0.004 of the BEIR paper, BM25 within 0.008.
- A terminal demo, `docs/assets/demo.gif`, regenerated from a real run by
  `docs/assets/demo/record.sh`.
- `examples/acceptance/offline_demo.py` takes the features above through the
  CLI in one run, offline, with a stub LLM and judge and MiniLM embeddings:
  synthesize, validate, wrap a LangChain compressor, dry-run, sweep the
  baseline arms, replay from the cache, bundle, view, LaTeX and the MCP safe
  set. It takes about 20 s, and the release gate (`python -m e2e.run`, phase
  p4) runs it.

### Changed

Defaults and naming:

- **One vector collection per embedding model.** The default collection name is
  `rag_documents_v2__<model>__<dim>`, so sessions on different profiles never
  share vectors. A name set explicitly (settings file, `RAG_COLLECTION_NAME`, a
  session) is used as given; the evaluation harness's per-arm names are
  unchanged. Each collection's model is recorded in a `collection_meta`
  table, and opening a collection with another model, even at the same size
  (bge-large and bge-m3 are both 1024), is refused with the command that
  switches back. An existing `rag_documents_v2` that holds documents is kept
  under its old name by the first model that opens it at its size; one at
  another size is left in place, and `lasagna status` names it.
- **Data lives in `~/.lasagna`.** The default data directory was
  `~/.rag_knowledge`. `LASAGNA_DATA_DIR` names another, and `LASAGNA_DB_DIR`
  and `LASAGNA_UPLOAD_DIR` place its parts; the `RAG_` names still work. An
  existing `~/.rag_knowledge` is renamed on first use when `~/.lasagna` does
  not exist. When both exist nothing is moved or deleted, and a warning gives
  the `mv` command.
- **Every setting has a `LASAGNA_` variable.** `LASAGNA_<NAME>` sets the same
  setting as `RAG_<NAME>` and wins when both are set; `RAG_` names keep
  working. `LASAGNA_SETTINGS_FILE` names the settings file.
- **A bad environment setting stops the process.** A `RAG_*` or `LASAGNA_*`
  value that does not parse, is out of range or is not one of the allowed
  values raises `ConfigError` naming the variable, its value and what it takes,
  instead of being logged and ignored. A `RAG_*` name that is no setting warns
  once with the closest one. `on` and `off` are booleans, an empty value means
  unset, and settings only the settings file defined (`use_agentic_rag`,
  `max_agent_iterations` and others) can be set from the environment.
- **Default models are current.** Defaults the providers retired or closed to
  new accounts were replaced with the model each provider names: OpenAI
  `gpt-6-luna`, Anthropic `claude-sonnet-5-5`, Azure `gpt-5.6-luna`, Google
  `gemini-3.6-flash`, Groq `openai/gpt-oss-120b`, Cohere `command-a-03-2025`,
  OpenRouter `anthropic/claude-sonnet-5.5`, Together
  `meta-models/Muse-Glimmer-30B` and DeepSeek `deepseek-flash`. `MODEL_NAME`
  and explicit settings work as before. The other defaults are still served
  and did not change.
- **The judge defaults to gpt-6-luna.** With no `judge_model:` or
  `RAG_EVAL_JUDGE_MODEL`, RAGAS used to pick GPT-4o mini (shut down on
  2027-01-14) on a client the harness did not meter. The default judge is
  built by the harness, so its tokens are counted, and `judge_model` in the
  run metadata and the comparability key names it. A retrieval-only run
  records no judge.
- The preset profiles name the chunker they run: `structure_aware` for `speed`
  and `standard`, `hierarchical` for `balanced` and `max_quality`. They said
  `semantic`, and each upload warned that it was overridden. The chunks are the
  same; with the structural settings turned off, both mean `semantic`.
- Session tokens are issued as `lasagna`; tokens issued as `rag-knowledge`
  before this release stay valid until they expire.
- Every title, banner, help text and guide that said "RAG Knowledge" says
  Lasagna, and the MCP server announces itself as `lasagna` (it was
  `rag_knowledge`). Clients name the server in their own config, so existing
  configs keep working.
- **`lasagna.__all__` is the public API: 20 names.** The component decorators,
  `register`, `registry`, `RAGConfig`, `KnowledgeClient`, `KnowledgeConfig`,
  `Experiment`, `ExperimentConfig`, `RAGBackend`, `get_rag_backend`,
  `create_rag_backend`, `VectorStoreAdapter`, `register_backend` and
  `__version__` (see Deprecated for the rest).
- Listing components no longer imports them: built-ins declare that they
  implement their stage's interface. `registry.list_*()` and
  `lasagna plugins list` load no model code, and `lasagna-mcp --help` takes
  under half a second (it took 8).

Metric definitions and results (`METRICS_VERSION` 1.5):

- An answer counts as an abstention only when it opens with one, so a hedge
  later in the answer no longer zeroes exact match and token F1. Answers are
  normalized in the official SQuAD order.
- Exact match and token F1 score the answer with its `[n]` citation markers
  removed, as ALCE does. NDCG gives a negative relevance grade zero gain, as
  `trec_eval` does. The judged set adds `factual_correctness`, and the
  answer-similarity model is part of the comparability key.
- `llm_listwise` runs RankGPT's sliding window: windows of 20 passages, moved by
  10 from the bottom of the list to the top. Its window default was 10.
- `query_decomposition` is least-to-most prompting's decomposition (Zhou et al.
  2022). Results recorded under that name before came from another prompt.
- BGE-M3 no longer gets the BGE v1.5 query instruction, which it was not
  trained with. This changes BGE-M3 query vectors on the sentence-transformers
  backend.
- `max_content_length` is a chunker setting: a sweep can vary it, each value
  gets a corpus of its own, and the chunker's stage digest includes it.
- **`results.json` records what its numbers were computed under:** the metrics
  version and package version (also at the top of the file), the settings the
  environment overrode (credentials and hosts redacted) and the Hugging Face
  commit of each model the tasks ran, read from the local cache. `determinism`
  says `seeded_per_task` when tasks ran one at a time and
  `seeded_global_concurrent` when they shared the process-wide generators; it
  said `seeded` for both.
- **Checkpoints from before this release do not resume.** The resume key
  includes a hash of the metrics version, package version, judge model and
  environment settings, so `--resume` never restores a result computed under
  other ones. A run resumed after upgrading starts its tasks again.

Command line:

- `-s` means `--session` in every command; `benchmark run` spells out
  `--split` and `--mode`. `benchmark run` measures retrieval unless `--tracks`
  names generation or `--generator` is given (generation needs an API key).
  `benchmark experiment` ends with the leaderboard, as `benchmark run` does,
  and says when every run of every arm was identical, so `n_runs: 1` would
  give the same answer.
- `lasagna serve` logs warnings only (`-v` or `LASAGNA_LOG_LEVEL=INFO` for
  more), prints the default session's profile, embedding model, vector store
  and data directory, and opens a browser only with `--open`.
- The examples that measure retrieval only (the README, the guide, the plugin
  template's `sweep.yaml` and the skills) run `n_runs: 1`; the ones with an
  LLM in the loop keep `n_runs: 3`.

Servers and deployment:

- **The MCP server speaks stdio by default and serves no deletion tools.**
  `lasagna-mcp` with no arguments uses stdio, which is how Claude Code, Claude
  Desktop and Cursor launch a server; `--transport sse` serves a port, and
  `--host` or `--port` without it is an error. The default tool set, `safe`,
  leaves out `delete_document`, `clear_session`, `bulk_delete_session`,
  `rebuild_collection`, `delete_rag_profile` and `reset_rag_config`;
  `--tools all` or `LASAGNA_MCP_TOOLS=all` serves them. Tool names,
  descriptions and input schemas are recorded in
  `tests/snapshots/mcp_tools.json`.
- **The compose stack defaults to a Milvus server.** `.env.example` sets
  `VECTOR_BACKEND=milvus`, which works with the default `celery,mcp` profiles;
  `make validate` refuses `milvus-lite` with either profile and requires
  `POSTGRES_PASSWORD`. Every long-running service has its own healthcheck, so
  `docker compose ... up --wait` returns when the stack can serve. The api runs
  gunicorn with `--preload` and honors `FORWARDED_ALLOW_IPS` (or
  `RAG_FORWARDED_ALLOW_IPS`). The models baked into the image are build
  arguments (`EMBEDDING_MODELS`, `RERANKER_MODELS`, `OPTIONAL_MODELS`).
- **With `RAG_REQUIRE_AUTH` on, creating a session needs an API key.** Send
  `X-API-Key` with one of the keys in `RAG_API_KEYS`; with none configured,
  `POST /api/v1/sessions` returns 503. Rate limits are counted per key rather
  than per session. With auth off nothing changes.
- **Request bodies are limited before they are read.** Uploads over
  `RAG_MAX_FILE_SIZE`, batches over `RAG_MAX_BATCH_BYTES` (200 MiB) and other
  bodies over 1 MiB get 413; a batch of more than `RAG_MAX_BATCH_FILES` (20)
  files gets 400.
- On any networked MCP transport, with or without auth, `upload_document` reads
  server-side paths only inside `RAG_MCP_UPLOAD_ROOT`.
- A search whose focused-document list cannot be read returns 503 instead of
  searching every document in the session, and listing a document's chunks
  while the vector store is unreachable returns 503 instead of an empty list.
- **Shutdown no longer leaves documents "processing".** The api waits up to
  `processing_timeout` (`RAG_PROCESSING_TIMEOUT`, 20 s in compose) for its
  processing threads, then marks unfinished documents failed with the reason.
  When every thread is busy, a REST upload gets 503 with `Retry-After` instead
  of being processed inside the request; scripts and the SDK still process
  inline. A REST search that runs past `RAG_SEARCH_TIMEOUT` (120 s) gets 504.
- **The metadata store records its version.** SQLite and PostgreSQL refuse a
  store written by a newer Lasagna, naming both versions. PostgreSQL creates and
  migrates its schema under an advisory lock and records the version.
- **Searching on Milvus no longer pays for a JSON filter.** A search was
  confined to its session through `metadata["session_id"]`, which Milvus parses
  row by row; it uses the `session_id` field every collection has and every
  writer fills with the same value. A search also stopped counting the
  collection's rows twice for a count it never used. Measured on Milvus Lite
  with 18,442 chunks in three sessions, the median search fell from 293 ms to
  32 ms without a reranker and from 415 ms to 144 ms with a MiniLM
  cross-encoder, returning the same ranked chunks and scores. A collection
  without the field keeps the old filter.

- `lasagna results leaderboard` takes one run's `results.json` (or its folder)
  as well as bundles; `-o` names a folder of `.md` or `.tex` pages plus
  `index.json`.
- `benchmark experiment` and `benchmark run` print `LLM cache: N hits, M
  misses (P%)` after the leaderboard when the run generated answers.
- Run metadata records a local benchmark's `path`, relative to the sweep file.
- MCP `get_system_status` names the chunking strategy that runs, and its
  `sota_features` block is now `features`.
- Nothing Lasagna prints, writes or describes uses an em dash; empty table
  cells print `-`.

Documentation:

- The README leads with the comparison, the reproduction, the quickstart and
  the agent setup; installation details moved to the guide. Package READMEs
  under `src/lasagna/` are short notes for contributors that point into
  `docs/`.

### Deprecated

Each of these still works in 0.1.x, says so when used, and is removed in 0.2.0.

- The 56 names `lasagna` exported besides `__all__` import with a
  `DeprecationWarning` naming the module to import them from.
- `lasagna benchmark compare` and `lasagna benchmark leaderboard`: use
  `lasagna results compare` and `lasagna results leaderboard`.
- `lasagna results interaction --bootstrap`: use `--resamples`.
- The MCP tools `search_knowledge` and `search_with_context`: use `search`.
  They still answer, but are listed only under `--tools all`.

### Fixed

Numbers that were wrong:

- **A bundle carried this build's metrics version, not the run's.** `results
  bundle` stamped the current `METRICS_VERSION` on any results file, so numbers
  computed under older definitions were published as current ones. A bundle
  carries the run's version, and `results bundle` refuses a file with a
  missing or different one, saying how to re-run it; `metrics_version_problems`
  gives `results validate` the same check.
- **Chunks cut at ingest without a word.** Benchmark ingest cut every chunk to
  `max_content_length` (4,000 characters) before embedding and indexing, and a
  sweep could not raise it. The default is unchanged and `chunk_size` 800 never
  reaches it, so no published number moves; a cut is logged.
- **NDCG with negative grades.** A negative grade lowered the score:
  `{a: 2, b: -1, c: 1}` ranked `[b, a, c]` scored 0.358 where `trec_eval` gives
  0.670. The registered BEIR loaders drop grades of 0 or below, so no harness
  number moves; custom qrels with negative grades do.
- **Per-arm corpora collided.** Arms with corpora of their own in one collection
  shared document and chunk ids, so the arm that ingested last took the others'
  documents. In a four-arm sweep the control scored 0.0, and a chunker arm
  drifted between 0.59 and 0.94 with task order. Each arm has its own ids,
  and two runs give identical numbers.
- **Embedder arms embedded the corpus with the built-in model.** Only the
  queries went through the arm's embedder, so the arm compared two models. On
  `smoke`, a bge-small function embedder scored 0.8712 before and 0.925 after,
  the same as setting `embedding_model` to bge-small.
- **Chunk-size sweeps did nothing.** The built-in chunker read `chunk_size`,
  `chunk_overlap`, `window_size` and the semantic thresholds from the
  process-wide config, so two arms sweeping `chunk_size` got identical chunks.
  Settings added with the new chunkers reused the control's corpus. Both take
  effect, and a setting of 0 is no longer read as unset.
- **The `semantic` component chunked by size.** It was built without an
  embedding model and ran fixed-size chunking under its name. It embeds with
  the arm's model and raises when none loads. `hierarchical` and
  `structure_aware` read the arm's settings too.
- **SQuAD v2's unanswerable half was dropped.** Only questions with a gold
  answer were scored, so a system that never abstains scored EM 1.000 where the
  official rule gives about 0.5. Unanswerable questions are scored, correct
  only when the system abstains, and every annotator's answer counts.
- **Latency stopped before the answer was written.** A live answer row records
  `retrieval_ms` and `generation_ms`, and `latency_ms` is their sum. Back-off
  sleeps between retries are not counted.
- **Paired tests rejected too often at small sample sizes.** Under a true null
  at alpha 0.05, the paired bootstrap rejected 9.1% of the time at 12 queries
  (the `smoke` benchmark) and 7.7% at 20. The significance, interaction and
  domain-stratified tests use a paired sign-flip randomization test, exact
  up to 30 non-zero differences, with a t interval. `p_value` and the interval
  keys (`confidence_interval_95`, `ci_lower`, `ci_upper`) hold those.
- **Agentic hops searched with the profile's components.** Each hop searches
  with the arm's reranker, query enhancer, retriever and embedder, and with the
  arm's per-query settings. Under evaluation, a failed agentic search fails the
  task instead of falling back to standard results.
- **A failing LLM reranker scored as the control.** When its LLM failed,
  `llm_listwise` kept the retrieval order. It raises, as the other LLM
  components do.
- **Late chunking truncated long documents.** Chunks past the model's window
  were embedded the standard way. Long documents are processed in overlapping
  windows (`late_chunking_overlap`), and under evaluation late chunking raises
  instead of falling back.
- **Markdown chunking cut code blocks at comments.** The structure-aware
  chunker, which every built-in profile runs, read a `#` comment inside a
  fenced code block as a header and split the block there. Fenced code is
  skipped when headers are found, and a chunk that is only a horizontal rule
  or table border (`---`, `|---|---|`) is dropped: searching the README
  returned `---` first, at score 1.0. Plain-text corpora chunk as before; the
  BEIR, smoke and plugin-template numbers are unchanged.
- **Configuring a second profile broke search in every other session.**
  `lasagna configure --profile speed -s A` and an upload rebuilt the shared
  collection at 384 dimensions, and the next search in any other session
  failed with "EMBEDDING DIMENSION MISMATCH" and a remedy that was a Python
  method. Each profile's model has its own collection, a mismatch message
  names both models and the command that fixes it, and rebuilding an empty
  collection is logged as information. Deleting a document, listing its chunks,
  native hybrid search and the orphan scan use the document's own collection.
- On Milvus, listing a collection's document ids past 16,384 chunks failed (the
  backend emulated an offset with `limit + offset`, over Milvus's cap), the
  error read as "corpus absent", and a large corpus was re-ingested on every
  run. Listing pages with `query_iterator`.

Results tooling:

- Experiment results and the LLM cache ignored `LASAGNA_DATA_DIR` and stayed
  under `~/.lasagna`. They are in `<data dir>/research/results` and
  `<data dir>/cache/llm.sqlite`.
- `results bundle -o newdir/run.json` (and `results schema`, `results
  per-query`, `export`) ended in a `FileNotFoundError` traceback when the
  folder did not exist. The folder is created; one that cannot be is a
  one-line error.
- **The LLM cache never applied to a custom provider.** It was attached in
  `SimpleLLMProvider` only, so a provider set with `set_llm_provider` was
  called again on every warm run while `results.json` recorded the cache as
  on with 0 hits and 0 misses. Any `LLMProvider` subclass's chat model now
  carries the cache and the token meter; a model that cannot carry the cache
  is recorded as `"applied": false` with the reason.
- A checkpoint is flushed to disk before it is renamed into place, so a crash
  cannot leave an empty checkpoint that `--resume` reads as a finished task.
- `results leaderboard` ranked latency and token counts highest first: the
  lower-is-better check read a field no row carried.
- LaTeX tables escape `\`, `{`, `}`, `~` and `^` in arm names; only `_ & % # $`
  were escaped, so such a name broke the table.
- `results significance` and the isolation summary said "1 pairs".
- **The short-list warning blamed the wrong setting.** It told users to raise
  `retrieval_chunk_overfetch` even when BM25 had simply run out of documents
  matching a one- or two-word query; it names the cause.

First run:

- `lasagna-mcp` over stdio answered `tools/list` after about 7 s: the
  `--session-backend` default made every start reset the backend, which
  imported torch. It answers in under a second; a reset happens only for a
  flag that was given.
- A misspelled benchmark, profile, stage or component name is one error with
  the likely name (`did you mean 'smoke'?`), before any model loads. A
  misspelled benchmark ran for seconds and failed inside its task.
- `component validate path/to/my_ideas.py:keyword_overlap` names the component
  `my_ideas:keyword_overlap`, as a sweep does.
- The `lasagna` command prints a bad setting as one message and exits 2;
  it was a traceback around the message.
- `lasagna doctor` names the session, profile, settings file and data
  directory in force, reports a bad setting as a failed check instead of
  crashing on it, and no longer checks for the GitHub CLI. `doctor --json`
  printed invalid JSON when piped: long lines were folded inside strings.
- `lasagna status` names a pre-0.1.0 collection that no configured model
  reads, and how to search it again.
- `lasagna init` states what each profile downloads (91 MB for `speed`,
  3.1 GB for `standard`), asks again for an empty or broken API key, and no
  longer cuts its title box off.
- **Another project's credentials leaked in.** pymilvus loads the nearest
  `.env` when imported: from a virtualenv inside a project, that project's
  keys appeared in `os.environ` and provider auto-detection used them. Lasagna
  imports pymilvus with `.env` loading off and the environment restored.
- A command run while `lasagna serve` holds the local store prints one
  sentence with the fix, not a 25-line traceback.
- One-shot commands no longer print gRPC fork notices, a "Loading weights" bar
  per model, or a warning about background GC, which only `serve` and the MCP
  server run. Progress bars that last longer than two seconds, such as a model
  download, still show.

Requests to current models:

- **Current Claude and OpenAI models rejected every request.** Lasagna sent
  `temperature=0` on every call. Claude 4.7 and later answer that with HTTP
  400, and gpt-6-luna rejects a temperature at its default reasoning effort.
  Each request is built from one table of what each model accepts
  (`lasagna.providers.model_capabilities`): Claude 4.7 and later get no
  `temperature`, `top_p` or `top_k`; OpenAI reasoning models that allow it get
  `reasoning_effort: "none"` with the temperature; `reasoning_effort` in the
  model settings wins. Unknown models are sent a temperature as before. The
  RAGAS judge's requests are built the same way, so gpt-6-luna judges at
  RAGAS's 0.01 where RAGAS sent 1.0.
- **A model that cannot be pinned says so.** `lasagna doctor` and the start of
  an experiment print one line (`claude-sonnet-5-5 does not accept temperature;
  answers can vary between runs; repeat runs (n_runs >= 3) to measure that
  variance`), and the run metadata records `sampling: provider-default` for the
  generator or judge. The LLM cache replays a call only when temperature 0 was
  actually sent, which also excludes models whose client drops it.
- Dateless ids that are snapshots (Claude from 4.6 on, and OpenAI models whose
  only snapshot is the dateless id, such as gpt-6-luna) no longer raise
  `UNRESOLVED_JUDGE_ALIAS` in a bundle.

Servers and deployment:

- **A stack configured for a Milvus server opened Milvus Lite.** The startup
  backend always passed a local `db_path`, which `MilvusBackend` took over the
  `uri`, so `backends/milvus.yml` and `MILVUS_URI` wrote to a file inside each
  container. A configured server address wins, the log names the store opened,
  and a `db_path` set together with a server address is an error naming both.
- **Uploads processed by the Celery worker failed.** The api stored a file
  inside its own container and passed the path to the worker, which had no such
  file. The api, worker and mcp share an `uploads-data` volume at
  `LASAGNA_UPLOAD_DIR=/data/uploads`.
- The worker and mcp containers inherited the image's healthcheck on port
  8000, which they do not serve, so they always reported unhealthy.
- `backends/milvus.yml` could not start: its image tag `v2.4-latest` does not
  exist, and it lacked the `milvus run standalone` command and the embedded
  etcd config. It follows Milvus's own standalone script, on `v2.4.24`.
- Values in `.env` other than those the compose file names (LLM API keys,
  `PINECONE_API_KEY`, `RAG_SETTINGS_FILE`, any `RAG_*` override) never reached
  the containers. They are passed in.
- The backend files configured the api, worker and mcp but not `celery-beat`;
  `milvus-lite.yml` configured only the api, so a worker started with it could
  not reach the store. Its worker, beat and mcp exit at once with the reason.
- The Dockerfile's unquoted `sentence-transformers>=5.1.2` was a shell
  redirection, so the version floors were never applied.
- A failed native Milvus hybrid search returned no results; it falls back to
  the BM25 index.
- Replicas or workers starting together against an empty PostgreSQL raced on
  schema creation: 35 of 40 concurrent first starts failed with a
  `UniqueViolation` in a local test, against 0 of 40 after the fix.
- A backend that could not open a data directory held by another process
  logged a full traceback; it logs one line at DEBUG, and other startup
  failures are logged with their traceback without breaking on braces in the
  message.
- The Redis and broker URLs, password included, were logged at startup.
- A URL no route matched became its own label value in the request histogram;
  it is `unmatched`.
- Every search logged "SOTA search: ... hybrid" at INFO, also when hybrid
  search was off. It is a DEBUG line naming the actual mode. The API
  description no longer says "Production-ready".
- Run records, the stage manifest and the MCP status name the chunker that
  runs. A structural strategy with its setting turned off records `semantic`;
  comparability keys change only for such arms.
- `benchmark leaderboard <folder>` failed with "Unknown format".
- `lasagna serve` and MCP tools on a store another process holds logged an
  ERROR line in addition to the reply that says so.

### Known issues

- **`pip` cannot resolve `[eval]` together with `[storage-all]`** (and therefore
  cannot resolve `[all]`). `pip install "lasagna-rag[eval,storage-all]"` exits
  with `ResolutionImpossible`; `[all]` backtracks for over fifteen minutes
  without converging. The constraint set is satisfiable: `uv pip compile`
  solves both in seconds, and `pip` succeeds when handed the versions it fails
  to find:

  ```bash
  uv pip install "lasagna-rag[eval,storage-all]"
  pip install "lasagna-rag[eval,storage-all]" s3fs==2026.6.0 fsspec==2026.6.0
  ```

  Cause: `s3fs`, `gcsfs` and `adlfs` release in lockstep with `fsspec` and pin
  it exactly, while `datasets` (via `ragas`) caps `fsspec`. Searching
  newest-first, pip explores a large `aiobotocore`/`botocore`/`boto3` space and
  gives up. Not "fixed" by pinning: any upper bound written here is a guess at
  when `datasets` will raise its own cap, and would strand users on an old
  `s3fs` once it does.

### Planned for 0.2.0

- Export and import of a session's documents and chunks.
- A prebuilt Wikipedia index.
- Multilingual benchmarks.
- Batched generation with vLLM.
- An OpenAI-compatible answer endpoint, and streamed answers.
- Weights & Biases and MLflow logging hooks.
- Claim-level diagnostics of generated answers.
- Calibration of the LLM judge against human labels.
- Removal of everything under Deprecated above.

## [0.1.0a1] - 2026-08-18

Internal pre-release, never published to PyPI. Its correctness work is part of
`0.1.0`.

### Changed: metric definitions (`METRICS_VERSION` 1.0 to 1.1)

Numbers produced before this release are **not comparable** with numbers
produced after it. `METRICS_VERSION` is part of the comparability key precisely
so the two cannot be grouped in one leaderboard table.

- **NDCG now uses linear gain** (`rel`), matching `trec_eval` and therefore
  every published BEIR number. It used `2^rel - 1`, the Burges convention:
  identical on binary judgments, different on graded ones (`beir/nfcorpus`,
  `beir/trec-covid` and `beir/dbpedia-entity`). `exponential_gain=True` is
  available for comparison against work that uses it.
- **Cutoff metrics are computed over `k` documents.** The harness requested `k`
  *chunks*, then de-duplicated to parent documents before scoring, so a
  "top-10" list was typically 4–7 documents and every `@k` metric was computed
  over a list shorter than its own name claimed: a systematic downward bias on
  every dataset. It now over-fetches chunks and cuts to `k` documents.
- **`k` is configurable** (`metric_k`), so NDCG@100 and Recall@1000 can be
  reported. It was hard-coded to 10 in six places. Metric keys follow the
  cutoff actually used.

### Fixed: numbers that were silently wrong

- **A test-set request could return train-set judgments.** The BEIR loader
  walked `[requested, test, validation, train]` and returned the first split
  that loaded, cached it under the *requested* split's key, and stamped
  `split: "test"` on the result. It now honors `allow_split_fallback`, reports
  the split it actually loaded, and warns when it substitutes.
- **`max_corpus_docs: 0`, documented as "no limit", capped at 100,000**,
  truncating 8 of 13 BEIR corpora and aborting 7 outright.
- **An aborting corpus ingest hung every sibling task** rather than failing it.
- **`sample_id_digest` was a function of `n_samples`.** Two benchmarks at the
  same sample count produced the same digest, in a field the bundle documents
  as proof that two runs scored the same questions.
- **`TRUNCATED_CORPUS` could never fire.** The adapter set the flag on itself
  and nothing carried it to the result, so truncated bundles validated clean.
  The flag is now set from the loop that actually truncates rather than from
  declared metadata, which for `beir/nfcorpus` claimed 170,000 documents
  against an actual 3,633.

### Fixed: install and tooling

- `langchain-ibm` was imported unguarded and declared nowhere, so
  `lasagna.infrastructure.llm_manager` raised `ImportError` on a clean install.
- The `pgvector` extra declared one of the three distributions its backend
  imports, so it installed cleanly and left the backend unavailable.
- `html2text` (GPL-3.0-or-later) was a core dependency and imported nowhere.
- `setuptools>=68` cannot build this package's PEP 639 metadata; the floor is
  now 77.
- The `compliance` CI job could not fail, for three separate reasons, and two of
  them survived the first attempt to fix them.
  - `pip-licenses` ran without `--partial-match`, so its copyleft gate matched
    only the literal string "GPL", so `GPL-3.0-or-later` sailed through. Adding
    the flag was not sufficient: `uvx` runs the tool in an ephemeral environment
    created with `include-system-site-packages = false`, and `pip-licenses`
    inspects the `sys.path` it runs under. Measured, it enumerated **zero**
    packages against 515 in a real environment. Now pinned to the CI interpreter
    with `--python`.
  - `detect-secrets scan --baseline` returns 0 unconditionally. Its replacement,
    `uvx detect-secrets-hook`, resolved the command string as a *distribution*
    name; no such distribution exists, so the step errored on every run, and
    being the job's first step it took the two checks after it down as well. Now
    `uvx --from detect-secrets detect-secrets-hook`, matching what
    `.pre-commit-config.yaml` already did. The unowned name was also a
    supply-chain hazard.
- The version is single-sourced from distribution metadata. It was a literal in
  three files, and `provenance.py` recorded the literal rather than the version
  installed.

### Added

The first standalone release of Lasagna; this is everything it brings.

- **End-to-end release gate** (`python -m e2e.run`). Everything in `tests/`
  runs from inside the repo with `src/` on `sys.path`; the gate is the only
  thing that checks the package the way a user meets it: built into a wheel,
  installed into a clean venv, driven from outside. It writes a dated report
  with metrics, failures and a GO/NO-GO verdict, and diffs against the previous
  run so a latency slide or a silently disabled phase is visible. A subset runs
  in CI on every PR. See `e2e/README.md`. The gate also covers the *second*
  day (upgrading over an existing install, and uninstalling), which is where a
  package leaves stale modules importable or a shadow copy behind.
- Stage-isolated RAG pipeline with seven pluggable component types, enforced by
  a conformance suite.
- Component registry accepting `paper` and `claims` metadata, returned by the
  listing API so results tables can show claimed against reproduced numbers.
- Six vector backends (Milvus, pgvector, Qdrant, Weaviate, Pinecone, OpenSearch)
  behind one interface with a shared contract test suite.
- Evaluation harness: 31 benchmark datasets, RAGAS metrics, NDCG/EM/F1, latency
  and cost accounting, multi-objective tradeoff analysis.
- Python SDK, REST API, MCP server, CLI, and a schema-driven configuration UI.
- `lasagna.core.model_licenses`: license metadata for every selectable model,
  warning once at load when a non-commercially-licensed model is used.
  See `docs/model-licenses.md`.

### Fixed
- **HyDE, query expansion, RAG Fusion and CRAG silently degraded to plain
  retrieval.** Each bridged to async work via `asyncio.run()`, which builds a
  fresh event loop per call. A cached LLM client holding an `asyncio.Lock` bound
  to an earlier loop then raised `RuntimeError`, which a broad `except` reduced
  to a one-line warning, so the feature produced nothing and the configuration
  scored the same as its baseline. Affected anyone driving search from a worker
  thread (e.g. `asyncio.to_thread`).
- **Evaluation produced empty answers and `faithfulness = 0.0`.** The judge and
  generator LLMs shared a single model string, so a model id valid for the
  scoring endpoint was passed to the generator's platform, where it did not
  exist. They now resolve independently, and the judge reads its own
  credentials so it cannot re-route the generator's provider detection.
- **CRAG graded chunks with an unbounded fan-out**, one LLM call per chunk. On
  large retrievals this overran endpoint concurrency limits; timed-out grades
  fell back to `ambiguous` and diluted the decision. Now capped
  (`RAG_CRAG_GRADING_CONCURRENCY`, default 4).
- Exception types and tracebacks are now logged at the failure sites above.
  Their absence is why these bugs stayed hidden across many evaluation runs.
- Removed `QueryEnhancer.enhance_query_sync()`, which silently returned an
  un-enhanced query when called inside a running event loop.

### Security
- Deployment manifests require every credential; none fall back to a default.
- The test suite is isolated from the real `~/.rag_knowledge` store.

### Notes
- Lasagna vendors no third-party source, bundles no model weights, and
  redistributes no benchmark corpora.

[Unreleased]: https://github.com/dolev31/lasagna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dolev31/lasagna/releases/tag/v0.1.0
