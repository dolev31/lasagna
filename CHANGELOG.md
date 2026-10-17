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

Its metric definitions are `METRICS_VERSION` 1.6: NDCG with linear gain and
zero gain for a negative grade (as `trec_eval`), cutoff metrics over `k`
documents in ranked order, exact match and token F1 normalized in the official SQuAD order with
`[n]` citation markers removed and abstention counted only when an answer opens
with one, and `factual_correctness` among the judged metrics
([metrics.md](src/lasagna/knowledge_evaluation_sdk/docs/metrics.md) defines
each). The version is part of every comparability key, so results from the
internal 0.1.0a1 pre-release (1.1) or from development builds since (1.2 to
1.5) never share a leaderboard group with results from this release.

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
  each with a 2 s timeout, and lists what did not answer in `failing` (Redis,
  which the cache, rate limits and uploads fall back from, in `degraded`).
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
  sets. The dense models land within 0.002 NDCG@10 of MTEB and BM25 with the
  cross-encoder within 0.004 of the BEIR paper; BM25 is 0.015 and 0.005 below
  Anserini's single-field BM25, the like-for-like run (0.008 below BEIR's
  multifield one).
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
  share vectors. A name set explicitly (settings file or `RAG_COLLECTION_NAME`)
  is used as given; the evaluation harness's per-arm names are
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

Metric definitions and results (`METRICS_VERSION` 1.6):

- The retrieval track scores the ranked list. Under a profile with context
  reordering on (standard, the default `base_profile`, balanced and
  max_quality) it scored the order the generator reads, best passages at both
  ends, so the second-ranked document fell below the cut to `k`. Retrieval
  numbers from those profiles change; the speed profile's do not. The stage
  manifest records the order each track used (`context_order`).
- Under those profiles the generator reads the top five passages reordered.
  The search reordered all ten it returned and the harness kept the first
  five, so the generator read ranks 1, 3, 5, 7 and 9 and never the second
  best. The judge sees the same five in ranked order. Query fusion
  (`enable_rag_fusion`, `enable_query_expansion`) fuses each variation's ranked
  list and reorders the fused list; it read reordered positions as ranks.
- A retriever arm with query fusion or expansion searches every variation with
  its own retriever. The variations searched the store's dense index, so the
  arm measured dense search under the retriever's name.
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

- `-p` means `--profile` wherever it exists; `serve --port` has no short flag
  (it was `-p` there). The Kubernetes manifests are named `lasagna` (namespace,
  deployments, labels and image), as the project is. The unversioned
  `/api/config/knowledge` routes are deprecated for any caller but the
  configurator page and move under `/api/v1` in 0.2.0; programs use
  `/api/v1/config`. `CITATION.cff` carries the release-date placeholder until
  the release stamps it with the changelog (it named 2026-08-18). The deploy
  guide's "Upgrade notes" for versions before this release are gone, and the
  claim that pip cannot resolve `[all]`: with `storage-all` bounding `fsspec`
  and `s3fs`, pip 26.2 resolves `[eval,storage-all]` in 31 s and `[all]` in
  about two minutes.
- `lasagna evaluate`, an undocumented command from before the benchmark
  harness, is removed: it checked for no API key, retried for 69 s and then
  failed. `lasagna benchmark run <file>` measures your own data
  ([docs/own-data.md](docs/own-data.md)). A core install no longer warns
  "Docling enabled in config but not installed" per document: the settings
  file turns Docling on and falls back without it by design, which is now
  said once, at INFO, with the install command.
- `-s` means `--session` in every command; `benchmark run` spells out
  `--split` and `--mode`. `benchmark run` measures retrieval unless `--tracks`
  names generation or `--generator` is given (generation needs an API key),
  and starts from the keyless `speed` profile (`--profile standard` for the
  3 GB one). A chunking error shared by every document prints once, with a
  count, and a run where every task failed no longer ends in a
  `RuntimeWarning`.
- Generation without an LLM key is refused before anything is ingested,
  naming the keys to set and `--tracks retrieval`; `benchmark validate-data`
  suggests `--tracks retrieval` when no key is set. In a sweep that also
  measures retrieval, a generation failure no longer discards the retrieval
  metrics: they are reported, and the run names what failed and exits 1.
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
  `rebuild_collection`, `delete_rag_profile`, `reset_rag_config` and
  `set_lifecycle_policy` (server-wide: it decides what garbage collection
  deletes in every session); `--tools all` or `LASAGNA_MCP_TOOLS=all` serves
  them. Tool names,
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
- **The Docker image runs torch's CPU-only build.** torch from PyPI brings the
  CUDA libraries on Linux, so the image the CI stack builds was 11.5 GB; it is
  3.8 GB. The build argument `TORCH_VARIANT=cuda` (`make build
  TORCH_VARIANT=cuda`, or `TORCH_VARIANT` in `.env`) builds the CUDA image for
  a GPU host; see `src/lasagna/deploy/README.md`.
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

- Text a user reads outside the docs says what the code does: the profile
  lists of `GET /api/v1/config/profiles` and the SDK's `list_profiles` (which
  said "SOTA features" and named models the profiles do not use) share the
  profiles' own descriptions, and the configurator, settings files, compose,
  Makefile, Dockerfile and Kubernetes comments, docstrings and help lose
  "SOTA", "production-ready", "seamless", "powerful" and their em dashes. A
  test covers those files and every string in the package.

### Deprecated

Each of these still works in 0.1.x, says so when used, and is removed in 0.2.0.

- The 56 names `lasagna` exported besides `__all__` import with a
  `DeprecationWarning` naming the module to import them from.
- `lasagna benchmark compare` and `lasagna benchmark leaderboard`: use
  `lasagna results compare` and `lasagna results leaderboard`.
- `lasagna results interaction --bootstrap`: use `--resamples`.

### Fixed

Server and deployment:

- **A session can no longer point the server at another vector store.**
  Session settings (REST `/api/v1/config`, `/api/config/knowledge/profile` and
  `/import`, the MCP config tools, the config manager) accepted any key, and
  search resolved them: `vector_backend_config` with any URL made the server
  connect there, and a session could name another collection, a database or
  upload path, a Redis URL or `reranker_trust_remote_code`. A session now sets
  retrieval and generation settings only; every other key, unknown keys and
  models named by a path or URL are refused with an error naming each key.
- **A foreign `Host` is refused while auth is off.** With `RAG_REQUIRE_AUTH`
  off, the REST API and the MCP server over HTTP answer only to localhost,
  127.0.0.1, [::1] and the names in `RAG_ALLOWED_HOSTS`, and refuse any other
  `Host` with 421, so a web page cannot reach them by DNS rebinding. Health and
  metrics paths answer any `Host`. Compose allows `api:*,mcp:*`.
- **Removing a key from `RAG_API_KEYS` ends the sessions it minted**, on REST
  and MCP. A token used to stay valid for its seven days.
- 413 and 429 responses to an allowed origin carry the CORS headers (CORS was
  the innermost middleware).
- The CLI follows `serve.json` only when the current user owns it, only that
  user can write it, and it names a loopback http URL with a live process of
  that user; `lasagna serve` writes it with mode 0600.
- **With Prometheus on, an agentic or CRAG search keeps its results.**
  `rag.search.total_ms` was registered with two label sets, the real client
  raised, and the routing fallback reran the standard pipeline after the agent
  had run. Each metric now has one label set (`rag_search_total_ms` is labelled
  by `tier` only), a metrics error never changes a request, and a routing
  failure is reported in `degraded_stages` as `routing`. Under evaluation a
  failed agentic search fails the task instead of being scored with standard
  results.
- **A Milvus collection whose row count cannot be read is no longer treated as
  empty and dropped.** `get_collection_stats` returned 0 on any error, which
  the identity check, an insert rebuild and MCP `reindex_document`
  with `force_rebuild` took for an empty collection. OpenSearch raises the same
  way.
- OpenSearch hosts and the tracing endpoint are logged without their passwords.
- MCP `reindex_document` and `rebuild_collection` no longer replace the
  server's shared embedder and reranker; they use the session's embedder for
  that request alone.
- A Milvus server's unflushed rows are counted before an emptiness check
  decides a drop: a server counts only flushed segments, so a collection
  written seconds earlier read as empty.
- **Kubernetes secrets stay out of the image and the wheel.** The deploy README
  told operators to edit `k8s/base/secrets.yaml`, which the Docker build
  context and the package data both included. The build context leaves out
  `src/lasagna/deploy/k8s/`, and secrets go in a git-ignored `overlays/local/`
  (copied from `overlays/local.example`) that the wheel and sdist exclude.
- **The Kubernetes defaults fit their models.** Each gunicorn worker and Celery
  process loads its own models after the fork (about 3 GiB), so the api runs
  one worker per pod and requests 4Gi, and the worker runs `--concurrency=2`
  and requests 5Gi; the api autoscaler scales on CPU (its memory target was
  always exceeded and pinned it at 10 replicas); a `lasagna-celery-beat`
  Deployment runs garbage collection, which never ran;
  `RAG_PROCESSING_TIMEOUT=20` keeps the shutdown drain inside the grace period,
  as in compose. `make validate-k8s` checks each of these.
- The Kubernetes worker's liveness probe pings its own pod's worker
  (`celery@$HOSTNAME`).
- CI pins detect-secrets 1.5.0, the version that wrote `.secrets.baseline`.
- The release workflow pins `pypa/gh-action-pypi-publish` to the commit v1.14.2
  names; the pin was the tag object.
- **`--resume` no longer restores results computed with another LLM.** The
  resume key was the same for gpt-6-luna on OpenAI and Claude on Anthropic; it
  now includes the LLM provider and model (`MODEL_NAME`, the detected API key),
  the judge's model and base URL, and the local commit of each model the arm
  loads. Earlier checkpoints are not restored.
- Archiving, restoring and GC expiry bump the session's content generation, so
  other workers stop serving an archived document (or keep missing a restored
  one) from their search cache.
- **Documents left "processing" by a dead worker are failed with the reason.**
  A new GC phase, also run when the lifecycle scheduler starts, fails documents
  still processing after twelve times the longest a document may take (the
  Celery task limits with Celery on, else `processing_timeout`; at least an
  hour). A redelivered Celery task skips a finished document.
- The orphan scan covers every embedding model's collection, not only the
  default model's. A scan that cannot read which collection a document was
  written to is marked incomplete and repair refuses it; repair re-checks each
  database-only orphan before archiving it.
- A `rag_documents_v2` collection from before 0.1.0 is adopted only by the
  model `LASAGNA_LEGACY_COLLECTION_MODEL` names. The first model with a
  matching size adopted it, so bge-large could claim bge-m3's vectors (both
  1024).

Numbers that were wrong:

- **The `mmr`, `combined` and `ensemble` rerankers ran as something else.** On
  the speed profile `mmr` and `ensemble` returned the retrieval order and
  `combined` scored exactly like `cross_encoder`: MMR never received the
  pipeline's embedder and was skipped with `enable_mmr` off, and the ensemble
  loaded the settings file's model, which failed offline. MMR now gets the
  search's embedder and runs when named; `ensemble` ranks with the arm's
  `ensemble_reranker_models` (and `ensemble_reranker_weights`), records them as
  what ran, and fails under evaluation with fewer than two models or one that
  cannot load. A reranker that returns its candidates in retrieval order on
  every query is flagged inert without a reranker-free control arm.
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

- **A sweep naming an unknown benchmark started anyway.** `benchmarks: [{name:
  smok}]` printed "Starting experiment", then failed every task with a quoted
  `KeyError` listing every benchmark; `--dry-run` printed the list three times
  and exited 0, and a missing local `path` also got as far as "Starting
  experiment". Benchmark names and data files are checked first, in the run
  and the dry run, with one error naming the closest benchmark and exit 1.
- **A claim printed beside any benchmark.** Claims were keyed by metric
  alone, so `benchmark run smoke --reranker my_ideas:keyword_overlap` printed
  "claimed 0.4200 (Someone et al. 2026)" next to a 24-document smoke test, in
  the LaTeX "Claimed" column too. A claim names its benchmark,
  `claims={"beir/scifact": {"ndcg@10": 0.456}}` (with `"split"` inside if the
  paper gives one), and tables show it only on a run of that benchmark and
  split. The flat form needs an explicit `"benchmark"` key; without one,
  registration fails with the form to use.
- **A typo in `-m` got a verdict.** `results significance -m ndgc@10` (or a
  metric no task recorded) printed "No pair survives Holm correction" and
  exited 0, and `results leaderboard`, `compare`, `power` and `interaction`
  also exited 0. Every results command now refuses a metric the run did not
  record, names the closest one and lists the rest, as `results view` did.
- **`--strict` and the hub refused single-run bundles while the docs
  recommend `n_runs: 1` for retrieval.** A run now records `nothing_samples`
  (only retrieval ran, no task called an LLM, repeats if any were identical);
  such a run raises no `SINGLE_RUN`, and the leaderboard verifies it when
  another lab's bundle equals its mean. A run with an LLM stage still needs
  repeats.
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
- **Claimed numbers never reached a results table.** The README and
  CONTRIBUTING promise that a component's `claims` let a table show the
  claimed number beside the reproduced one; nothing did, and a sweep naming
  `my_ideas:longer_first` re-registered the function without the claims its
  decorator declared. A run records each component's paper and claims
  (`components` in its metadata and bundle), and the leaderboard
  `benchmark experiment` prints, `results leaderboard` in markdown and LaTeX,
  and `results view` show the claim and its paper beside the measured value.
- **The stage isolation proof mostly skipped.** `pytest -rs
  tests/test_stage_isolation_behavioural.py` skipped 17 cases: the
  cross-encoder, combined and MMR rerankers and the semantic and
  structure-aware chunkers tried their gigabyte defaults offline, and the LLM
  components had no LLM. They run with cross-encoder/ms-marco-TinyBERT-L-2-v2
  and all-MiniLM-L6-v2 through config overrides, or a stub LLM, and fail
  rather than skip when they cannot run. `needs_embedding_weights` can name
  the models a test loads, so CI's weights job runs these and the offline
  job skips them only when the models are not cached. Four stay skipped,
  each naming why: colbert (colbertv2.0, 440 MB), monot5 (990 MB), cohere
  (its API) and flashrank (its package).
- **A dataset missing offline failed in every task.** With SciFact not
  cached and offline mode on, the Colab example printed the dataset error once
  per arm and again in the failed-task list, then a green "Experiment
  Complete ... Failed: 3" box and a "Next:" command. A run checks once, before
  any task, that each benchmark's Hugging Face data is cached when offline
  mode is on, and stops with one error naming the benchmark and the fix. When
  every task fails, the summary box is red, says so, and prints no "Next:".
- `benchmark run ./my.jsonl` no longer warns that your own file is "an easier
  task than the published benchmark", names its results folder by the file's
  stem (`benchmark_my_<time>`, not `benchmark_._my.jsonl_<time>`), and says
  "1 run", "embedded in 0.4 s" and "latency: 0.42 ms" instead of "1 run(s)",
  "0.0 min" and "0.0 ms". `validate-data` on data nothing can be measured on
  is one error with exit 1; it printed "valid jsonl data" first. A factorial
  with an inert arm prints the isolation verdict before the inert line.
  `component validate cross_encoder` checks the registered component, with a
  likely name for a typo; it was refused as not being module:function.
- `lasagna status` without `-s` said "No documents uploaded yet." while
  another session held documents, and took about 7 s without a server. It
  names the sessions with documents and how to show one, and reads the
  metadata database directly, opening the vector store only when a document
  could predate per-model collections (0.3 s).
- `lasagna doctor` said "ok, 4 model(s) cached" while the profile's own models
  were not downloaded. It checks the session's embedder and reranker (`-s`
  picks the session) and names any that are missing, with their sizes.
- **Chunks rewrote their text and pointed at the wrong place.** The semantic
  and structure-aware chunkers joined sentences with single spaces, so
  "**Is the widget free?** Yes" came back as "**Is the widget free? ** Yes"
  and paragraph breaks as spaces; spans counted stripped whitespace, ran
  past the text (0-500 for 233 characters), or were missing for a
  one-sentence text. A chunk is now the source text at the span it reports.
  The places where text is split are unchanged, and so is every benchmark
  number (the BERT-family tokenizers the benchmarks use ignore the
  whitespace that changed).
- Every upload on the default profile warned that `chunking_strategy='semantic'`
  was overridden and `hierarchical` would run. The settings file and the
  built-in defaults declare `hierarchical`, which is what runs with
  `enable_hierarchical` on, as the presets already declare theirs.

First run:

- The README said its component comparison "runs as is, with no model,
  corpus or key"; with an empty Hugging Face cache the sweep failed
  ("all-MiniLM-L6-v2 is not downloaded"). It says `python my_ideas.py`
  downloads nothing and the sweep downloads 182 MB once, the size a test
  computes from the sweep the README shows.
- `lasagna doctor` said the first search downloads "~5 GB" and the speed
  profile "an 80 MB model" (measured: 3.1 GB and 91 MB), and that retrieval
  works without an API key, while the default configuration sends complex
  queries through agentic retrieval. Both hints are computed from the
  profiles and the measured sizes, and name the speed profile as the one
  that retrieves without a key. The configurator, the settings file
  comments and the support skill state the same sizes.
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
- Messages from the first hour: a malformed sweep file is one line with its
  line and column (it was a 164-line traceback); a misspelled component or
  arm key gets the likely name, without the advice to install an entry point
  or a list of every known setting; warnings print as `WARNING: <message>`
  without loguru's timestamp and source line (`-v` keeps them); progress
  lines name the benchmark, arm and run instead of wrapping task hashes; the
  isolation verdict prints before the summary; `results significance` with
  nothing significant prints no empty table and says "1 non-significant pair
  is not shown"; the LaTeX leaderboard drops columns that say nothing for the
  rows (Varied equal to Config, an anonymous submitter, Verified), and no
  table prints "± 0.0000" for one run; the dry run says "cached" rather than
  "downloaded"; names with colons print literally (`nope:x` printed as
  "nope❌"); a function reranker that returns something other than a number
  is named with the value; `component validate` ends by showing the sweep
  line, not "register it". The README's component is `my_ideas.py`, the file
  its sweep names.
- `lasagna doctor` fails a vector backend nothing registers, with the
  likely name: `RAG_VECTOR_BACKEND=milvuss` was "ok vector_backend:
  configured: milvuss", and the first upload failed.
- `lasagna doctor` names the session, profile, settings file and data
  directory in force, reports a bad setting as a failed check instead of
  crashing on it, and no longer checks for the GitHub CLI. `doctor --json`
  printed invalid JSON when piped: long lines were folded inside strings.
- `lasagna status` names a pre-0.1.0 collection that no configured model
  reads, and how to search it again.
- `lasagna init` states what each profile downloads (91 MB for `speed`,
  3.1 GB for `standard`), and no longer cuts its title box off.
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
- The README's Python quickstart wrote gRPC fork notices (121 lines in one
  run) and "Loading weights" bars to stderr. The fork was tqdm's: its first
  bar creates a cross-process lock, which starts multiprocessing's resource
  tracker while Milvus Lite's gRPC runs. Lasagna keeps tqdm on its thread lock,
  and hides the weight-loading bar for a cached model and the encoding bar for
  a few batches, without setting any environment variable.
- On macOS with transformers 5.20 or later (what an install from PyPI gets),
  every command and the quickstart printed torch's "NOTE: Redirects are
  currently not supported in MacOs." It concerns torchelastic workers, which
  Lasagna never starts. Importing `lasagna` drops that one message from that
  one torch logger; other torch warnings still show.
- **The Python quickstart failed silently offline, then hung.** On the
  default profile with BGE-M3 not downloaded and `HF_HUB_OFFLINE` set,
  `wait_for_processing` returned the failed document without a word, the
  search after it failed with a traceback, and Python then waited about 58 s
  to exit while the worker retried the missing model twice, 30 s apart.
  `wait_for_processing` raises `DocumentProcessingError` with the reason and
  the fix (`raise_on_failure=False` returns the document instead); a model
  that is not downloaded while offline is refused before any load is tried; a
  missing model, a missing or unreadable file, a missing package and a
  rejected credential are not retried; a document stays "processing" between
  retries instead of reading as failed; and a pending retry no longer holds
  the interpreter at exit. `lasagna upload` in the same state took 67 s and
  printed three ERROR lines after its own; it prints one error with the fix
  after about 10 s.
- **Rerunning the quickstart duplicated the corpus.** Each run uploaded
  paper.pdf again into the same session, `lasagna status` listed every copy,
  and search returned the same passage once per copy. An upload whose bytes
  the session already holds (ready or processing) returns that document with
  `duplicate_of` set, from the SDK, the CLI ("Already uploaded"), REST and
  MCP; `allow_duplicates=True`, `lasagna upload --force` or the REST form
  field `allow_duplicates=true` index it again. Uploads with an explicit
  `document_id` (reindex, benchmark corpora) or a component override are
  never deduplicated. Search also returns a chunk at most once.
- **Setting only `embedding_model` broke ingestion.**
  `KnowledgeConfig(embedding_model="BAAI/bge-small-en-v1.5")` (or
  `RAG_EMBEDDING_MODEL` alone) kept the settings file's 1024 dimensions: the
  store refused the 384-dimension vectors with a pymilvus traceback, the
  document failed and search returned nothing. A model set without its size
  takes its own, from a table of common models or its files in the Hugging
  Face cache; when neither knows it, the upload is refused before anything is
  stored, naming `embedding_dim`.
- **The CLI ignored environment variables.** `RAG_ENABLE_RERANKING=false
  lasagna status` still listed reranking, and `RAG_EMBEDDING_MODEL=...
  lasagna upload` embedded with BGE-M3: the CLI passed the default profile's
  stored snapshot of the settings file to its client as explicit settings,
  which beat the environment and were saved over the session. Every command
  now resolves session, environment, settings file and defaults in the order
  docs/configuration.md gives, and `status` and the `serve` banner print the
  settings a search on that session runs with.
- **`lasagna init` wrote an API key Lasagna then did not read.** It saved
  OPENAI_API_KEY to ./.env (readable by everyone), said Lasagna reads no
  other key from .env, and `doctor` reported no key. It no longer asks for or
  writes a key: it says which variable to export, and that `doctor` will see
  it. The "Cloud API" embedding question is gone (nothing read its answer;
  the profile chooses the model), the summary names the profile's model,
  which is what `status` shows, and each provider is labeled with the model
  Lasagna calls by default (it read "OpenAI (GPT-4o)").
- **The README's `serve --mcp` could not search.** Every MCP tool call over
  `/mcp/sse` answered "Either session_id or session_token is required".
  `lasagna serve --session/-s <session>` names the session that MCP tool
  calls and REST requests without `X-Session-ID` use (it sets
  `RAG_SESSION_ID`, which REST now honors too), the banner names it, and the
  README block passes `-s my-corpus`. A call that names no session says
  how to name one.
- **MCP errors leaked queries and confused agents.** A failed search wrote a
  134-line traceback to the client's log with every frame's local variables,
  the query text among them six times; every log sink in the package now has
  `diagnose=False`. The missing-LLM error ended "or call
  configure_providers()", a Python call an agent cannot make; it says which
  variable to set where Lasagna runs. And on the default profile every MCP
  search went to the agentic loop and failed without an LLM key, while the
  README said only complex queries need one: without an LLM the MCP `search`
  runs the session's standard pipeline and names `agentic` in
  `degraded_stages`, as REST does for a query it routes there. The README
  and docs/configuration.md say so.
- **Health answers contradicted each other.** `/api/v1/health` answered 200
  with `"unhealthy"` and an empty `failing`, `/api/v1/health/ready` 503 naming
  the embedding model, the MCP `get_system_status` tool "healthy", and the
  deployment guide both "always with 200" and "503 while ... not loaded".
  One model now: liveness (`/api/v1/health/live`, and the MCP service's
  `/health`) checks nothing; readiness answers 503 naming each failing
  component, and counts Redis down as degraded, since the cache, rate limits
  and uploads fall back; `/api/v1/health` and `get_system_status` report the
  same component states with `failing` and `degraded`, `/api/v1/health`
  always with 200. The embedding check is for the model the served session
  (`serve -s`, else `default`) searches with.
- `lasagna upload` took one file, so the README looped over a folder, one
  process and one model load per file. It takes several paths and glob
  patterns (`lasagna upload ./docs/*.md -s my-corpus`) through one client,
  reports a row per file, and exits 1 when any failed.
- `RAG_VECTOR_BACKEND=milvuss lasagna benchmark run smoke` ran, failed its
  task and advised pointing RAG_DB_DIR at a new directory. `upload`,
  `search`, `serve`, `benchmark run` and `benchmark experiment` check the
  backend name first and stop with the likely name and the list, as `doctor`
  does. `RAG_SETTINGS_FILE` (or `LASAGNA_SETTINGS_FILE`) naming a file that
  does not exist is an error; it logged a warning and ran on the packaged
  settings. "1 task(s) failed" and "plugin(s)" are pluralized.

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

- **`configure --profile` stranded a session's documents.** On a `speed`
  session holding documents, `configure --profile standard` printed "Applied
  profile: standard" and exited 0; switching bge-small to `speed` (both 384
  dimensions) did the same, and search then returned 0 results while `status`
  said "Documents: 1". The guard compared embedding sizes in the new model's
  empty collection, so `--force` never had anything to override. Any write to
  a session's configuration (profile switch, patch, save, import, reset) is
  now refused when the session's documents are in another collection than the
  one it would search, naming the choices: a new session, or `--force` and a
  reindex.
- **Eleven MCP tools required `session_id` with `RAG_SESSION_ID` set** (stdio
  and `serve --mcp -s`): `get_document_status`, `get_document_info`,
  `set_active_profile`, `patch_rag_config` and seven more answered a raw
  pydantic error with a URL, and `list_profiles` and `get_profile_stats` used
  the session "default" instead of the server's. No tool requires `session_id`
  now: every tool takes the argument, else the server's session, else says
  how to set one. `set_active_profile`, `patch_rag_config`, `save_rag_profile`,
  `delete_rag_profile` and `set_lifecycle_policy` list `session_id` after
  their required arguments (MCP clients pass arguments by name).
- **The MCP server over stdio wrote 144 lines to the client's log per
  search**, at DEBUG, in ANSI color, under the name `rag_knowledge_mcp`, with
  FastMCP's "Processing request" lines and a "suspicious citations" warning for
  every Markdown file (only a PDF has pages). It logs at WARNING on stdio (INFO
  on SSE; `--log-level` or `LASAGNA_LOG_LEVEL` to change), in color only on a
  terminal, as `lasagna.mcp`, and leaves the stdlib root logger alone, also
  inside `lasagna serve --mcp`. `serverInfo.version` is Lasagna's, not the MCP
  SDK's; the startup line counts the tools a client is shown (it said 25 while
  `tools/list` returned 23); `--help` reads `lasagna-mcp` and names
  `RAG_SESSION_ID`; tool descriptions name only profiles that exist. Search
  results no longer carry an invented example sentence ("The cost was $1,370
  [1]") among their citation instructions.
- `set_lifecycle_policy` moves out of the default `safe` MCP tool set: it is
  server-wide, and a grace period of zero or a document TTL makes the next
  garbage collection delete other sessions' documents. `patch_rag_config`
  stays: it changes only the caller's session and refuses a change that would
  leave stored documents unsearchable.
- `lasagna serve` on a busy port printed its banner and failed 6 s later with
  "[Errno 48] address already in use"; it checks the port first and names
  `--port`. It no longer warns on every start that APScheduler is missing
  (garbage collection runs on request; the `lifecycle` extra schedules it),
  says how to build the configurator page from a source checkout instead of
  linking a 404, names the sessions that have documents beside the default
  one, and stops without "1 leaked semaphore" (tqdm's cross-process lock,
  which it no longer creates). `/docs` loads Swagger UI from a CDN, as FastAPI
  does; endpoints.md says so and points offline readers at `/openapi.json`.
- **Nothing else could use the store while `lasagna serve` ran.** `status`,
  `search`, `upload` and `lasagna-mcp` loaded their models for about 7 s, then
  failed with "Another process is using the local store". `serve` records its
  address in `<data dir>/serve.json`; the CLI then goes through its REST API,
  and stops at once, naming the fix, when another process holds the store.
  `lasagna serve --mcp` serves the MCP tools at `/mcp/sse` from the same
  process, and a `lasagna-mcp` tool call answers at once with that URL. An
  MCP client disconnecting no longer logs a traceback.
- **A corpus built from Python could not be searched from anywhere else.**
  After an SDK upload with `config_for_speed()`, `lasagna status` showed
  BGE-M3 and the CLI, REST and MCP searched with it: the client's settings
  lived in its process only. A client saves its settings as its session's on
  the session's first upload, as `lasagna configure` does; a session with
  settings of its own keeps them. `lasagna status` shows them.
- **An unconfigured session answered `500 Internal server error`.** The
  README's curl with no session header, or on a session not yet configured,
  started a 2.3 GB BGE-M3 download inside the request or, offline, failed with
  a 500, while `/health/ready` reported the model up from its name alone. A
  server (`lasagna serve`, `lasagna-mcp`) now answers `503` naming the model,
  the sessions that have documents and `lasagna configure --profile speed -s
  <session>`, and never downloads a model inside a request unless
  `allow_model_download_in_requests` is set; readiness reports the served
  session's model as down when it cannot load. MCP `get_system_status` reports the
  session's collection and model (it read the default one: `row_count 0` for a
  session with 76 chunks), and `lasagna search` on a session with no documents
  names the sessions that have some instead of loading a model.
- **The top search result always scored 1.00, and each surface returned a
  different number of results.** Scores were min-max normalized per query, and
  REST and MCP dropped results below `similarity_threshold` on that relative
  score: one query returned 5 results from the CLI, 3 from REST at `top_k` 5
  and 2 from MCP at `top_k` 3. A result's `score` is absolute (the reranker's
  probability, or (cosine + 1) / 2 for dense search), REST results carry their
  `rank`, MCP citations report the same score, and no surface applies a
  threshold unless the request sets `min_score`. The evaluation ranks by order
  and keeps its thresholds; only the per-document score it records changes.
- **A stack configured for a Milvus server opened Milvus Lite.** The startup
  backend always passed a local `db_path`, which `MilvusBackend` took over the
  `uri`, so `backends/milvus.yml` and `MILVUS_URI` wrote to a file inside each
  container. A configured server address wins, the log names the store opened,
  and a `db_path` set together with a server address is an error naming both.
- **Orphan repair deleted the vectors of documents it should keep.** Detection
  counted only `ready` rows, so a document still uploading, processing or
  indexing, one in `error` and an archived one were vector-only orphans, and a
  repair (the Celery scan with `dry_run=False`, `POST /admin/gc/repair`)
  deleted their vectors. A vector is an orphan only when no row in any status
  owns it, and repair checks again before deleting.
- **Without the `celery` profile, uploads waited for a worker that never
  ran.** Compose set `RAG_CELERY_ENABLED=true` whatever the profiles, and
  with Redis reachable the api queued every upload. `make` now derives the
  setting from `COMPOSE_PROFILES`, `make validate` refuses `true` without the
  profile, and the api queues an upload only when a worker answers on the
  broker, otherwise processing it in its own threads and logging why. GC is
  scheduled inside the api in the same case; it was left to a Beat that was
  not running.
- **The Docker image had no Qdrant, Weaviate or Pinecone client**, so the
  stack built with those fragments never passed its readiness check. The
  Dockerfile's `VECTOR_BACKEND` build argument installs the client (and fails
  the build if it does not import), the three fragments set it, and `make
  validate` refuses an existing image built without that client.
- **No upload ever reached the Celery worker, and the worker ran no task.**
  The api dispatches from a threadpool thread, where Celery's current app was
  its unconfigured default (AMQP on localhost), so every `.delay()` failed
  with "Connection refused" and the api processed the upload itself. And the
  worker's `autodiscover_tasks` imported a module that does not exist, so it
  discarded every upload and GC run as an unregistered task while its
  healthcheck passed. The app is the default for all threads and lists the
  task module; found running the live tests against the compose stack.
- **A session's settings overrode the environment with the settings file's.**
  Search reads a session's settings above the environment, and was handed the
  active profile's whole configuration, settings file included. With the
  persistent session backend (the default of `lasagna` and `lasagna-mcp`),
  every session searched with the file's `vector_backend` and
  `embedding_model` whatever `RAG_VECTOR_BACKEND` and `RAG_EMBEDDING_MODEL`
  said; with the in-memory one, any session that applied a profile did. A
  profile now sets only its own settings. The compose stack and the
  Kubernetes ConfigMap set `RAG_SESSION_BACKEND=persistent`, so the api's
  workers, the Celery worker and the MCP server see one session's settings
  alike; in memory, two requests of a session could get different settings.
- **A process that had read a session never saw later changes to it** with
  session settings in PostgreSQL, so a profile applied through one api worker
  missed the other workers, the Celery worker and the MCP server; and the
  worker indexed with the model it had loaded, not the session's, into a
  collection nobody had created. Sessions are re-read once a second, and the
  task makes the session's embedder and collection current before indexing.
- **On Qdrant and Pinecone a profile could not be applied once its model's
  collection existed** (409, "could not read the embedding size"): the size
  was read only from a field named `embedding`, Milvus's name. Any dense
  vector field is read now. Weaviate, pgvector and OpenSearch still report no
  size.
- **`KnowledgeClient.from_url` failed against a server with auth on, and its
  `close()` raised `AttributeError`.** It never sent `X-Session-Token`, so
  `RAG_REQUIRE_AUTH=true` answered 401 to every call. With `api_key` (a key
  from `RAG_API_KEYS`) it mints a session token on the first 401;
  `session_token=` sends one minted already.
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
- The Redis and broker URLs, password included, were logged at startup. So
  were the Redis cache and session store URLs on connect, the Milvus, Qdrant
  and Weaviate URLs, and the SDK's server URL; every logged URL, URI and DSN
  now has its password replaced by `***`, a PostgreSQL key=value DSN
  included.
- A URL no route matched became its own label value in the request histogram;
  it is `unmatched`. The rate-limit counter `rag.http.rate_limited` had the
  same problem for every URL, matched or not (one series per document id):
  it is labelled with the route template or `unmatched`.
- The request id reached only the JSON log format. The server's default text
  format shows it as `[<id>]` before the message, and takes its level from
  `RAG_LOG_LEVEL` as the JSON format does.
- Every search logged "SOTA search: ... hybrid" at INFO, also when hybrid
  search was off. It is a DEBUG line naming the actual mode. The API
  description no longer says "Production-ready".
- Run records, the stage manifest and the MCP status name the chunker that
  runs. A structural strategy with its setting turned off records `semantic`;
  comparability keys change only for such arms.
- `benchmark leaderboard <folder>` failed with "Unknown format".
- `lasagna serve` and MCP tools on a store another process holds logged an
  ERROR line in addition to the reply that says so.
- **Orphan repair archived every document when the vector scan failed.** A
  scan error was read as "no vectors", so every ready document looked
  orphaned and `POST /admin/gc/repair?dry_run=false` (or the weekly scan)
  archived all of them; a scan that stopped at its 50,000-document limit did
  the same past the limit. Such a report is `incomplete`, names the reason
  and lists no orphans, and repair returns `aborted` without changing
  anything.
- **Two sessions on different embedding models, served at once, could mix
  their vectors.** Uploads and searches switched an embedder every request
  shares and read it back a few lines later, so concurrent uploads on
  bge-large and bge-m3 (both 1024 dimensions) could write one model's vectors
  into the other's collection, and a search could embed its query with the
  other session's model. Each request now holds its own session's embedder.
  A process keeps each model it has loaded (up to five) instead of reloading
  one whenever the session changes. The same held for MMR: each search set its
  embedder on the reranker all searches share, so a concurrent search on
  another model could make MMR rank this search's results with that model's
  embeddings. The embedder now goes with the search's rerank call.
- The CI lint job failed on its first run (two mypy errors in
  `sdk/integrations/inbound.py`), the secret scan on nine false positives
  missing from `.secrets.baseline`, and the step that runs the tests loading
  real embedding weights on the openai, cohere and voyage embedders, which the
  conformance test constructed without their package or key (it now passes
  them a stand-in client). `scripts/ci-local.sh` now also runs the
  license check, the dependency audit and the internal-reference grep as CI
  does, and with `--full` the release build, the distribution check
  (`scripts/check_dist.py`, shared with the release workflow) and the wheel
  smoke test.
- **Each API worker's keyword index went stale.** The in-memory BM25 index
  was filled from the store once per session per process, and only the
  process that uploaded or deleted a document updated its own. With several
  workers and Celery, a document processed elsewhere never reached a worker's
  keyword search, and a deleted document kept matching in the others. The
  metadata database keeps a content generation per session, bumped on every
  upload and delete, and a process whose index is behind reads the store
  again on its next search. Keyword-only hits whose chunk is gone from the
  store are dropped. The search cache, also per process, follows the same
  generation: on the compose stack another worker served a deleted document
  from its cache for up to the 300 s TTL.
- After an upload, the uploading process indexed the chunks for keyword
  search under their position instead of their stored id, held each chunk
  twice once it read the store, and could return the same chunk twice.
- **Batch uploads mostly failed without Celery.** A batch may hold 20 files
  but the processing pool has 5 threads, and the batch route refused a file
  whenever every thread was busy, so files 6 to 20 came back "threads are
  busy". Each file of a batch now waits for a thread (at most
  `processing_timeout`); a single upload into a full pool is still a 503.
- Hybrid search on a Milvus server reported `degraded_stages: ["hybrid"]` on
  every search: the backend claimed native hybrid, which needs a
  `sparse_embedding` field the collections Lasagna creates do not have, so
  each search tried it, failed and fell back, and the BM25 index it fell back
  to was never filled. Native hybrid is claimed per collection, only where
  that field exists; otherwise BM25 fusion is the normal path.
- On/off environment variables accept the same spellings everywhere (`1`,
  `true`, `yes`, `on` and their opposites, any case). `OPENSEARCH_USE_SSL` and
  `OPENSEARCH_VERIFY_CERTS` read only `true`, so `yes` or `1` left SSL and
  certificate checks off; `on` was ignored by the lifecycle settings,
  `RAG_EMBED_MCP`, `RAG_CONFIGURATOR`, `RAG_ALLOW_HEADER_IDENTITY` and
  `RAG_LANGFUSE_TRACING`.
- Values in the settings file (`LASAGNA_SETTINGS_FILE`, or the packaged
  `settings.lasagna.toml`) were merged without the checks environment values
  get, so `chunk_size = 5` or `database_backend = "postgres"` ran as written.
  They are checked the same way (type, range, choices), and the error names
  the file, each key and its value.
- `docs/reference/settings.md` lists the variables that are not settings
  (`RAG_REQUIRE_AUTH`, `RAG_API_KEYS`, the batch and search limits,
  `LASAGNA_LOG_FORMAT`, `LASAGNA_MCP_TOOLS`, `MILVUS_URI` and the other vector
  store connections, the paths), generated from one list in
  `core/config.py`. A `RAG_` variable the package reads that is missing from
  the list fails the suite.
- `get_backend("milvus")` with no arguments ignored `RAG_VECTOR_BACKEND_CONFIG`
  and the settings file's `vector_backend_config` and opened Milvus Lite: it
  read the session configuration through a module path that no longer
  existed. It follows the rule the server's startup backend follows (`uri` in
  `vector_backend_config`, else `MILVUS_URI`, else Milvus Lite).
- `--resume` restored checkpoints made under other library versions (torch,
  transformers, sentence-transformers, ragas, pymilvus), another settings
  file, another `variance_source` or another `max_concurrency`. The resume
  key includes them, so such checkpoints are not restored.
- `results.json` was written in place, so a crash or a full disk mid-write
  left a truncated file where the last good one had been. It is written to a
  temporary file, synced and renamed into place, like checkpoints.
- **Every upload failed on the Celery worker in Kubernetes.** The manifests
  turned Celery on but mounted no volume, so a worker pod could not open a
  file an api pod stored. Both Deployments mount a ReadWriteMany claim,
  `lasagna-uploads`, at `LASAGNA_UPLOAD_DIR`, and `make validate-k8s`
  checks a rendered overlay for it.
- The MCP `search` tool and the SDK's `SearchResults` did not say when a
  stage fell back (hybrid running dense-only, HyDE without an LLM), which the
  REST API reports in `degraded_stages`. The MCP result's `<search_metadata>`
  and `SearchResults.degraded_stages` name them, also for a remote client.
- The image installed whatever torch, transformers and sentence-transformers
  were newest at build time; it installs the versions in
  `constraints/0.1.0.txt`, which the release was tested with. The jaeger
  service is pinned to `1.76.0` instead of `latest`.
- **The image carried `deploy/.env`.** With no `.dockerignore`, `docker build`
  copied `src/lasagna/deploy/.env` (the JWT secret, the Postgres password, API
  keys) into an image layer. A root `.dockerignore` keeps every `.env` except
  `.env.example`, key files, virtual environments, `.git`, build output,
  caches and test artifacts out of the build.
- `GET /` advertised `/configurator`, which answered 404 in a source checkout
  where the page is not built; it now says it is not built and how to build
  it, as the `serve` banner does. `/docs` loads Swagger UI from a CDN and was
  blank offline: it serves the `swagger-ui-bundle` package's copy when that
  is installed, and otherwise shows, when the bundle cannot load, where the
  schema and the endpoint reference are.
- The deployment guide's quick start begins with cloning the repository,
  which the stack builds from, and says why Kubernetes embeds with
  `BAAI/bge-large-en-v1.5` while compose uses `BAAI/bge-m3`, what that means
  for a corpus moved between them, and how to align them.
- The docs disagreed: the default first-run download was 4.5 GB in the
  README and the guide, 3.1 GB in docs/configuration.md and `lasagna init`,
  and 5.4 GB in code comments. Every page states 3.1 GB, the default
  profile's weights as `core/setup_text.py` records them, and the README and
  guide add the 2.3 GB the `flag-embedding` extra adds (FlagEmbedding also
  fetches BGE-M3's ONNX copy). CONTRIBUTING no longer cites a CHANGELOG
  "Known issues" entry that does not exist, and gives the guide's measured
  pip time. The SDK's getting-started page no longer makes an LLM key a
  prerequisite, and the RAG skill says a keyless complex query falls back.
- Release hygiene: the MCP server no longer registers the deprecated
  `search_knowledge` and `search_with_context` aliases in a first release (29
  tools; `--tools safe` still serves 22), `lasagna.TYPE_CHECKING` is no longer
  importable from the top-level package, and runtime messages (the model
  license warning, the power verdict, several errors) use plain punctuation
  instead of " -- ".
- MCP tool descriptions were reST docstrings, `:param:` fields and double
  backticks included. A description is the docstring's prose as plain text,
  ending with what the tool returns, and each parameter's text is that
  parameter's `description` in the input schema, where clients look for it.

### Known issues

- **Quotas are counted per session, not per API key.** With auth on, rate
  limits are counted per key, but the document, storage and chunk quotas are
  per session, and one key can mint any number of sessions. Counting them per
  key needs a durable record of which key made each session, readable by the
  Celery worker and by garbage collection, which rebuild the counters per
  session; it is planned. `src/lasagna/deploy/README.md` (Request limits) says
  what to do meanwhile.

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
