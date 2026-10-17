# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

The first public release. Lasagna is a RAG framework built for comparing one
part of a pipeline with everything else held fixed: eight swappable stages and
55 built-in components (each one that implements a published method names its
paper), an evaluation harness that says whether a difference is real (paired
significance and interaction tests, power, a stage isolation check, checkable
result bundles), and the same pipeline served to agents over MCP, to programs
over REST and the Python SDK, and in a Docker Compose stack.

Its metric definitions are `METRICS_VERSION` 1.7
([metrics.md](src/lasagna/knowledge_evaluation_sdk/docs/metrics.md) defines
each metric). The version is part of every comparability key, so results from
the internal 0.1.0a1 pre-release (1.1) or from development builds since (1.2 to
1.6) never share a leaderboard group with results from this release.

Users of 0.1.0a1: read [Changed for users of 0.1.0a1](#changed-for-users-of-010a1)
below. Every fix made between 0.1.0a1 and this release is described in the git
history.

### Added

Components, by stage (`lasagna component list [stage]` lists them without
loading any):

- **Generator, an eighth stage.** `concise` (the harness's own prompt), `cited`
  (ALCE), `chain_of_thought`, `chain_of_note`, and the baselines `closed_book`
  (the question alone) and `long_context` (all ten retrieved passages). A
  function `f(query, contexts) -> str` becomes one with `@lasagna.generator`.
  An arm with `no_rag: true` retrieves nothing and answers with `closed_book`.
- **Retriever:** `dense`, `bm25`, `splade`, `hybrid_rrf` and `hybrid_convex`.
- **Reranker:** `cross_encoder` (any current Hugging Face reranker by name),
  `colbert`, `monot5`, `llm_listwise` (RankGPT's sliding window),
  `llm_pointwise`, `flashrank`, `cohere`, `mmr`, `combined` and `ensemble`.
- **Query enhancer:** `hyde`, `query_expansion`, `query_decomposition`
  (least-to-most), `step_back`, `query2doc`, `rewrite` and `multi_query`.
- **Agentic pipeline:** `self_ask`, `ircot`, `flare`, `adaptive` and
  `self_rag`, each with its paper's defaults as settings; `loop_trace` records
  what each loop did.
- **Chunker:** `semantic` (with Kamradt's breakpoint methods),
  `structure_aware`, `hierarchical`, `fixed`, `recursive`, `sentence_window`,
  `markdown_header`, `proposition` and `contextual`.
- **Embedder:** `sentence_transformers`, `bge_m3`, `openai`, `cohere` and
  `voyage`, with each model family's query and document prefixes.
- **Document processor:** built-in extraction, and `docling`, `markitdown` and
  `pymupdf4llm` (each an extra; `pymupdf4llm` is AGPL-3.0 and not in `all`).
- **Your own component in one function.** `@lasagna.reranker` (and a decorator
  per stage) turns a function into a component a sweep names as
  `module:function`; `lasagna component validate` checks it keeps its stage's
  contract. `lasagna.from_langchain` and `lasagna.from_llamaindex` register
  LangChain and LlamaIndex objects as stages (extra: `llamaindex`).

Experiments:

- **Sweeps** (`lasagna benchmark experiment sweep.yaml`) over components and
  settings, factorial designs, repeated runs and baseline arms; chunker and
  embedder arms get corpora of their own. A key the harness does not read is
  an error naming the likely intended one.
- **Your own data as a benchmark.** A JSONL or CSV file, or a BEIR folder:
  `benchmarks: [{path: data/my.jsonl}]` or `lasagna benchmark run ./my.jsonl`.
  It registers as `local/<stem>@<12 hex digits of its sha256>`, so an edited
  file is a different benchmark. `lasagna benchmark validate-data` names each
  bad row and field. See [docs/own-data.md](docs/own-data.md).
- **`lasagna benchmark synthesize`** writes factoid, multi-hop and unanswerable
  questions from your documents with the configured LLM, for development
  rather than for reported results.
- **`--dry-run`** prints the arms, tasks, models and whether each is
  downloaded, with token, time and (from a price table, `prices:`) dollar
  estimates from a 3-sample probe.
- **A re-run replays its LLM responses.** Requests at temperature 0 are
  answered from `~/.lasagna/cache/llm.sqlite` inside an experiment; the hit
  rate is in `results.json` and the bundle. Off with `llm_cache: false` or
  `--no-cache`.
- **Metrics:** NDCG, recall, precision, MRR, MAP, R-precision, complete
  recall, hit rate and F1 at `k`; exact match, token F1, ROUGE-L, answer
  containment and similarity, ALCE citation recall and precision, abstention
  rate and SQuAD v2's answerable and unanswerable splits; the RAGAS judged
  metrics with `factual_correctness`; latency, tokens and LLM calls per query.

Results:

- **`lasagna results significance`** (paired sign-flip test, Holm correction)
  ends in a verdict naming each pair that survives and who won;
  `results power` gives the minimum detectable effect; `results interaction`
  tests a 2x2 design.
- **`lasagna results view`** writes a run as one offline HTML page:
  leaderboard, significance, every metric, the isolation check and a per-query
  diff with what each arm retrieved.
- Every `lasagna results` command takes a run's folder (the one a run prints
  after "Output:") or the `results.json` in it.
- **Bundles** (`results bundle`, `results validate`) carry what a reader needs
  to decide whether two numbers may be compared, with one comparability key per
  benchmark and configuration; `results leaderboard` groups by it, as markdown
  pages or LaTeX tables (`--format latex`).
- **Stage isolation.** Every run records a manifest of each stage's settings;
  arms that differ outside the stages they name block the bundle, and an arm
  whose named change never moved a ranking (on at least ten queries) is
  flagged inert.
- **A reproduction:** on the full SciFact and NFCorpus test sets the dense
  models land within 0.002 NDCG@10 of MTEB, and BM25 with a cross-encoder
  within 0.004 of the BEIR paper (`examples/reproduction/`).

Serving:

- **MCP server** (`lasagna-mcp`, stdio by default, or `lasagna serve --mcp`)
  for Claude Code, Claude Desktop and Cursor. The default tool set, `safe`,
  serves no deletion tools (`--tools all` does).
- **REST API** (`lasagna serve`, or the Compose stack) under `/api/v1`,
  covered by the deprecation policy in [GOVERNANCE.md](GOVERNANCE.md); search
  responses name the stages that fell back in `degraded_stages`.
- **Profiles** `speed` (92 MB, no LLM), `standard`, `balanced` and
  `max_quality`, applied per session from any surface. A switch that moves a
  session's documents to another model (`lasagna configure --reindex`,
  `reindex=true` over REST and MCP) checks the new models load, reindexes every
  document first and switches only when all are reindexed.
- **Deployment:** a Compose stack (Postgres, Redis, a Milvus server or Qdrant,
  Celery, MCP) with per-service healthchecks, Kubernetes manifests, Prometheus
  metrics, JSON logs with request ids, and readiness that pings each store.
- **Six vector stores** behind one interface: Milvus (and Milvus Lite),
  pgvector, Qdrant, Weaviate, Pinecone and OpenSearch.

For contributors: the test suite keeps the Hugging Face hub offline unless
`HF_HUB_OFFLINE=0`, and tests that load real weights skip when their model is
not cached.

Documentation and examples: the [guide](docs/guide.md),
[configuration](docs/configuration.md) (which setting wins, profiles, where
data lives), the generated [settings reference](docs/reference/settings.md),
[model licenses](docs/model-licenses.md), a Colab notebook comparing a
one-function reranker with the built-in cross-encoder, the plugin template,
and an offline acceptance demo (`examples/acceptance/offline_demo.py`).

### Changed for users of 0.1.0a1

#### Metric definitions (`METRICS_VERSION` 1.1 to 1.7)

Each step changes what a number means, so each is part of the comparability
key; `results bundle` refuses a `results.json` whose metrics version is missing
or not this build's, saying how to run it again.

- **1.2:** the held-fixed block of a result describes the run, not the
  machine. It read the process's settings, so a `speed` run (all-MiniLM-L6-v2)
  published the settings file's embedder, and two runs on different models
  shared a key.
- **1.3:** held-fixed fields come from what each task ran, and BM25 stemming is
  part of the key.
- **1.4:** exact match and token F1 count an answer as an abstention only when
  it opens with one, and normalize in the official SQuAD order. The key covers
  the settings of every stage a run does not vary (`fixed_stages`), so a
  dense-only and a hybrid run no longer share one.
- **1.5:** exact match and token F1 score the answer with its `[n]` citation
  markers removed (as ALCE does); NDCG gives a negative grade zero gain (as
  `trec_eval` does); the judge is also asked for `factual_correctness`; the
  answer-similarity model (`similarity_model`) is in the key. New per-query
  metrics are added, not redefined.
- **1.6:** the retrieval track scores the ranked list. Under a profile with
  context reordering on (`standard`, the default `base_profile`, `balanced`
  and `max_quality`) it scored the order the generator reads, so the
  second-ranked document fell below the cut to `k`; the generator read ranks
  1, 3, 5, 7 and 9 of ten, and now reads the top five reordered. `speed`
  numbers do not move.
- **1.7:** a chunk id that repeats in a search's candidate list takes one place
  before the reranker and the cut to `top_k`. Numbers move only for a corpus
  whose store held a chunk twice (earlier builds could ingest a corpus on top
  of itself), and only for dense-only arms; BM25, SPLADE, hybrid and fused
  arms merge by chunk id and cannot move.

The key changed in three more ways, without a new version: a run on a corpus
that failed partly to ingest gets a key of its own (`corpus_ingest_gap`);
`corpus_provenance` defaults to `native`, so older bundles keep their keys; and
each benchmark of a multi-benchmark run has its own key (later benchmarks
carried the first one's). A single-benchmark run's key is unchanged.

Other numbers move because 0.1.0a1 measured something else under a name: the
`mmr`, `combined` and `ensemble` rerankers, embedder and chunker arms (which
embedded or chunked the corpus with the built-in model or settings), the
`semantic` chunker (which chunked by size), `llm_listwise` (window 20, step
10, as RankGPT), `query_decomposition` (least-to-most prompting), BGE-M3
queries (no BGE v1.5 instruction), SQuAD v2 (its unanswerable half is scored),
latency (it includes writing the answer) and the paired tests (a sign-flip
randomization test, where the bootstrap rejected 9.1% of true nulls at 12
queries). Checkpoints from 0.1.0a1 do not resume.

#### Renamed: the old name works through 0.1.x, with a warning

Each is removed in 0.2.0.

| Old | New |
| --- | --- |
| `lasagna benchmark compare` | `lasagna results compare` |
| `lasagna benchmark leaderboard` | `lasagna results leaderboard` |
| `lasagna results interaction --bootstrap` | `--resamples` |
| `from lasagna import <name>`, for the 56 names dropped from `lasagna.__all__` | the module the `DeprecationWarning` names |

The unversioned `/api/config/knowledge` routes are for the configurator page;
other callers use `/api/v1/config`, and they move under `/api/v1` in 0.2.0.

#### Renamed or removed without an alias

- The data directory is `~/.lasagna` (was `~/.rag_knowledge`, renamed on first
  use when `~/.lasagna` does not exist; `LASAGNA_DATA_DIR` names another).
- `-s` means `--session` in every command and `-p` means `--profile`:
  `benchmark run` spells out `--split` and `--mode`, and `serve --port` has no
  short flag.
- The MCP server announces itself as `lasagna` (was `rag_knowledge`); session
  tokens are issued as `lasagna`, and tokens issued as `rag-knowledge` stay
  valid until they expire. The `sota_features` block of MCP
  `get_system_status` is `features`.
- `lasagna evaluate` is removed: `lasagna benchmark run <file>` measures your
  own data.
- Every setting has a `LASAGNA_` variable, which wins over the `RAG_` one; the
  `RAG_` names keep working with no warning. `config_for_quality()` is an alias
  of `config_for_max_quality()`.
- `lasagna.sdk.__all__` lists the 33 names a client of a knowledge base
  imports; every other name it listed still imports from `lasagna.sdk`.

#### Defaults that changed

- **Sessions' settings are kept in one store for every surface.** The CLI, the
  MCP server, `lasagna serve` and the SDK read and write the metadata
  database, so a profile set with `lasagna configure` applies to a
  `KnowledgeClient` for that session in another process.
  `RAG_SESSION_BACKEND=memory` keeps a process's sessions to itself. Changes
  that several processes make at once to one session, or to the saved
  profiles, are all kept. A
  session's settings stay while it holds documents; those of a session without
  documents are deleted after `session_idle_days` (30) without a read or write.
- **One vector collection per embedding model**
  (`rag_documents_v2__<model>__<dim>`). An existing `rag_documents_v2` is kept
  under its name by the first model that opens it at its size. A switch that
  would leave a session's documents in another model's collection is refused
  unless it reindexes them (`--reindex`) or is forced (`--force`), and a
  search or `lasagna doctor` on such a session says so.
- **`lasagna benchmark run`** measures retrieval unless `--tracks` names
  generation, and starts from the `speed` profile.
- **`lasagna-mcp`** speaks stdio and serves the `safe` tool set; `lasagna
  serve` logs warnings only and opens no browser unless `--open`.
- **A bad environment setting stops the process** with `ConfigError` naming the
  variable, its value and what it takes; a misspelled `RAG_` or `LASAGNA_`
  name warns with the setting it is close to.
- **The judge defaults to gpt-6-luna**, built by the harness so its tokens are
  counted, and the default LLM of each provider is that provider's current
  model.
- **`lasagna component validate`** builds a registered component with the
  `speed` profile's model (`--profile` chooses another).
- **With `RAG_REQUIRE_AUTH` on, quotas count per API key.** Minting a session
  needs a key from `RAG_API_KEYS`, a key mints at most
  `RAG_MAX_SESSIONS_PER_KEY` sessions (100), and its sessions' documents,
  storage and chunks count toward its quota. The MCP `get_system_status` tool
  needs a session token, and its paths and counts the admin token. Auth off:
  unchanged.
- Request bodies are limited before they are read (413), a REST search that
  runs past `RAG_SEARCH_TIMEOUT` (120 s) gets 504, and every numeric session
  setting has a maximum (`SESSION_MAXIMUMS`) that applies to REST, MCP and
  imports, never to the evaluation harness.
- A model a session names (`monot5_model`, `splade_model`, `colbert_model`,
  `flashrank_model`, as the embedder and the cross-encoder already did) follows
  the server's download policy instead of downloading inside a request.
- Uploads, reindexes and deletes of one document wait for each other across
  processes (a PostgreSQL advisory lock, or a file lock beside a SQLite store).
  A document that fails to index leaves no chunks behind, search returns
  chunks of ready documents only, and uploading a file again whose copies all
  failed retries the newest one under its id.
- The Compose stack no longer publishes Redis on the host
  (`-f expose-redis.yml` publishes it on loopback for debugging).
- The Docker image runs torch's CPU build (3.8 GB; `TORCH_VARIANT=cuda` builds
  the GPU one), and the Compose stack defaults to a Milvus server.
- `sentencepiece` is a core dependency, for the `monot5` reranker's tokenizer.

### Known issues

- **Windows needs WSL.** The default store, Milvus Lite, does not install on
  Windows, so `pip install lasagna-rag` succeeds and the first upload fails.
  Use WSL, or point Lasagna at a Milvus server or another vector store
  (`RAG_VECTOR_BACKEND`).
- **mcp 2.x is not supported.** `mcp` is pinned below 2.0 (2.0 removed
  `mcp.server.fastmcp`, which the MCP server imports), so an environment that
  needs mcp 2.x cannot also install Lasagna. Install it in an environment of
  its own (`pipx install lasagna-rag` or `uv tool install lasagna-rag`) and run
  `lasagna-mcp` from there.
- **Docling's models are not under the download policy.** With the `docling`
  extra installed (the settings file turns Docling on), the first document it
  converts downloads Docling's layout models, inside the request on a server
  that otherwise refuses downloads. Download them before serving
  (`docling-tools models download -o /path/to/models`, then set
  `DOCLING_ARTIFACTS_PATH=/path/to/models` for the server), or turn Docling off
  with `RAG_DOCLING_ENABLED=false`.
- **Chunk boundaries depend on nltk's `punkt_tab`.** Lasagna never downloads
  it; without it sentences are split with a regular expression, so the same
  corpus can chunk differently on two machines (a run's bundle records the
  fallback). Run `python -m nltk.downloader punkt_tab` once on every machine
  whose numbers you compare.
- **A local Milvus Lite store opens in one process at a time.** While
  `lasagna-mcp` (stdio) or a benchmark holds it, `lasagna status`, `search` and
  `upload` stop with the fix; run `lasagna serve --mcp` so the CLI and agents
  share it, or use a Milvus server.
- **pip takes about two minutes to resolve `.[all]`.** uv resolves it in
  seconds (`uv pip install "lasagna-rag[all]"`).

### Planned for 0.2.0

- Export and import of a session's documents and chunks.
- A prebuilt Wikipedia index.
- Multilingual benchmarks.
- Batched generation with vLLM.
- An OpenAI-compatible answer endpoint, and streamed answers.
- Weights & Biases and MLflow logging hooks.
- Claim-level diagnostics of generated answers.
- Calibration of the LLM judge against human labels.
- Removal of the old names in "Renamed: the old name works through 0.1.x"
  above.

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
