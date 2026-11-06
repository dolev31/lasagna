# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-06

Chat with a corpus from any OpenAI client, a knowledge graph every process
shares, and benchmarks that say when a stage fell back. Thank you to
[@segevshlomovIBM](https://github.com/segevshlomovIBM), for the stage isolation
check that now excuses only the settings an arm names (#25), and for the
evidence runs that found the hybrid arms running dense-only in the Compose
stack (#50, #52), fixed here.

### Added

- A benchmark's run record says how each graph arm's corpus got its knowledge
  graph: `system.graph.state` is `built`, `reused`, or `rebuilt` when a reused
  corpus had none (ingested before the graph was stored in the metadata
  database), whose documents are then extracted again from the chunks
  already stored (#55), as are documents another extraction model did.
  `incomplete`, with how many documents are `missing`, when some could not be
  extracted (no LLM, a failed call); every query of that arm counts as a
  `graphrag` fallback, so the bundle says so. A document that names no entity,
  or whose replies held no readable JSON (`unparsed_chunks`), counts as
  extracted. The benchmark ingest stores a
  document's graph only once the document is in, so a corpus whose ingest
  failed leaves none. Two graph arms that differ in `graph_extraction_model`
  or `graph_entity_types` get a corpus each, and an arm that names no model is
  keyed by the default model. That key is new: a graph arm's corpus ingested
  before it is ingested once more.
- `POST /api/v1/chat/completions` and `GET /api/v1/models`: chat with a
  session's documents from any OpenAI client library or chat front end (#13).
  The route searches with the session's pipeline, as `/search/citations` does,
  under the same session header, auth, deadline and quota, and writes the
  answer with the `cited` generator from the configured LLM. It answers in the
  OpenAI format, at once or streamed as server-sent events ending with
  `data: [DONE]`, with the passages as `citations` (`[n]` in the answer is
  `citations[n-1]`) and `degraded_stages`. The answer runs under the new
  `RAG_CHAT_TIMEOUT` (120 s); the LLM is checked before the search, and an
  answer that fails is not retried by the client (`x-should-retry: false`), so
  nothing is searched for an answer that cannot be written. The endpoint reference, the REST
  guide and the README show how; every stated endpoint count is now checked
  against the API's OpenAPI schema, and every route against the reference.

### Fixed

- The knowledge graph `enable_graph_retrieval` builds is kept in the metadata
  database every process shares, so a search finds the graph whichever process
  built it (Part of #55). It lived in the memory of the process that ingested:
  in the Compose stack the Celery worker built it and every API worker searched
  an empty one, silently, and a restart lost it. It is now stored per session
  and collection (SQLite, or PostgreSQL in the stack), each mention and edge
  with its document, after the document's chunks are stored, and each process
  loads it again once the session's documents change (an upload, a delete, a
  benchmark's ingest). Storing a document again replaces its part, so a
  reindex leaves no old chunk ids behind, and an upload that fails keeps none.
  A search no longer creates an empty graph. Concurrent writers lock entities
  in one order, so PostgreSQL does not deadlock them. In
  `lasagna.retrieval.graph`, `get_or_create_session_graph` is deprecated (it
  builds a graph no other process sees, and retrieval no longer reads it); use
  `graph_for` and `index_document_graph`. `GraphRetriever.expand` takes
  `collection_name`, defaulting to the session's.
- Deleting a document removes its part of the session's knowledge graphs, on
  every delete path, and deleting a session drops what is left; a reset
  forgets the graphs a process had loaded, and a stale benchmark corpus drops
  its graph with its vectors (#55).
- Hybrid search asks about the collection it searches, and a connection pool
  passes the question on (#50, #52). The Compose stack wraps its Milvus server
  in a pool (`RAG_CONNECTION_POOL_SIZE=10`) that did not pass on
  `collection_supports_native_hybrid`, so every collection counted as able to
  run native hybrid search. The collections Lasagna creates have no
  `sparse_embedding` field, so every hybrid search tried the native call,
  failed and fell back (all 12 queries of a `standard` run on `smoke`). The
  same answer skipped filling the BM25 index from the store, so wherever a
  process had not indexed the chunks itself, the fallback found it empty and
  returned dense results only. In the stack, the Celery worker ingests
  uploads, so this was every REST and MCP hybrid search an API worker ran: in
  a test of 22 searches on one upload, all 4 that ran (one per worker) fell
  back to dense-only, and the search cache served the other 18 from them. In a
  benchmark, it was a hybrid arm on a corpus ingested with hybrid off: under `speed`
  it scored exactly the control, ndcg@10 0.9464 on `smoke`, where Milvus Lite
  gives 0.9505. In the stack it now gives 0.9505, with map@10 and R-precision
  as on Milvus Lite. The base adapter answers per collection from its
  capabilities (pgvector, OpenSearch and Weaviate as before), Milvus reads the
  collection's fields, and the pool asks one of its instances. The searcher
  never answers yes without asking, and asks about the collection a session
  searches, which the benchmark names before its first search. The legacy
  Milvus searcher serves only the collection it was built for. On a collection
  without the field, `MilvusBackend.hybrid_search` returns dense results and
  reports a degraded `hybrid` stage instead of failing. Numbers move only for
  hybrid search on a Milvus server where it had run dense-only.
- A stage that falls back while a benchmark's queries run is named in the
  results and the bundle (Part of #50). Hybrid search, the graph expansion
  and other stages catch their own failure, carry on with less and record it;
  REST, MCP and the SDK reported those stages, and the benchmark harness
  never collected them. In the Compose stack, every query of a hybrid arm
  under `speed` failed the native call and then ran dense-only, scoring
  exactly the control (#52), and its bundle carried no fallback. Each task now
  records `system.degraded_stages`: per track, the stage, on how many of its
  queries, and up to three reasons with the queries each covered, so a
  dense-only fallback is told apart from one that changed nothing. The bundle
  raises `ACTIVE_FALLBACKS` with one line per arm, track and stage
  (`hybrid fell back during the run: smoke, arm 'hybrid', retrieval track,
  12 of 12 queries (12 queries: ...; 12 queries: BM25 index empty; dense
  results only)`), which `lasagna results validate` prints and `--strict`
  refuses; `lasagna benchmark run`, `benchmark experiment` and
  `lasagna results view` show the same lines. A task where nothing fell back
  records nothing new, so its bundle is what it was. Reasons are redacted
  like the rest of a bundle. A `degraded.collecting()` block opened inside
  another now hands its stages to the outer one.
- A document turns `ready` only once a search finds it. On a Milvus server,
  where searches read at the Bounded level, a search sent the moment a
  document turned ready missed it: in the default Compose stack 27 of 30 such
  searches came back without it, and each found it a second later. Before it
  marks a document ready, an upload or a reindex now waits until the server
  has applied its chunks (one Strong count per document, about 200 ms; a
  benchmark's ingest waits once per corpus);
  searches keep the fast default. Milvus Lite, Qdrant, OpenSearch, pgvector
  and Weaviate searches already saw a write when it returned, so nothing
  changes there. Pinecone is eventually consistent and is not covered: there a
  search sent right after ready can still miss the document.
- The stage isolation check excuses only the settings an arm declares, and
  what the harness sets on their account, rather than every setting in a
  stage an arm names, and judges an inert arm setting by setting. An arm that
  named `hybrid_alpha` hid a `default_top_k` that leaked into the retriever
  stage; a reranker sweep, where every arm names a reranker, hid a leaked
  `rerank_top_k`; an arm that named `hybrid_alpha` and `rrf_k` passed as live
  when only `rrf_k` arrived. The first two now mark the bundle
  `UNDECLARED_STAGE_CHANGE`, naming the settings and the track each leaked on
  that acts on it (`default_top_k on the generation track`); the third is
  flagged inert for the setting that never arrived. The run's isolation line
  and `results view` say the arms differ only in the *settings* they name,
  which is what is checked.
- The stage manifest records the routing and retrieval-depth settings
  (`use_agentic_rag`, `agentic_pipeline`, `routing_default_tier`,
  `similarity_threshold`, `chunk_cite_retrieval`, `default_top_k`,
  `enable_context_reordering`) per track, as `{"retrieval": ...,
  "generation": ...}`, the way `context_order` already was, and
  `system.tracks` records which tracks ran. It digested the values the
  retrieval track forces on every arm, so on a generation track an arm that
  set `use_agentic_rag: true` ran the loop, digested like its control, and
  was called inert; two arms that differed only in `routing_default_tier`
  could not be told apart. The check now judges such an arm per track: inert
  on retrieval, where the setting is forced, and live on generation. For
  anyone parsing `results.json`: each of these settings, under
  `system.stages.<stage>.settings`, is now that dict where it was a single
  value. Stage digests, and so comparability keys, change for every run from
  this change on: a bundle written before `system.tracks` was recorded keeps
  its verdict under `results validate`, which reads the integrity block the
  bundle records, and groups on a leaderboard only with bundles written
  before it; `results bundle` on a results.json written before it runs the
  new check, so it can refuse what passed then.
- The isolation check calls a setting only the agentic loop acts on
  (`max_agent_iterations`, `enable_crag`, `self_ask_*`, `ircot_*`, `flare_*`
  and the rest) inert on the retrieval track, which runs no loop, and so
  speculative RAG's (`enable_speculative_rag`, `speculative_*`), which only the
  answer step reads, since that track makes no answers; it called such an arm
  live there. An arm inert on one track only marks the bundle `INERT_ARM`, a
  warning, so `results validate --strict` fails a full-run sweep of such a
  setting. What the harness sets on one track only (the depth `metric_k` asks
  retrieval for) is excused on that track only, so a `default_top_k` leaked
  into a `metric_k` arm's generation config is caught.
  A pair of arms that shares no track is not compared, and neither is a row
  written before `system.tracks` was recorded paired with a newer one (a
  sweep resumed across the upgrade); each is listed as unverified, and with
  no pair compared the run says why.

## [0.2.1] - 2026-11-04

A patch release: uploads to the default Docker Compose stack are indexed again.
Thank you to [@tnalbant](https://github.com/tnalbant), for the CLI reference
that no longer changes with the installed Typer version, and to
[@segevshlomovIBM](https://github.com/segevshlomovIBM), whose evidence runs on
Milvus Lite and in the Compose stack found seven harness bugs, filed as #46 to
#52 (still open).

### Added

- A dev container (`.devcontainer/devcontainer.json`): a GitHub Codespace or
  the Dev Containers extension sets up Python 3.12, the full development
  install (with CPU-only torch, under the release's constraints) and Docker on
  first start, so the Compose stack a pull request is tested in runs inside
  it.

### Fixed

- Uploads to the default Docker Compose stack (Milvus server, Celery worker)
  are indexed again. Before indexing, the worker clears what an earlier attempt
  at the document left, and a Milvus server refuses a filtered delete on a
  collection that is not loaded, so before any search had loaded it every upload
  ended in `error`. `MilvusBackend.delete` and `delete_raw` now load the
  collection first, as its reads already did.
- An AUTOINDEX index is accepted by a Milvus server, whether it comes from an
  `IndexConfig` (which carried its default HNSW params) or from the collection
  manager with `index_params` set. A server refuses any param but the metric
  with AUTOINDEX; only Milvus Lite, which builds a FLAT index instead, took it.
- The generated CLI reference labels an option's type the same way whichever
  Typer version is installed, so a checkout installed under the release's
  constraints file (Typer 0.26.8) no longer reports the page as out of date and
  rewrites it.
- The evaluation SDK's benchmark catalog lists `2wikimultihopqa` and says 32
  public benchmarks; it had kept 31 since 0.2.0. Every page that states how many
  benchmarks, components or published loops there are is now checked against
  the code, so a pull request that adds one updates each page that counts it.

## [0.2.0] - 2026-11-01

The first release with outside contributors. Thank you to
[@ect2000](https://github.com/ect2000), for the `2wikimultihopqa` benchmark
with its distractor paragraphs and the agentic-pipeline checks in
`lasagna component validate`, and to [@juliasadler](https://github.com/juliasadler),
for the fix to two services racing to create a collection, measured against
the stack's own Qdrant.

### Deprecated

- The old names renamed in 0.1.0 (`lasagna benchmark compare`, `lasagna
  benchmark leaderboard`, `results interaction --bootstrap`, and the names
  dropped from `lasagna.__all__`) and the unversioned `/api/config/knowledge`
  routes stay for one more release, with their warnings, and are removed in
  0.3.0. 0.1.0 said 0.2.0; their warnings now name 0.3.0.

### Changed

- The PyPI page links the documentation site and the paper.

### Added

- A pinned `2wikimultihopqa` benchmark adapter with a validation corpus that
  retains gold paragraphs and distractors, with version-specific IDs that deduplicate
  formatting-only whitespace differences while keeping the first original text.
- A "PR quality" check on every pull request (`scripts/check_pr_body.py`). It
  fails until the description fills in the template, a person names themselves
  as having read the whole diff, every suppression in the added lines gives a
  reason, every TODO links an issue, and a change to `src/lasagna/` comes with a
  test and a line here. The `no-changelog` and `no-new-test` labels waive the
  last two. CONTRIBUTING.md has the full bar under "Quality bar".
- A pull request template, and `CODEOWNERS`.
- Issue forms for a benchmark request and a research proposal. The component
  form now asks for the stage, the paper and the number it reports, and a
  comparison with a built-in control. Blank issues are off; questions go to
  Discussions.
- A [Wanted](https://dolev31.github.io/lasagna/docs/project/wanted/) page: the
  components, benchmarks, studies and deployment work the project would like to
  have, each with what "done" means.
- A pull request checklist in `docs/contributing-components.md`.
- An `IMAGE_EXTRAS` build argument for the Docker image (set in `.env`): extras
  installed beyond the production set, empty by default. `IMAGE_EXTRAS=eval`
  runs `lasagna benchmark` and sweeps inside the Compose stack.
- Every pull request is tested in the production environment, the Compose
  stack built from its branch (CONTRIBUTING.md, "Tested in the production
  environment"). The "PR quality" check fails while "How it was tested" shows
  no `docker compose` command, and the template has a box for it.

### Fixed

- On a Milvus server, the benchmark's corpus census and pre-flight see the
  rows written just before them. A server reads with the collection's Bounded
  consistency, so a count right after ingestion came up short (0 of 24
  documents), and every cold `lasagna benchmark run` in the Compose stack was
  refused as "changed while this run used it". Backends gain `read_fresh()`,
  a block whose filter reads see every earlier write (a no-op by default; on a
  Milvus server it reads Strong, about 200 ms a query, so only these checks
  use it and search keeps its speed). `iter_all` on a server first runs one
  Strong query, since pymilvus's query iterator ignores the consistency level,
  so a scan right after an insert could come up short. `PooledVectorBackend`,
  which the Compose stack runs in front of a Milvus server, passes both on: it
  enters `read_fresh` on every pooled instance, and `iter_all` uses an
  instance's own cursor instead of paging by offset, which read stale and
  stopped at Milvus's 16,384-row limit. The pool also passes on `is_lite`, so
  a pool over Milvus Lite (`connection_pool_size` above 1) is no longer taken
  for a server.
- Two services starting together no longer fail when the other creates the
  collection first: `create_collection` raises `CollectionExistsError` (a
  `ValueError` subclass) on an existing name, and collection initialization
  continues as the "already exists" branch does instead of failing to start.
  The same lost-race handling now covers the third check-then-create path,
  `CollectionManager.ensure_backend_collection`, which the API process and a
  Celery worker can reach at the same time.
- Validate `BaseAgenticPipeline` components with an async contract check and
  report incorrect results or input mutation instead of rejecting the stage.

## [0.1.0] - 2026-10-21

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
below. The fixes made since 0.1.0a1 are described in the git history.

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
  an error naming the likely intended one, and a value out of its setting's
  range is an error naming the arm, before anything is ingested and in
  `--dry-run` too. The ranges are the same for an arm, a session, a settings
  file and the environment.
- **Your own data as a benchmark.** A JSONL or CSV file, or a BEIR folder:
  `benchmarks: [{path: data/my.jsonl}]` or `lasagna benchmark run ./my.jsonl`.
  It registers as `local/<stem>@<12 hex digits of its sha256>`, so an edited
  file is a different benchmark. `lasagna benchmark validate-data` names each
  bad row and field. See [docs/own-data.md](docs/own-data.md).
- **`lasagna benchmark synthesize`** writes factoid, multi-hop and unanswerable
  questions from your documents with the configured LLM, for development
  rather than for reported results. The plan warns when the documents are too
  few for the number of questions asked.
- **Benchmarks run beside `lasagna serve`.** While another process holds the
  local Milvus Lite store, `benchmark run` and `benchmark experiment` use one
  of their own in `~/.lasagna/bench/dbs` and say so
  ([configuration.md](docs/configuration.md#where-data-lives)).
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
  tests a 2x2 design. A sweep ends with the command to run next on its
  results: one line per benchmark, and `results interaction` after a factorial.
- **`lasagna results view`** writes a run as one offline HTML page:
  leaderboard, significance, every metric, the isolation check and a per-query
  diff with what each arm retrieved.
- Every `lasagna results` command takes a run's folder (the one a run prints
  after "Output:") or the `results.json` in it, and reads `--benchmark` as the
  benchmark's id or the `name:` the sweep gave it, listing the run's
  benchmarks when it matches none.
- **Bundles** (`results bundle`, `results validate`) carry what a reader needs
  to decide whether two numbers may be compared, with one comparability key per
  benchmark and configuration; `results leaderboard` groups by it, as markdown
  pages or LaTeX tables (`--format latex`), and `results compare` prints it
  for each side. Every results command refuses a bundle that `results
  validate` rejects; `--allow-invalid` shows one marked INVALID.
- **Stage isolation.** Every run records a manifest of each stage's settings;
  arms that differ outside the stages they name block the bundle, and an arm
  whose named change never moved a ranking (on at least ten queries) is
  flagged inert.
- **A reproduction:** on the full SciFact and NFCorpus test sets the dense
  models land within 0.002 NDCG@10 of MTEB, and BM25 with a cross-encoder
  within 0.004 of the BEIR paper (`examples/reproduction/`). Its
  `results.csv` records the commit, metrics version and date of the run that
  produced the numbers, and `results_csv.py` writes the same file from a rerun
  to compare.

Serving:

- **MCP server** (`lasagna-mcp`, stdio by default, or `lasagna serve --mcp`)
  for Claude Code, Claude Desktop and Cursor. The default tool set, `safe`,
  serves no deletion tools (`--tools all` does). `search` names the stages that
  fell back and why, with the fix, also when it finds nothing (for example,
  when the session's documents are in another model's collection). An
  argument of the wrong type, or one the tool does not take (`session` for
  `session_id`), is refused in one sentence naming the one it takes.
- **REST API** (`lasagna serve`, or the Compose stack) under `/api/v1`,
  covered by the deprecation policy in [GOVERNANCE.md](GOVERNANCE.md); search
  responses name the stages that fell back in `degraded_stages` (a reranker
  not downloaded: retrieval order and the download, also when switching to it).
- **`lasagna upload`** takes files, globs and folders (a folder's readable
  files, subfolders included, saying what it skipped).
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
- The SDK's `upload(metadata=...)` and `upload_batch(metadata=...)` raise
  `ValueError`: no surface stores per-document metadata, and it was dropped.
- Every setting has a `LASAGNA_` variable, which wins over the `RAG_` one; the
  `RAG_` names keep working with no warning. `config_for_quality()` is an alias
  of `config_for_max_quality()`.
- `lasagna.sdk.__all__` lists the 40 names a client of a knowledge base
  imports, among them the `KnowledgeError`s a settings change it cannot apply
  raises (`DocumentsLeftBehind`, `ProfileCannotLoad`, `SettingsNotSaved`; it
  logged a warning); every other name it listed still imports from it.

#### Defaults that changed

- **Sessions' settings are kept in one store for every surface.** The CLI, the
  MCP server, `lasagna serve` and the SDK read and write the metadata
  database, so a profile set with `lasagna configure` applies to a
  `KnowledgeClient` for that session in another process, and a client's
  `config=` is that session's settings everywhere (refused, as by `lasagna
  configure`, when it would leave the session's documents unsearched).
  `RAG_SESSION_BACKEND=memory` keeps a process's sessions to itself.
  Concurrent changes from several processes are all kept; the evaluation
  harness keeps its sessions in its own process. A session's settings stay
  while it holds documents, else `session_idle_days` (30; 0 keeps them) after
  last use. A change the store does not save is refused (503); on PostgreSQL
  every worker reads a session as last saved. A saved profile is its owner's,
  and a re-save or delete that would strand other sessions' documents is refused.
- **One vector collection per embedding model**
  (`rag_documents_v2__<model>__<dim>`). An existing `rag_documents_v2` is kept
  under its name by the first model that opens it at its size. A switch that
  would leave a session's documents in another model's collection is refused
  unless it reindexes them (`--reindex`) or is forced (`--force`), and a
  search or `lasagna doctor` on such a session says so. A `--reindex` switch
  holds the session (other writes get 409) and moves documents back if
  stopped; `lasagna configure --resume-switch` finishes one whose process died,
  and `lasagna configure --reset` drops a session's own settings.
- **`lasagna benchmark run`** measures retrieval unless `--tracks` names
  generation, and starts from the `speed` profile.
- **`lasagna-mcp`** speaks stdio and serves the `safe` tool set; `lasagna
  serve` logs warnings only and opens no browser unless `--open`.
- **`lasagna search` says which session it searched** when a server answers for another.
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
  setting has a maximum and every list a maximum length (`SESSION_MAXIMUMS`),
  for REST, MCP and imports, never the evaluation harness; `PATCH
  /api/v1/config` uses the same bounds as the settings check.
- A model a session names (`monot5_model`, `splade_model`, `colbert_model`,
  `flashrank_model`, as the embedder and the cross-encoder already did) follows
  the server's download policy instead of downloading inside a request.
- Uploads, reindexes and deletes of one document wait for each other across
  processes (a PostgreSQL advisory lock, or a file lock beside a SQLite store).
  Such a lock is waited for `RAG_STORE_LOCK_TIMEOUT` (300 s), then the
  operation fails with 503. Two experiments on one vector store wait for each
  other's corpus ingest, and a task whose corpus changed under it fails.
  A document that fails to index leaves no chunks behind, search returns
  chunks of ready documents only, and uploading a file again whose copies all
  failed retries the newest one under its id.
- The Compose stack no longer publishes Redis (`-f expose-redis.yml` does, on loopback).
- The Docker image runs torch's CPU build (3.8 GB; `TORCH_VARIANT=cuda` builds
  the GPU one), and the Compose stack defaults to a Milvus server.
- `sentencepiece` is a core dependency, for the `monot5` reranker's tokenizer.

### Known issues

- **Windows needs WSL.** Milvus Lite, the default store, does not install on
  Windows: `pip install lasagna-rag` succeeds and the first upload fails. Use
  WSL, a Milvus server or another vector store (`RAG_VECTOR_BACKEND`).
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
- **`beir/nfcorpus` is not pinned to a data revision** (`beir/scifact` is), so
  its runs record a moving reference. Pin one with `LASAGNA_DATASET_REVISIONS`,
  a JSON map from benchmark name to Hugging Face commit.

### Planned for 0.2.0

- Export and import of a session's documents and chunks.
- A prebuilt Wikipedia index.
- Multilingual benchmarks.
- Batched generation with vLLM.
- An OpenAI-compatible answer endpoint, and streamed answers.
- Weights & Biases and MLflow logging hooks.
- Claim-level diagnostics of generated answers.
- Calibration of the LLM judge against human labels.
- Removal of the old names listed under "Renamed" above.

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
- Evaluation harness: 31 benchmark entries over about 22 datasets, RAGAS metrics, NDCG/EM/F1, latency
  and cost accounting, multi-objective tradeoff analysis.
- Python SDK, REST API, MCP server, CLI, and a schema-driven configuration UI.
- `lasagna.core.model_licenses`: license metadata for every selectable model,
  warning once at load when a non-commercially-licensed model is used.
  See `docs/model-licenses.md`.

### Fixed
- **The first PDF Docling converted could time out.** Docling loads (and on the
  first run downloads) its layout models inside the first conversion, and that
  time counted against `docling_timeout` (300 seconds). The models now load when
  the converter is built, before the timer starts, and a timeout names
  `docling-tools models download` and `DOCLING_ARTIFACTS_PATH`.
- **The first document of a benchmark corpus could fail with no reason given.**
  Ingest gave each document 120 seconds to chunk, counted from the first wait,
  and the first document of a semantic or structure-aware corpus loaded the
  embedding model inside that time. On a busy machine or a first run the
  document failed with `Chunking failed for doc_...:` and an empty reason. The
  model now loads before any document is timed, and a chunking failure always
  names its exception type (a timeout also says which limit was hit).
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

[Unreleased]: https://github.com/dolev31/lasagna/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dolev31/lasagna/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dolev31/lasagna/releases/tag/v0.1.0
