# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

Lasagna now has eight swappable stages and 53 built-in components; each one that
implements a published method names its paper. Several fixes below change
measured numbers, so `METRICS_VERSION` is now 1.5 (it was 1.3), and results from
before this release do not share a leaderboard group with results after it.

### Added

Components, by stage:

- **Generator, a new eighth stage.** How the answer is written from the
  retrieved passages can now be swapped with everything else held fixed.
  Built-ins: `concise` (the harness's previous prompt, word for word), `cited`
  (ALCE), `chain_of_thought` and `chain_of_note`. A function
  `f(query, contexts) -> str` becomes a generator with `@lasagna.generator`.
  Sweep it with `pipeline.generators`, run one with
  `lasagna benchmark run --generator`, or call `KnowledgePipeline.generate()`.
  A run that names no generator writes answers exactly as before.
- **Retriever:** `dense`, `bm25`, `splade`, `hybrid_rrf` and `hybrid_convex`.
  The stage had no built-in component before. `dense` is the default vector
  search as a component, the control for a retriever sweep. New settings:
  `bm25_k1`, `bm25_b`, `splade_model`, `hybrid_fusion_depth` and
  `hybrid_convex_normalization`.
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
  in characters, words or tokens. `chunking_strategy: recursive` now runs the
  recursive algorithm; it used to be refused. `semantic` gains Kamradt's
  breakpoint methods (`semantic_breakpoint_method`); the default, `absolute`, is
  unchanged.
- **Embedder:** `sentence_transformers`, `bge_m3`, `openai`, `cohere` and
  `voyage`. Query and document prefixes follow each model family's model card.
- **Document processor:** `docling`, `markitdown` and `pymupdf4llm`, each with an
  extra of the same name. `pymupdf4llm` is AGPL-3.0 and is not part of `all`.

Sweeps, validation and metrics:

- Chunker and embedder components sweep. Each arm's corpus is chunked or
  embedded by its own component, in a corpus of its own; an embedder also gets
  its own collection. Document processors do not sweep, since benchmark corpora
  are loaded as text.
- Every chunking, embedding and document-processing setting gets its own
  corpus in a sweep.
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
  tokens and LLM calls per query. The judge is now also asked for
  `factual_correctness`.
- `lasagna --version` prints the installed version.
- **`lasagna results significance` ends in a verdict.** A green box names each
  pair that survives Holm correction, which side won (lower-is-better metrics
  included) and by how much; when nothing survives, a yellow box says so and
  points at `lasagna results power`.
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
- **Your own data as a benchmark, with no adapter.** A JSONL or CSV file, or a
  BEIR folder, runs with `benchmarks: [{path: data/my.jsonl}]` in a sweep or
  `lasagna benchmark run ./my.jsonl`. It registers as
  `local/<file name>@<12 hex digits of its sha256>`, so an edited file never
  resumes the old file's checkpoints or shares its leaderboard group.
  `lasagna benchmark validate-data` names each bad row and field and says which
  tracks the data supports. See `docs/own-data.md`.

### Changed

- **Searching on Milvus no longer pays for a JSON filter.** A search was
  confined to its session through `metadata["session_id"]`, which Milvus parses
  row by row; it now uses the `session_id` field every collection has and every
  writer fills with the same value. A search also stopped counting the
  collection's rows twice for a count it never used. Measured on Milvus Lite
  with 18,442 chunks in three sessions, the median search fell from 293 ms to
  32 ms without a reranker and from 415 ms to 144 ms with a MiniLM
  cross-encoder, returning the same ranked chunks and scores. A collection
  without the field keeps the old filter.
- Every title, banner, help text and guide that said "RAG Knowledge" says
  Lasagna, and the MCP server announces itself as `lasagna` (it was
  `rag_knowledge`). Clients name the server in their own config, so existing
  configs keep working.
- `max_content_length` is a chunker setting: a sweep can vary it, each value
  gets a corpus of its own, and the chunker's stage digest includes it.
- **`METRICS_VERSION` 1.4:** an answer counts as an abstention only when it opens
  with one, so a hedge later in the answer no longer zeroes exact match and
  token F1. Answers are normalized in the official SQuAD order.
- **`METRICS_VERSION` 1.5:** exact match and token F1 score the answer with its
  `[n]` citation markers removed, as ALCE does. NDCG gives a negative relevance
  grade zero gain, as `trec_eval` does. The judged set adds
  `factual_correctness`, and the answer-similarity model is part of the
  comparability key.
- `llm_listwise` runs RankGPT's sliding window: windows of 20 passages, moved by
  10 from the bottom of the list to the top. Its window default was 10.
- `query_decomposition` is least-to-most prompting's decomposition (Zhou et al.
  2022). Results recorded under that name before came from another prompt.
- BGE-M3 no longer gets the BGE v1.5 query instruction, which it was not
  trained with. This changes BGE-M3 query vectors on the sentence-transformers
  backend.

### Fixed

Numbers that were wrong:

- **Chunks cut at ingest without a word.** Benchmark ingest cut every chunk to
  `max_content_length` (4,000 characters) before embedding and indexing, and a
  sweep could not raise it. The default is unchanged and `chunk_size` 800 never
  reaches it, so no published number moves; a cut is now logged.

- **NDCG with negative grades.** A negative grade lowered the score:
  `{a: 2, b: -1, c: 1}` ranked `[b, a, c]` scored 0.358 where `trec_eval` gives
  0.670. The registered BEIR loaders drop grades of 0 or below, so no harness
  number moves; custom qrels with negative grades do.
- **Per-arm corpora collided.** Arms with corpora of their own in one collection
  shared document and chunk ids, so the arm that ingested last took the others'
  documents. In a four-arm sweep the control scored 0.0, and a chunker arm
  drifted between 0.59 and 0.94 with task order. Each arm now has its own ids,
  and two runs give identical numbers.
- **Embedder arms embedded the corpus with the built-in model.** Only the
  queries went through the arm's embedder, so the arm compared two models. On
  `smoke`, a bge-small function embedder scored 0.8712 before and 0.925 now,
  the same as setting `embedding_model` to bge-small.
- **Chunk-size sweeps did nothing.** The built-in chunker read `chunk_size`,
  `chunk_overlap`, `window_size` and the semantic thresholds from the
  process-wide config, so two arms sweeping `chunk_size` got identical chunks.
  Settings added with the new chunkers reused the control's corpus. Both now
  take effect, and a setting of 0 is no longer read as unset.
- **The `semantic` component chunked by size.** It was built without an
  embedding model and ran fixed-size chunking under its name. It now embeds with
  the arm's model and raises when none loads. `hierarchical` and
  `structure_aware` read the arm's settings too.
- **SQuAD v2's unanswerable half was dropped.** Only questions with a gold
  answer were scored, so a system that never abstains scored EM 1.000 where the
  official rule gives about 0.5. Unanswerable questions are now scored, correct
  only when the system abstains, and every annotator's answer counts.
- **Latency stopped before the answer was written.** A live answer row now
  records `retrieval_ms` and `generation_ms`, and `latency_ms` is their sum.
  Back-off sleeps between retries are not counted.
- **Paired tests rejected too often at small sample sizes.** Under a true null
  at alpha 0.05, the paired bootstrap rejected 9.1% of the time at 12 queries
  (the `smoke` benchmark) and 7.7% at 20. The significance, interaction and
  domain-stratified tests now use a paired sign-flip randomization test, exact
  up to 30 non-zero differences, with a t interval. `p_value` and the interval
  keys (`confidence_interval_95`, `ci_lower`, `ci_upper`) now hold those.
- **Agentic hops searched with the profile's components.** Each hop now searches
  with the arm's reranker, query enhancer, retriever and embedder, and with the
  arm's per-query settings. Under evaluation, a failed agentic search fails the
  task instead of falling back to standard results.
- **A failing LLM reranker scored as the control.** When its LLM failed,
  `llm_listwise` kept the retrieval order. It now raises, as the other LLM
  components do.
- **Late chunking truncated long documents.** Chunks past the model's window
  were embedded the standard way. Long documents are now processed in
  overlapping windows (`late_chunking_overlap`), and under evaluation late
  chunking raises instead of falling back.
- **The short-list warning blamed the wrong setting.** It told users to raise
  `retrieval_chunk_overfetch` even when BM25 had simply run out of documents
  matching a one- or two-word query; it now names the cause.
- `results significance` and the isolation summary said "1 pairs".

## 0.1.0a1 — 2026-08-18

Internal pre-release, never published to PyPI. Its correctness work is part of
`0.1.0`.

### Changed — metric definitions (`METRICS_VERSION` 1.0 → 1.1)

Numbers produced before this release are **not comparable** with numbers
produced after it. `METRICS_VERSION` is part of the comparability key precisely
so the two cannot be grouped in one leaderboard table.

- **NDCG now uses linear gain** (`rel`), matching `trec_eval` and therefore
  every published BEIR number. It used `2^rel - 1`, the Burges convention:
  identical on binary judgements, different on graded ones — `beir/nfcorpus`,
  `beir/trec-covid` and `beir/dbpedia-entity`. `exponential_gain=True` is
  available for comparison against work that uses it.
- **Cutoff metrics are computed over `k` documents.** The harness requested `k`
  *chunks*, then de-duplicated to parent documents before scoring, so a
  "top-10" list was typically 4–7 documents and every `@k` metric was computed
  over a list shorter than its own name claimed — a systematic downward bias on
  every dataset. It now over-fetches chunks and cuts to `k` documents.
- **`k` is configurable** (`metric_k`), so NDCG@100 and Recall@1000 can be
  reported. It was hard-coded to 10 in six places. Metric keys follow the
  cutoff actually used.

### Fixed — numbers that were silently wrong

- **A test-set request could return train-set judgements.** The BEIR loader
  walked `[requested, test, validation, train]` and returned the first split
  that loaded, cached it under the *requested* split's key, and stamped
  `split: "test"` on the result. It now honours `allow_split_fallback`, reports
  the split it actually loaded, and warns when it substitutes.
- **`max_corpus_docs: 0`, documented as "no limit", capped at 100,000** —
  truncating 8 of 13 BEIR corpora and aborting 7 outright.
- **An aborting corpus ingest hung every sibling task** rather than failing it.
- **`sample_id_digest` was a function of `n_samples`.** Two benchmarks at the
  same sample count produced the same digest, in a field the bundle documents
  as proof that two runs scored the same questions.
- **`TRUNCATED_CORPUS` could never fire.** The adapter set the flag on itself
  and nothing carried it to the result, so truncated bundles validated clean.
  The flag is now set from the loop that actually truncates rather than from
  declared metadata — which for `beir/nfcorpus` claimed 170,000 documents
  against an actual 3,633.

### Fixed — install and tooling

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
    only the literal string "GPL" — `GPL-3.0-or-later` sailed through. Adding
    the flag was not sufficient: `uvx` runs the tool in an ephemeral environment
    created with `include-system-site-packages = false`, and `pip-licenses`
    inspects the `sys.path` it runs under. Measured, it enumerated **zero**
    packages against 515 in a real environment. Now pinned to the CI interpreter
    with `--python`.
  - `detect-secrets scan --baseline` returns 0 unconditionally. Its replacement,
    `uvx detect-secrets-hook`, resolved the command string as a *distribution*
    name — no such distribution exists, so the step errored on every run, and
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
  thing that checks the package the way a user meets it — built into a wheel,
  installed into a clean venv, driven from outside. It writes a dated report
  with metrics, failures and a GO/NO-GO verdict, and diffs against the previous
  run so a latency slide or a silently disabled phase is visible. A subset runs
  in CI on every PR. See `e2e/README.md`. The gate also covers the *second*
  day — upgrading over an existing install and uninstalling — which is where a
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
  to a one-line warning — so the feature produced nothing and the configuration
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

### Known issues
- **`pip` cannot resolve `[eval]` together with `[storage-all]`** (and therefore
  cannot resolve `[all]`). `pip install "lasagna-rag[eval,storage-all]"` exits
  with `ResolutionImpossible`; `[all]` backtracks for over fifteen minutes
  without converging.

  The constraint set is satisfiable — this is a resolver search failure, not a
  real conflict. `uv pip compile` solves both in seconds (216 and 290 packages),
  and `pip` itself succeeds the moment it is handed the answer:

  ```bash
  # works
  uv pip install "lasagna-rag[eval,storage-all]"
  # also works: give pip the versions it fails to find on its own
  pip install "lasagna-rag[eval,storage-all]" s3fs==2026.6.0 fsspec==2026.6.0
  ```

  Cause: `s3fs`/`gcsfs`/`adlfs` release in lockstep with `fsspec` and pin it
  exactly, while `datasets` (via `ragas`) caps `fsspec`. Searching newest-first,
  pip explores a large `aiobotocore`/`botocore`/`boto3` space and gives up.
  Capping `fsspec` alone does *not* help it. Each extra installs fine by itself.

  Deliberately not "fixed" by pinning: any upper bound written here is a guess
  at when `datasets` will raise its own cap, and would strand users on an old
  `s3fs` once it does.
- `list_document_ids()` fails past roughly 16k chunks on Milvus. The backend
  emulates offset with `limit + offset`, which exceeds Milvus's 16384 cap, and
  the error is caught as "corpus absent" — so a large corpus is re-ingested on
  every run. Does not affect correctness, only iteration speed. The fix is to
  override `list_document_ids` in `milvus_backend.py` using
  `client.query_iterator()`, which supports offset natively.

### Security
- Deployment manifests require every credential; none fall back to a default.
- The test suite is isolated from the real `~/.rag_knowledge` store.

### Notes
- Lasagna vendors no third-party source, bundles no model weights, and
  redistributes no benchmark corpora.

[Unreleased]: https://github.com/dolev31/lasagna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dolev31/lasagna/releases/tag/v0.1.0
