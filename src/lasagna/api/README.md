# api: the RAGBackend facade and the REST app

`RAGBackend` is the in-process entry point every surface goes through: the SDK client, the
`lasagna` CLI, the MCP server and the REST app in `rest/`. It owns the vector backend, the
embedding manager, upload and indexing, and the search pipeline, and delegates each to one of
the private modules below. `rest/` wraps it in a FastAPI app.

How to call it as a user: [README.md](../../../README.md) and
[docs/guide.md](../../../docs/guide.md); deployment: [deploy/README.md](../deploy/README.md).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Lazy exports (`RAGBackend`, `get_rag_backend`, `initialize`, `reset_instance`, `close_backend`); importing a submodule must not load torch |
| `rag_backend.py` | `RAGBackend` singleton; builds and wires the managers below and the `DocumentLifecycleManager` |
| `_collection_manager.py` | `CollectionManager`: default and per-session backend resolution, collection creation, index type, schema migration, embedding-size checks (`CollectionIdentityError`) |
| `_document_pipeline.py` | `DocumentPipeline`: upload, extract (OCR fallback), chunk, embed, index; delete (`keep_file`), lookup |
| `_search_orchestrator.py` | `SearchOrchestrator`: routing tiers, cache, query enhancement, dense and hybrid search, reranking, MMR, auto-merge, compression, citations; multi-query search; `_search_comprehensive` (agentic tier) |
| `_auto_merger.py` | `AutoMerger`: replaces enough sibling child chunks with their parent chunk |
| `_profile_guard.py` | `refuse_embedding_change`: refuses a config write that would change a session's embedding size away from the stored vectors |
| `_validation.py` | Session and document id checks, filename sanitizing, Milvus string escaping |
| `rest/app.py` | `create_app()`, middleware, lifespan, `/metrics`, optional MCP and configurator mounts; module-level `app` |
| `rest/auth_mode.py` | `RAG_REQUIRE_AUTH`, API keys, secret checks, the refuse-to-start rule for public binds |
| `rest/dependencies.py` | `get_backend`, `get_session_id`, `get_user_id`, `get_profile_owner` |
| `rest/session_token.py` | Session JWT create and decode, `extract_session_id`; no FastAPI import, so MCP can use it |
| `rest/limits.py` | Body size limit middleware, batch and search-timeout limits, log format |
| `rest/models.py` | Pydantic request and response models |
| `rest/routes/` | `health`, `sessions`, `documents`, `search`, `citations`, `config`, `pipelines`, `gc_admin` under `/api/v1`; `knowledge_config` and `reindex` outside it |

The routes a running server exposes are listed at its `/docs` (OpenAPI). Identity, middleware
order and auth: [rest/DESIGN.md](rest/DESIGN.md). Further notes: [docs/architecture.md](docs/architecture.md),
[docs/endpoints.md](docs/endpoints.md), [docs/config.md](docs/config.md).

## Design notes

**One backend per process.** `RAGBackend.__new__` and `__init__` both take `_instance_lock`
(double-checked), since two threads can receive the same instance from `__new__` and both enter
`__init__`. A failed init resets `_initialized`, so the next call retries. `close_backend()`
releases connections and models before dropping the instance; `reset_instance()` only drops it.

**Per-session config, shared collections.** A search reads its session's resolved `RAGConfig`
(`SearchOrchestrator._get_session_config`). Each embedding model has its own collection, shared
by every session on that model; `CollectionManager` resolves which backend and collection a
session uses and remembers initialized collections in a bounded map (500 entries) to keep a
multi-tenant process from growing without limit. See
[docs/configuration.md](../../../docs/configuration.md#one-collection-per-embedding-model).

**Embedding size is checked before a write, not after.** `KnowledgeConfigManager` calls
`refuse_embedding_change` before any profile switch, patch, save, import or reset that changes
the embedding size, so no surface (CLI, REST, MCP, configurator) can leave a session pointed at
a collection of another size.

**Arm components reach every hop.** An agentic pipeline issues its own searches and passes only
settings. `_INNER_SEARCH_COMPONENTS` (a `ContextVar`) carries the arm's reranker, enhancer,
retriever and embedder into those inner searches, so an arm that swaps the reranker swaps it
in every agentic hop too. Routing decisions are kept in thread-local storage so concurrent requests do not share
metadata.

**Failures are loud under evaluation.** A comprehensive-tier search that fails falls back to
standard search and records the degradation, except under `strict_stages()`, where it raises.
Configuration errors (`KeyError`, `ValueError`, `TypeError`) always propagate.

**Upload flow.** `DocumentPipeline.upload_document()` checks quota, file size and type, then
reserves quota before any I/O, stores the file through the storage backend, and writes the
metadata row with status `processing`. A failure at either step releases the reservation (and
deletes the stored file). It then calls `infrastructure.tasks.document_tasks.dispatch_processing()`:

1. Celery, when `celery_enabled` is on and the broker accepts the task; a broker error falls
   through.
2. A bounded thread pool, when `async_processing` is on (the default): `max_concurrent_uploads`
   threads (`RAG_MAX_CONCURRENT_UPLOADS`, default 5). The REST routes pass
   `refuse_when_busy=True`, so a full pool is a 503 with `Retry-After: 5` and the upload is
   rolled back; the batch route also passes `wait_when_busy` (`processing_timeout`), so each
   of its files waits for a thread instead. Other callers (SDK, CLI, MCP) fall through.
3. Inline, in the caller's thread.

A dispatched upload returns at once with status `processing`; clients poll (the SDK's
`documents.wait_for_processing`). `process_document()` extracts text (Docling, a named
`document_processor`, or `DocumentProcessor` with OCR), chunks it, optionally adds contextual
prefixes, GraphRAG entities and RAPTOR summaries, embeds (late chunking when enabled), inserts
into the session's backend and collection, updates the BM25 index and parent chunks, and sets
`ready` with `chunks_count`. A failure sets `error` and releases the reservation. On shutdown,
`close_backend()` calls `drain_document_processing()`, which waits up to `processing_timeout`
(300 s) and marks what is still running as failed.

**Document lifecycle.** Archive, restore, garbage collection and orphan repair live in
`lasagna.lifecycle`; `RAGBackend` wires its manager and scheduler, and `rest/routes/gc_admin.py`
exposes them behind `X-Admin-Token` (`RAG_ADMIN_TOKEN`).
