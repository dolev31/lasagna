# core: configuration, registry, pipeline, metadata store

The layer every other package imports: the resolved configuration (`RAGConfig`), the component
registry and stage ABCs, `KnowledgePipeline`, the Pydantic data models, the metadata database
(SQLite or PostgreSQL), and per-session profile state (`session_config/`).

How a setting is resolved: [docs/configuration.md](../../../docs/configuration.md#where-a-setting-comes-from).
Every setting and its default: [docs/reference/settings.md](../../../docs/reference/settings.md).
Writing a component against the ABCs: [docs/guide.md](../../../docs/guide.md#4-write-a-component).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `config.py` | `RAGConfig` (one property per setting; `with_session()` for a session-scoped view), `get_rag_config`, `reset_rag_config`, lazy module-level `rag_config`, `ConfigError` |
| `registry.py` | `ComponentRegistry` singleton (`registry`), `COMPONENT_TYPES`, built-in registrations as lazy `module:Class` factories, entry-point discovery (group `lasagna_components`), graph-node callables |
| `base_components.py` | Stage ABCs (`BaseEmbedder`, `BaseChunker`, `BaseReranker`, `BaseRetriever`, `BaseDocumentProcessor`, `BaseQueryEnhancer`, `BaseAgenticPipeline`) and their data types |
| `pipeline.py` | `KnowledgePipeline` (compose, swap, run, evaluate, serialize), `PipelineResult`, `QUERY_TIME_STAGES` |
| `schema.py` | Pydantic models (`Document`, `DocumentChunk`, `SearchResult`, ...) and vector schema helpers |
| `database_backend.py` | `DatabaseBackend` Protocol that both metadata stores implement |
| `database.py` | `RAGDatabase` (SQLite, `rag.db`), `get_rag_database` (picks the backend), `reset_rag_database` |
| `database_pg.py` | `PostgresRAGDatabase` (connection pool) |
| `session_config/` | `KnowledgeConfigManager`: profiles, per-session state, persistence, last-search citations, vector backend choice, validation (one mixin per file) |
| `paths.py` | The data directory (`~/.lasagna`, or `LASAGNA_DATA_DIR`), `dbs/` and `uploads/`, and the one-time rename from `~/.rag_knowledge` |
| `strict.py` | `strict_stages()`: during evaluation a stage that cannot run raises instead of falling back |
| `provenance.py` | `capture()` of environment and versions for results bundles; `redact`, `redact_url` |
| `testing.py` | `ComponentTestHarness`, used by `lasagna component validate` and the experiment runner |
| `quota.py`, `rate_limiter.py`, `tenant.py` | Multi-tenant quotas, adaptive token-bucket rate limiting, partition-based isolation; all off by default |
| `accelerator.py` | `device_guard`: one process-wide lock for Apple MPS, taken around each model load or batch |
| `model_licenses.py` | License metadata for downloadable weights; `warn_if_restricted` |
| `diagnostics.py`, `setup_text.py`, `_text.py` | Text and checks behind `lasagna doctor` and `lasagna init`; `did_you_mean` |

Deeper pages: [docs/architecture.md](docs/architecture.md), [docs/config.md](docs/config.md),
[docs/session_config.md](docs/session_config.md), [docs/database.md](docs/database.md),
[docs/backend_parity.md](docs/backend_parity.md).

## Singletons

| Object | Accessor | Reset | Lock |
| ------ | -------- | ----- | ---- |
| `RAGConfig` | `get_rag_config()`, `rag_config` | `reset_rag_config()` | `_rag_config_lock` |
| `ComponentRegistry` | `registry`, `ComponentRegistry()` | `ComponentRegistry.reset()` | class `_lock`; `_data_lock` for registrations |
| `KnowledgeConfigManager` | `get_knowledge_config_manager()` | none (module global `_knowledge_config_manager`) | `_knowledge_config_manager_lock` |
| metadata database | `get_rag_database()` | `reset_rag_database()` (closes the old one) | `_rag_database_lock` |
| `QuotaManager`, `RateLimiter`, `TenantManager` | `get_quota_manager()`, `get_rate_limiter()`, `get_tenant_manager()` | `configure_*()` | `_instance_lock` in each module |

All use double-checked locking. With `RAG_DATABASE_BACKEND=postgresql`,
`get_knowledge_config_manager()` returns a new request-scoped manager per call instead of the
singleton. `reset_rag_config()` makes the next access re-read environment, TOML and provider
settings; tests rely on it.

## Design notes

**Public API.** Names in `core/__init__.py` `__all__` are the stable import surface
(`from lasagna.core import ...`). Everything else (`get_rag_database`, `DatabaseBackend`, the
quota, rate limiter and tenant managers, `ComponentTestHarness`) is imported from its submodule.

**Two metadata stores, one contract.** `RAGDatabase` and `PostgresRAGDatabase` both implement
`DatabaseBackend`; `get_rag_database()` chooses by `rag_config.database_backend`.
`tests/test_backend_parity.py` runs the same tests on both (PostgreSQL only when
`DATABASE_DSN` is set). Both create their tables on first access: `rag_documents`, `citations`,
`parent_chunks`, `rag_profiles`, `rag_sessions`, `schema_meta`, `collection_meta`, `gc_lock`,
`gc_audit_log`; PostgreSQL adds `rag_quotas`. When you change a table, change both DDLs.

**The session-config persistence mixin is chosen at import.** `session_config/__init__.py`
reads `RAG_DATABASE_BACKEND` from the environment when it is imported and mixes in
`_persistence.py` (tables in `rag.db`) or `_persistence_db.py` (PostgreSQL with a Redis cache).
The last search's citations per session are also written to `citations_<session>.json` in the
data directory (`_citations_store.py`), so the MCP process and the REST app can share them.

**Config is read through properties.** Code reads `config.<setting>`, never the TOML or the
environment directly, so session overrides, the `LASAGNA_`/`RAG_` environment names and the
TOML file are applied in one place (`RAGConfig._get_knowledge_setting`). A new setting needs a
default in the defaults dict of `config.py` (its comment becomes the description) and a
property. Then run `python scripts/gen_settings_doc.py` to regenerate
`docs/reference/settings.md`; `tests/test_settings_doc_current.py` fails when the page is stale.

**Built-ins load lazily.** The registry stores built-in components as `module:Class` strings
and imports them on first `get`, so `import lasagna` does not load torch. Entry-point discovery
is best-effort; failures are recorded in `discovery_log()` (shown by `lasagna plugins list`).
