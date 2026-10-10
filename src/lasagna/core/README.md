# `core/` — Lasagna foundation package

Foundational layer for the RAG knowledge system. Every other module
(`agentic/`, `sdk/`, `api/`, `retrieval/`, `processing/`) depends on `core/`.

## Quick Start

```python
from lasagna.core import (
    RAGConfig, get_rag_config, reset_rag_config,
    ComponentRegistry, registry,
    KnowledgePipeline, PipelineResult,
    KnowledgeConfigManager, get_knowledge_config_manager,
    BaseReranker,
)

# Access global config
cfg = get_rag_config()
print(cfg.vector_backend, cfg.embedding_model)

# Register a custom component
@registry.register("reranker", "my_reranker")
class MyReranker(BaseReranker): ...

# Build and run a pipeline
pipeline = KnowledgePipeline.from_profile("balanced")
pipeline.ingest("/path/to/docs", session_id="sess_123")
results = pipeline.query("What is X?", session_id="sess_123")
```

## Module Map

| File | Purpose | Key Exports |
|---|---|---|
| `config.py` | 160-property RAG configuration with 4-level resolution | `RAGConfig`, `get_rag_config`, `rag_config` |
| `database.py` | SQLite metadata backend (documents, citations, chunks, profiles, sessions) | `RAGDatabase`, `get_rag_database` |
| `database_pg.py` | PostgreSQL metadata backend (drop-in replacement) | `PostgresRAGDatabase` |
| `database_backend.py` | `DatabaseBackend` Protocol — contract both backends implement | `DatabaseBackend` |
| `schema.py` | Pydantic models + vector schema DDL | `Document`, `DocumentChunk`, `SearchResult` |
| `registry.py` | Singleton component registry with auto-discovery | `ComponentRegistry`, `registry` |
| `pipeline.py` | Composable RAG pipeline with hooks and evaluation | `KnowledgePipeline`, `PipelineResult` |
| `base_components.py` | ABCs for pluggable components | `BaseEmbedder`, `BaseChunker`, `BaseReranker`, ... |
| `quota.py` | Per-tenant resource quota enforcement | `QuotaManager`, `get_quota_manager` |
| `rate_limiter.py` | Adaptive token-bucket rate limiter | `RateLimiter`, `get_rate_limiter` |
| `tenant.py` | Partition-based tenant isolation | `TenantManager`, `get_tenant_manager` |
| `testing.py` | Component contract validation harness | `ComponentTestHarness` |
| `session_config/` | Session/profile management (6 mixins) | `KnowledgeConfigManager` |

## Singletons

| Singleton | Accessor | Reset | Lock |
|---|---|---|---|
| `RAGConfig` | `get_rag_config()` | `reset_rag_config()` | `_rag_config_lock` |
| `ComponentRegistry` | `ComponentRegistry()` / `registry` | `ComponentRegistry.reset()` | `_lock` (class-level) |
| `KnowledgeConfigManager` | `get_knowledge_config_manager()` | re-assign internal `_knowledge_config_manager` | `_knowledge_config_manager_lock` |
| `RAGDatabase` | `get_rag_database()` | `reset_rag_database()` | `_rag_database_lock` |
| `QuotaManager` | `get_quota_manager()` | `configure_quota_manager()` | `_instance_lock` |
| `RateLimiter` | `get_rate_limiter()` | `configure_rate_limiter()` | `_instance_lock` |
| `TenantManager` | `get_tenant_manager()` | `configure_tenant_manager()` | `_instance_lock` |

All singletons use double-check locking for thread safety.

## API Stability Policy

- **Stable public API**: Symbols in `__all__` (31 symbols) — import from `lasagna.core`
- **Semi-public API**: Symbols in submodules but NOT in `__all__` — import from the submodule directly:
  - `DatabaseBackend`, `get_rag_database` — from `core.database` / `core.database_backend`
  - `QuotaManager`, `get_quota_manager` — from `core.quota`
  - `RateLimiter`, `get_rate_limiter` — from `core.rate_limiter`
  - `TenantManager`, `get_tenant_manager` — from `core.tenant`
  - `ComponentTestHarness` — from `core.testing`

## Config Resolution Chain

Properties resolve in priority order (highest wins):

```
Session override  →  Environment variable (RAG_*)  →  TOML setting  →  Default
```

See [docs/config.md](docs/config.md) for the full 160-property reference.

## Database Backend Parity

Both SQLite and PostgreSQL backends implement the `DatabaseBackend` Protocol.
See [docs/database.md](docs/database.md) for the parity contract and behavioral differences.

### Data Layout

All state lives in a single database — `rag.db` (SQLite) or equivalent PostgreSQL tables.
No JSON files, no external state. Tables are created automatically on first access.

| Table | What it stores |
|---|---|
| `rag_documents` | Document metadata and processing status |
| `citations` | Citation tracking with TTL-based expiry |
| `parent_chunks` | Parent content for hierarchical chunking |
| `rag_profiles` | User-created RAG configuration profiles |
| `rag_sessions` | Per-session config state (active profile, overrides, focused docs) |
| `schema_meta` | Internal schema versioning |

## Further Documentation

- [docs/architecture.md](docs/architecture.md) — Dependency graph, config chain, session lifecycle
- [docs/config.md](docs/config.md) — RAGConfig property reference
- [docs/database.md](docs/database.md) — Database backend guide and parity contract
- [docs/session_config.md](docs/session_config.md) — Session/profile management guide
- [docs/backend_parity.md](docs/backend_parity.md) — What the SQLite and PostgreSQL backends must do identically
