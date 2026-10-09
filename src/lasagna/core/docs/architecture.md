# Core Package Architecture

## Dependency Graph

```
                        ┌──────────────────┐
                        │   __init__.py    │  (public API surface)
                        └──────┬───────────┘
                               │ re-exports
          ┌────────────┬───────┼───────┬────────────┬──────────┐
          ▼            ▼       ▼       ▼            ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ ┌──────────┐
    │ config.py│ │schema.py │ │registry│ │ pipeline  │ │session/  │
    │ (1618L)  │ │ (274L)   │ │(844L)  │ │ (870L)    │ │__init__ │
    └──┬───────┘ └──────────┘ └───┬────┘ └─────┬─────┘ └──┬──────┘
       │                          │             │          │
       │              ┌───────────┘       ┌─────┘    ┌─────┘
       ▼              ▼                   ▼          ▼
  ┌──────────────────────────┐    ┌────────────┐  ┌──────────────────┐
  │  base_components.py      │    │ database.py│  │ session_config/  │
  │  (ABCs: 6 components)    │    │ (1039L)    │  │ (6 mixins, 9 files)│
  └──────────────────────────┘    │database_pg │  └──────────────────┘
                                  │ (1023L)    │
                                  └──────┬─────┘
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                         ┌────────┐ ┌────────┐ ┌────────┐
                         │quota.py│ │rate_   │ │tenant  │
                         │(549L)  │ │limiter │ │.py     │
                         └────────┘ │(455L)  │ │(159L)  │
                                    └────────┘ └────────┘
```

External consumers (57 files import from `core/`):
- `api/` — REST routes, dependencies, backend orchestration
- `sdk/` — Python SDK client, managers
- `agentic/` — Agentic RAG graph nodes
- `retrieval/` — Search, ranking, enhancement
- `processing/` — Document processing, chunking
- `infrastructure/` — Embedding, caching
- `providers/` — Config, LLM, session providers

## Config Resolution Chain

```
┌─────────────────────────────────────────────────────┐
│                 _get_knowledge_setting()              │
│                                                       │
│  Priority 1: Session overrides                        │
│  ┌──────────────────────────────────────────────┐    │
│  │ session_provider.get_resolved_config(sess_id) │    │
│  │ → exact key → strip 'rag' prefix → flat leaf  │    │
│  └──────────────────────────────────────────────┘    │
│                    │ miss                             │
│                    ▼                                  │
│  Priority 2: Environment variables (RAG_*)            │
│  ┌──────────────────────────────────────────────┐    │
│  │ self._env_overrides[leaf_key]                  │    │
│  │ Type-coerced to match default type             │    │
│  │ Numeric range validation for known keys        │    │
│  └──────────────────────────────────────────────┘    │
│                    │ miss                             │
│                    ▼                                  │
│  Priority 3: TOML settings                            │
│  ┌──────────────────────────────────────────────┐    │
│  │ settings.lasagna.toml [rag] section          │    │
│  │ Or custom file via RAG_SETTINGS_FILE env var   │    │
│  └──────────────────────────────────────────────┘    │
│                    │ miss                             │
│                    ▼                                  │
│  Priority 4: Hardcoded default                        │
│  ┌──────────────────────────────────────────────┐    │
│  │ default_config dict in config.py               │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

At init time, `_load_config()` merges sources bottom-up:
```python
config = {**default_config, **knowledge_toml, **rag_settings, **env_overrides}
```

Session overrides are applied at runtime via `with_session(session_id)`.

## Session Lifecycle

```
                   new request
                       │
                       ▼
            ┌──────────────────┐
            │ get_session_state│
            │  (session_id)    │
            └────────┬─────────┘
                     │
           ┌─────────┴──────────┐
           │ exists in memory?   │
           └─────────┬──────────┘
              yes    │    no
              │      │     │
              ▼      │     ▼
         return it   │  create default
                     │  KnowledgeConfigState
                     │     │
                     │     ▼
                     │  load from persistence
                     │  (file or PG+Redis)
                     │     │
                     ▼     ▼
            ┌──────────────────┐
            │ patch_config()   │  ← session overrides
            │ set_active_      │  ← profile switches
            │   profile()      │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ persist state    │
            │ (atomic write or │
            │  PG upsert +     │
            │  Redis cache)    │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ cleanup_old_     │  ← TTL=24h, throttled to 1/min
            │  sessions()      │
            └──────────────────┘
```

## Database Backend Architecture

```
          ┌─────────────────────────┐
          │   DatabaseBackend       │ (@runtime_checkable Protocol)
          │   16 methods            │
          └──────┬──────────────────┘
                 │ implements
       ┌─────────┴──────────┐
       ▼                    ▼
┌──────────────┐    ┌──────────────────┐
│ RAGDatabase  │    │PostgresRAGDatabase│
│ (SQLite)     │    │ (PostgreSQL)      │
│ per-call conn│    │ connection pool   │
│ WAL mode     │    │ psycopg3/2       │
│ RLock        │    │ extra tables:     │
│              │    │  profiles,sessions│
│              │    │  quotas           │
└──────────────┘    └──────────────────┘
```

Both backends share:
- 4 tables: `rag_documents`, `citations`, `parent_chunks`, `schema_meta`
- `store_citations()` with per-session limit enforcement (max 5000)
- Full-row upsert on citation conflict
- Composite index `(session_id, created_at DESC, id DESC)`

See [database.md](database.md) for the full parity contract.

## Component Registry

```
                ┌─────────────────────────────┐
                │     ComponentRegistry       │
                │     (singleton via __new__)  │
                └──────────┬──────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────┐
        │ Built-in │ │ Decorator│    │ Entry    │
        │ (lazy)   │ │ @register│    │ Points   │
        │ 15 comps │ │          │    │ lasagna_ │
        └──────────┘ └──────────┘    │components│
                                     └──────────┘

Categories: embedder, chunker, reranker, retriever,
            document_processor, query_enhancer
```

Registration: `@registry.register("category", "name")` decorator or
`registry.register_class("category", "name", cls)` imperative.

Lookup: `registry.get("category", "name", **kwargs)` with adaptive parameter filtering.

## Mixin Composition (KnowledgeConfigManager)

```
KnowledgeConfigManager
  ├── ProfileMixin          (_profiles.py)
  │     profile CRUD, built-in presets (5 profiles)
  ├── SessionStateMixin     (_session_state.py)
  │     get/patch/reset/import/export config
  ├── PersistenceMixin      (_persistence.py — SQLite mode)
  │   OR DatabasePersistenceMixin (_persistence_db.py — PG mode)
  │     load/save profiles & session states
  ├── CitationStoreMixin    (_citations_store.py)
  │     cross-process search results sharing
  ├── BackendConfigMixin    (_backend_config.py)
  │     BYOVDB vector backend config
  └── ValidationMixin       (_validation_mixin.py)
        config validation gate
```

Mode selection: `RAG_DATABASE_BACKEND=postgresql` → DatabasePersistenceMixin,
otherwise → PersistenceMixin (file-based).

## Built-in RAG Profiles

| Profile | Hybrid | Reranking | HyDE | Agentic | Embedding |
|---|---|---|---|---|---|
| `speed` | No | No | No | No | MiniLM-L6-v2 (384d) |
| `standard` | Yes | Yes | No | Yes | BGE-m3 (1024d) |
| `balanced` | Yes | Yes | No | Yes | BGE-m3 + adaptive routing |
| `max_quality` | Yes | Yes | Yes | Yes | BGE-m3 + all features |
