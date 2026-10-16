# Core Package Architecture

## Dependency Graph

```
                        ┌──────────────────┐
                        │   __init__.py    │  (public API surface)
                        └──────┬───────────┘
                               │ re-exports
          ┌────────────┬───────┼───────┬────────────┬──────────┐
          ▼            ▼       ▼       ▼            ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ ┌───────────────┐
    │ config.py│ │schema.py │ │registry│ │ pipeline  │ │session_config/│
    │          │ │          │ │  .py   │ │   .py     │ │  __init__.py  │
    └──┬───────┘ └──────────┘ └───┬────┘ └─────┬─────┘ └──┬────────────┘
       │                          │             │          │
       │              ┌───────────┘       ┌─────┘    ┌─────┘
       ▼              ▼                   ▼          ▼
  ┌──────────────────────────┐    ┌────────────┐  ┌────────────────────┐
  │  base_components.py      │    │ database.py│  │ session_config/    │
  │  (ABCs: 8 components)    │    │ (SQLite)   │  │ (6 mixins, 9 files)│
  └──────────────────────────┘    │database_pg │  └────────────────────┘
                                  │ (Postgres) │
                                  └──────┬─────┘
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                         ┌────────┐ ┌────────┐ ┌────────┐
                         │quota.py│ │rate_   │ │tenant  │
                         │        │ │limiter │ │.py     │
                         └────────┘ │.py     │ └────────┘
                                    └────────┘
```

External consumers (55 modules outside `core/` import from it), among them:
- `api/`: REST routes, dependencies, backend orchestration
- `sdk/`: Python SDK client, managers
- `agentic/`: Agentic RAG graph nodes
- `retrieval/`: Search, ranking, enhancement
- `processing/`: Document processing, chunking
- `infrastructure/`: Embedding, caching
- `providers/`: Config, LLM, session providers
- `knowledge_evaluation_sdk/`: the experiment runner

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
│  Priority 2: Environment variables (LASAGNA_*, RAG_*) │
│  ┌──────────────────────────────────────────────┐    │
│  │ self._env_overrides[leaf_key]                  │    │
│  │ LASAGNA_<NAME> wins over RAG_<NAME>            │    │
│  │ Type-coerced to match default type             │    │
│  │ Numeric range validation for known keys        │    │
│  └──────────────────────────────────────────────┘    │
│                    │ miss                             │
│                    ▼                                  │
│  Priority 3: TOML settings                            │
│  ┌──────────────────────────────────────────────┐    │
│  │ settings.lasagna.toml [rag] section          │    │
│  │ Or a file named by LASAGNA_SETTINGS_FILE      │    │
│  │ (or RAG_SETTINGS_FILE)                         │    │
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
            │ _cleanup_old_    │  ← TTL=24h, throttled to 1/min
            │  sessions()      │
            └──────────────────┘
```

## Database Backend Architecture

```
          ┌─────────────────────────┐
          │   DatabaseBackend       │ (@runtime_checkable Protocol)
          │   41 methods            │
          └──────┬──────────────────┘
                 │ implements
       ┌─────────┴──────────┐
       ▼                    ▼
┌──────────────┐    ┌──────────────────┐
│ RAGDatabase  │    │PostgresRAGDatabase│
│ (SQLite)     │    │ (PostgreSQL)      │
│ per-call conn│    │ connection pool   │
│ WAL mode     │    │ psycopg3/2        │
│ RLock        │    │ extra table:      │
│              │    │  rag_quotas       │
└──────────────┘    └──────────────────┘
```

Both backends share:
- 9 tables: `rag_documents`, `citations`, `parent_chunks`, `schema_meta`, `rag_profiles`, `rag_sessions`, `collection_meta`, `gc_audit_log`, `gc_lock`
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
        │          │ │          │    │ lasagna_ │
        └──────────┘ └──────────┘    │components│
                                     └──────────┘

Categories (COMPONENT_TYPES): embedder, chunker, reranker, retriever,
            document_processor, query_enhancer, generator, agentic_pipeline
```

Registration: `@registry.register("category", "name")` decorator or
`registry.register_class("category", "name", cls)` imperative.

Lookup: `registry.get("category", "name", **kwargs)` with adaptive parameter filtering.

## Mixin Composition (KnowledgeConfigManager)

```
KnowledgeConfigManager
  ├── ProfileMixin          (_profiles.py)
  │     profile CRUD, built-in presets (default plus 4 profiles)
  ├── SessionStateMixin     (_session_state.py)
  │     get/patch/reset/import/export config
  ├── PersistenceMixin      (_persistence.py, SQLite mode)
  │   OR DatabasePersistenceMixin (_persistence_db.py, PG mode)
  │     load/save profiles & session states
  ├── CitationStoreMixin    (_citations_store.py)
  │     cross-process search results sharing
  ├── BackendConfigMixin    (_backend_config.py)
  │     BYOVDB vector backend config
  └── ValidationMixin       (_validation_mixin.py)
        config validation gate
```

Mode selection: `RAG_DATABASE_BACKEND=postgresql` in the environment when `core.session_config` is imported → DatabasePersistenceMixin,
otherwise → PersistenceMixin (file-based). This choice reads that one variable, not `LASAGNA_DATABASE_BACKEND` or the
`database_backend` setting in a settings file.

## Built-in RAG Profiles

| Profile | Hybrid | Reranking | HyDE | Agentic | Embedding |
|---|---|---|---|---|---|
| `speed` | No | No | No | No | MiniLM-L6-v2 (384d) |
| `standard` | Yes | Yes | No | Yes | BGE-m3 (1024d) |
| `balanced` | Yes | Yes | No | Yes | BGE-m3 + adaptive routing |
| `max_quality` | Yes | Yes | Yes | Yes | BGE-m3 + all features |
