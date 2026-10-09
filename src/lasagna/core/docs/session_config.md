# Session Configuration Guide

## Overview

The `session_config/` subpackage manages per-session RAG configuration state,
user profiles, and cross-process citation sharing. It uses a mixin composition
pattern to separate concerns while allowing two persistence backends.

## Mixin Composition

`KnowledgeConfigManager` is composed from 6 mixins:

```
KnowledgeConfigManager(
    ProfileMixin,           # Profile CRUD + built-in presets
    SessionStateMixin,      # Session state get/patch/reset/import/export
    _PersistenceBase,       # File-based OR PostgreSQL+Redis persistence
    CitationStoreMixin,     # Cross-process search results sharing
    BackendConfigMixin,     # BYOVDB vector backend config
    ValidationMixin,        # Config validation gate
)
```

### Mixin Contract

Each mixin expects the host class to provide:

| Attribute/Method | Used By | Purpose |
|---|---|---|
| `self._lock` (RLock) | All mixins | Thread-safe state mutations |
| `self._data_dir` (str) | PersistenceMixin, CitationStoreMixin | File storage directory |
| `self._session_states` (dict) | SessionStateMixin, PersistenceMixin | In-memory session cache |
| `self._global_custom_profiles` (list) | ProfileMixin, PersistenceMixin | Custom profile store |
| `self._global_profile_names` (set) | ProfileMixin | Fast name lookup |
| `self._default_profiles` (list) | ProfileMixin | Built-in profiles from TOML |
| `self.get_session_state()` | CitationStoreMixin | Access session state |
| `self._persist_session_states()` | CitationStoreMixin, SessionStateMixin | Trigger persistence |
| `self._state_from_dict()` | PersistenceMixin | Deserialize state |

## Persistence Backends

Selected at import time via `RAG_DATABASE_BACKEND` env var:

### File-based (default, SQLite mode)

- Profiles: `{data_dir}/rag_profiles.json` (atomic write via tempfile + os.replace)
- Sessions: `{data_dir}/rag_sessions.json` (same atomic write)
- Citations: `{data_dir}/citations_{session_id}.json` (per-session files)
- Reload check: File mtime comparison, max every 2 seconds

### PostgreSQL + Redis mode

- Profiles: `rag_profiles` table (user_id + profile_name unique)
- Sessions: `rag_sessions` table (session_id primary key)
- Read path: Redis (L1, <1ms) → PostgreSQL (L2, ~5ms)
- Write path: PostgreSQL → Redis cache update
- Request-scoped: Only loads the target session, not all sessions
- Cleanup: Throttled to once per 60 seconds per worker

## Data Model

### `KnowledgeConfigState` (per-session state)

| Field | Type | Default | Description |
|---|---|---|---|
| `active_profile` | str | `"default"` | Currently active profile name |
| `session_overrides` | dict | `{}` | Per-session config overrides |
| `rag_enabled` | bool | `True` | Whether RAG is active |
| `focused_document_ids` | list | `[]` | Document focus filter |
| `last_accessed` | float | `time.time()` | For TTL-based cleanup |
| `last_citations` | list | `[]` | Last search citations (cross-process) |
| `last_chunks` | list | `[]` | Last search chunks (cross-process) |

### `RAGProfile`

| Field | Type | Description |
|---|---|---|
| `name` | str | Profile identifier (unique per user) |
| `description` | str | Human-readable description |
| `config` | dict | Config overrides applied when profile is active |

### Constants

| Constant | Value | Description |
|---|---|---|
| `SESSION_TTL_SECONDS` | 86400 (24h) | Expired session cleanup threshold |
| `MAX_CACHED_SESSIONS` | 100 | In-memory session cache limit |
| `MAX_GLOBAL_PROFILES` | 50 | Max custom profiles per user |
| `MAX_CITATIONS_PER_SESSION` | 500 | In-memory citation cap |
| `MAX_CHUNKS_PER_SESSION` | 500 | In-memory chunk cap |
| `MAX_CHUNK_CONTENT_LENGTH` | 10000 | Per-chunk content truncation |

## Built-in Profiles

Loaded from `settings.lasagna.toml` via `_load_defaults()`:

| Profile | Key Settings |
|---|---|
| `default` | Base settings from TOML |
| `speed` | No hybrid/reranking/HyDE; MiniLM-L6-v2 (384d); fast routing |
| `standard` | Hybrid + reranking + MMR + agentic; BGE-m3 (1024d) |
| `balanced` | Standard + structured graph + hierarchical chunking + adaptive routing |
| `max_quality` | All features enabled; HyDE, expansion, RAG fusion, CRAG |

## Key Operations

### Profile Lifecycle

```python
mgr = get_knowledge_config_manager()

# List all profiles (built-in + custom)
profiles = mgr.list_profiles()

# Switch active profile for a session
mgr.set_active_profile("balanced", session_id="sess_123")

# Create custom profile
mgr.save_profile("my_profile", config={"top_k": 10, "reranking_enabled": True})

# Delete custom profile (cannot delete built-in)
mgr.delete_profile("my_profile")
```

### Session Config Operations

```python
# Get resolved config (profile + session overrides merged)
config = mgr.get_resolved_config(session_id="sess_123")

# Apply session-level overrides
mgr.patch_config({"top_k": 20}, session_id="sess_123")

# Reset session to profile defaults
mgr.reset_config(session_id="sess_123")

# Export/import for portability
exported = mgr.export_config(session_id="sess_123")
mgr.import_config(exported, session_id="sess_456")
```

### Citation Sharing (Cross-Process)

```python
# MCP subprocess stores search results
mgr.set_last_search_results(citations, chunks, session_id="sess_123")

# Main server reads them
citations, chunks = mgr.get_last_search_results(session_id="sess_123")
```

Uses atomic file writes (`tempfile.mkstemp` + `os.replace`) for cross-process
safety. Path hardening prevents traversal attacks via regex allowlist +
collision-safe hashing + path containment check.

## Citation Limit Scopes

There are two distinct citation limit scopes:

| Scope | Limit | Location | Trigger |
|---|---|---|---|
| Database (hot path) | 5000 per session | `database.py` / `database_pg.py` `store_citations()` | Every `store_citations()` call |
| Database (standalone) | 1000 per session (default) | `enforce_citation_limit()` | Manual/admin cleanup |
| In-memory (session state) | 500 per session | `_data_models.py` `MAX_CITATIONS_PER_SESSION` | `set_last_search_results()` |

## Thread Safety

- All state mutations go through `self._lock` (RLock)
- File persistence uses atomic write (tempfile + os.replace)
- PostgreSQL singleton (`_get_pg_db()`) uses double-check locking with `_pg_db_lock`
- Session cleanup throttle uses module-level `_cleanup_lock`
