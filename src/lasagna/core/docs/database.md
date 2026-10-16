# Database Backend Guide

## Overview

The RAG system stores document metadata, citations, and parent chunks in a
relational database. Two backends are supported:

| Feature | SQLite (`RAGDatabase`) | PostgreSQL (`PostgresRAGDatabase`) |
|---|---|---|
| Use case | Single process: local use, development, testing | Several processes or replicas sharing one store |
| Connection model | Per-call with WAL mode, `timeout=30` | Connection pool (psycopg3) or per-call (psycopg2) |
| Thread safety | `threading.RLock` + `connection()`/`transaction()` context managers | Connection-per-operation (no shared cursors) |
| Schema version | 6 (`SCHEMA_VERSION`, with migrations v1-v6) | 6 (the same `STORE_VERSION`; inline DDL blocks, idempotent, under an advisory lock) |
| Driver | `sqlite3` (stdlib) | `psycopg` v3 (preferred) or `psycopg2` (fallback) |

## Parity Contract

Both backends MUST exhibit identical behavior for these operations:

1. **Citation limit enforcement**: `store_citations()` with N > 5000 retains only the most recent 5000 citations per session (atomic, within the same transaction)
2. **Citation upsert**: `store_citations()` on conflict (same `citation_id`) updates ALL 14 mutable fields
3. **Statistics**: `get_stats()` returns shared keys: `documents.{total, sessions, by_status}`, `citations.{total, expired_pending_cleanup, sessions}`, `schema_version`
4. **Protocol conformance**: All `DatabaseBackend` Protocol methods produce identical outputs for identical inputs
5. **Schema parity**: Shared tables (`rag_documents`, `citations`, `parent_chunks`, `schema_meta`, `rag_profiles`, `rag_sessions`, `collection_meta`, `gc_audit_log`, `gc_lock`) have semantically equivalent DDL

## Intended Differences

These are NOT bugs:

- **Extra PG table**: `rag_quotas` (multi-tenant quota storage shared across replicas)
- **Extra PG methods** (3): `get_profile`, `get_user_quota`, `save_user_quota`. Both backends have `save_profile`, `get_profiles`, `delete_profile`, `save_session`, `get_session`, `get_active_sessions` and `cleanup_expired_sessions`.
- **Type mappings**: `REAL` vs `DOUBLE PRECISION`, `INTEGER` vs `BIGINT`, `TIMESTAMP` vs `TIMESTAMPTZ`, `TEXT '{}'` vs `JSONB '{}'::jsonb`
- **Backend-specific stats keys**: SQLite returns `db_path`, PG returns `backend` + `dsn`
- **Connection lifecycle**: SQLite opens/closes per call; PG uses a connection pool with min/max sizing

## Shared Schema (v6)

The three central tables follow. `rag_profiles` and `rag_sessions` hold session config (see [session_config.md](session_config.md)), `gc_audit_log` and `gc_lock` the garbage collector's audit trail and lock, and `collection_meta` the embedding model each vector collection was built with.

### `rag_documents`

```sql
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER DEFAULT 0,
    file_type TEXT DEFAULT '',
    status TEXT DEFAULT 'processing',
    chunks_count INTEGER DEFAULT 0,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    processing_time REAL,
    text_length INTEGER,
    preview_text TEXT,
    vector_backend_name TEXT DEFAULT 'milvus',
    storage_backend_name TEXT DEFAULT 'local',
    deleted_at TIMESTAMP DEFAULT NULL,
    expires_at TIMESTAMP DEFAULT NULL,
    collection_name TEXT DEFAULT NULL
);
```

### `citations`

```sql
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- SERIAL in PG
    citation_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    query_id TEXT,
    chunk_id TEXT,
    document_id TEXT NOT NULL,
    document_name TEXT,
    content_preview TEXT,
    content_hash TEXT,
    chunk_index INTEGER,
    page_number INTEGER,
    section_header TEXT,
    score REAL,               -- DOUBLE PRECISION in PG
    normalized_score REAL,    -- DOUBLE PRECISION in PG
    hop_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- TIMESTAMPTZ in PG
    expires_at TIMESTAMP,     -- TIMESTAMPTZ in PG
    FOREIGN KEY (document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
);
```

### `parent_chunks`

```sql
CREATE TABLE IF NOT EXISTS parent_chunks (
    stable_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',  -- JSONB in PG
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
);
```

## Indexes

Both backends share these indexes:

| Index | Columns | Purpose |
|---|---|---|
| `idx_documents_session_id` | `rag_documents(session_id)` | Session document lookup |
| `idx_documents_status` | `rag_documents(status)` | Status filtering |
| `idx_documents_deleted_at` | `rag_documents(deleted_at) WHERE deleted_at IS NOT NULL` | Soft-delete GC (partial) |
| `idx_documents_expires_at` | `rag_documents(expires_at) WHERE expires_at IS NOT NULL` | TTL expiry (partial) |
| `idx_documents_backend` | `rag_documents(vector_backend_name)` | Backend tracking |
| `idx_documents_storage` | `rag_documents(storage_backend_name)` | Storage tracking |
| `idx_citations_session` | `citations(session_id)` | Session citation lookup |
| `idx_citations_query` | `citations(query_id)` | Query-scoped citation lookup |
| `idx_citations_created` | `citations(created_at)` | Age-based cleanup |
| `idx_citations_expires` | `citations(expires_at)` | TTL-based cleanup |
| `idx_citations_content_hash` | `citations(content_hash)` | Dedup lookup |
| `idx_citations_document` | `citations(document_id)` | Document cascade |
| `idx_citations_session_created` | `citations(session_id, created_at DESC, id DESC)` | Citation limit enforcement (composite) |
| `idx_parent_chunks_document` | `parent_chunks(document_id)` | Document cascade |
| `idx_parent_chunks_session` | `parent_chunks(session_id)` | Session lookup |
| `idx_profiles_user` | `rag_profiles(user_id)` | Profile lookup |
| `idx_sessions_user` | `rag_sessions(user_id)` | Session lookup by user |
| `idx_sessions_accessed` | `rag_sessions(last_accessed)` | Idle-session cleanup |
| `idx_gc_audit_run_id` | `gc_audit_log(run_id)` | GC run lookup |
| `idx_gc_audit_started` | `gc_audit_log(started_at)` | GC history |

PostgreSQL adds `idx_profiles_builtin` on `rag_profiles(is_builtin)`.

## Citation Limit Enforcement

Both backends enforce a per-session limit of 5000 citations within the
`store_citations()` transaction:

- **SQLite**: `_enforce_citation_limit_on_conn()` runs `DELETE WHERE id NOT IN (SELECT id ... ORDER BY created_at DESC LIMIT 5000)`
- **PostgreSQL**: CTE-based eviction. It finds the cutoff row at offset 5000 and deletes everything older using `(created_at, id) < (cutoff)` for deterministic ordering

The limit is only enforced when `stored_count > 0` (no-op for empty batches).
The standalone `enforce_citation_limit(session_id, max_citations=1000)` method
is available for manual/admin cleanup.

## Migration History

| Version | Changes |
|---|---|
| v0 → v1 | Initial schema (the `CREATE TABLE` statements) |
| v1 → v2 | Add the `parent_chunks` table for hierarchical chunking |
| v2 → v3 | Add `vector_backend_name` column to `rag_documents` |
| v3 → v4 | Add `storage_backend_name` column to `rag_documents` |
| v4 → v5 | Add the `rag_profiles` and `rag_sessions` tables (session config persistence) |
| v5 → v6 | Add the lifecycle columns `deleted_at`, `expires_at` and `collection_name` to `rag_documents`, and the `gc_audit_log` and `gc_lock` tables |

A store written by a newer version than the code is refused (`refuse_newer_store`) rather than migrated down.

SQLite migrations run via `_run_migrations()` with `ALTER TABLE ADD COLUMN`
guarded by error checks for pre-existing columns.

PostgreSQL migrations run as inline `DO $$ ... $$` blocks during
`_create_schema()`, using `information_schema.columns` checks for idempotency.

## Singleton Access

```python
from lasagna.core.database import get_rag_database, reset_rag_database

# Auto-selects backend based on rag_config.database_backend
db = get_rag_database()

# For testing: reset calls close() on the old instance
reset_rag_database()
```

Backend selection logic in `get_rag_database()`:
- `database_backend == "postgresql"` → `PostgresRAGDatabase(dsn=...)`
- Otherwise → `RAGDatabase(db_path)` (SQLite)
