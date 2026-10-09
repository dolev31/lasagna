# Database Backend Guide

## Overview

The RAG system stores document metadata, citations, and parent chunks in a
relational database. Two backends are supported:

| Feature | SQLite (`RAGDatabase`) | PostgreSQL (`PostgresRAGDatabase`) |
|---|---|---|
| Use case | Single-user, development, testing | Multi-instance production |
| Connection model | Per-call with WAL mode, `timeout=30` | Connection pool (psycopg3) or per-call (psycopg2) |
| Thread safety | `threading.RLock` + `connection()`/`transaction()` context managers | Connection-per-operation (no shared cursors) |
| Schema version | 4 (with migrations v0-v4) | 4 (inline DDL blocks, idempotent) |
| Driver | `sqlite3` (stdlib) | `psycopg` v3 (preferred) or `psycopg2` (fallback) |

## Parity Contract

Both backends MUST exhibit identical behavior for these operations:

1. **Citation limit enforcement**: `store_citations()` with N > 5000 retains only the most recent 5000 citations per session (atomic, within the same transaction)
2. **Citation upsert**: `store_citations()` on conflict (same `citation_id`) updates ALL 14 mutable fields
3. **Statistics**: `get_stats()` returns shared keys: `documents.{total, sessions, by_status}`, `citations.{total, expired_pending_cleanup, sessions}`, `schema_version`
4. **Protocol conformance**: All `DatabaseBackend` Protocol methods produce identical outputs for identical inputs
5. **Schema parity**: Shared tables (`rag_documents`, `citations`, `parent_chunks`, `schema_meta`) have semantically equivalent DDL

## Intended Differences

These are NOT bugs:

- **Extra PG tables**: `rag_profiles`, `rag_sessions`, `rag_quotas` (for distributed session persistence and multi-tenant quota storage)
- **Extra PG methods** (10): `save_profile`, `get_profiles`, `delete_profile`, `get_profile`, `save_session`, `get_session`, `get_active_sessions`, `cleanup_expired_sessions`, `get_user_quota`, `save_user_quota`
- **Type mappings**: `REAL` vs `DOUBLE PRECISION`, `INTEGER` vs `BIGINT`, `TIMESTAMP` vs `TIMESTAMPTZ`, `TEXT '{}'` vs `JSONB '{}'::jsonb`
- **Backend-specific stats keys**: SQLite returns `db_path`, PG returns `backend` + `dsn`
- **Connection lifecycle**: SQLite opens/closes per call; PG uses a connection pool with min/max sizing

## Shared Schema (v4)

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
    storage_backend_name TEXT DEFAULT 'local'
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Indexes

Both backends share these indexes:

| Index | Columns | Purpose |
|---|---|---|
| `idx_documents_session_id` | `rag_documents(session_id)` | Session document lookup |
| `idx_documents_status` | `rag_documents(status)` | Status filtering |
| `idx_citations_session` | `citations(session_id)` | Session citation lookup |
| `idx_citations_query` | `citations(query_id)` | Query-scoped citation lookup |
| `idx_citations_created` | `citations(created_at)` | Age-based cleanup |
| `idx_citations_expires` | `citations(expires_at)` | TTL-based cleanup |
| `idx_citations_content_hash` | `citations(content_hash)` | Dedup lookup |
| `idx_citations_document` | `citations(document_id)` | Document cascade |
| `idx_citations_session_created` | `citations(session_id, created_at DESC, id DESC)` | Citation limit enforcement (composite) |
| `idx_parent_chunks_document` | `parent_chunks(document_id)` | Document cascade |
| `idx_parent_chunks_session` | `parent_chunks(session_id)` | Session lookup |

## Citation Limit Enforcement

Both backends enforce a per-session limit of 5000 citations within the
`store_citations()` transaction:

- **SQLite**: `_enforce_citation_limit_on_conn()` — `DELETE WHERE id NOT IN (SELECT id ... ORDER BY created_at DESC LIMIT 5000)`
- **PostgreSQL**: CTE-based eviction — finds the cutoff row at offset 5000, deletes everything older using `(created_at, id) < (cutoff)` for deterministic ordering

The limit is only enforced when `stored_count > 0` (no-op for empty batches).
The standalone `enforce_citation_limit(session_id, max_citations=1000)` method
is available for manual/admin cleanup.

## Migration History

| Version | Changes |
|---|---|
| v0 → v1 | Add `preview_text` column to `rag_documents` |
| v1 → v2 | Add `vector_backend_name` column |
| v2 → v3 | Migrate `chunk_id` from INTEGER to TEXT |
| v3 → v4 | Add `storage_backend_name` column |

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
