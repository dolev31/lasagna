"""PostgreSQL backend for RAG metadata storage.

Drop-in replacement for ``RAGDatabase`` (SQLite) that uses PostgreSQL
for multi-instance, concurrent-write production deployments.

Requires ``psycopg2-binary>=2.9.0`` (sync) or ``psycopg[binary]>=3.1``
(sync+async-ready).  Falls back to psycopg2 when psycopg3 is absent.

Usage::

    database_backend = "postgresql"
    database_dsn = "postgresql://user:pass@localhost:5432/rag"

The implementation conforms to the ``DatabaseBackend`` protocol defined
in ``database_backend.py``.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from lasagna.core.provenance import redact_url

# ---------------------------------------------------------------------------
# Driver import with fallback
# ---------------------------------------------------------------------------

_psycopg = None
_psycopg2 = None

try:
    import psycopg  # psycopg 3

    _psycopg = psycopg
except ImportError:
    pass

if _psycopg is None:
    try:
        import psycopg2
        import psycopg2.extras

        _psycopg2 = psycopg2
    except ImportError:
        pass


def _check_driver():
    if _psycopg is None and _psycopg2 is None:
        raise ImportError(
            "PostgreSQL backend requires either psycopg (v3) or psycopg2. "
            "Install with: pip install psycopg[binary] or pip install psycopg2-binary"
        )


# ---------------------------------------------------------------------------
# Schema DDL (PostgreSQL dialect)
# ---------------------------------------------------------------------------

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_size BIGINT DEFAULT 0,
    file_type TEXT DEFAULT '',
    status TEXT DEFAULT 'processing',
    chunks_count INTEGER DEFAULT 0,
    upload_date TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT,
    processing_time DOUBLE PRECISION,
    text_length INTEGER,
    preview_text TEXT,
    vector_backend_name TEXT DEFAULT 'milvus',
    storage_backend_name TEXT DEFAULT 'local',
    deleted_at TIMESTAMPTZ DEFAULT NULL,
    expires_at TIMESTAMPTZ DEFAULT NULL,
    collection_name TEXT DEFAULT NULL
);
"""

_CREATE_GC_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS gc_audit_log (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    docs_scanned INTEGER DEFAULT 0,
    docs_cleaned INTEGER DEFAULT 0,
    bytes_freed BIGINT DEFAULT 0,
    errors INTEGER DEFAULT 0,
    details JSONB DEFAULT '{}'::jsonb
);
"""

_CREATE_GC_LOCK = """
CREATE TABLE IF NOT EXISTS gc_lock (
    lock_name TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
"""

_CREATE_CITATIONS = """
CREATE TABLE IF NOT EXISTS citations (
    id SERIAL PRIMARY KEY,
    citation_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    query_id TEXT,
    chunk_id TEXT,
    document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
    document_name TEXT,
    content_preview TEXT,
    content_hash TEXT,
    chunk_index INTEGER,
    page_number INTEGER,
    section_header TEXT,
    score DOUBLE PRECISION,
    normalized_score DOUBLE PRECISION,
    hop_index INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);
"""

_CREATE_PARENT_CHUNKS = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    stable_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS rag_profiles (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '__global__',
    profile_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    config JSONB NOT NULL DEFAULT '{}',
    is_builtin BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, profile_name)
);
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS rag_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '__anonymous__',
    active_profile TEXT NOT NULL DEFAULT 'default',
    session_overrides JSONB DEFAULT '{}',
    rag_enabled BOOLEAN DEFAULT true,
    focused_document_ids JSONB DEFAULT '[]',
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_QUOTAS = """
CREATE TABLE IF NOT EXISTS rag_quotas (
    user_id TEXT PRIMARY KEY,
    tier TEXT DEFAULT 'free',
    max_documents INT DEFAULT 1000,
    max_storage_bytes BIGINT DEFAULT 5368709120,
    max_searches_per_minute INT DEFAULT 600,
    max_uploads_per_hour INT DEFAULT 200,
    max_chunks INT DEFAULT 100000,
    max_profiles INT DEFAULT 50
);
"""

# Which embedding model made each vector collection (see database.py).
_CREATE_COLLECTION_META = """
CREATE TABLE IF NOT EXISTS collection_meta (
    vector_backend TEXT NOT NULL DEFAULT 'milvus',
    collection_name TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dim INT NOT NULL,
    lasagna_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (vector_backend, collection_name)
);
"""

# A counter per session, bumped on every upload and delete (see database.py).
_CREATE_CONTENT_GENERATIONS = """
CREATE TABLE IF NOT EXISTS content_generations (
    session_id TEXT PRIMARY KEY,
    generation BIGINT NOT NULL DEFAULT 0
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON rag_documents(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON rag_documents(status);",
    "CREATE INDEX IF NOT EXISTS idx_documents_backend ON rag_documents(vector_backend_name);",
    "CREATE INDEX IF NOT EXISTS idx_documents_storage ON rag_documents(storage_backend_name);",
    # Lifecycle indexes (partial for efficiency — only non-null rows)
    "CREATE INDEX IF NOT EXISTS idx_documents_deleted_at ON rag_documents(deleted_at) WHERE deleted_at IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_documents_expires_at ON rag_documents(expires_at) WHERE expires_at IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_citations_session ON citations(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_query ON citations(query_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_created ON citations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_citations_expires ON citations(expires_at);",
    "CREATE INDEX IF NOT EXISTS idx_citations_content_hash ON citations(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_citations_document ON citations(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_session_created ON citations(session_id, created_at DESC, id DESC);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_document ON parent_chunks(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_session ON parent_chunks(session_id);",
    # Config profiles & sessions indexes
    "CREATE INDEX IF NOT EXISTS idx_profiles_user ON rag_profiles(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_profiles_builtin ON rag_profiles(is_builtin);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON rag_sessions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_accessed ON rag_sessions(last_accessed);",
    # GC audit log indexes
    "CREATE INDEX IF NOT EXISTS idx_gc_audit_run_id ON gc_audit_log(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_gc_audit_started ON gc_audit_log(started_at);",
]


# ---------------------------------------------------------------------------
# Store version
# ---------------------------------------------------------------------------

#: Key of the transaction-scoped advisory lock that serializes schema creation
#: and migration ("lasagna" in ASCII, as a bigint).
_SCHEMA_LOCK_KEY = 0x6C617361676E61


def _lock_and_check_store_version(cur, where: str) -> None:
    """Take the schema lock for this transaction, then refuse a newer store.

    Every replica and worker creates and migrates the schema at startup. Run
    concurrently, two of them raced on ``CREATE TABLE`` / ``ALTER TABLE`` and one
    failed its start with a duplicate-object error. The advisory lock is
    released when the transaction commits or rolls back.
    """
    from lasagna.core.database import refuse_newer_store

    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
    cur.execute(_CREATE_SCHEMA_META)
    cur.execute("SELECT value FROM schema_meta WHERE key = 'version'")
    row = cur.fetchone()
    refuse_newer_store(int(row[0]) if row else 0, where)


def _store_version() -> int:
    from lasagna.core.database import STORE_VERSION

    return STORE_VERSION


def _record_store_version(cur) -> None:
    """Write this code's store version; it was never written on PostgreSQL."""
    from lasagna.core.database import STORE_VERSION

    cur.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('version', %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (str(STORE_VERSION),),
    )


# ---------------------------------------------------------------------------
# PostgreSQL RAG Database
# ---------------------------------------------------------------------------


class PostgresRAGDatabase:
    """PostgreSQL implementation of the DatabaseBackend protocol.

    Thread-safe via connection-per-operation (no shared cursors).
    Connection pooling is handled by psycopg's built-in pool or can be
    configured externally.
    """

    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10):
        _check_driver()
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool = None
        self._lock = threading.Lock()
        self._initialized = False
        self._initialize()

    def _initialize(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._setup_pool()
            self._create_schema()
            self._initialized = True
            logger.info(f"PostgreSQL RAG database initialized: {redact_url(self._dsn)}")

    def _setup_pool(self):
        if _psycopg is not None:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
            self._driver = "psycopg3"
        else:
            # psycopg2 doesn't have a built-in pool in the base package;
            # use simple connection-per-call.
            self._pool = None
            self._driver = "psycopg2"
        logger.debug(f"PostgreSQL driver: {self._driver}")

    def close(self) -> None:
        """Shut down connection pool. Idempotent."""
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None
        self._initialized = False

    @contextmanager
    def _conn(self):
        """Yield a connection from the pool (psycopg3) or create one (psycopg2)."""
        if self._driver == "psycopg3" and self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            conn = _psycopg2.connect(self._dsn)
            try:
                yield conn
            finally:
                conn.close()

    def _create_schema(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                _lock_and_check_store_version(
                    cur, self._dsn.split("@")[-1] if "@" in self._dsn else "(local)"
                )
                cur.execute(_CREATE_DOCUMENTS)
                cur.execute(_CREATE_CITATIONS)
                cur.execute(_CREATE_PARENT_CHUNKS)
                cur.execute(_CREATE_PROFILES)
                cur.execute(_CREATE_SESSIONS)
                cur.execute(_CREATE_QUOTAS)
                cur.execute(_CREATE_GC_AUDIT_LOG)
                cur.execute(_CREATE_GC_LOCK)
                cur.execute(_CREATE_CONTENT_GENERATIONS)
                for idx_sql in _CREATE_INDEXES:
                    cur.execute(idx_sql)
                # Add storage_backend_name column if missing (v3->v4 migration)
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'rag_documents'
                              AND column_name = 'storage_backend_name'
                        ) THEN
                            ALTER TABLE rag_documents ADD COLUMN storage_backend_name TEXT DEFAULT 'local';
                        END IF;
                    END $$;
                """)
                # v5->v6: Add lifecycle columns if missing
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'rag_documents' AND column_name = 'deleted_at'
                        ) THEN
                            ALTER TABLE rag_documents ADD COLUMN deleted_at TIMESTAMPTZ DEFAULT NULL;
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'rag_documents' AND column_name = 'expires_at'
                        ) THEN
                            ALTER TABLE rag_documents ADD COLUMN expires_at TIMESTAMPTZ DEFAULT NULL;
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'rag_documents' AND column_name = 'collection_name'
                        ) THEN
                            ALTER TABLE rag_documents ADD COLUMN collection_name TEXT DEFAULT NULL;
                        END IF;
                    END $$;
                """)
                # Migrate citations.chunk_id from INTEGER to TEXT if needed
                cur.execute("""
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'citations'
                              AND column_name = 'chunk_id'
                              AND data_type = 'integer'
                        ) THEN
                            ALTER TABLE citations ALTER COLUMN chunk_id TYPE TEXT USING chunk_id::TEXT;
                        END IF;
                    END $$;
                """)
                _record_store_version(cur)
            conn.commit()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def save_document(
        self, document: Dict[str, Any], file_path: str, collection_name: Optional[str] = None
    ) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_documents
                    (document_id, session_id, filename, file_path, file_size,
                     file_type, status, chunks_count, upload_date, error_message,
                     processing_time, text_length, preview_text, vector_backend_name,
                     storage_backend_name, collection_name, expires_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        session_id=EXCLUDED.session_id, filename=EXCLUDED.filename,
                        file_path=EXCLUDED.file_path, file_size=EXCLUDED.file_size,
                        file_type=EXCLUDED.file_type, status=EXCLUDED.status,
                        chunks_count=EXCLUDED.chunks_count, upload_date=EXCLUDED.upload_date,
                        error_message=EXCLUDED.error_message, processing_time=EXCLUDED.processing_time,
                        text_length=EXCLUDED.text_length, preview_text=EXCLUDED.preview_text,
                        vector_backend_name=EXCLUDED.vector_backend_name,
                        storage_backend_name=EXCLUDED.storage_backend_name,
                        collection_name=COALESCE(EXCLUDED.collection_name, rag_documents.collection_name),
                        expires_at=COALESCE(EXCLUDED.expires_at, rag_documents.expires_at)
                    """,
                    (
                        document.get("document_id"),
                        document.get("session_id"),
                        document.get("filename"),
                        file_path,
                        document.get("file_size"),
                        document.get("file_type"),
                        document.get("status", "processing"),
                        document.get("chunks_count", 0),
                        document.get("upload_date", datetime.now(timezone.utc).isoformat()),
                        document.get("error_message"),
                        document.get("processing_time"),
                        document.get("text_length"),
                        document.get("preview_text"),
                        document.get("vector_backend_name", "milvus"),
                        document.get("storage_backend_name", "local"),
                        collection_name or document.get("collection_name"),
                        document.get("expires_at"),
                    ),
                )
            conn.commit()
        return True

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM rag_documents WHERE document_id = %s", (document_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def get_documents(
        self,
        session_id: str,
        status: Optional[str] = None,
        include_soft_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                conditions = ["session_id = %s"]
                params: list = [session_id]

                if status:
                    conditions.append("status = %s")
                    params.append(status)
                elif not include_soft_deleted:
                    conditions.append("status != 'soft_deleted'")

                where = " AND ".join(conditions)
                cur.execute(
                    f"SELECT * FROM rag_documents WHERE {where} ORDER BY upload_date DESC",
                    params,
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    # Backward-compatible alias
    get_session_documents = get_documents

    def get_documents_batch(
        self,
        document_ids: List[str],
        include_soft_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get multiple documents by their IDs in a single query."""
        if not document_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(document_ids))
                soft_filter = "" if include_soft_deleted else " AND status != 'soft_deleted'"
                cur.execute(
                    f"SELECT * FROM rag_documents WHERE document_id IN ({placeholders}){soft_filter}",
                    document_ids,
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
        chunks_count: Optional[int] = None,
        processing_time: Optional[float] = None,
        text_length: Optional[int] = None,
        preview_text: Optional[str] = None,
    ) -> bool:
        updates = ["status = %s"]
        params: list = [status]

        if error_message is not None:
            updates.append("error_message = %s")
            params.append(error_message)
        if chunks_count is not None:
            updates.append("chunks_count = %s")
            params.append(chunks_count)
        if processing_time is not None:
            updates.append("processing_time = %s")
            params.append(processing_time)
        if text_length is not None:
            updates.append("text_length = %s")
            params.append(text_length)
        if preview_text is not None:
            updates.append("preview_text = %s")
            params.append(preview_text)

        params.append(document_id)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE rag_documents SET {', '.join(updates)} WHERE document_id = %s",
                    params,
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def delete_document(self, document_id: str) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_documents WHERE document_id = %s", (document_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    # ------------------------------------------------------------------
    # Citation operations
    # ------------------------------------------------------------------

    def store_citations(
        self,
        citations: List[Dict[str, Any]],
        session_id: str,
        query_id: Optional[str] = None,
        ttl_hours: int = 168,
    ) -> int:
        if not citations:
            return 0
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        stored = 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for c in citations:
                    try:
                        cur.execute(
                            """
                            INSERT INTO citations
                            (citation_id, session_id, query_id, chunk_id, document_id,
                             document_name, content_preview, content_hash, chunk_index,
                             page_number, section_header, score, normalized_score,
                             hop_index, expires_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (citation_id) DO UPDATE SET
                                session_id=EXCLUDED.session_id, query_id=EXCLUDED.query_id,
                                chunk_id=EXCLUDED.chunk_id, document_id=EXCLUDED.document_id,
                                document_name=EXCLUDED.document_name,
                                content_preview=EXCLUDED.content_preview,
                                content_hash=EXCLUDED.content_hash,
                                chunk_index=EXCLUDED.chunk_index,
                                page_number=EXCLUDED.page_number,
                                section_header=EXCLUDED.section_header,
                                score=EXCLUDED.score, normalized_score=EXCLUDED.normalized_score,
                                hop_index=EXCLUDED.hop_index, expires_at=EXCLUDED.expires_at
                            """,
                            (
                                c.get("citation_id"),
                                session_id,
                                query_id or c.get("query_id"),
                                c.get("chunk_id"),
                                c.get("document_id"),
                                c.get("document_name"),
                                c.get("content_preview"),
                                c.get("content_hash"),
                                c.get("chunk_index"),
                                c.get("page_number"),
                                c.get("section_header"),
                                c.get("score"),
                                c.get("normalized_score"),
                                c.get("hop_index"),
                                expires_at.isoformat(),
                            ),
                        )
                        stored += 1
                    except Exception as e:
                        logger.error(f"Failed to store citation: {e}")

                # Enforce per-session citation limit within the same transaction
                # (mirrors SQLite's atomic insert + eviction in store_citations).
                if stored > 0:
                    try:
                        cur.execute(
                            "SELECT COUNT(*) FROM citations WHERE session_id = %s",
                            (session_id,),
                        )
                        total = cur.fetchone()[0]
                        max_citations = 5000
                        if total > max_citations:
                            # CTE-based eviction: find the cutoff row, delete
                            # everything older.  Deterministic ordering via
                            # (created_at DESC, id DESC) breaks ties on id.
                            cur.execute(
                                """
                                WITH cutoff AS (
                                    SELECT created_at, id FROM citations
                                    WHERE session_id = %s
                                    ORDER BY created_at DESC, id DESC
                                    OFFSET %s LIMIT 1
                                )
                                DELETE FROM citations
                                WHERE session_id = %s
                                  AND (created_at, id) <= (SELECT created_at, id FROM cutoff)
                                """,
                                (session_id, max_citations, session_id),
                            )
                            evicted = cur.rowcount
                            if evicted > 0:
                                logger.debug(
                                    f"Evicted {evicted} old citations for session {session_id}"
                                )
                    except Exception as e:
                        logger.warning(f"Citation limit enforcement failed: {e}")
            conn.commit()
        return stored

    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM citations WHERE citation_id = %s AND (expires_at IS NULL OR expires_at > NOW())",
                    (citation_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def get_session_citations(
        self,
        session_id: str,
        query_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if query_id:
                    cur.execute(
                        "SELECT * FROM citations WHERE session_id = %s AND query_id = %s "
                        "AND (expires_at IS NULL OR expires_at > NOW()) ORDER BY created_at DESC LIMIT %s",
                        (session_id, query_id, limit),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM citations WHERE session_id = %s "
                        "AND (expires_at IS NULL OR expires_at > NOW()) ORDER BY created_at DESC LIMIT %s",
                        (session_id, limit),
                    )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def clear_session_citations(self, session_id: str) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM citations WHERE session_id = %s", (session_id,))
                count = cur.rowcount
            conn.commit()
        return count

    def cleanup_expired_citations(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM citations WHERE expires_at < NOW()")
                count = cur.rowcount
            conn.commit()
        if count > 0:
            logger.info(f"Cleaned up {count} expired citations")
        return count

    def cleanup_citations_by_age(self, max_age_hours: int) -> int:
        """Delete citations older than *max_age_hours* (safety-net cleanup)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM citations WHERE created_at < NOW() - (%s * INTERVAL '1 hour')",
                    (max_age_hours,),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def get_citation_by_content_hash(
        self, content_hash: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM citations WHERE content_hash = %s AND session_id = %s "
                    "AND (expires_at IS NULL OR expires_at > NOW()) "
                    "ORDER BY created_at DESC LIMIT 1",
                    (content_hash, session_id),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def get_citation_aggregate_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT AVG(normalized_score) FROM citations WHERE normalized_score IS NOT NULL"
                )
                avg_score = cur.fetchone()[0] or 0.0

                cur.execute("SELECT COUNT(*) FROM citations WHERE page_number IS NULL")
                missing_page = cur.fetchone()[0]
        return {"avg_score": float(avg_score), "missing_page": missing_page}

    def enforce_citation_limit(self, session_id: str, max_citations: int = 1000) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM citations
                    WHERE session_id = %s
                    AND id NOT IN (
                        SELECT id FROM citations
                        WHERE session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    )
                    """,
                    (session_id, session_id, max_citations),
                )
                count = cur.rowcount
            conn.commit()
        return count

    # ------------------------------------------------------------------
    # Parent chunk operations
    # ------------------------------------------------------------------

    def save_parent_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    meta = c.get("metadata", {})
                    meta_str = json.dumps(meta) if isinstance(meta, dict) else str(meta)
                    cur.execute(
                        """
                        INSERT INTO parent_chunks
                        (stable_id, document_id, session_id, content, chunk_index,
                         start_char, end_char, metadata)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (stable_id) DO UPDATE SET
                            content=EXCLUDED.content, metadata=EXCLUDED.metadata
                        """,
                        (
                            c["stable_id"],
                            c["document_id"],
                            c["session_id"],
                            c["content"],
                            c["chunk_index"],
                            c.get("start_char", 0),
                            c.get("end_char", 0),
                            meta_str,
                        ),
                    )
            conn.commit()
        return len(chunks)

    def get_parent_chunks(self, stable_ids: List[str]) -> Dict[str, str]:
        if not stable_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(stable_ids))
                cur.execute(
                    f"SELECT stable_id, content FROM parent_chunks WHERE stable_id IN ({placeholders})",
                    stable_ids,
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_parent_chunks_with_metadata(self, stable_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not stable_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(stable_ids))
                cur.execute(
                    f"SELECT stable_id, content, chunk_index, start_char, end_char, metadata "
                    f"FROM parent_chunks WHERE stable_id IN ({placeholders})",
                    stable_ids,
                )
                result = {}
                for row in cur.fetchall():
                    meta = row[5]
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                    elif not isinstance(meta, dict):
                        meta = {}
                    result[row[0]] = {
                        "content": row[1],
                        "chunk_index": row[2],
                        "start_char": row[3] or 0,
                        "end_char": row[4] or 0,
                        "metadata": meta,
                    }
                return result

    def delete_parent_chunks_by_document(self, document_id: str) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM parent_chunks WHERE document_id = %s", (document_id,))
                count = cur.rowcount
            conn.commit()
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag_documents")
                total_docs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT session_id) FROM rag_documents")
                doc_sessions = cur.fetchone()[0]

                cur.execute("SELECT status, COUNT(*) FROM rag_documents GROUP BY status")
                status_counts = {row[0]: row[1] for row in cur.fetchall()}

                cur.execute("SELECT COUNT(*) FROM citations")
                total_citations = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM citations WHERE expires_at < NOW()")
                expired_citations = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT session_id) FROM citations")
                citation_sessions = cur.fetchone()[0]

        return {
            "documents": {
                "total": total_docs,
                "sessions": doc_sessions,
                "by_status": status_counts,
            },
            "citations": {
                "total": total_citations,
                "expired_pending_cleanup": expired_citations,
                "sessions": citation_sessions,
            },
            "schema_version": _store_version(),
            "backend": "postgresql",
            "dsn": self._dsn.split("@")[-1] if "@" in self._dsn else "(local)",
        }

    # ------------------------------------------------------------------
    # Profile operations (config management)
    # ------------------------------------------------------------------

    def save_profile(
        self,
        user_id: str,
        profile_name: str,
        config: Dict[str, Any],
        description: str = "",
        is_builtin: bool = False,
    ) -> bool:
        """Save or update a RAG configuration profile."""
        config_json = json.dumps(config) if isinstance(config, dict) else config
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_profiles
                    (user_id, profile_name, description, config, is_builtin, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, profile_name) DO UPDATE SET
                        description = EXCLUDED.description,
                        config = EXCLUDED.config,
                        updated_at = NOW()
                    """,
                    (user_id, profile_name, description, config_json, is_builtin),
                )
            conn.commit()
        return True

    def get_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all profiles visible to a user (built-in + user's custom)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, profile_name, description, config, is_builtin,
                           created_at, updated_at
                    FROM rag_profiles
                    WHERE user_id = %s OR is_builtin = true
                    ORDER BY is_builtin DESC, profile_name
                    """,
                    (user_id,),
                )
                cols = [d[0] for d in cur.description]
                rows = []
                for row in cur.fetchall():
                    d = dict(zip(cols, row))
                    if isinstance(d.get("config"), str):
                        try:
                            d["config"] = json.loads(d["config"])
                        except (json.JSONDecodeError, TypeError):
                            d["config"] = {}
                    rows.append(d)
                return rows

    def delete_profile(self, user_id: str, profile_name: str) -> bool:
        """Delete a user's custom profile. Cannot delete built-in profiles."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM rag_profiles
                    WHERE user_id = %s AND profile_name = %s AND is_builtin = false
                    """,
                    (user_id, profile_name),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def delete_all_user_profiles(self, user_id: str) -> int:
        """Delete all non-builtin profiles for a user."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_profiles WHERE user_id = %s AND is_builtin = false",
                    (user_id,),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def get_profile(self, user_id: str, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile by user_id and name."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, profile_name, description, config, is_builtin,
                           created_at, updated_at
                    FROM rag_profiles
                    WHERE (user_id = %s OR is_builtin = true) AND profile_name = %s
                    LIMIT 1
                    """,
                    (user_id, profile_name),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                d = dict(zip(cols, row))
                if isinstance(d.get("config"), str):
                    try:
                        d["config"] = json.loads(d["config"])
                    except (json.JSONDecodeError, TypeError):
                        d["config"] = {}
                return d

    # ------------------------------------------------------------------
    # Session operations (config management)
    # ------------------------------------------------------------------

    def save_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Save or update session config state."""
        overrides_json = json.dumps(state.get("session_overrides", {}))
        focused_json = json.dumps(state.get("focused_document_ids", []))
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_sessions
                    (session_id, user_id, active_profile, session_overrides,
                     rag_enabled, focused_document_ids, last_accessed)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        active_profile = EXCLUDED.active_profile,
                        session_overrides = EXCLUDED.session_overrides,
                        rag_enabled = EXCLUDED.rag_enabled,
                        focused_document_ids = EXCLUDED.focused_document_ids,
                        last_accessed = NOW()
                    """,
                    (
                        session_id,
                        state.get("user_id", "__anonymous__"),
                        state.get("active_profile", "default"),
                        overrides_json,
                        state.get("rag_enabled", True),
                        focused_json,
                    ),
                )
            conn.commit()
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a single session's config state."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM rag_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                d = dict(zip(cols, row))
                # Parse JSONB fields
                for field in ("session_overrides", "focused_document_ids"):
                    if isinstance(d.get(field), str):
                        try:
                            d[field] = json.loads(d[field])
                        except (json.JSONDecodeError, TypeError):
                            d[field] = {} if field == "session_overrides" else []
                return d

    def get_active_sessions(self, max_age_seconds: int = 86400) -> List[Dict[str, Any]]:
        """Get all sessions accessed within *max_age_seconds*."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM rag_sessions
                    WHERE last_accessed > NOW() - make_interval(secs => %s)
                    ORDER BY last_accessed DESC
                    """,
                    (max_age_seconds,),
                )
                cols = [d[0] for d in cur.description]
                rows = []
                for row in cur.fetchall():
                    d = dict(zip(cols, row))
                    for field in ("session_overrides", "focused_document_ids"):
                        if isinstance(d.get(field), str):
                            try:
                                d[field] = json.loads(d[field])
                            except (json.JSONDecodeError, TypeError):
                                d[field] = {} if field == "session_overrides" else []
                    rows.append(d)
                return rows

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session by ID."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_sessions WHERE session_id = %s",
                    (session_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def cleanup_expired_sessions(self, max_age_seconds: int = 86400) -> int:
        """Delete sessions older than *max_age_seconds*."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM rag_sessions
                    WHERE last_accessed < NOW() - make_interval(secs => %s)
                    """,
                    (max_age_seconds,),
                )
                count = cur.rowcount
            conn.commit()
        if count > 0:
            logger.info(f"Cleaned up {count} expired config sessions")
        return count

    # ------------------------------------------------------------------
    # Lifecycle / Garbage Collection operations
    # ------------------------------------------------------------------

    def soft_delete_document(self, document_id: str) -> bool:
        """Mark a document as soft-deleted (status=soft_deleted, deleted_at=NOW())."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE rag_documents SET status = 'soft_deleted', deleted_at = NOW() WHERE document_id = %s AND status != 'soft_deleted'",
                    (document_id,),
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def restore_document(self, document_id: str, grace_cutoff: Optional[datetime] = None) -> bool:
        """Restore a soft-deleted document to READY status (clears deleted_at).

        Args:
            grace_cutoff: If provided, only restore if deleted_at >= grace_cutoff
                (atomic grace-window enforcement — eliminates TOCTOU race).
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                if grace_cutoff is not None:
                    cur.execute(
                        "UPDATE rag_documents SET status = 'ready', deleted_at = NULL"
                        " WHERE document_id = %s AND status = 'soft_deleted'"
                        " AND (deleted_at IS NULL OR deleted_at >= %s)",
                        (document_id, grace_cutoff),
                    )
                else:
                    cur.execute(
                        "UPDATE rag_documents SET status = 'ready', deleted_at = NULL"
                        " WHERE document_id = %s AND status = 'soft_deleted'",
                        (document_id,),
                    )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def list_soft_deleted_before(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids soft-deleted before *cutoff* (past grace window).

        Optionally scoped to a single session.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE status = 'soft_deleted' AND deleted_at < %s AND session_id = %s",
                        (cutoff, session_id),
                    )
                else:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE status = 'soft_deleted' AND deleted_at < %s",
                        (cutoff,),
                    )
                return [row[0] for row in cur.fetchall()]

    def list_expired_documents(self, session_id: Optional[str] = None) -> List[str]:
        """Return document_ids whose expires_at has passed (TTL-based expiry).

        Optionally scoped to a single session.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE expires_at IS NOT NULL AND expires_at < NOW() AND status != 'soft_deleted' AND session_id = %s",
                        (session_id,),
                    )
                else:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE expires_at IS NOT NULL AND expires_at < NOW() AND status != 'soft_deleted'",
                    )
                return [row[0] for row in cur.fetchall()]

    def list_error_documents_older_than(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids with status=error whose upload_date < cutoff.

        Optionally scoped to a single session.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE status = 'error' AND upload_date < %s AND session_id = %s",
                        (cutoff, session_id),
                    )
                else:
                    cur.execute(
                        "SELECT document_id FROM rag_documents WHERE status = 'error' AND upload_date < %s",
                        (cutoff,),
                    )
                return [row[0] for row in cur.fetchall()]

    def list_all_active_document_ids(
        self,
        session_id: Optional[str] = None,
        only_ready: bool = False,
    ) -> List[str]:
        """Return all document_ids that are not soft-deleted.

        Optionally scoped to a single session. If only_ready=True, restricts
        to documents with status=ready only.
        """
        status_clause = "status = 'ready'" if only_ready else "status != 'soft_deleted'"
        with self._conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        f"SELECT document_id FROM rag_documents WHERE {status_clause} AND session_id = %s",
                        (session_id,),
                    )
                else:
                    cur.execute(
                        f"SELECT document_id FROM rag_documents WHERE {status_clause}",
                    )
                return [row[0] for row in cur.fetchall()]

    def rebuild_quota_counters(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recompute quota usage by scanning the documents table."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        """
                        SELECT COUNT(*), COALESCE(SUM(file_size), 0), COALESCE(SUM(chunks_count), 0)
                        FROM rag_documents
                        WHERE status NOT IN ('soft_deleted', 'error') AND session_id = %s
                        """,
                        (session_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT COUNT(*), COALESCE(SUM(file_size), 0), COALESCE(SUM(chunks_count), 0)
                        FROM rag_documents
                        WHERE status NOT IN ('soft_deleted', 'error')
                        """,
                    )
                row = cur.fetchone()
                return {
                    "documents": row[0] if row else 0,
                    "storage_bytes": row[1] if row else 0,
                    "chunks": row[2] if row else 0,
                }

    def get_all_session_quota_counters(self) -> Dict[str, Dict[str, int]]:
        """Return per-session document/storage/chunk counts from the documents table.

        Only active (non-deleted, non-error) documents are counted.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id,
                           COUNT(*),
                           COALESCE(SUM(file_size), 0),
                           COALESCE(SUM(chunks_count), 0)
                    FROM rag_documents
                    WHERE status NOT IN ('soft_deleted', 'error')
                      AND session_id IS NOT NULL
                    GROUP BY session_id
                    """
                )
                result: Dict[str, Dict[str, int]] = {}
                for row in cur.fetchall():
                    result[row[0]] = {
                        "documents": row[1],
                        "storage_bytes": row[2],
                        "chunks": row[3],
                    }
                return result

    def acquire_gc_lock(self, owner_id: str, ttl_seconds: int = 3600) -> bool:
        """Try to acquire the global GC advisory lock."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Delete expired lock first
                cur.execute("DELETE FROM gc_lock WHERE lock_name = 'gc' AND expires_at < NOW()")
                # Try to insert (fails on PK conflict if another owner holds it)
                try:
                    cur.execute(
                        """
                        INSERT INTO gc_lock (lock_name, owner_id, acquired_at, expires_at)
                        VALUES ('gc', %s, NOW(), NOW() + make_interval(secs => %s))
                        """,
                        (owner_id, ttl_seconds),
                    )
                    conn.commit()
                    return True
                except Exception:
                    conn.rollback()
                    return False

    def release_gc_lock(self, owner_id: str) -> None:
        """Release the GC advisory lock held by *owner_id*."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM gc_lock WHERE lock_name = 'gc' AND owner_id = %s",
                    (owner_id,),
                )
            conn.commit()

    def log_gc_run(self, run_id: str, phase: str, result: Dict[str, Any]) -> None:
        """Append a GC phase result to gc_audit_log."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                details = result.get("details", {})
                cur.execute(
                    """
                    INSERT INTO gc_audit_log
                    (run_id, phase, started_at, completed_at, docs_scanned, docs_cleaned,
                     bytes_freed, errors, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        phase,
                        result.get("started_at", datetime.now(timezone.utc)),
                        result.get("completed_at"),
                        result.get("docs_scanned", 0),
                        result.get("docs_cleaned", 0),
                        result.get("bytes_freed", 0),
                        result.get("errors", 0),
                        json.dumps(details) if isinstance(details, dict) else details,
                    ),
                )
            conn.commit()

    def get_gc_audit_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent GC audit log entries (newest first)."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT run_id, phase, started_at, completed_at,
                               docs_scanned, docs_cleaned, bytes_freed, errors, details
                        FROM gc_audit_log
                        ORDER BY started_at DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                    cols = [d[0] for d in cur.description]
                    rows = []
                    for row in cur.fetchall():
                        d = dict(zip(cols, row))
                        if d.get("details") and isinstance(d["details"], str):
                            try:
                                d["details"] = json.loads(d["details"])
                            except Exception:
                                pass
                        rows.append(d)
                    return rows
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Quota operations
    # ------------------------------------------------------------------

    def get_user_quota(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get quota tier for a user."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM rag_quotas WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))

    def save_user_quota(self, user_id: str, quota: Dict[str, Any]) -> bool:
        """Save or update quota tier for a user."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_quotas
                    (user_id, tier, max_documents, max_storage_bytes,
                     max_searches_per_minute, max_uploads_per_hour,
                     max_chunks, max_profiles)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        tier = EXCLUDED.tier,
                        max_documents = EXCLUDED.max_documents,
                        max_storage_bytes = EXCLUDED.max_storage_bytes,
                        max_searches_per_minute = EXCLUDED.max_searches_per_minute,
                        max_uploads_per_hour = EXCLUDED.max_uploads_per_hour,
                        max_chunks = EXCLUDED.max_chunks,
                        max_profiles = EXCLUDED.max_profiles
                    """,
                    (
                        user_id,
                        quota.get("tier", "free"),
                        quota.get("max_documents", 1000),
                        quota.get("max_storage_bytes", 5368709120),
                        quota.get("max_searches_per_minute", 600),
                        quota.get("max_uploads_per_hour", 200),
                        quota.get("max_chunks", 100000),
                        quota.get("max_profiles", 50),
                    ),
                )
            conn.commit()
        return True

    # ------------------------------------------------------------------
    # Content generations (mirrors RAGDatabase; created with the schema)
    # ------------------------------------------------------------------

    def get_content_generation(self, session_id: str) -> int:
        """How many times the session's documents have changed; 0 before the first."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT generation FROM content_generations WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0]) if row else 0

    def bump_content_generation(self, session_id: str) -> int:
        """Record a change to the session's documents and return the new generation."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO content_generations (session_id, generation) VALUES (%s, 1) "
                    "ON CONFLICT (session_id) DO UPDATE "
                    "SET generation = content_generations.generation + 1 RETURNING generation",
                    (session_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0])

    # ------------------------------------------------------------------
    # Collection identity (mirrors RAGDatabase; created on first use)
    # ------------------------------------------------------------------

    def _ensure_collection_meta(self, cur) -> None:
        if not getattr(self, "_collection_meta_ready", False):
            cur.execute(_CREATE_COLLECTION_META)
            self._collection_meta_ready = True

    @staticmethod
    def _collection_meta_row(cur) -> Optional[Dict[str, Any]]:
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    def get_collection_meta(
        self, collection_name: str, vector_backend: str = "milvus"
    ) -> Optional[Dict[str, Any]]:
        """The stamp of one collection: the model and size its vectors were made at."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._ensure_collection_meta(cur)
                cur.execute(
                    "SELECT * FROM collection_meta "
                    "WHERE vector_backend = %s AND collection_name = %s",
                    (vector_backend, collection_name),
                )
                found = self._collection_meta_row(cur)
            conn.commit()
        return found

    def stamp_collection(
        self,
        collection_name: str,
        embedding_model: Optional[str],
        embedding_dim: int,
        vector_backend: str = "milvus",
        lasagna_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record which model a collection holds; an existing stamp is kept."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._ensure_collection_meta(cur)
                cur.execute(
                    "INSERT INTO collection_meta "
                    "(vector_backend, collection_name, embedding_model, embedding_dim, "
                    "lasagna_version, created_at) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (vector_backend, collection_name) DO NOTHING",
                    (
                        vector_backend,
                        collection_name,
                        embedding_model,
                        int(embedding_dim),
                        lasagna_version,
                        datetime.now(timezone.utc),
                    ),
                )
                cur.execute(
                    "SELECT * FROM collection_meta "
                    "WHERE vector_backend = %s AND collection_name = %s",
                    (vector_backend, collection_name),
                )
                found = self._collection_meta_row(cur)
            conn.commit()
        return found or {}

    def forget_collection_meta(self, collection_name: str, vector_backend: str = "milvus") -> bool:
        """Remove a collection's stamp, after the collection itself was dropped."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._ensure_collection_meta(cur)
                cur.execute(
                    "DELETE FROM collection_meta "
                    "WHERE vector_backend = %s AND collection_name = %s",
                    (vector_backend, collection_name),
                )
                removed = cur.rowcount > 0
            conn.commit()
        return removed

    def list_collection_meta(self) -> List[Dict[str, Any]]:
        """Every stamped collection, oldest first."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._ensure_collection_meta(cur)
                cur.execute("SELECT * FROM collection_meta ORDER BY created_at, collection_name")
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
        return rows
