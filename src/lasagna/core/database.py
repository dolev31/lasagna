"""Unified RAG SQLite Database Manager.

This module provides a single SQLite database for the RAG system,
containing document metadata and citation tracking in one file (rag.db).

Benefits:
- Single database file simplifies deployment and backup
- Enables foreign key relationships (citations -> documents)
- Atomic transactions across document and citation operations
- Simpler connection pooling and resource management

The database is created automatically on first access and handles:
- Table creation with proper schema
- Index creation for performance
- Thread-safe access via connection pooling
"""

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

SCHEMA_VERSION = 6

#: The metadata store's version, shared by the SQLite and PostgreSQL stores and
#: recorded in their ``schema_meta`` table. A store recorded at a higher version
#: was written by newer code, whose columns and meanings this code does not know.
STORE_VERSION = SCHEMA_VERSION

#: The ``rag_documents`` columns ``update_document_fields`` may set: what a new
#: copy of a document changes. Never the id, the session or lifecycle state.
DOCUMENT_FIELDS = frozenset(
    {
        "filename",
        "file_path",
        "file_size",
        "file_type",
        "status",
        "chunks_count",
        "upload_date",
        "error_message",
        "processing_time",
        "text_length",
        "preview_text",
        "vector_backend_name",
        "storage_backend_name",
        "collection_name",
    }
)


class DocumentIdTaken(RuntimeError):
    """A new document's id is already in the store; REST answers 409 (or 404 for another session's)."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(
            f"document_id {document_id!r} is already taken; choose another id, or upload "
            "again to replace your document under it."
        )


class StoreVersionError(RuntimeError):
    """The store was written by a newer Lasagna than the one opening it."""


def refuse_newer_store(recorded: int, where: str) -> None:
    """Raise when the store at ``where`` is newer than this code.

    Opening it anyway used to work silently: migrations only ran upward, so an
    older install read and wrote rows whose columns it did not know, after a
    downgrade or with two versions sharing one database.
    """
    if recorded > STORE_VERSION:
        raise StoreVersionError(
            f"The metadata store at {where} is at store version {recorded}, written by a "
            f"newer Lasagna; this version reads store versions up to {STORE_VERSION}. "
            f"Upgrade (pip install -U lasagna-rag), or point RAG_DB_DIR or "
            f"RAG_DATABASE_DSN at a different store."
        )


# Document metadata table
CREATE_DOCUMENTS_TABLE = """
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
"""

# GC audit log — records each phase of each GC run for observability
CREATE_GC_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS gc_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    docs_scanned INTEGER DEFAULT 0,
    docs_cleaned INTEGER DEFAULT 0,
    bytes_freed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    details TEXT DEFAULT '{}'
);
"""

# Distributed advisory lock — prevents duplicate GC runs across processes
CREATE_GC_LOCK_TABLE = """
CREATE TABLE IF NOT EXISTS gc_lock (
    lock_name TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);
"""

# Citations table
CREATE_CITATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    score REAL,
    normalized_score REAL,
    hop_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
);
"""

# Parent chunks table for hierarchical chunking (auto-merge retrieval)
# Parent chunks are context containers stored in SQLite, NOT in Milvus,
# because they don't need embeddings — they are fetched by stable_id
# when auto-merge determines enough child chunks were retrieved.
CREATE_PARENT_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    stable_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
);
"""

# Schema metadata table for version tracking
CREATE_SCHEMA_META_TABLE = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# Session config profiles table (mirrors PostgresRAGDatabase.rag_profiles)
CREATE_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS rag_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '__global__',
    profile_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    config TEXT NOT NULL DEFAULT '{}',
    is_builtin INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, profile_name)
);
"""

# Session config state table (mirrors PostgresRAGDatabase.rag_sessions)
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS rag_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '__anonymous__',
    active_profile TEXT NOT NULL DEFAULT 'default',
    session_overrides TEXT DEFAULT '{}',
    rag_enabled INTEGER DEFAULT 1,
    focused_document_ids TEXT DEFAULT '[]',
    last_accessed REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# All indexes
CREATE_INDEXES = [
    # Document indexes
    "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON rag_documents(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON rag_documents(status);",
    # Lifecycle indexes (partial indexes on non-null columns for efficiency)
    "CREATE INDEX IF NOT EXISTS idx_documents_deleted_at ON rag_documents(deleted_at) WHERE deleted_at IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_documents_expires_at ON rag_documents(expires_at) WHERE expires_at IS NOT NULL;",
    # Citation indexes
    "CREATE INDEX IF NOT EXISTS idx_citations_session ON citations(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_query ON citations(query_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_created ON citations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_citations_expires ON citations(expires_at);",
    "CREATE INDEX IF NOT EXISTS idx_citations_content_hash ON citations(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_citations_document ON citations(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_citations_session_created ON citations(session_id, created_at DESC, id DESC);",
    # Parent chunk indexes
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_document ON parent_chunks(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_session ON parent_chunks(session_id);",
    # Backend tracking indexes
    "CREATE INDEX IF NOT EXISTS idx_documents_backend ON rag_documents(vector_backend_name);",
    "CREATE INDEX IF NOT EXISTS idx_documents_storage ON rag_documents(storage_backend_name);",
    # Profile and session indexes
    "CREATE INDEX IF NOT EXISTS idx_profiles_user ON rag_profiles(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON rag_sessions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_accessed ON rag_sessions(last_accessed);",
    # GC audit log indexes
    "CREATE INDEX IF NOT EXISTS idx_gc_audit_run_id ON gc_audit_log(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_gc_audit_started ON gc_audit_log(started_at);",
]


# =============================================================================
# RAG DATABASE MANAGER
# =============================================================================


class RAGDatabase:
    """Unified SQLite database manager for RAG system.

    Manages a single rag.db file containing:
    - rag_documents: Document metadata and processing status
    - citations: Citation tracking with TTL-based lifecycle
    - schema_meta: Schema versioning for migrations

    Features:
    - Thread-safe via RLock
    - Automatic table and index creation
    - Schema versioning
    - WAL mode for better concurrent performance
    - Foreign key enforcement

    Usage:
        # Get singleton instance
        db = get_rag_database()

        # Or create with custom path
        db = RAGDatabase("/path/to/rag.db")

        # Use connection context manager
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_documents")
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the RAG database.

        Args:
            db_path: Path to the SQLite database file. If None, uses
                     the path from rag_config.
        """
        if db_path is None:
            from .config import rag_config

            # Use the configured db_dir with rag.db name
            db_dir = Path(rag_config._config.get("db_path", "")).parent
            self.db_path = str(db_dir / "rag.db")
        else:
            self.db_path = db_path

        self._lock = threading.RLock()
        self._initialized = False

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize()

    def _initialize(self):
        """Initialize database tables and indexes."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.info(f"Initializing RAG database: {self.db_path}")

            # One process at a time, as on PostgreSQL: two workers starting on
            # a new file at once failed the second one's switch to WAL with
            # "database is locked".
            with self.exclusive("schema"), self._get_connection() as conn:
                cursor = conn.cursor()

                # Enable WAL mode for better concurrent performance
                cursor.execute("PRAGMA journal_mode=WAL;")

                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys=ON;")

                # Create schema meta table first
                cursor.execute(CREATE_SCHEMA_META_TABLE)

                # Check current schema version
                cursor.execute("SELECT value FROM schema_meta WHERE key = 'version'")
                row = cursor.fetchone()
                current_version = int(row[0]) if row else 0
                refuse_newer_store(current_version, self.db_path)

                # Create tables
                cursor.execute(CREATE_DOCUMENTS_TABLE)
                cursor.execute(CREATE_CITATIONS_TABLE)
                cursor.execute(CREATE_PARENT_CHUNKS_TABLE)
                cursor.execute(CREATE_PROFILES_TABLE)
                cursor.execute(CREATE_SESSIONS_TABLE)
                cursor.execute(CREATE_GC_AUDIT_LOG_TABLE)
                cursor.execute(CREATE_GC_LOCK_TABLE)

                # Run migrations if needed (before indexes, since migrations may add columns)
                if current_version < SCHEMA_VERSION:
                    self._run_migrations(cursor, current_version)
                    cursor.execute(
                        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                        ("version", str(SCHEMA_VERSION)),
                    )

                # Create indexes (after migrations so all columns exist)
                for index_sql in CREATE_INDEXES:
                    cursor.execute(index_sql)

                conn.commit()

            self._initialized = True
            logger.info(f"RAG database initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings.

        Returns:
            SQLite connection with row_factory and timeout configured.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys for this connection
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def connection(self):
        """Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection

        Example:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM rag_documents")
        """
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions.

        Automatically commits on success, rolls back on exception.
        Uses lock for thread safety.

        Yields:
            sqlite3.Connection: Database connection in transaction

        Example:
            with db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
                cursor.execute("UPDATE ...")
                # Auto-commits if no exception
        """
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @contextmanager
    def exclusive(self, name: str):
        """Hold the lock ``name`` against every thread and process using this database.

        A file lock in ``<database>.locks/`` (see `lasagna.core.store_locks`):
        the processes that share a SQLite file are on one host. Replacing,
        retrying and deleting a document, and the duplicate check of an upload,
        take it.
        """
        from .store_locks import file_lock

        with file_lock(f"{self.db_path}.locks", name):
            yield

    def _run_migrations(self, cursor: sqlite3.Cursor, from_version: int):
        """Run schema migrations.

        Args:
            cursor: Database cursor
            from_version: Current schema version in database
        """
        logger.info(f"Running migrations from version {from_version} to {SCHEMA_VERSION}")

        # Migration v0 -> v1: Initial schema (handled by CREATE statements)

        # Migration v1 -> v2: Add parent_chunks table for hierarchical chunking
        if from_version < 2:
            cursor.execute(CREATE_PARENT_CHUNKS_TABLE)
            # Indexes handled by CREATE_INDEXES after migrations
            logger.info("Migration v1->v2: Created parent_chunks table")

        # Migration v2 -> v3: Add vector_backend_name to track which backend holds each doc's vectors
        if from_version < 3:
            try:
                cursor.execute(
                    "ALTER TABLE rag_documents ADD COLUMN vector_backend_name TEXT DEFAULT 'milvus';"
                )
            except sqlite3.OperationalError as e:
                err_msg = str(e).lower()
                if "already exists" not in err_msg and "duplicate column" not in err_msg:
                    raise
                # Column already exists (e.g., fresh install with v3 CREATE TABLE)
            # Index handled by CREATE_INDEXES after migrations
            logger.info("Migration v2->v3: Added vector_backend_name column")

        # Migration v3 -> v4: Add storage_backend_name to track which storage holds each doc's files
        if from_version < 4:
            try:
                cursor.execute(
                    "ALTER TABLE rag_documents ADD COLUMN storage_backend_name TEXT DEFAULT 'local';"
                )
            except sqlite3.OperationalError as e:
                err_msg = str(e).lower()
                if "already exists" not in err_msg and "duplicate column" not in err_msg:
                    raise
                # Column already exists (e.g., fresh install with v4 CREATE TABLE)
            # Index handled by CREATE_INDEXES after migrations
            logger.info("Migration v3->v4: Added storage_backend_name column")

        # Migration v4 -> v5: Add rag_profiles and rag_sessions tables
        # for session config persistence (replacing JSON files)
        if from_version < 5:
            cursor.execute(CREATE_PROFILES_TABLE)
            cursor.execute(CREATE_SESSIONS_TABLE)
            logger.info("Migration v4->v5: Created rag_profiles and rag_sessions tables")

        # Migration v5 -> v6: Add lifecycle columns (soft-delete, TTL, collection tracking)
        # and GC infrastructure tables
        if from_version < 6:
            lifecycle_cols = [
                (
                    "deleted_at",
                    "ALTER TABLE rag_documents ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL;",
                ),
                (
                    "expires_at",
                    "ALTER TABLE rag_documents ADD COLUMN expires_at TIMESTAMP DEFAULT NULL;",
                ),
                (
                    "collection_name",
                    "ALTER TABLE rag_documents ADD COLUMN collection_name TEXT DEFAULT NULL;",
                ),
            ]
            for col_name, sql in lifecycle_cols:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError as e:
                    err_msg = str(e).lower()
                    if "already exists" not in err_msg and "duplicate column" not in err_msg:
                        raise
            cursor.execute(CREATE_GC_AUDIT_LOG_TABLE)
            cursor.execute(CREATE_GC_LOCK_TABLE)
            logger.info("Migration v5->v6: Added lifecycle columns and GC tables")

    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================

    def save_document(
        self,
        document: Dict[str, Any],
        file_path: str,
        collection_name: Optional[str] = None,
        *,
        replace: bool = True,
    ) -> bool:
        """Save document metadata.

        Args:
            document: Document metadata dictionary
            file_path: Path to the stored file
            collection_name: Vector collection this document was written to (for orphan detection)
            replace: Overwrite a row with the same id (the evaluation harness
                registering its corpus again). False for a new upload: an id
                taken meanwhile raises `DocumentIdTaken` instead of replacing
                that row and its ``session_id``.

        Returns:
            True if saved successfully
        """
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        with self.transaction() as conn:
            cursor = conn.cursor()
            try:
                self._insert_document(cursor, verb, document, file_path, collection_name)
            except sqlite3.IntegrityError as exc:
                if replace or "UNIQUE" not in str(exc).upper():
                    raise
                raise DocumentIdTaken(str(document.get("document_id"))) from None
        return True

    @staticmethod
    def _insert_document(cursor, verb, document, file_path, collection_name) -> None:
        cursor.execute(
            f"""
                {verb} INTO rag_documents
                (document_id, session_id, filename, file_path, file_size,
                 file_type, status, chunks_count, upload_date, error_message,
                 processing_time, text_length, preview_text, vector_backend_name,
                 storage_backend_name, collection_name, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        return True

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document metadata dict or None if not found
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rag_documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_documents(
        self,
        session_id: str,
        status: Optional[str] = None,
        include_soft_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get all documents for a session.

        Args:
            session_id: Session identifier
            status: Optional status filter
            include_soft_deleted: If False (default), excludes soft-deleted documents

        Returns:
            List of document metadata dicts
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            conditions = ["session_id = ?"]
            params: List[Any] = [session_id]

            if status:
                conditions.append("status = ?")
                params.append(status)
            elif not include_soft_deleted:
                conditions.append("status != 'soft_deleted'")

            where = " AND ".join(conditions)
            cursor.execute(
                f"SELECT * FROM rag_documents WHERE {where} ORDER BY upload_date DESC",
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    # Backward-compatible alias
    get_session_documents = get_documents

    def get_documents_batch(
        self,
        document_ids: List[str],
        include_soft_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get multiple documents by their IDs in a single query.

        Args:
            document_ids: List of document IDs to fetch
            include_soft_deleted: If False (default), excludes soft-deleted documents
        """
        if not document_ids:
            return []
        with self.connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(document_ids))
            soft_delete_filter = "" if include_soft_deleted else " AND status != 'soft_deleted'"
            cursor.execute(
                f"SELECT * FROM rag_documents WHERE document_id IN ({placeholders}){soft_delete_filter}",
                document_ids,
            )
            return [dict(row) for row in cursor.fetchall()]

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
        """Update document status and related fields.

        Args:
            document_id: Document identifier
            status: New status value
            error_message: Error message if status is 'error'
            chunks_count: Number of chunks (if processing complete)
            processing_time: Processing duration in seconds
            text_length: Total text length in characters
            preview_text: Preview text for document awareness

        Returns:
            True if updated, False if document not found
        """
        updates = ["status = ?"]
        params = [status]

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if chunks_count is not None:
            updates.append("chunks_count = ?")
            params.append(chunks_count)
        if processing_time is not None:
            updates.append("processing_time = ?")
            params.append(processing_time)
        if text_length is not None:
            updates.append("text_length = ?")
            params.append(text_length)
        if preview_text is not None:
            updates.append("preview_text = ?")
            params.append(preview_text)

        params.append(document_id)

        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE rag_documents SET {', '.join(updates)} WHERE document_id = ?", params
            )
            return cursor.rowcount > 0

    def update_document_fields(self, document_id: str, fields: Dict[str, Any]) -> bool:
        """Set the given columns of one document row, None included.

        For a document whose new copy replaced the old one: the file, the
        collection and the counts change in one statement, and a stale
        ``error_message`` can be cleared, which ``update_document_status`` never
        does. Only the columns in ``DOCUMENT_FIELDS`` may be set.

        Returns:
            True if updated, False if document not found
        """
        unknown = set(fields) - DOCUMENT_FIELDS
        if unknown:
            raise ValueError(f"Not document columns: {sorted(unknown)}")
        if not fields:
            return self.get_document(document_id) is not None
        columns = sorted(fields)
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE rag_documents SET {', '.join(f'{c} = ?' for c in columns)} "
                "WHERE document_id = ?",
                [fields[c] for c in columns] + [document_id],
            )
            return cursor.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its citations (via cascade).

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rag_documents WHERE document_id = ?", (document_id,))
            return cursor.rowcount > 0

    # =========================================================================
    # CITATION OPERATIONS
    # =========================================================================

    def store_citations(
        self,
        citations: List[Dict[str, Any]],
        session_id: str,
        query_id: Optional[str] = None,
        ttl_hours: int = 168,
    ) -> int:
        """Store citations in database.

        Args:
            citations: List of citation dictionaries
            session_id: Session identifier
            query_id: Optional query identifier
            ttl_hours: Time-to-live in hours (default 168 = 1 week)

        Returns:
            Number of citations stored
        """
        if not citations:
            return 0

        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        stored_count = 0

        with self.transaction() as conn:
            cursor = conn.cursor()
            for citation in citations:
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO citations
                        (citation_id, session_id, query_id, chunk_id, document_id,
                         document_name, content_preview, content_hash, chunk_index,
                         page_number, section_header, score, normalized_score,
                         hop_index, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            citation.get("citation_id"),
                            session_id,
                            query_id or citation.get("query_id"),
                            citation.get("chunk_id"),
                            citation.get("document_id"),
                            citation.get("document_name"),
                            citation.get("content_preview"),
                            citation.get("content_hash"),
                            citation.get("chunk_index"),
                            citation.get("page_number"),
                            citation.get("section_header"),
                            citation.get("score"),
                            citation.get("normalized_score"),
                            citation.get("hop_index"),
                            expires_at.isoformat(),
                        ),
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Failed to store citation: {e}")

            # Enforce per-session limit within the same transaction
            # so inserts + eviction are atomic (commit or rollback together).
            if stored_count > 0:
                try:
                    deleted = self._enforce_citation_limit_on_conn(
                        conn,
                        session_id,
                        max_citations=5000,
                    )
                    if deleted > 0:
                        logger.debug(f"Evicted {deleted} old citations for session {session_id}")
                except Exception as e:
                    logger.warning(f"Citation limit enforcement failed: {e}")

        return stored_count

    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Get citation by ID.

        Args:
            citation_id: Citation identifier

        Returns:
            Citation dict or None if not found or expired
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM citations
                WHERE citation_id = ?
                AND (expires_at IS NULL OR expires_at > datetime('now'))
                """,
                (citation_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_citations(
        self,
        session_id: str,
        query_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get citations for a session.

        Args:
            session_id: Session identifier
            query_id: Optional query filter
            limit: Maximum results

        Returns:
            List of citation dicts
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            if query_id:
                cursor.execute(
                    """
                    SELECT * FROM citations
                    WHERE session_id = ? AND query_id = ?
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, query_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM citations
                    WHERE session_id = ?
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )

            return [dict(row) for row in cursor.fetchall()]

    def clear_session_citations(self, session_id: str) -> int:
        """Clear all citations for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of citations deleted
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM citations WHERE session_id = ?", (session_id,))
            return cursor.rowcount

    def cleanup_expired_citations(self) -> int:
        """Remove expired citations.

        Returns:
            Number of citations deleted
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM citations WHERE expires_at < datetime('now')")
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired citations")
        return deleted

    def cleanup_citations_by_age(self, max_age_hours: int) -> int:
        """Delete citations older than *max_age_hours* (safety-net cleanup)."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM citations WHERE created_at < datetime('now', ? || ' hours')",
                (f"-{max_age_hours}",),
            )
            return cursor.rowcount

    def get_citation_by_content_hash(
        self, content_hash: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a citation by content hash within a session."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM citations
                WHERE content_hash = ? AND session_id = ?
                AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (content_hash, session_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_citation_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate citation statistics (avg score, missing page count)."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(normalized_score) FROM citations WHERE normalized_score IS NOT NULL"
            )
            row = cursor.fetchone()
            avg_score = row[0] if row and row[0] else 0.0

            cursor.execute("SELECT COUNT(*) FROM citations WHERE page_number IS NULL")
            missing_page = cursor.fetchone()[0]
        return {"avg_score": float(avg_score), "missing_page": missing_page}

    def _enforce_citation_limit_on_conn(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        max_citations: int = 1000,
    ) -> int:
        """Enforce per-session citation limit using an existing connection.

        Intended for use within an already-open transaction so the caller
        controls commit/rollback semantics.

        Args:
            conn: Active SQLite connection (caller manages the transaction).
            session_id: Session identifier.
            max_citations: Maximum citations to keep.

        Returns:
            Number of citations deleted.
        """
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM citations
            WHERE session_id = ?
            AND id NOT IN (
                SELECT id FROM citations
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (session_id, session_id, max_citations),
        )
        return cursor.rowcount

    def enforce_citation_limit(self, session_id: str, max_citations: int = 1000) -> int:
        """Enforce per-session citation limit.

        Keeps only the most recent citations up to the limit.

        Args:
            session_id: Session identifier
            max_citations: Maximum citations to keep

        Returns:
            Number of citations deleted
        """
        with self.transaction() as conn:
            return self._enforce_citation_limit_on_conn(conn, session_id, max_citations)

    # =========================================================================
    # PARENT CHUNK OPERATIONS (Hierarchical Chunking)
    # =========================================================================

    def save_parent_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Batch-insert parent chunks for hierarchical chunking.

        Parent chunks are context containers used by the auto-merge algorithm.
        When enough child chunks from the same parent are retrieved, the parent
        content replaces them to provide more coherent context to the LLM.

        Args:
            chunks: List of dicts with keys: stable_id, document_id, session_id,
                    content, chunk_index, start_char, end_char, metadata

        Returns:
            Number of chunks inserted.
        """
        if not chunks:
            return 0

        import json

        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO parent_chunks
                (stable_id, document_id, session_id, content, chunk_index,
                 start_char, end_char, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["stable_id"],
                        c["document_id"],
                        c["session_id"],
                        c["content"],
                        c["chunk_index"],
                        c.get("start_char", 0),
                        c.get("end_char", 0),
                        json.dumps(c.get("metadata", {})),
                    )
                    for c in chunks
                ],
            )
            return len(chunks)

    def get_parent_chunks(self, stable_ids: List[str]) -> Dict[str, str]:
        """Fetch parent chunk content by stable_ids.

        Used by the auto-merge algorithm to replace child chunks with
        their parent when enough siblings are retrieved.

        Args:
            stable_ids: List of parent stable_id values.

        Returns:
            Dict mapping stable_id -> content. Missing IDs are omitted.
        """
        if not stable_ids:
            return {}

        with self.connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(stable_ids))
            cursor.execute(
                f"SELECT stable_id, content FROM parent_chunks WHERE stable_id IN ({placeholders})",
                stable_ids,
            )
            return {row["stable_id"]: row["content"] for row in cursor.fetchall()}

    def get_parent_chunks_with_metadata(self, stable_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch parent chunk content AND positional metadata by stable_ids.

        Used by the auto-merge algorithm to produce accurate merged metadata
        for citations (line ranges, character offsets, chunk index).

        Args:
            stable_ids: List of parent stable_id values.

        Returns:
            Dict mapping stable_id -> {content, chunk_index, start_char, end_char, metadata}.
            Missing IDs are omitted.
        """
        if not stable_ids:
            return {}

        import json

        with self.connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(stable_ids))
            cursor.execute(
                f"SELECT stable_id, content, chunk_index, start_char, end_char, metadata "
                f"FROM parent_chunks WHERE stable_id IN ({placeholders})",
                stable_ids,
            )
            result = {}
            for row in cursor.fetchall():
                meta_str = row["metadata"] or "{}"
                try:
                    meta = json.loads(meta_str)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                result[row["stable_id"]] = {
                    "content": row["content"],
                    "chunk_index": row["chunk_index"],
                    "start_char": row["start_char"] or 0,
                    "end_char": row["end_char"] or 0,
                    "metadata": meta,
                }
            return result

    def delete_parent_chunks_by_document(self, document_id: str) -> int:
        """Delete all parent chunks for a document.

        Called during document deletion to clean up hierarchical data.

        Args:
            document_id: Document identifier

        Returns:
            Number of parent chunks deleted.
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM parent_chunks WHERE document_id = ?", (document_id,))
            return cursor.rowcount

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with document and citation counts
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            # Document stats
            cursor.execute("SELECT COUNT(*) as total FROM rag_documents")
            total_docs = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(DISTINCT session_id) as sessions FROM rag_documents")
            doc_sessions = cursor.fetchone()["sessions"]

            cursor.execute("SELECT status, COUNT(*) as count FROM rag_documents GROUP BY status")
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Citation stats
            cursor.execute("SELECT COUNT(*) as total FROM citations")
            total_citations = cursor.fetchone()["total"]

            cursor.execute(
                "SELECT COUNT(*) as expired FROM citations WHERE expires_at < datetime('now')"
            )
            expired_citations = cursor.fetchone()["expired"]

            cursor.execute("SELECT COUNT(DISTINCT session_id) as sessions FROM citations")
            citation_sessions = cursor.fetchone()["sessions"]

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
                "db_path": self.db_path,
                "schema_version": SCHEMA_VERSION,
            }

    # =========================================================================
    # PROFILE OPERATIONS (Session Config)
    # =========================================================================

    def save_profile(
        self,
        user_id: str,
        profile_name: str,
        config: Dict[str, Any],
        description: str = "",
        is_builtin: bool = False,
    ) -> bool:
        """Save or update a RAG configuration profile."""
        import json as _json

        config_str = _json.dumps(config) if isinstance(config, dict) else config
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rag_profiles
                (user_id, profile_name, description, config, is_builtin, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, profile_name) DO UPDATE SET
                    description = excluded.description,
                    config = excluded.config,
                    is_builtin = excluded.is_builtin,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, profile_name, description, config_str, int(is_builtin)),
            )
        return True

    def get_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all profiles visible to a user (built-in + user's custom)."""
        import json as _json

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, profile_name, description, config, is_builtin,
                       created_at, updated_at
                FROM rag_profiles
                WHERE user_id = ? OR is_builtin = 1
                ORDER BY is_builtin DESC, profile_name
                """,
                (user_id,),
            )
            rows = []
            for row in cursor.fetchall():
                d = dict(row)
                d["is_builtin"] = bool(d.get("is_builtin", 0))
                if isinstance(d.get("config"), str):
                    try:
                        d["config"] = _json.loads(d["config"])
                    except (ValueError, TypeError):
                        d["config"] = {}
                rows.append(d)
            return rows

    def delete_profile(self, user_id: str, profile_name: str) -> bool:
        """Delete a user's custom profile. Cannot delete built-in profiles."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM rag_profiles WHERE user_id = ? AND profile_name = ? AND is_builtin = 0",
                (user_id, profile_name),
            )
            return cursor.rowcount > 0

    def delete_all_user_profiles(self, user_id: str) -> int:
        """Delete all non-builtin profiles for a user.

        Used by _persist_global_profiles() to do a snapshot-style persist:
        delete all then re-insert current state.
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM rag_profiles WHERE user_id = ? AND is_builtin = 0",
                (user_id,),
            )
            return cursor.rowcount

    # =========================================================================
    # SESSION OPERATIONS (Session Config)
    # =========================================================================

    def save_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Save or update session config state."""
        import json as _json

        overrides_str = _json.dumps(state.get("session_overrides", {}))
        focused_str = _json.dumps(state.get("focused_document_ids", []))
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rag_sessions
                (session_id, user_id, active_profile, session_overrides,
                 rag_enabled, focused_document_ids, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    active_profile = excluded.active_profile,
                    session_overrides = excluded.session_overrides,
                    rag_enabled = excluded.rag_enabled,
                    focused_document_ids = excluded.focused_document_ids,
                    last_accessed = excluded.last_accessed
                """,
                (
                    session_id,
                    state.get("user_id", "__anonymous__"),
                    state.get("active_profile", "default"),
                    overrides_str,
                    int(state.get("rag_enabled", True)),
                    focused_str,
                    state.get("last_accessed", time.time()),
                ),
            )
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a single session's config state."""
        import json as _json

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM rag_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            d = dict(row)
            d["rag_enabled"] = bool(d.get("rag_enabled", 1))
            for field in ("session_overrides", "focused_document_ids"):
                if isinstance(d.get(field), str):
                    try:
                        d[field] = _json.loads(d[field])
                    except (ValueError, TypeError):
                        d[field] = {} if field == "session_overrides" else []
            return d

    def get_active_sessions(self, max_age_seconds: int = 86400) -> List[Dict[str, Any]]:
        """Get all sessions accessed within max_age_seconds."""
        import json as _json

        cutoff = time.time() - max_age_seconds
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM rag_sessions WHERE last_accessed > ? ORDER BY last_accessed DESC",
                (cutoff,),
            )
            rows = []
            for row in cursor.fetchall():
                d = dict(row)
                d["rag_enabled"] = bool(d.get("rag_enabled", 1))
                for field in ("session_overrides", "focused_document_ids"):
                    if isinstance(d.get(field), str):
                        try:
                            d[field] = _json.loads(d[field])
                        except (ValueError, TypeError):
                            d[field] = {} if field == "session_overrides" else []
                rows.append(d)
            return rows

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session by ID."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM rag_sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

    def cleanup_expired_sessions(self, max_age_seconds: int = 86400) -> int:
        """Delete sessions older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM rag_sessions WHERE last_accessed < ?",
                (cutoff,),
            )
            count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired config sessions")
        return count

    # =========================================================================
    # LIFECYCLE / GARBAGE COLLECTION OPERATIONS
    # =========================================================================

    def soft_delete_document(self, document_id: str) -> bool:
        """Mark a document as soft-deleted (sets status=soft_deleted, deleted_at=now).

        Does NOT remove vectors, files, or free quota.  The GC will hard-delete
        once the grace window expires. Bumps the session's content generation
        (``record_visibility_change``).

        Returns:
            True if the document was found and updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE rag_documents SET status = 'soft_deleted', deleted_at = ? WHERE document_id = ? AND status != 'soft_deleted'",
                (now, document_id),
            )
            session_id = self._session_of(cursor, document_id) if cursor.rowcount > 0 else None
        return record_visibility_change(self, session_id)

    def restore_document(self, document_id: str, grace_cutoff: Optional[datetime] = None) -> bool:
        """Restore a soft-deleted document to READY status (clears deleted_at).

        Args:
            document_id: Document to restore.
            grace_cutoff: If provided, only restore if deleted_at >= grace_cutoff
                (i.e. still within the grace window).  This makes the grace-window
                check atomic with the UPDATE — eliminates the TOCTOU race between
                the Python-level check and the DB write.

        Returns:
            True if the document was found and restored.
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            if grace_cutoff is not None:
                cursor.execute(
                    "UPDATE rag_documents SET status = 'ready', deleted_at = NULL"
                    " WHERE document_id = ? AND status = 'soft_deleted'"
                    " AND (deleted_at IS NULL OR deleted_at >= ?)",
                    (document_id, grace_cutoff.isoformat()),
                )
            else:
                cursor.execute(
                    "UPDATE rag_documents SET status = 'ready', deleted_at = NULL"
                    " WHERE document_id = ? AND status = 'soft_deleted'",
                    (document_id,),
                )
            session_id = self._session_of(cursor, document_id) if cursor.rowcount > 0 else None
        return record_visibility_change(self, session_id)

    @staticmethod
    def _session_of(cursor, document_id: str) -> Optional[str]:
        cursor.execute("SELECT session_id FROM rag_documents WHERE document_id = ?", (document_id,))
        row = cursor.fetchone()
        return row[0] if row else ""

    def list_soft_deleted_before(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids soft-deleted before *cutoff* (past grace window).

        Used by GC Phase 1 to find hard-delete candidates.
        Optionally scoped to a single session.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE status = 'soft_deleted' AND deleted_at < ? AND session_id = ?",
                    (cutoff.isoformat(), session_id),
                )
            else:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE status = 'soft_deleted' AND deleted_at < ?",
                    (cutoff.isoformat(),),
                )
            return [row[0] for row in cursor.fetchall()]

    def list_expired_documents(self, session_id: Optional[str] = None) -> List[str]:
        """Return document_ids whose expires_at has passed (TTL-based expiry).

        Used by GC Phase 2.
        Optionally scoped to a single session.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE expires_at IS NOT NULL AND expires_at < ? AND status != 'soft_deleted' AND session_id = ?",
                    (now, session_id),
                )
            else:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE expires_at IS NOT NULL AND expires_at < ? AND status != 'soft_deleted'",
                    (now,),
                )
            return [row[0] for row in cursor.fetchall()]

    def list_error_documents_older_than(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids with status=error whose upload_date < cutoff.

        Used by GC Phase 3 (error retention cleanup).
        Optionally scoped to a single session.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE status = 'error' AND upload_date < ? AND session_id = ?",
                    (cutoff.isoformat(), session_id),
                )
            else:
                cursor.execute(
                    "SELECT document_id FROM rag_documents WHERE status = 'error' AND upload_date < ?",
                    (cutoff.isoformat(),),
                )
            return [row[0] for row in cursor.fetchall()]

    def fail_stale_processing_documents(
        self, cutoff: datetime, reason: str, session_id: Optional[str] = None
    ) -> List[str]:
        """Mark documents still uploading, processing or indexing since before *cutoff* as error.

        Each update re-checks the status, so a document that finished between
        the read and the write keeps its status. Used by GC Phase 0.
        """
        stale = f"status IN ({_UNFINISHED_SQL}) AND upload_date < ?"
        params: List[Any] = [cutoff.isoformat()]
        if session_id:
            stale += " AND session_id = ?"
            params.append(session_id)
        failed: List[str] = []
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT document_id FROM rag_documents WHERE {stale}", params)
            for (document_id,) in cursor.fetchall():
                cursor.execute(
                    f"UPDATE rag_documents SET status = 'error', error_message = ? "
                    f"WHERE document_id = ? AND {stale}",
                    [reason, document_id, *params],
                )
                if cursor.rowcount > 0:
                    failed.append(document_id)
        return failed

    def list_all_active_document_ids(
        self,
        session_id: Optional[str] = None,
        only_ready: bool = False,
    ) -> List[str]:
        """Return all document_ids that are not soft-deleted.

        Used by OrphanDetector to compare against the vector backend.
        Optionally scoped to a single session. If only_ready=True, restricts
        to documents with status=ready only.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            status_clause = "status = 'ready'" if only_ready else "status != 'soft_deleted'"
            if session_id:
                cursor.execute(
                    f"SELECT document_id FROM rag_documents WHERE {status_clause} AND session_id = ?",
                    (session_id,),
                )
            else:
                cursor.execute(
                    f"SELECT document_id FROM rag_documents WHERE {status_clause}",
                )
            return [row[0] for row in cursor.fetchall()]

    def rebuild_quota_counters(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recompute quota usage by scanning the documents table.

        Returns a dict with keys: documents, storage_bytes, chunks.
        Used by GC Phase 6 to correct counter drift after process restarts.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(file_size), 0), COALESCE(SUM(chunks_count), 0)
                    FROM rag_documents
                    WHERE status NOT IN ('soft_deleted', 'error') AND session_id = ?
                    """,
                    (session_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(file_size), 0), COALESCE(SUM(chunks_count), 0)
                    FROM rag_documents
                    WHERE status NOT IN ('soft_deleted', 'error')
                    """,
                )
            row = cursor.fetchone()
            return {
                "documents": row[0] if row else 0,
                "storage_bytes": row[1] if row else 0,
                "chunks": row[2] if row else 0,
            }

    def get_all_session_quota_counters(self) -> Dict[str, Dict[str, int]]:
        """Return per-session document/storage/chunk counts from the documents table.

        Only active (non-deleted, non-error) documents are counted.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
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
            for row in cursor.fetchall():
                result[row[0]] = {
                    "documents": row[1],
                    "storage_bytes": row[2],
                    "chunks": row[3],
                }
            return result

    def acquire_gc_lock(self, owner_id: str, ttl_seconds: int = 3600) -> bool:
        """Try to acquire the global GC advisory lock.

        Returns True if the lock was acquired, False if already held by another owner.
        The lock expires automatically after *ttl_seconds*.
        """
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        now_iso = now.isoformat()

        with self.transaction() as conn:
            cursor = conn.cursor()
            # Delete expired lock first (allow re-acquisition after TTL)
            cursor.execute(
                "DELETE FROM gc_lock WHERE lock_name = 'gc' AND expires_at < ?",
                (now_iso,),
            )
            # Try to insert (fails silently if lock exists and is not expired)
            try:
                cursor.execute(
                    "INSERT INTO gc_lock (lock_name, owner_id, acquired_at, expires_at) VALUES ('gc', ?, ?, ?)",
                    (owner_id, now_iso, expires_at),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def release_gc_lock(self, owner_id: str) -> None:
        """Release the GC advisory lock held by *owner_id*.

        No-op if the lock is held by a different owner or does not exist.
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM gc_lock WHERE lock_name = 'gc' AND owner_id = ?",
                (owner_id,),
            )

    def log_gc_run(self, run_id: str, phase: str, result: Dict[str, Any]) -> None:
        """Append a GC phase result to gc_audit_log."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            import json

            cursor.execute(
                """
                INSERT INTO gc_audit_log
                (run_id, phase, started_at, completed_at, docs_scanned, docs_cleaned,
                 bytes_freed, errors, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    phase,
                    result.get("started_at", datetime.now(timezone.utc).isoformat()),
                    result.get("completed_at"),
                    result.get("docs_scanned", 0),
                    result.get("docs_cleaned", 0),
                    result.get("bytes_freed", 0),
                    result.get("errors", 0),
                    json.dumps(result.get("details", {})),
                ),
            )

    def get_gc_audit_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent GC audit log entries (newest first)."""
        import json as _json

        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT run_id, phase, started_at, completed_at,
                           docs_scanned, docs_cleaned, bytes_freed, errors, details
                    FROM gc_audit_log
                    ORDER BY started_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                cols = [c[0] for c in cursor.description]
                rows = []
                for row in cursor.fetchall():
                    d = dict(zip(cols, row))
                    if d.get("details") and isinstance(d["details"], str):
                        try:
                            d["details"] = _json.loads(d["details"])
                        except Exception:
                            pass
                    rows.append(d)
                return rows
        except Exception:
            return []

    # =========================================================================
    # COLLECTION IDENTITY (which model embedded each vector collection)
    # =========================================================================
    #
    # Created on first use rather than in _initialize, so a store from any
    # earlier version gains the table without a schema migration.

    def _collection_meta_conn(self):
        conn = self._get_connection()
        if not getattr(self, "_collection_meta_ready", False):
            conn.execute(CREATE_COLLECTION_META_TABLE)
            conn.commit()
            self._collection_meta_ready = True
        return conn

    def get_collection_meta(
        self, collection_name: str, vector_backend: str = "milvus"
    ) -> Optional[Dict[str, Any]]:
        """The stamp of one collection: the model and size its vectors were made at."""
        with self._lock:
            conn = self._collection_meta_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM collection_meta WHERE vector_backend = ? AND collection_name = ?",
                    (vector_backend, collection_name),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def stamp_collection(
        self,
        collection_name: str,
        embedding_model: Optional[str],
        embedding_dim: int,
        vector_backend: str = "milvus",
        lasagna_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record which model a collection holds; an existing stamp is kept.

        Returns the stamp in force, which is the earlier one when there was one:
        the first model to write a collection decides what it holds.
        """
        with self._lock:
            conn = self._collection_meta_conn()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO collection_meta "
                    "(vector_backend, collection_name, embedding_model, embedding_dim, "
                    "lasagna_version, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        vector_backend,
                        collection_name,
                        embedding_model,
                        int(embedding_dim),
                        lasagna_version,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM collection_meta WHERE vector_backend = ? AND collection_name = ?",
                    (vector_backend, collection_name),
                ).fetchone()
                return dict(row)
            finally:
                conn.close()

    def forget_collection_meta(self, collection_name: str, vector_backend: str = "milvus") -> bool:
        """Remove a collection's stamp, after the collection itself was dropped."""
        with self._lock:
            conn = self._collection_meta_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM collection_meta WHERE vector_backend = ? AND collection_name = ?",
                    (vector_backend, collection_name),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def list_collection_meta(self) -> List[Dict[str, Any]]:
        """Every stamped collection, oldest first."""
        with self._lock:
            conn = self._collection_meta_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM collection_meta ORDER BY created_at, collection_name"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # =========================================================================
    # SESSION PRINCIPALS (which API key minted a session)
    # =========================================================================
    #
    # With auth on, quotas count per API key over all its sessions, and a key
    # mints a bounded number of sessions. Every process that counts usage (API
    # workers, the Celery worker, garbage collection) reads the record here.

    def _session_principals_conn(self):
        conn = self._get_connection()
        if not getattr(self, "_session_principals_ready", False):
            conn.execute(CREATE_SESSION_PRINCIPALS_TABLE)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_principals_principal "
                "ON session_principals(principal, created_at);"
            )
            conn.commit()
            self._session_principals_ready = True
        return conn

    def bind_session_principal(
        self,
        session_id: str,
        principal: str,
        *,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> bool:
        """Record that ``principal`` minted ``session_id``; False when it already has ``limit``.

        ``limit`` counts the principal's sessions minted at or after ``since``
        (the ones whose tokens may still be valid). The count and the insert
        hold the principal's lock, so two mints at once cannot both pass.
        """
        with self.exclusive(f"sessions:{principal}"), self._lock:
            conn = self._session_principals_conn()
            try:
                if limit is not None:
                    (count,) = conn.execute(
                        "SELECT COUNT(*) FROM session_principals "
                        "WHERE principal = ? AND created_at >= ?",
                        (
                            principal,
                            (since or datetime.min.replace(tzinfo=timezone.utc)).isoformat(),
                        ),
                    ).fetchone()
                    if count >= limit:
                        return False
                conn.execute(
                    "INSERT INTO session_principals (session_id, principal, created_at) "
                    "VALUES (?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                    (session_id, principal, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def session_principal(self, session_id: str) -> Optional[str]:
        """The principal that minted ``session_id``; None when none is recorded."""
        with self._lock:
            conn = self._session_principals_conn()
            try:
                row = conn.execute(
                    "SELECT principal FROM session_principals WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    # =========================================================================
    # CONTENT GENERATIONS (when a session's searchable chunks last changed)
    # =========================================================================
    #
    # Each process keeps its own in-memory BM25 index of a session's chunks.
    # Every upload and delete bumps the session's generation here, and a process
    # whose index was built at another generation rebuilds it from the vector
    # store. Created on first use, like collection_meta.

    def _content_generation_conn(self):
        conn = self._get_connection()
        if not getattr(self, "_content_generation_ready", False):
            conn.execute(CREATE_CONTENT_GENERATIONS_TABLE)
            conn.commit()
            self._content_generation_ready = True
        return conn

    def get_content_generation(self, session_id: str) -> int:
        """How many times the session's documents have changed; 0 before the first."""
        with self._lock:
            conn = self._content_generation_conn()
            try:
                row = conn.execute(
                    "SELECT generation FROM content_generations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

    def bump_content_generation(self, session_id: str) -> int:
        """Record a change to the session's documents and return the new generation.

        The increment and the read are one transaction, so two processes
        bumping at once get different generations.
        """
        with self._lock:
            conn = self._content_generation_conn()
            try:
                conn.execute(
                    "INSERT INTO content_generations (session_id, generation) VALUES (?, 1) "
                    "ON CONFLICT(session_id) DO UPDATE SET generation = generation + 1",
                    (session_id,),
                )
                row = conn.execute(
                    "SELECT generation FROM content_generations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                conn.commit()
                return int(row[0])
            finally:
                conn.close()


#: Statuses of a document that some process is still working on.
_UNFINISHED_SQL = "'uploading', 'processing', 'indexing'"


def record_visibility_change(db: Any, session_id: Optional[str]) -> bool:
    """Bump ``session_id``'s content generation after a document was archived or restored.

    Each process caches search results and checks this generation on a cache
    hit, which skips the database. Without the bump, another worker kept
    serving an archived document from its cache until the entry expired, and
    kept leaving a restored one out. Returns whether a document changed (``None``:
    none did); a failed bump is logged, and the status change stands.
    """
    if session_id is None:
        return False
    try:
        if session_id:
            db.bump_content_generation(session_id)
    except Exception as e:
        logger.warning(
            f"Could not record the change to session {session_id[:8]}'s documents: {e}. "
            "Other API or worker processes may serve cached results for up to "
            "search_cache_ttl seconds."
        )
    return True


#: One row per session: a counter bumped on every upload and delete, which tells
#: each process whether its in-memory BM25 index of the session is current.
CREATE_CONTENT_GENERATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS content_generations (
    session_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0
);
"""

#: One row per session minted with an API key: the key's principal (a digest,
#: never the key) and when. Quotas count per principal with auth on.
CREATE_SESSION_PRINCIPALS_TABLE = """
CREATE TABLE IF NOT EXISTS session_principals (
    session_id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);
"""

#: One row per vector collection: the embedding model and size its vectors were
#: made with, and the version that made it. Two models at the same size (bge-m3
#: and bge-large are both 1024) are told apart only by this.
CREATE_COLLECTION_META_TABLE = """
CREATE TABLE IF NOT EXISTS collection_meta (
    vector_backend TEXT NOT NULL DEFAULT 'milvus',
    collection_name TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dim INTEGER NOT NULL,
    lasagna_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (vector_backend, collection_name)
);
"""


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_rag_database = None
_rag_database_lock = threading.Lock()


def get_rag_database(db_path: Optional[str] = None):
    """Get or create the RAG database singleton.

    Selects the backend based on ``rag_config.database_backend``:

    - ``"sqlite"`` (default): Uses ``RAGDatabase`` with local file.
    - ``"postgresql"``: Uses ``PostgresRAGDatabase`` with connection pool.

    Args:
        db_path: Optional custom database path / DSN.  If provided on first
                 call, sets the database location.  Ignored on subsequent calls.

    Returns:
        A ``DatabaseBackend``-conformant instance (SQLite or PostgreSQL).
    """
    global _rag_database

    if _rag_database is None:
        with _rag_database_lock:
            if _rag_database is None:
                from .config import rag_config

                backend = getattr(rag_config, "database_backend", "sqlite")

                if backend == "postgresql":
                    dsn = db_path or getattr(rag_config, "database_dsn", "")
                    if not dsn:
                        raise ValueError(
                            "database_backend='postgresql' requires database_dsn to be set"
                        )
                    pool_min = getattr(rag_config, "database_pool_min", 2)
                    pool_max = getattr(rag_config, "database_pool_max", 10)
                    from .database_pg import PostgresRAGDatabase

                    _rag_database = PostgresRAGDatabase(
                        dsn=dsn,
                        pool_min=pool_min,
                        pool_max=pool_max,
                    )
                    logger.info("Using PostgreSQL database backend")
                else:
                    _rag_database = RAGDatabase(db_path)

    return _rag_database


def content_generation(session_id: str) -> Optional[int]:
    """The session's content generation in the metadata database; None if unreadable.

    Bumped on every upload and delete (``bump_content_generation``). A process
    compares it with the one its in-memory state (the BM25 index, the search
    cache) was built at, so a change made by another process is noticed. None
    leaves that state as it is: a database that cannot be read must not empty
    it on every search.
    """
    try:
        read = getattr(get_rag_database(), "get_content_generation", None)
        return None if read is None else int(read(session_id))
    except Exception as e:
        logger.debug(f"Could not read the content generation of session {session_id[:8]}: {e}")
        return None


def reset_rag_database():
    """Reset the RAG database singleton.

    Used primarily for testing.  Calls ``close()`` on the old instance
    (if it supports it) to cleanly shut down connection pools.
    """
    global _rag_database
    with _rag_database_lock:
        old = _rag_database
        _rag_database = None
        if old is not None and hasattr(old, "close"):
            try:
                old.close()
            except Exception:
                pass
