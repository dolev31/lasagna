"""Database backend protocol for the RAG knowledge base.

Defines the abstract interface that all database backends (SQLite,
PostgreSQL, etc.) must implement.  The protocol covers three resource
types — documents, citations, and parent chunks — plus statistics.

The existing ``RAGDatabase`` (SQLite) is the default implementation.
Production deployments can opt into PostgreSQL via config::

    database_backend = "postgresql"
    database_dsn = "postgresql://user:pass@host:5432/rag"

Both implementations share identical method signatures so the rest of
the codebase does not know or care which one is in use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol for RAG metadata storage backends.

    Every method must be implemented by concrete backends.
    All methods are synchronous; async wrappers are planned for Branch 6.
    """

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def save_document(
        self, document: Dict[str, Any], file_path: str, collection_name: Optional[str] = None
    ) -> bool:
        """Persist document metadata.  Upserts by document_id."""
        ...

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document by ID.  Returns None if not found."""
        ...

    def get_documents(
        self,
        session_id: str,
        status: Optional[str] = None,
        include_soft_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """List documents for a session, optionally filtered by status.

        Soft-deleted documents are excluded by default (include_soft_deleted=False).
        """
        ...

    def get_documents_batch(
        self, document_ids: List[str], include_soft_deleted: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch multiple documents by ID in a single query.

        Soft-deleted documents are excluded unless ``include_soft_deleted``.
        """
        ...

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
        """Update a document's processing status and related fields."""
        ...

    def update_document_fields(self, document_id: str, fields: Dict[str, Any]) -> bool:
        """Set the given columns of one row (``database.DOCUMENT_FIELDS``), None included."""
        ...

    def delete_document(self, document_id: str) -> bool:
        """Delete a document (cascades to citations)."""
        ...

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
        """Batch-insert citations with TTL.  Returns count stored."""
        ...

    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single non-expired citation by ID."""
        ...

    def get_session_citations(
        self,
        session_id: str,
        query_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List non-expired citations for a session."""
        ...

    def clear_session_citations(self, session_id: str) -> int:
        """Delete all citations for a session.  Returns count deleted."""
        ...

    def cleanup_expired_citations(self) -> int:
        """Remove expired citations.  Returns count deleted."""
        ...

    def cleanup_citations_by_age(self, max_age_hours: int) -> int:
        """Delete citations older than *max_age_hours*.  Returns count deleted."""
        ...

    def get_citation_by_content_hash(
        self, content_hash: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a non-expired citation by content hash within a session."""
        ...

    def get_citation_aggregate_stats(self) -> Dict[str, Any]:
        """Return ``{'avg_score': float, 'missing_page': int}``."""
        ...

    def enforce_citation_limit(self, session_id: str, max_citations: int = 1000) -> int:
        """Keep only the most recent *max_citations* for a session."""
        ...

    # ------------------------------------------------------------------
    # Parent chunk operations (hierarchical chunking)
    # ------------------------------------------------------------------

    def save_parent_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Batch-insert parent chunks.  Returns count inserted."""
        ...

    def get_parent_chunks(self, stable_ids: List[str]) -> Dict[str, str]:
        """Fetch parent chunk content by stable_ids.  Returns {id: content}."""
        ...

    def get_parent_chunks_with_metadata(self, stable_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch parent chunk content + positional metadata by stable_ids."""
        ...

    def delete_parent_chunks_by_document(self, document_id: str) -> int:
        """Delete all parent chunks for a document.  Returns count deleted."""
        ...

    # ------------------------------------------------------------------
    # Profile operations (session config)
    # ------------------------------------------------------------------

    def save_profile(
        self,
        user_id: str,
        profile_name: str,
        config: Dict[str, Any],
        description: str = "",
        is_builtin: bool = False,
    ) -> bool:
        """Save or update a configuration profile.  Upserts by (user_id, profile_name)."""
        ...

    def get_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all profiles visible to a user (built-in + user's custom)."""
        ...

    def delete_profile(self, user_id: str, profile_name: str) -> bool:
        """Delete a user's custom profile.  Cannot delete built-in profiles."""
        ...

    def delete_all_user_profiles(self, user_id: str) -> int:
        """Delete all non-builtin profiles for a user.  Returns count deleted."""
        ...

    # ------------------------------------------------------------------
    # Session operations (session config)
    # ------------------------------------------------------------------

    def save_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Save or update session config state.  Upserts by session_id."""
        ...

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a single session's config state.  Returns None if not found."""
        ...

    def get_active_sessions(self, max_age_seconds: int = 86400) -> List[Dict[str, Any]]:
        """Get all sessions accessed within *max_age_seconds*."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session by ID.  Returns True if deleted."""
        ...

    def cleanup_expired_sessions(self, max_age_seconds: int = 86400) -> int:
        """Delete sessions older than *max_age_seconds*.  Returns count deleted."""
        ...

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return database-level statistics (document/citation counts etc.)."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle / Garbage Collection operations
    # ------------------------------------------------------------------

    def soft_delete_document(self, document_id: str) -> bool:
        """Mark a document as soft-deleted (sets status=soft_deleted, deleted_at=now).

        Does NOT remove vectors, files, or free quota. Returns True if updated.
        """
        ...

    def restore_document(self, document_id: str, grace_cutoff: Optional[datetime] = None) -> bool:
        """Restore a soft-deleted document to READY status (clears deleted_at).

        Args:
            grace_cutoff: If provided, only restore if deleted_at >= grace_cutoff.
                Closes the TOCTOU race between the Python-level grace check and
                the DB write — the condition is enforced atomically in the UPDATE.

        Returns True if the document was found and restored.
        """
        ...

    def list_soft_deleted_before(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids that were soft-deleted before *cutoff*.

        Used by GC Phase 1 to find candidates for hard deletion.
        Optionally scoped to a single session.
        """
        ...

    def list_expired_documents(self, session_id: Optional[str] = None) -> List[str]:
        """Return document_ids whose expires_at has passed (TTL-based expiry).

        Used by GC Phase 2.
        Optionally scoped to a single session.
        """
        ...

    def list_error_documents_older_than(
        self, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[str]:
        """Return document_ids with status=error whose upload_date < cutoff.

        Used by GC Phase 3 (error retention cleanup).
        Optionally scoped to a single session.
        """
        ...

    def fail_stale_processing_documents(
        self, cutoff: datetime, reason: str, session_id: Optional[str] = None
    ) -> List[str]:
        """Mark documents still uploading, processing or indexing since before *cutoff* as error.

        Sets ``error_message`` to *reason*. A document whose status changed in
        the meantime keeps it. Returns the ids it marked. Used by GC Phase 0.
        """
        ...

    def list_all_active_document_ids(
        self,
        session_id: Optional[str] = None,
        only_ready: bool = False,
    ) -> List[str]:
        """Return all document_ids that are not soft-deleted.

        Used by OrphanDetector to compare against the vector backend.
        Optionally scoped to a single session. If only_ready=True, only
        returns documents with status=ready (excludes processing/error states).
        """
        ...

    def rebuild_quota_counters(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recompute quota usage by scanning the documents table.

        Returns a dict with keys: documents, storage_bytes, chunks.
        Used by GC Phase 6 to correct counter drift after process restarts.
        """
        ...

    def get_all_session_quota_counters(self) -> Dict[str, Dict[str, int]]:
        """Return per-session document/storage/chunk counts from the documents table.

        Returns a dict keyed by session_id, each value a dict with keys:
        ``documents``, ``storage_bytes``, ``chunks``.  Only active (non-deleted,
        non-error) documents are counted.  Used by GC Phase 6 to sync all
        in-memory quota counters after a restart.
        """
        ...

    def acquire_gc_lock(self, owner_id: str, ttl_seconds: int = 3600) -> bool:
        """Try to acquire the global GC advisory lock.

        Returns True if the lock was acquired, False if already held by another owner.
        The lock expires automatically after *ttl_seconds*.
        """
        ...

    def release_gc_lock(self, owner_id: str) -> None:
        """Release the GC advisory lock held by *owner_id*.

        No-op if the lock is held by a different owner or does not exist.
        """
        ...

    def log_gc_run(self, run_id: str, phase: str, result: Dict[str, Any]) -> None:
        """Append a GC phase result to gc_audit_log.

        *result* should contain: started_at, completed_at, docs_scanned,
        docs_cleaned, bytes_freed, errors, and an optional details dict.
        """
        ...

    def get_gc_audit_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the *limit* most recent rows from gc_audit_log (newest first).

        Each row is a dict with keys: run_id, phase, started_at, completed_at,
        docs_scanned, docs_cleaned, bytes_freed, errors, details.
        Returns an empty list if the table doesn't exist (pre-migration).
        """
        ...
