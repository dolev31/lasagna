"""DocumentLifecycleManager — high-level orchestrator.

Wraps ``GarbageCollector``, ``OrphanDetector``, and bulk operations into a
single, easy-to-use facade.  This is the primary entry point for the
``RAGBackend`` and REST/MCP/SDK layers.

Responsibilities:
- Soft-delete (archive) a document
- Restore a soft-deleted document within the grace window
- Bulk-delete all documents in a session
- Bulk-delete a list of specific document IDs
- Trigger GC or orphan detection on demand

Usage::

    lifecycle = DocumentLifecycleManager(db=db, pipeline=pipeline, backend=vector_backend, policy=policy)
    doc = lifecycle.soft_delete("doc_123", "session_abc")
    doc = lifecycle.restore("doc_123", "session_abc")
    result = lifecycle.bulk_delete_session("session_abc", confirm=True)
    report = lifecycle.run_gc()
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from ._results import BulkDeleteResult, ConsistencyReport, GCReport, RepairReport
from .gc import GarbageCollector
from .orphan_detector import OrphanDetector

if TYPE_CHECKING:
    from ..core.database_backend import DatabaseBackend
    from ..api._document_pipeline import DocumentPipeline
    from ..vector_backends.base import VectorStoreAdapter
    from .policy import LifecyclePolicy


class DocumentExpiredError(Exception):
    """Raised when a soft-deleted document cannot be restored because its grace window has expired."""


class DocumentLifecycleManager:
    """High-level manager for all document lifecycle operations."""

    def __init__(
        self,
        db: "DatabaseBackend",
        pipeline: "DocumentPipeline",
        backend: "VectorStoreAdapter",
        policy: "LifecyclePolicy",
        collection_name: str,
    ) -> None:
        self._db = db
        self._pipeline = pipeline
        self._backend = backend
        self._policy = policy
        self._collection_name = collection_name
        self._gc = GarbageCollector(db=db, pipeline=pipeline, policy=policy)
        self._orphan_detector = OrphanDetector(db=db, backend=backend)

    # =========================================================================
    # Soft delete / restore
    # =========================================================================

    def soft_delete(self, document_id: str, session_id: str) -> dict:
        """Archive a document (soft-delete).

        Sets ``status = soft_deleted`` and ``deleted_at = now()``.
        Vectors and files are NOT touched — the GC hard-deletes them after
        the grace window expires.

        Quota is NOT freed here; it is freed when GC runs Phase 1.

        Returns:
            Updated document metadata dict.

        Raises:
            ValueError: If the document does not exist or does not belong to the session.
        """
        doc = self._db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        if doc.get("session_id") != session_id:
            raise ValueError(f"Document {document_id} does not belong to session {session_id}")
        if doc.get("status") == "soft_deleted":
            return doc  # Already archived — idempotent

        # Bumps the session's content generation, so the other processes'
        # search caches stop serving it too.
        self._db.soft_delete_document(document_id)
        self._keep_local_keyword_index(session_id)

        # Invalidate search cache and session focus list
        try:
            self._pipeline._invalidate_search_cache(session_id)
        except Exception:
            pass
        try:
            self._pipeline._remove_from_session_focus(document_id, session_id)
        except Exception:
            pass

        return self._db.get_document(document_id) or {}

    def restore(self, document_id: str, session_id: str) -> dict:
        """Restore a soft-deleted document to READY status.

        Only possible within the soft-delete grace window.

        Returns:
            Updated document metadata dict.

        Raises:
            ValueError: If the document doesn't exist or isn't soft-deleted.
            DocumentExpiredError: If the grace window has expired.
        """
        doc = self._db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        if doc.get("session_id") != session_id:
            raise ValueError(f"Document {document_id} does not belong to session {session_id}")
        if doc.get("status") != "soft_deleted":
            raise ValueError(
                f"Document {document_id} is not soft-deleted (status={doc.get('status')})"
            )

        # Check grace window (Python pre-check for a helpful error message)
        deleted_at_raw = doc.get("deleted_at")
        deleted_at: Optional[datetime] = None
        if deleted_at_raw:
            if isinstance(deleted_at_raw, str):
                try:
                    deleted_at = datetime.fromisoformat(deleted_at_raw.replace("Z", "+00:00"))
                except ValueError:
                    deleted_at = None
            elif isinstance(deleted_at_raw, datetime):
                deleted_at = deleted_at_raw

        if deleted_at:
            grace_expiry = deleted_at + timedelta(hours=self._policy.soft_delete_grace_hours)
            if datetime.now(timezone.utc) > grace_expiry:
                raise DocumentExpiredError(
                    f"Cannot restore document {document_id}: grace window expired at {grace_expiry.isoformat()}"
                )

        # Pass grace_cutoff to the DB so the WHERE clause atomically enforces the
        # grace window — closes the TOCTOU race between the Python check above and
        # the actual UPDATE.
        grace_cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=self._policy.soft_delete_grace_hours)
            if self._policy.soft_delete_grace_hours
            else None
        )
        restored = self._db.restore_document(document_id, grace_cutoff=grace_cutoff)
        if not restored:
            # Could happen if the grace window expired between our Python check and
            # the DB write (very narrow race).  Surface a clear error.
            raise DocumentExpiredError(
                f"Cannot restore document {document_id}: grace window expired during restore"
            )

        self._keep_local_keyword_index(session_id)
        # Re-add to search cache
        try:
            self._pipeline._invalidate_search_cache(session_id)
        except Exception:
            pass

        return self._db.get_document(document_id) or {}

    def _keep_local_keyword_index(self, session_id: str) -> None:
        """Archive and restore leave the chunks in the store, so this process's
        BM25 index is still current at the generation they bumped; other
        processes cannot tell and read theirs again."""
        try:
            generation = self._db.get_content_generation(session_id)
            self._pipeline.hybrid_searcher.note_local_change(session_id, generation)
        except Exception:
            pass

    # =========================================================================
    # Bulk operations
    # =========================================================================

    def bulk_delete_session(self, session_id: str, *, confirm: bool = False) -> BulkDeleteResult:
        """Delete ALL documents in a session efficiently.

        For backends that support partitions, uses a single ``drop_partition``
        call (O(1)).  For others, a single vector ``delete(filter=session_id)``
        call followed by a batch DB delete.

        Args:
            session_id: Session whose documents should be purged.
            confirm: Must be True.  Safety guard.

        Returns:
            ``BulkDeleteResult`` with counts.
        """
        if not confirm:
            raise ValueError("bulk_delete_session requires confirm=True")

        doc_rows = self._db.get_documents(session_id, include_soft_deleted=True)
        total = len(doc_rows)

        # Start with zero counts; accumulate only for docs successfully deleted.
        # Pre-aggregation before the loop would overcount on partial failures.
        result = BulkDeleteResult(
            session_id=session_id,
            requested=total,
            deleted=0,
            bytes_freed=0,
            chunks_freed=0,
        )

        if total == 0:
            return result

        # Delegate each deletion to pipeline.delete_document() so that vectors,
        # stored files, BM25 index, DB record, and quota are all cleaned up
        # correctly — including per-document backend routing for multi-backend
        # deployments.  The pipeline already handles quota per-doc, so we do NOT
        # call quota_manager.record_delete() separately.
        for doc in doc_rows:
            doc_id = doc.get("document_id", "")
            doc_session = doc.get("session_id", session_id)
            file_size = doc.get("file_size", 0) or 0
            chunks = doc.get("chunks_count", 0) or 0
            if not doc_id:
                continue
            try:
                ok = self._pipeline.delete_document(doc_id, doc_session)
                if ok is not False:  # None (void) or True both count as success
                    result.deleted += 1
                    result.bytes_freed += file_size
                    result.chunks_freed += chunks
                else:
                    logger.warning(
                        f"bulk_delete_session: delete_document returned False for {doc_id}"
                    )
                    result.errors += 1
            except Exception as exc:
                logger.warning(f"bulk_delete_session: delete failed for {doc_id}: {exc}")
                result.errors += 1

        logger.info(
            f"bulk_delete_session: deleted {result.deleted}/{total} docs from session {session_id}"
        )
        return result

    def bulk_delete(self, document_ids: List[str], session_id: str) -> BulkDeleteResult:
        """Delete a specific list of documents.

        Falls through to ``DocumentPipeline.delete_document`` for each item,
        which handles quota, vectors, files, and cache in one consistent operation.
        """
        result = BulkDeleteResult(
            session_id=session_id,
            requested=len(document_ids),
            deleted=0,
        )
        for doc_id in document_ids:
            try:
                self._pipeline.delete_document(doc_id, session_id)
                result.deleted += 1
            except Exception as exc:
                logger.warning(f"bulk_delete: failed to delete {doc_id}: {exc}")
                result.errors += 1
        return result

    # =========================================================================
    # GC / orphan detection
    # =========================================================================

    def run_gc(self, *, session_id: Optional[str] = None) -> GCReport:
        """Trigger a manual GC cycle."""
        return self._gc.run_full_cycle(session_id=session_id)

    def detect_orphans(
        self,
        session_id: Optional[str] = None,
    ) -> ConsistencyReport:
        """Run orphan detection over every collection, and return a consistency report.

        The default collection and every other one ``collection_meta`` records
        for this vector backend: each embedding model writes its own.
        """
        return self._orphan_detector.detect(
            collection_name=self._collection_name,
            session_id=session_id,
            other_collections=self._recorded_collections(),
        )

    def _recorded_collections(self) -> List[str]:
        try:
            rows = list(self._db.list_collection_meta() or [])
        except Exception as exc:  # noqa: BLE001 - the default collection is still scanned
            logger.warning(f"Could not list the collections to scan for orphans: {exc}")
            return []
        backend = getattr(self._backend, "backend_name", None)
        return [
            r["collection_name"]
            for r in rows
            if isinstance(r, dict)
            and r.get("collection_name")
            and (not isinstance(backend, str) or r.get("vector_backend", backend) == backend)
        ]

    def repair_orphans(
        self,
        report: ConsistencyReport,
        *,
        dry_run: bool = True,
    ) -> RepairReport:
        """Repair orphans found in *report*."""
        return self._orphan_detector.repair(
            report,
            collection_name=self._collection_name,
            dry_run=dry_run,
        )
