"""OrphanDetector — cross-store consistency checker.

Compares document IDs in three stores:
  - **SQLite/PostgreSQL** (``DatabaseBackend``) — ground truth
  - **Vector backend** (``VectorStoreAdapter``): every vector belongs to a DB record

Identifies two classes of orphans:

db_only
    A DB record exists but no chunks are in the vector store.
    Typical cause: document failed mid-ingestion (vectors never written or
    partially written then the process died).

vector_only
    Chunks exist in the vector store but no DB record, in any status, owns
    them. Typical cause: DB record was hard-deleted (the only permanent
    deletion; an archived document is ``soft_deleted`` and can be restored) or
    the document was created in a previous DB schema that was wiped, leaving
    leaked vectors. A document still uploading, processing or indexing, one in
    ``error``, and an archived one all own their vectors.

Usage::

    detector = OrphanDetector(db=get_rag_database(), backend=vector_backend)
    report = detector.detect(collection_name="rag_documents")
    if not report.is_consistent:
        repair_report = detector.repair(report, collection_name="rag_documents", dry_run=False)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from loguru import logger

from ._results import ConsistencyReport, OrphanRecord, RepairReport
from ..vector_backends.filters import FilterExpression

if TYPE_CHECKING:
    from ..core.database_backend import DatabaseBackend
    from ..vector_backends.base import VectorStoreAdapter


class OrphanDetector:
    """Detects and optionally repairs cross-store inconsistencies."""

    def __init__(
        self,
        db: "DatabaseBackend",
        backend: "VectorStoreAdapter",
    ) -> None:
        self._db = db
        self._backend = backend

    def _recorded_in(self, doc_ids: set, collection_name: str) -> set:
        """The documents among ``doc_ids`` whose vectors belong in ``collection_name``.

        Each embedding model has its own collection, so a document missing from
        this one is an orphan only if it was written here. A record with no
        collection predates the field and counts as this one's.
        """
        if not doc_ids:
            return doc_ids
        ordered = sorted(doc_ids)
        try:
            rows = [
                row
                for start in range(0, len(ordered), 500)
                for row in self._db.get_documents_batch(ordered[start : start + 500]) or []
            ]
        except Exception as exc:  # noqa: BLE001 - keep the old, wider answer
            logger.debug(f"OrphanDetector: could not read document collections: {exc}")
            return doc_ids
        elsewhere = {
            r.get("document_id")
            for r in rows
            if r.get("collection_name") and r.get("collection_name") != collection_name
        }
        return doc_ids - elsewhere

    def _unowned(self, doc_ids: set) -> set:
        """The documents among ``doc_ids`` that no DB record owns, in any status.

        Repair deletes these vectors, so a record that cannot be read counts
        as an owner: "could not tell" must not delete a document's vectors.
        """
        if not doc_ids:
            return doc_ids
        ordered = sorted(doc_ids)
        try:
            owned = {
                row.get("document_id")
                for start in range(0, len(ordered), 500)
                for row in self._db.get_documents_batch(
                    ordered[start : start + 500], include_soft_deleted=True
                )
                or []
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"OrphanDetector: could not read the records of {len(doc_ids)} documents "
                f"found only in the vector store, so none is reported as an orphan: {exc}"
            )
            return set()
        return doc_ids - owned

    # =========================================================================
    # Public API
    # =========================================================================

    def detect(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        vector_scan_limit: int = 50_000,
    ) -> ConsistencyReport:
        """Scan both stores and return a consistency report.

        Args:
            collection_name: Vector collection to scan.
            session_id: Optional — scope the scan to a single session.
            vector_scan_limit: Maximum document IDs to read from the vector
                backend in one call.  Raise if you have more than 50 k unique
                documents in a single collection.

        Returns:
            ``ConsistencyReport`` listing all orphan records.
        """
        scanned_at = datetime.now(timezone.utc)
        logger.info(
            f"OrphanDetector: scanning collection='{collection_name}'"
            + (f" session='{session_id}'" if session_id else "")
        )

        # db-only compares READY docs: in-flight docs (UPLOADING/PROCESSING/INDEXING)
        # haven't had all their vectors written yet and would appear as false-positive
        # db-only orphans. DB implementations that don't support the flag fall back
        # to all-active (graceful degradation). vector-only does not use this set: a
        # document in any status owns its vectors (see _unowned).
        try:
            db_ids = set(self._db.list_all_active_document_ids(session_id, only_ready=True))
        except TypeError:
            db_ids = set(self._db.list_all_active_document_ids(session_id))

        try:
            vec_ids = set(
                self._backend.list_document_ids(
                    collection_name,
                    session_id=session_id,
                    limit=vector_scan_limit,
                )
            )
        except Exception as exc:
            logger.warning(f"OrphanDetector: vector backend scan failed: {exc}")
            vec_ids = set()

        scan_limited = len(vec_ids) >= vector_scan_limit
        if scan_limited:
            logger.warning(
                f"OrphanDetector: vector scan hit limit ({vector_scan_limit:,} docs). "
                "Results may be incomplete. Increase vector_scan_limit or scope by session_id."
            )

        db_only = self._recorded_in(db_ids - vec_ids, collection_name)
        vector_only = self._unowned(vec_ids - db_ids)

        orphans = [
            OrphanRecord(document_id=doc_id, in_db=True, in_vector=False)
            for doc_id in sorted(db_only)
        ] + [
            OrphanRecord(document_id=doc_id, in_db=False, in_vector=True)
            for doc_id in sorted(vector_only)
        ]

        report = ConsistencyReport(
            scanned_at=scanned_at,
            session_id=session_id,
            db_count=len(db_ids),
            vector_count=len(vec_ids),
            orphans=orphans,
            scan_limited=scan_limited,
            vector_scan_limit=vector_scan_limit,
            estimated_unscanned=(len(db_ids) - vector_scan_limit) if scan_limited else None,
        )

        if report.is_consistent:
            logger.info(
                f"OrphanDetector: consistent; {len(db_ids)} DB docs, {len(vec_ids)} vec docs"
            )
        else:
            logger.warning(
                f"OrphanDetector: {report.db_only_count} db-only orphans, "
                f"{report.vector_only_count} vector-only orphans"
            )

        return report

    def repair(
        self,
        report: ConsistencyReport,
        collection_name: str,
        *,
        dry_run: bool = True,
    ) -> RepairReport:
        """Repair the orphans found in *report*.

        - **db-only**: the DB record is soft-deleted (it has no vectors, so
          hard-deletion would leave it permanently lost; soft-delete gives the
          admin a chance to review first).
        - **vector-only**: the chunks are deleted from the vector backend.

        Args:
            report: Consistency report from ``detect()``.
            collection_name: Vector collection name (for vector-only deletes).
            dry_run: If True (default), log what would happen but make no changes.

        Returns:
            ``RepairReport`` with counts of repaired items.
        """
        result = RepairReport(dry_run=dry_run)
        # The report may be older than the records: a document saved since
        # then owns its vectors, so ownership is read again before deleting.
        vector_only = {o.document_id for o in report.orphans if o.kind == "vector_only"}
        unowned = self._unowned(vector_only)
        for doc_id in sorted(vector_only - unowned):
            logger.info(f"Orphan repair: {doc_id} has a document record now; vectors kept")

        for orphan in report.orphans:
            if orphan.kind == "vector_only" and orphan.document_id not in unowned:
                continue
            if orphan.kind == "db_only":
                if not dry_run:
                    try:
                        self._db.soft_delete_document(orphan.document_id)
                        result.repaired_db_only += 1
                    except Exception as exc:
                        logger.warning(f"Orphan repair db-only {orphan.document_id}: {exc}")
                        result.errors += 1
                else:
                    logger.info(f"[dry_run] Would soft-delete db-only orphan: {orphan.document_id}")
                    result.repaired_db_only += 1

            elif orphan.kind == "vector_only":
                if not dry_run:
                    try:
                        self._backend.delete(
                            collection_name,
                            FilterExpression.eq("document_id", orphan.document_id),
                        )
                        result.repaired_vector_only += 1
                    except Exception as exc:
                        logger.warning(f"Orphan repair vector-only {orphan.document_id}: {exc}")
                        result.errors += 1
                else:
                    logger.info(f"[dry_run] Would delete vector-only orphan: {orphan.document_id}")
                    result.repaired_vector_only += 1

        if not dry_run:
            logger.info(
                f"OrphanDetector repair: db_only={result.repaired_db_only} "
                f"vector_only={result.repaired_vector_only} errors={result.errors}"
            )

        return result
