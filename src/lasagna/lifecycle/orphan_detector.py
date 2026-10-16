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
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set

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

    def _recorded_in(self, doc_ids: set, collection_name: str, *, default: bool = True) -> set:
        """The documents among ``doc_ids`` whose vectors belong in ``collection_name``.

        Each embedding model has its own collection, so a document missing from
        this one is an orphan only if it was written here. A record with no
        collection predates the field and counts as the ``default`` one's.
        Raises when the records cannot be read: assuming this collection then
        made every document of another model look like one with no vectors.
        """
        if not doc_ids:
            return doc_ids
        ordered = sorted(doc_ids)
        rows = [
            row
            for start in range(0, len(ordered), 500)
            for row in self._db.get_documents_batch(ordered[start : start + 500]) or []
        ]
        recorded = {r.get("document_id"): r.get("collection_name") for r in rows}
        return {
            d
            for d in doc_ids
            if recorded.get(d) == collection_name or (default and not recorded.get(d))
        }

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

    def _still_without_vectors(self, report: ConsistencyReport, collection_name: str) -> set:
        """The db-only orphans of ``report`` that are still ready, still recorded in
        their collection, and still have no vectors there.

        Each collection is scanned again, since its vectors may have been written
        after the report. "Could not tell" keeps the document.
        """
        by_name: Dict[str, Set[str]] = {}
        for o in report.orphans:
            if o.kind == "db_only":
                by_name.setdefault(o.collection_name or collection_name, set()).add(o.document_id)
        if not by_name:
            return set()
        try:
            try:
                ready = set(
                    self._db.list_all_active_document_ids(report.session_id, only_ready=True)
                )
            except TypeError:
                ready = set(self._db.list_all_active_document_ids(report.session_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Orphan repair: could not read the documents again, none archived: {exc}"
            )
            return set()
        still: Set[str] = set()
        for name, doc_ids in by_name.items():
            try:
                present = set(
                    self._backend.list_document_ids(
                        name, session_id=report.session_id, limit=report.vector_scan_limit
                    )
                )
                if len(present) >= report.vector_scan_limit:
                    raise RuntimeError("the scan stopped at vector_scan_limit")
                missing = (doc_ids & ready) - present
                still |= self._recorded_in(missing, name, default=name == collection_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Orphan repair: could not check the {len(doc_ids)} db-only document(s) "
                    f"of {name} again, so none is archived: {exc}"
                )
        for doc_id in sorted(set().union(*by_name.values()) - still):
            logger.info(f"Orphan repair: {doc_id} is not a db-only orphan now; kept")
        return still

    # =========================================================================
    # Public API
    # =========================================================================

    def detect(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        vector_scan_limit: int = 50_000,
        other_collections: Iterable[str] = (),
    ) -> ConsistencyReport:
        """Scan both stores and return a consistency report.

        Args:
            collection_name: The default collection to scan; a document whose
                record names no collection belongs to it.
            session_id: Optional — scope the scan to a single session.
            vector_scan_limit: Maximum document IDs to read from the vector
                backend in one call.  Raise if you have more than 50 k unique
                documents in a single collection.
            other_collections: More collections to scan (one per embedding
                model, from ``collection_meta``). One the vector store does
                not have is skipped.

        Returns:
            ``ConsistencyReport`` listing all orphan records.
        """
        scanned_at = datetime.now(timezone.utc)
        names = [collection_name] + [
            c for c in dict.fromkeys(other_collections) if c and c != collection_name
        ]
        has = getattr(self._backend, "has_collection", None)
        names = [n for n in names if n == collection_name or has is None or _exists(has, n)]
        logger.info(
            f"OrphanDetector: scanning collection(s) {', '.join(repr(n) for n in names)}"
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

        # A scan that failed or stopped at the limit says nothing about the
        # documents it did not reach: compared with it, each of them would look
        # like a document with no vectors, and repair would archive it. Such a
        # report names the reason and lists no orphans.
        incomplete_reason: Optional[str] = None
        scan_limited = False
        vec_by_name: Dict[str, Set[str]] = {}
        for name in names:
            try:
                vec_by_name[name] = set(
                    self._backend.list_document_ids(
                        name,
                        session_id=session_id,
                        limit=vector_scan_limit,
                    )
                )
            except Exception as exc:
                incomplete_reason = f"the vector store scan of {name} failed: {exc}"
                break
            if len(vec_by_name[name]) >= vector_scan_limit:
                scan_limited = True
                incomplete_reason = (
                    f"the vector scan of {name} stopped at vector_scan_limit "
                    f"({vector_scan_limit:,} documents); scope the scan by session_id "
                    "or raise the limit"
                )
                break

        orphans = []
        if incomplete_reason is None:
            try:
                for name, vec_ids in vec_by_name.items():
                    db_only = self._recorded_in(
                        db_ids - vec_ids, name, default=name == collection_name
                    )
                    orphans += [
                        OrphanRecord(doc_id, in_db=True, in_vector=False, collection_name=name)
                        for doc_id in sorted(db_only)
                    ]
            except Exception as exc:  # noqa: BLE001 - reported as incomplete
                incomplete_reason = (
                    f"the database could not say which collection each document was "
                    f"written to: {exc}"
                )
                orphans = []
        if incomplete_reason is None:
            for name, vec_ids in vec_by_name.items():
                orphans += [
                    OrphanRecord(doc_id, in_db=False, in_vector=True, collection_name=name)
                    for doc_id in sorted(self._unowned(vec_ids - db_ids))
                ]
        else:
            logger.warning(f"OrphanDetector: no orphans reported because {incomplete_reason}")
        # db-only first, then vector-only, as one collection's report lists them.
        orphans.sort(key=lambda o: (o.kind != "db_only", names.index(o.collection_name)))

        report = ConsistencyReport(
            scanned_at=scanned_at,
            session_id=session_id,
            db_count=len(db_ids),
            vector_count=sum(len(v) for v in vec_by_name.values()),
            orphans=orphans,
            scan_limited=scan_limited,
            vector_scan_limit=vector_scan_limit,
            estimated_unscanned=(len(db_ids) - vector_scan_limit) if scan_limited else None,
            incomplete_reason=incomplete_reason,
            collections=names,
        )

        if report.is_consistent:
            logger.info(
                f"OrphanDetector: consistent; {len(db_ids)} DB docs, {len(vec_ids)} vec docs"
            )
        elif not report.incomplete:
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

        An incomplete report (the vector scan failed or stopped at the limit)
        is refused: nothing changes, and ``aborted``/``abort_reason`` say why.

        Args:
            report: Consistency report from ``detect()``.
            collection_name: Vector collection name (for vector-only deletes).
            dry_run: If True (default), log what would happen but make no changes.

        Returns:
            ``RepairReport`` with counts of repaired items.
        """
        result = RepairReport(dry_run=dry_run)
        if report.incomplete:
            result.aborted = True
            reason = report.incomplete_reason or "the vector scan stopped at vector_scan_limit"
            result.abort_reason = f"nothing was repaired because {reason}"
            logger.warning(f"Orphan repair: {result.abort_reason}")
            return result
        # The report may be older than the records: a document saved since
        # then owns its vectors, so ownership is read again before deleting.
        vector_only = {o.document_id for o in report.orphans if o.kind == "vector_only"}
        unowned = self._unowned(vector_only)
        for doc_id in sorted(vector_only - unowned):
            logger.info(f"Orphan repair: {doc_id} has a document record now; vectors kept")
        # Likewise a db-only document may have its vectors now.
        still_db_only = self._still_without_vectors(report, collection_name)

        for orphan in report.orphans:
            if orphan.kind == "vector_only" and orphan.document_id not in unowned:
                continue
            if orphan.kind == "db_only" and orphan.document_id not in still_db_only:
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
                            orphan.collection_name or collection_name,
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


def _exists(has_collection, name: str) -> bool:
    try:
        return bool(has_collection(name))
    except Exception as exc:  # noqa: BLE001 - an unreachable store is scanned and reported
        logger.debug(f"OrphanDetector: could not tell whether {name} exists: {exc}")
        return True
