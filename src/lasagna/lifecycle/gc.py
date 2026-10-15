"""Garbage Collector for the knowledge package.

``GarbageCollector`` runs a 6-phase cleanup cycle against the database and
document pipeline.  It is designed to be backend-agnostic: it receives a
``DatabaseBackend`` (protocol) and a ``DocumentPipeline`` reference and
delegates all actual deletion to the pipeline's existing ``delete_document``
method — so all cleanup side-effects (quota release, vector cleanup, cache
invalidation) are handled correctly without duplication.

Phases
------
1. hard_delete_soft_deleted  — docs past soft-delete grace window
2. expire_by_ttl             — docs whose ``expires_at`` has passed
3. cleanup_error_documents   — ERROR docs older than ``error_retention_days``
4. cleanup_expired_citations — delegates to existing DB cleanup method
5. cleanup_idle_sessions     — delegates to existing DB cleanup method
6. reconcile_quotas          — rebuild quota counters from source-of-truth DB

Usage::

    gc = GarbageCollector(db=get_rag_database(), pipeline=pipeline, policy=policy)
    report = gc.run_full_cycle()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from ._results import GCPhaseResult, GCReport
from ..core.quota import get_quota_manager

if TYPE_CHECKING:
    from ..core.database_backend import DatabaseBackend
    from ..api._document_pipeline import DocumentPipeline
    from .policy import LifecyclePolicy


class GarbageCollector:
    """6-phase document lifecycle garbage collector.

    Thread-safe: each ``run_full_cycle()`` call acquires the DB advisory lock
    before proceeding so that concurrent processes (multiple API workers,
    APScheduler jobs) cannot run duplicate GC cycles simultaneously.
    """

    def __init__(
        self,
        db: "DatabaseBackend",
        pipeline: "DocumentPipeline",
        policy: "LifecyclePolicy",
    ) -> None:
        self._db = db
        self._pipeline = pipeline
        self._policy = policy

    # =========================================================================
    # Public API
    # =========================================================================

    def run_full_cycle(self, *, session_id: Optional[str] = None) -> GCReport:
        """Run all 6 GC phases and return an aggregate report.

        Acquires the DB advisory lock before running.  Returns immediately
        (with ``aborted=True``) if another process holds the lock.

        Args:
            session_id: If given, scope phases 1-3 to a single session.
                        Phase 4-6 always run globally.
        """
        run_id = str(uuid.uuid4())
        owner_id = f"gc-{run_id}"
        report = GCReport(run_id=run_id, started_at=datetime.now(timezone.utc))

        if not self._db.acquire_gc_lock(owner_id, ttl_seconds=3 * 3600):
            report.aborted = True
            report.abort_reason = "Another GC process is running (lock held)."
            report.completed_at = datetime.now(timezone.utc)
            logger.info(f"GC skipped: lock held by another process (run_id={run_id})")
            return report

        try:
            logger.info(f"GC cycle started (run_id={run_id})")

            # Phases 1-3 accept session_id for scoped GC; phases 4-6 always global.
            scoped_phases: List = [
                lambda: self._phase1_hard_delete_soft_deleted(session_id=session_id),
                lambda: self._phase2_expire_by_ttl(session_id=session_id),
                lambda: self._phase3_cleanup_error_documents(session_id=session_id),
                self._phase4_cleanup_citations,
                self._phase5_cleanup_idle_sessions,
                self._phase6_reconcile_quotas,
            ]
            for phase_fn in scoped_phases:
                result = phase_fn()
                report.phases.append(result)
                self._db.log_gc_run(run_id, result.phase, result.to_dict())

        except Exception as exc:
            logger.exception(f"GC cycle error (run_id={run_id}): {exc}")
        finally:
            self._db.release_gc_lock(owner_id)
            report.completed_at = datetime.now(timezone.utc)

        logger.info(
            f"GC cycle complete (run_id={run_id}): "
            f"cleaned={report.total_docs_cleaned} docs, "
            f"freed={report.total_bytes_freed / 1024:.1f} KB, "
            f"errors={report.total_errors}"
        )
        return report

    # =========================================================================
    # Phase implementations
    # =========================================================================

    def _phase1_hard_delete_soft_deleted(
        self, *, session_id: Optional[str] = None
    ) -> GCPhaseResult:
        """Hard-delete documents that have been soft-deleted past the grace window."""
        phase = "phase1_hard_delete_soft_deleted"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._policy.soft_delete_grace_hours)
        candidate_ids = self._db.list_soft_deleted_before(cutoff, session_id=session_id)

        if not candidate_ids:
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Safety: abort if this GC run would hard-delete an unusually large fraction.
        # Use get_stats() with a fallback so the phase still works when the method is
        # unavailable (e.g., in test mocks that don't define it).
        try:
            stats = self._db.get_stats()
            total_docs = stats.get("documents", {}).get("total", 0) or len(candidate_ids)
        except Exception:
            total_docs = len(candidate_ids)
        fraction = len(candidate_ids) / max(total_docs, 1)
        if fraction > self._policy.safety_threshold_percent:
            result.completed_at = datetime.now(timezone.utc)
            result.details["safety_abort"] = True
            result.details["candidate_count"] = len(candidate_ids)
            result.details["fraction"] = fraction
            logger.warning(
                f"GC Phase 1 safety threshold exceeded: {len(candidate_ids)} candidates "
                f"({fraction:.1%} of {total_docs} docs): aborting hard deletion. "
                f"Threshold: {self._policy.safety_threshold_percent:.1%}"
            )
            return result

        result.docs_scanned = len(candidate_ids)
        for doc_id in candidate_ids:
            try:
                doc = self._db.get_document(doc_id)
                file_size = (doc or {}).get("file_size", 0) or 0
                chunks = (doc or {}).get("chunks_count", 0) or 0
                # Use the pipeline's existing hard-delete (releases quota, clears vectors/files)
                ok = self._pipeline.delete_document(doc_id, (doc or {}).get("session_id", ""))
                if ok is not False:  # None (void return) or True both count as success
                    result.docs_cleaned += 1
                    result.bytes_freed += file_size
                    result.details.setdefault("chunks_freed", 0)
                    result.details["chunks_freed"] += chunks
                else:
                    logger.warning(f"GC Phase 1: delete_document returned False for {doc_id}")
                    result.errors += 1
            except Exception as exc:
                logger.warning(f"GC Phase 1: failed to hard-delete {doc_id}: {exc}")
                result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        logger.info(f"GC Phase 1: hard-deleted {result.docs_cleaned}/{result.docs_scanned} docs")
        return result

    def _phase2_expire_by_ttl(self, *, session_id: Optional[str] = None) -> GCPhaseResult:
        """Soft-delete documents whose TTL (expires_at) has passed."""
        phase = "phase2_expire_by_ttl"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        if not self._policy.document_ttl_days:  # None or 0 both mean TTL disabled
            result.completed_at = datetime.now(timezone.utc)
            result.details["skipped"] = "document_ttl_days is None or 0 (disabled)"
            return result

        expired_ids = self._db.list_expired_documents(session_id=session_id)
        result.docs_scanned = len(expired_ids)

        for doc_id in expired_ids:
            try:
                self._db.soft_delete_document(doc_id)
                result.docs_cleaned += 1
            except Exception as exc:
                logger.warning(f"GC Phase 2: failed to soft-delete {doc_id}: {exc}")
                result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        logger.info(
            f"GC Phase 2: soft-deleted {result.docs_cleaned}/{result.docs_scanned} expired docs"
        )
        return result

    def _phase3_cleanup_error_documents(self, *, session_id: Optional[str] = None) -> GCPhaseResult:
        """Soft-delete ERROR-status documents older than error_retention_days."""
        phase = "phase3_cleanup_error_documents"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        if self._policy.error_retention_days == 0:
            result.completed_at = datetime.now(timezone.utc)
            result.details["skipped"] = "error_retention_days is 0"
            return result

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._policy.error_retention_days)
        error_ids = self._db.list_error_documents_older_than(cutoff, session_id=session_id)
        result.docs_scanned = len(error_ids)

        for doc_id in error_ids:
            try:
                self._db.soft_delete_document(doc_id)
                result.docs_cleaned += 1
            except Exception as exc:
                logger.warning(f"GC Phase 3: failed to soft-delete error doc {doc_id}: {exc}")
                result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        logger.info(
            f"GC Phase 3: soft-deleted {result.docs_cleaned}/{result.docs_scanned} error docs"
        )
        return result

    def _phase4_cleanup_citations(self) -> GCPhaseResult:
        """Remove expired citations (delegates to existing DB method)."""
        phase = "phase4_cleanup_citations"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        try:
            cleaned = self._db.cleanup_expired_citations()
            result.docs_cleaned = cleaned  # re-using docs_cleaned for citation count
            result.details["citations_removed"] = cleaned
        except Exception as exc:
            logger.warning(f"GC Phase 4: citation cleanup failed: {exc}")
            result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        logger.debug(f"GC Phase 4: removed {result.docs_cleaned} expired citations")
        return result

    def _phase5_cleanup_idle_sessions(self) -> GCPhaseResult:
        """Remove sessions idle for longer than session_idle_days."""
        phase = "phase5_cleanup_idle_sessions"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        max_age_seconds = self._policy.session_idle_days * 86400
        try:
            cleaned = self._db.cleanup_expired_sessions(max_age_seconds=max_age_seconds)
            result.docs_cleaned = cleaned
            result.details["sessions_removed"] = cleaned
        except Exception as exc:
            logger.warning(f"GC Phase 5: session cleanup failed: {exc}")
            result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        logger.debug(f"GC Phase 5: removed {result.docs_cleaned} idle sessions")
        return result

    def _phase6_reconcile_quotas(self) -> GCPhaseResult:
        """Rebuild in-memory quota counters from the DB (repairs counter drift).

        Reads per-session document/storage/chunk counts directly from the
        documents table and pushes them into the live QuotaManager so that
        drift accumulated during a crash or rolling restart is corrected.
        """
        phase = "phase6_reconcile_quotas"
        started = datetime.now(timezone.utc)
        result = GCPhaseResult(phase=phase, started_at=started)

        try:
            # Single GROUP-BY query returns all sessions' authoritative counters.
            per_session = self._db.get_all_session_quota_counters()
            result.details["sessions_scanned"] = len(per_session)

            # Push counters into the running QuotaManager (both in-memory and Redis).
            # sync_from_counters() writes unconditionally regardless of enabled state,
            # so counters are accurate if quota enforcement is turned on later.
            qm = get_quota_manager()
            synced = 0
            for sid, counters in per_session.items():
                qm.sync_from_counters(
                    sid,
                    documents=counters["documents"],
                    storage_bytes=counters["storage_bytes"],
                    chunks=counters["chunks"],
                )
                synced += 1

            result.details["sessions_synced"] = synced
            result.details["quota_enabled"] = qm.is_enabled
            result.details["synced"] = synced > 0
            logger.info(
                f"GC Phase 6: quota counters written for {synced} sessions "
                f"(enforcement {'on' if qm.is_enabled else 'off'})"
            )

        except Exception as exc:
            logger.warning(f"GC Phase 6: quota reconciliation failed: {exc}")
            result.errors += 1

        result.completed_at = datetime.now(timezone.utc)
        return result
