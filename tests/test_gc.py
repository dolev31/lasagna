"""Tests for the GarbageCollector — 6-phase cleanup engine.

Tests cover:
- Phase 1: hard-delete expired soft-deleted docs
- Phase 2: expire docs past their TTL
- Phase 3: soft-delete old ERROR docs
- Safety threshold: abort if too many docs would be deleted
- Batching: large sets processed in batches
- DB advisory lock: skip if another process holds the lock
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from lasagna.lifecycle.gc import GarbageCollector
from lasagna.lifecycle.policy import LifecyclePolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def policy():
    return LifecyclePolicy(
        soft_delete_grace_hours=24,
        document_ttl_days=30,
        error_retention_days=7,
        safety_threshold_percent=0.25,
        batch_size=10,
    )


@pytest.fixture()
def mock_db():
    db = MagicMock()
    db.acquire_gc_lock.return_value = True
    db.release_gc_lock.return_value = None
    db.log_gc_run.return_value = None
    db.list_soft_deleted_before.return_value = []
    db.list_expired_documents.return_value = []
    db.list_error_documents_older_than.return_value = []
    db.soft_delete_document.return_value = True
    db.cleanup_expired_citations.return_value = 0
    db.cleanup_expired_sessions.return_value = 0
    db.rebuild_quota_counters.return_value = {"sessions_updated": 0, "drift_percent": 0.0}
    db.get_all_session_quota_counters.return_value = {}
    # get_stats() used for Phase 1 safety threshold check
    db.get_stats.return_value = {"documents": {"total": 100}}
    # get_document() must return a plain dict so bytes_freed stays an int
    db.get_document.return_value = {"file_size": 0, "chunks_count": 0, "session_id": ""}
    return db


@pytest.fixture()
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.delete_document.return_value = True
    return pipeline


@pytest.fixture()
def gc(policy, mock_db, mock_pipeline):
    return GarbageCollector(db=mock_db, pipeline=mock_pipeline, policy=policy)


# ---------------------------------------------------------------------------
# Advisory Lock
# ---------------------------------------------------------------------------


class TestAdvisoryLock:
    def test_gc_skips_if_lock_held(self, gc, mock_db):
        """GC returns aborted=True without running if another process holds the lock."""
        mock_db.acquire_gc_lock.return_value = False

        report = gc.run_full_cycle()

        assert report.aborted is True
        _mock_pipeline = gc._pipeline
        # No actual deletion calls made
        mock_db.list_soft_deleted_before.assert_not_called()

    def test_gc_acquires_lock_before_running(self, gc, mock_db):
        """GC acquires the advisory lock before any phase executes."""
        gc.run_full_cycle()
        mock_db.acquire_gc_lock.assert_called_once()

    def test_gc_releases_lock_after_completion(self, gc, mock_db):
        """GC releases the lock even if a phase raises."""
        mock_db.list_soft_deleted_before.side_effect = RuntimeError("db error")

        _report = gc.run_full_cycle()

        mock_db.release_gc_lock.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 1: hard-delete expired soft-deleted docs
# ---------------------------------------------------------------------------


class TestPhase1HardDelete:
    def test_phase1_calls_delete_for_expired_docs(self, gc, mock_db, mock_pipeline):
        """Phase 1 calls pipeline.delete_document() for each expired doc."""
        mock_db.list_soft_deleted_before.return_value = ["doc_1", "doc_2", "doc_3"]

        _report = gc.run_full_cycle()

        assert mock_pipeline.delete_document.call_count == 3
        calls = [c[0][0] for c in mock_pipeline.delete_document.call_args_list]
        assert set(calls) == {"doc_1", "doc_2", "doc_3"}

    def test_phase1_safety_threshold_abort(self, gc, mock_db, mock_pipeline, policy):
        """Phase 1 aborts if the number to delete exceeds safety_threshold_percent."""
        # 100 total docs, 30 to delete = 30% > 25% threshold
        mock_db.get_stats.return_value = {"documents": {"total": 100}}
        mock_db.list_soft_deleted_before.return_value = [f"doc_{i}" for i in range(30)]

        report = gc.run_full_cycle()

        # Phase 1 should be aborted (safety_abort set in details, not a top-level field)
        phase1_result = next(
            (p for p in report.phases if p.phase == "phase1_hard_delete_soft_deleted"), None
        )
        assert phase1_result is not None
        assert phase1_result.details.get("safety_abort") is True
        mock_pipeline.delete_document.assert_not_called()

    def test_phase1_below_threshold_proceeds(self, gc, mock_db, mock_pipeline):
        """Phase 1 proceeds when deletion count is below the safety threshold."""
        mock_db.get_stats.return_value = {"documents": {"total": 100}}
        # 10 to delete = 10% < 25% threshold
        mock_db.list_soft_deleted_before.return_value = [f"doc_{i}" for i in range(10)]

        _report = gc.run_full_cycle()

        assert mock_pipeline.delete_document.call_count == 10

    def test_phase1_processes_in_batches(self, gc, mock_db, mock_pipeline, policy):
        """Phase 1 processes docs in batches of policy.batch_size."""
        # 25 docs, batch_size=10 → 3 batches
        doc_ids = [f"doc_{i}" for i in range(25)]
        mock_db.list_soft_deleted_before.return_value = doc_ids

        _report = gc.run_full_cycle()

        # All 25 processed
        assert mock_pipeline.delete_document.call_count == 25


# ---------------------------------------------------------------------------
# Phase 2: TTL expiry
# ---------------------------------------------------------------------------


class TestPhase2TTLExpiry:
    def test_phase2_soft_deletes_expired_docs(self, gc, mock_db):
        """Phase 2 soft-deletes docs with expires_at in the past."""
        mock_db.list_expired_documents.return_value = ["doc_ttl_1", "doc_ttl_2"]

        gc.run_full_cycle()

        calls = [c[0][0] for c in mock_db.soft_delete_document.call_args_list]
        assert "doc_ttl_1" in calls
        assert "doc_ttl_2" in calls

    def test_phase2_skipped_when_ttl_disabled(self, mock_db, mock_pipeline):
        """Phase 2 is skipped when document_ttl_days is None."""
        policy = LifecyclePolicy(document_ttl_days=None, soft_delete_grace_hours=24)
        gc = GarbageCollector(db=mock_db, pipeline=mock_pipeline, policy=policy)

        gc.run_full_cycle()

        mock_db.list_expired_documents.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 3: ERROR document cleanup
# ---------------------------------------------------------------------------


class TestPhase3ErrorCleanup:
    def test_phase3_soft_deletes_old_error_docs(self, gc, mock_db):
        """Phase 3 soft-deletes ERROR docs older than error_retention_days."""
        mock_db.list_error_documents_older_than.return_value = ["err_doc_1"]

        gc.run_full_cycle()

        calls = [c[0][0] for c in mock_db.soft_delete_document.call_args_list]
        assert "err_doc_1" in calls

    def test_phase3_cutoff_uses_error_retention_days(self, gc, mock_db, policy):
        """Phase 3 computes cutoff as now() - error_retention_days."""
        gc.run_full_cycle()

        # Verify list_error_documents_older_than was called with a cutoff
        assert mock_db.list_error_documents_older_than.called
        cutoff_arg = mock_db.list_error_documents_older_than.call_args[0][0]
        expected_approx = datetime.now(timezone.utc) - timedelta(days=policy.error_retention_days)
        assert abs((cutoff_arg - expected_approx).total_seconds()) < 5


# ---------------------------------------------------------------------------
# GC Report
# ---------------------------------------------------------------------------


class TestGCReport:
    def test_report_contains_all_phases(self, gc, mock_db):
        """GCReport contains a result entry for each phase that ran."""
        report = gc.run_full_cycle()
        phase_names = {p.phase for p in report.phases}
        # Phase names include the phase number prefix (e.g. "phase1_...")
        assert "phase1_hard_delete_soft_deleted" in phase_names
        assert "phase2_expire_by_ttl" in phase_names
        assert "phase3_cleanup_error_documents" in phase_names

    def test_report_total_cleaned(self, gc, mock_db, mock_pipeline):
        """GCReport.total_docs_cleaned sums across all phases."""
        mock_db.list_soft_deleted_before.return_value = ["doc_1", "doc_2"]
        mock_db.list_expired_documents.return_value = ["doc_3"]

        report = gc.run_full_cycle()

        assert report.total_docs_cleaned >= 3

    def test_report_has_run_id(self, gc):
        """GCReport has a non-empty run_id."""
        report = gc.run_full_cycle()
        assert report.run_id
        assert len(report.run_id) > 4

    def test_report_logs_to_db(self, gc, mock_db):
        """GC writes phase results to the gc_audit_log table."""
        gc.run_full_cycle()
        assert mock_db.log_gc_run.called


# ---------------------------------------------------------------------------
# Phase 6: quota reconciliation
# ---------------------------------------------------------------------------


class TestPhase6QuotaReconciliation:
    def test_phase6_syncs_counters_when_quota_disabled(self, gc, mock_db):
        """Phase 6 writes DB counters into QuotaManager even when quota is disabled.

        sync_from_counters() must be unconditional — _enabled only controls
        enforcement, not counter population.
        """
        from lasagna.core.quota import QuotaManager

        mock_db.get_all_session_quota_counters.return_value = {
            "sess_abc": {"documents": 5, "storage_bytes": 1024, "chunks": 20},
        }

        qm = QuotaManager(enabled=False)  # quota disabled
        with patch("lasagna.lifecycle.gc.get_quota_manager", return_value=qm):
            gc.run_full_cycle()

        # Counters must be written regardless of enabled state
        with qm._lock:
            usage = qm._usage.get("sess_abc")
        assert usage is not None, "Usage entry not created — sync was a no-op"
        assert usage.documents == 5
        assert usage.storage_bytes == 1024
        assert usage.chunks == 20

    def test_phase6_synced_flag_reflects_actual_writes(self, gc, mock_db):
        """synced=True only when at least one session was written."""
        from lasagna.core.quota import QuotaManager

        mock_db.get_all_session_quota_counters.return_value = {
            "sess_xyz": {"documents": 2, "storage_bytes": 512, "chunks": 8},
        }

        qm = QuotaManager(enabled=False)
        with patch("lasagna.lifecycle.gc.get_quota_manager", return_value=qm):
            report = gc.run_full_cycle()

        phase6 = next(p for p in report.phases if p.phase == "phase6_reconcile_quotas")
        assert phase6.details.get("sessions_synced") == 1
        assert phase6.details.get("synced") is True

    def test_phase6_synced_false_when_no_sessions(self, gc, mock_db):
        """synced=False when DB returns no sessions (nothing to reconcile)."""
        from lasagna.core.quota import QuotaManager

        mock_db.get_all_session_quota_counters.return_value = {}

        qm = QuotaManager(enabled=False)
        with patch("lasagna.lifecycle.gc.get_quota_manager", return_value=qm):
            report = gc.run_full_cycle()

        phase6 = next(p for p in report.phases if p.phase == "phase6_reconcile_quotas")
        assert phase6.details.get("sessions_synced") == 0
        assert phase6.details.get("synced") is False
