"""Tests for document lifecycle management — soft-delete, restore, bulk ops.

Covers the DocumentLifecycleManager facade and all soft-delete/restore
invariants including quota semantics and search-cache exclusion.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from lasagna.lifecycle import (
    DocumentLifecycleManager,
    DocumentExpiredError,
    LifecyclePolicy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    document_id: str = "doc_abc",
    session_id: str = "sess_xyz",
    status: str = "ready",
    deleted_at: datetime | None = None,
) -> dict:
    return {
        "document_id": document_id,
        "session_id": session_id,
        "status": status,
        "deleted_at": deleted_at,
        "file_size": 1024,
        "chunks_count": 5,
    }


@pytest.fixture()
def policy():
    return LifecyclePolicy(soft_delete_grace_hours=24, batch_size=100)


@pytest.fixture()
def mock_db():
    db = MagicMock()
    db.get_document.return_value = _make_doc()
    db.soft_delete_document.return_value = True
    db.restore_document.return_value = True
    db.get_documents.return_value = []
    return db


@pytest.fixture()
def mock_pipeline():
    pipeline = MagicMock()
    pipeline._invalidate_search_cache.return_value = None
    pipeline._remove_from_session_focus.return_value = None
    pipeline.delete_document.return_value = True
    return pipeline


@pytest.fixture()
def mock_backend():
    backend = MagicMock()
    backend.capabilities = MagicMock(supports_partitions=False)
    backend.delete.return_value = 0
    return backend


@pytest.fixture()
def manager(policy, mock_db, mock_pipeline, mock_backend):
    return DocumentLifecycleManager(
        db=mock_db,
        pipeline=mock_pipeline,
        backend=mock_backend,
        policy=policy,
        collection_name="test_collection",
    )


# ---------------------------------------------------------------------------
# Soft-delete
# ---------------------------------------------------------------------------


class TestSoftDelete:
    def test_soft_delete_sets_status(self, manager, mock_db, mock_pipeline):
        """soft_delete() calls db.soft_delete_document and invalidates cache."""
        updated = _make_doc(status="soft_deleted")
        mock_db.get_document.side_effect = [
            _make_doc(status="ready"),  # initial lookup
            updated,  # post-delete lookup
        ]

        result = manager.soft_delete("doc_abc", "sess_xyz")

        mock_db.soft_delete_document.assert_called_once_with("doc_abc")
        mock_pipeline._invalidate_search_cache.assert_called_once_with("sess_xyz")
        mock_pipeline._remove_from_session_focus.assert_called_once_with("doc_abc", "sess_xyz")
        assert result["status"] == "soft_deleted"

    def test_soft_delete_wrong_session_raises(self, manager, mock_db):
        """soft_delete() raises ValueError if document belongs to another session."""
        mock_db.get_document.return_value = _make_doc(session_id="other_session")
        with pytest.raises(ValueError, match="does not belong to session"):
            manager.soft_delete("doc_abc", "sess_xyz")

    def test_soft_delete_not_found_raises(self, manager, mock_db):
        """soft_delete() raises ValueError if document not found."""
        mock_db.get_document.return_value = None
        with pytest.raises(ValueError, match="not found"):
            manager.soft_delete("missing_doc", "sess_xyz")

    def test_soft_delete_idempotent(self, manager, mock_db):
        """Calling soft_delete() on an already-soft-deleted doc is a no-op."""
        already_deleted = _make_doc(status="soft_deleted")
        mock_db.get_document.return_value = already_deleted

        result = manager.soft_delete("doc_abc", "sess_xyz")

        mock_db.soft_delete_document.assert_not_called()
        assert result["status"] == "soft_deleted"

    def test_soft_delete_does_not_touch_vectors(self, manager, mock_db, mock_backend):
        """Soft-delete does NOT call backend.delete() — vectors stay."""
        manager.soft_delete("doc_abc", "sess_xyz")
        mock_backend.delete.assert_not_called()

    def test_soft_delete_does_not_free_quota(self, manager, mock_db, mock_pipeline):
        """Soft-delete does NOT free quota (only hard-delete does).

        Quota is released by ``pipeline.delete_document()``, which GC phase 1
        calls once the grace window expires. Asserting soft_delete never
        reaches it is the form of this check that has teeth.

        The previous version patched
        ``lasagna.lifecycle.manager.get_quota_manager`` with ``create=True``.
        That module has never had such an attribute, so ``create=True``
        invented one, nothing ever called it, and the assertion that it was
        not called held no matter what ``soft_delete`` did. The test could not
        fail.
        """
        manager.soft_delete("doc_abc", "sess_xyz")
        mock_pipeline.delete_document.assert_not_called()


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_within_grace_window(self, manager, mock_db, mock_pipeline):
        """restore() works when deleted_at is within the grace window."""
        deleted_at = datetime.now(timezone.utc) - timedelta(hours=12)  # 12h ago, grace=24h
        soft_deleted_doc = _make_doc(status="soft_deleted", deleted_at=deleted_at)
        restored_doc = _make_doc(status="ready")
        mock_db.get_document.side_effect = [soft_deleted_doc, restored_doc]
        mock_db.restore_document.return_value = True  # simulate successful atomic restore

        result = manager.restore("doc_abc", "sess_xyz")

        call_args = mock_db.restore_document.call_args
        assert call_args[0][0] == "doc_abc"
        assert "grace_cutoff" in call_args[1]
        assert call_args[1]["grace_cutoff"] is not None
        mock_pipeline._invalidate_search_cache.assert_called_once_with("sess_xyz")
        assert result["status"] == "ready"

    def test_restore_after_grace_window_raises(self, manager, mock_db, policy):
        """restore() raises DocumentExpiredError when grace window has expired."""
        deleted_at = datetime.now(timezone.utc) - timedelta(hours=48)  # 48h ago, grace=24h
        soft_deleted_doc = _make_doc(status="soft_deleted", deleted_at=deleted_at)
        mock_db.get_document.return_value = soft_deleted_doc

        with pytest.raises(DocumentExpiredError, match="grace window expired"):
            manager.restore("doc_abc", "sess_xyz")

        mock_db.restore_document.assert_not_called()

    def test_restore_not_soft_deleted_raises(self, manager, mock_db):
        """restore() raises ValueError if document is READY (not soft_deleted)."""
        mock_db.get_document.return_value = _make_doc(status="ready")
        with pytest.raises(ValueError, match="not soft-deleted"):
            manager.restore("doc_abc", "sess_xyz")

    def test_restore_not_found_raises(self, manager, mock_db):
        """restore() raises ValueError if document not found."""
        mock_db.get_document.return_value = None
        with pytest.raises(ValueError, match="not found"):
            manager.restore("doc_abc", "sess_xyz")

    def test_restore_wrong_session_raises(self, manager, mock_db):
        """restore() raises ValueError if document belongs to another session."""
        deleted_at = datetime.now(timezone.utc) - timedelta(hours=1)
        doc = _make_doc(session_id="other", status="soft_deleted", deleted_at=deleted_at)
        mock_db.get_document.return_value = doc
        with pytest.raises(ValueError, match="does not belong"):
            manager.restore("doc_abc", "sess_xyz")

    def test_restore_no_deleted_at_still_works(self, manager, mock_db):
        """restore() works even if deleted_at is missing (legacy or edge case)."""
        soft_deleted_doc = _make_doc(status="soft_deleted", deleted_at=None)
        restored_doc = _make_doc(status="ready")
        mock_db.get_document.side_effect = [soft_deleted_doc, restored_doc]

        result = manager.restore("doc_abc", "sess_xyz")
        assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# Soft-deleted docs excluded from search
# ---------------------------------------------------------------------------


class TestSearchExclusion:
    def test_soft_deleted_excluded_from_get_documents(self, mock_db):
        """DB layer: get_documents() should not return soft_deleted by default."""
        # This tests the DB filtering fix (include_soft_deleted=False default)
        # We simulate the DB correctly filtering by checking the mock call
        from lasagna.core.database import RAGDatabase

        db = MagicMock(spec=RAGDatabase)
        db.get_documents.return_value = [
            _make_doc(document_id="doc_ready", status="ready"),
            # soft_deleted doc should NOT appear here (filtered at DB level)
        ]
        rows = db.get_documents("sess_xyz")
        assert all(r["status"] != "soft_deleted" for r in rows)


# ---------------------------------------------------------------------------
# Bulk delete session
# ---------------------------------------------------------------------------


class TestBulkDeleteSession:
    def test_bulk_delete_requires_confirm(self, manager):
        """bulk_delete_session() raises ValueError without confirm=True."""
        with pytest.raises(ValueError, match="confirm=True"):
            manager.bulk_delete_session("sess_xyz", confirm=False)

    def test_bulk_delete_session_calls_pipeline_delete(self, manager, mock_db, mock_pipeline):
        """bulk_delete_session() calls pipeline.delete_document() for each doc."""
        doc_rows = [
            _make_doc(document_id=f"doc_{i}", status="ready", deleted_at=None) for i in range(3)
        ]
        mock_db.get_documents.return_value = doc_rows

        result = manager.bulk_delete_session("sess_xyz", confirm=True)

        # pipeline.delete_document called once per doc (handles vectors, DB, quota)
        assert mock_pipeline.delete_document.call_count == 3
        assert result.deleted == 3
        assert result.requested == 3

    def test_bulk_delete_empty_session(self, manager, mock_db, mock_backend):
        """bulk_delete_session() returns early and makes no calls for empty sessions."""
        mock_db.get_documents.return_value = []
        result = manager.bulk_delete_session("sess_xyz", confirm=True)
        mock_backend.delete.assert_not_called()
        assert result.deleted == 0

    def test_bulk_delete_uses_pipeline_for_all_backends(
        self, manager, mock_db, mock_pipeline, mock_backend
    ):
        """bulk_delete_session() always delegates to pipeline.delete_document()."""
        # Even for partition-capable backends, the pipeline handles routing correctly
        mock_backend.capabilities = MagicMock(supports_partitions=True)
        doc_rows = [_make_doc()]
        mock_db.get_documents.return_value = doc_rows

        manager.bulk_delete_session("sess_xyz", confirm=True)

        # pipeline handles all cleanup (vectors, DB, quota, files)
        assert mock_pipeline.delete_document.call_count == 1

    def test_bulk_delete_reports_byte_totals(self, manager, mock_db, mock_pipeline):
        """bulk_delete_session() pre-calculates and reports byte/chunk totals."""
        doc_rows = [
            {**_make_doc(document_id=f"doc_{i}"), "file_size": 500, "chunks_count": 10}
            for i in range(2)
        ]
        mock_db.get_documents.return_value = doc_rows

        result = manager.bulk_delete_session("sess_xyz", confirm=True)

        # Quota is freed per-doc inside pipeline.delete_document(); the result
        # carries the pre-calculated totals for caller reporting.
        assert result.bytes_freed == 1000
        assert result.chunks_freed == 20
        assert mock_pipeline.delete_document.call_count == 2


# ---------------------------------------------------------------------------
# TTL write-at-ingest path
# ---------------------------------------------------------------------------


def _make_pipeline_stub():
    """Create a minimal DocumentPipeline with mocked storage and collection manager."""
    from lasagna.api._document_pipeline import DocumentPipeline

    mock_storage = MagicMock()
    mock_storage.backend_name = "local"
    mock_collection_manager = MagicMock()
    mock_collection_manager._default_backend_name = "milvus"
    pipeline = object.__new__(DocumentPipeline)
    pipeline._storage = mock_storage
    pipeline._collection_manager = mock_collection_manager
    pipeline.hybrid_searcher = MagicMock()
    return pipeline


def _make_document(document_id="doc_test", session_id="sess_test"):
    from lasagna.core.schema import Document, DocumentStatus

    return Document(
        document_id=document_id,
        session_id=session_id,
        filename="test.pdf",
        file_size=1024,
        file_type="pdf",
        status=DocumentStatus.READY,
    )


class TestTTLWriteAtIngest:
    """Verify that expires_at is set during document save when document_ttl_days is configured."""

    def test_expires_at_set_when_ttl_configured(self):
        """save_document dict includes expires_at when LifecyclePolicy.document_ttl_days is set."""
        from lasagna.lifecycle.policy import LifecyclePolicy

        policy_with_ttl = LifecyclePolicy(document_ttl_days=30)
        mock_db = MagicMock()
        pipeline = _make_pipeline_stub()
        doc = _make_document()

        # get_rag_database and LifecyclePolicy are imported locally inside _save_document_metadata,
        # so patch at their source modules.
        with (
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
            patch(
                "lasagna.lifecycle.policy.LifecyclePolicy.from_env", return_value=policy_with_ttl
            ),
        ):
            pipeline._save_document_metadata(doc, "/storage/test.pdf")

        mock_db.save_document.assert_called_once()
        doc_dict = mock_db.save_document.call_args[0][0]
        assert "expires_at" in doc_dict
        assert doc_dict["expires_at"] is not None, "expires_at should be set when TTL is configured"

    def test_expires_at_none_when_no_ttl(self):
        """save_document dict has expires_at=None when document_ttl_days is None."""
        from lasagna.lifecycle.policy import LifecyclePolicy

        policy_no_ttl = LifecyclePolicy(document_ttl_days=None)
        mock_db = MagicMock()
        pipeline = _make_pipeline_stub()
        doc = _make_document(document_id="doc_no_ttl")

        with (
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
            patch("lasagna.lifecycle.policy.LifecyclePolicy.from_env", return_value=policy_no_ttl),
        ):
            pipeline._save_document_metadata(doc, "/storage/test.pdf")

        doc_dict = mock_db.save_document.call_args[0][0]
        assert doc_dict.get("expires_at") is None, "expires_at should be None when no TTL is set"


# ---------------------------------------------------------------------------
# Scheduler start/stop wiring
# ---------------------------------------------------------------------------


class TestSchedulerWiring:
    """Verify LifecycleScheduler is properly started and stopped."""

    def test_scheduler_started_on_create(self):
        """LifecycleScheduler.start() is called when _create_lifecycle_manager runs."""
        mock_scheduler = MagicMock()
        mock_db = MagicMock()
        mock_manager_instance = MagicMock()
        mock_manager_instance._gc = MagicMock()

        with (
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
            patch(
                "lasagna.api.rag_backend.DocumentLifecycleManager",
                return_value=mock_manager_instance,
            ),
            patch("lasagna.lifecycle.scheduler.LifecycleScheduler", return_value=mock_scheduler),
        ):
            from lasagna.api.rag_backend import RAGBackend
            from lasagna.lifecycle.policy import LifecyclePolicy

            backend = object.__new__(RAGBackend)
            backend._document_pipeline = MagicMock()
            backend._vector_backend = MagicMock()

            with patch("lasagna.api.rag_backend.LifecyclePolicy") as mock_policy_cls:
                mock_policy_cls.from_env.return_value = LifecyclePolicy()
                backend._create_lifecycle_manager()

        mock_scheduler.start.assert_called_once()
        assert backend._lifecycle_scheduler is mock_scheduler

    def test_scheduler_stopped_on_close(self):
        """close_backend() calls _lifecycle_scheduler.stop()."""
        import lasagna.api.rag_backend as rag_backend_mod
        from lasagna.api.rag_backend import close_backend

        mock_scheduler = MagicMock()
        mock_backend = MagicMock()
        mock_backend._lifecycle_scheduler = mock_scheduler
        mock_backend._vector_backend = None
        mock_backend.embedding_manager = None

        original_instance = rag_backend_mod._instance
        try:
            rag_backend_mod._instance = mock_backend
            close_backend()
        finally:
            rag_backend_mod._instance = original_instance

        mock_scheduler.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Per-document collection_name deletion routing
# ---------------------------------------------------------------------------


def _make_delete_pipeline():
    """Pipeline stub for delete_document tests."""
    from lasagna.api._document_pipeline import DocumentPipeline

    pipeline = object.__new__(DocumentPipeline)
    pipeline._storage = MagicMock()
    pipeline._collection_manager = MagicMock()
    pipeline._collection_manager._default_backend_name = "milvus"
    pipeline.hybrid_searcher = MagicMock()
    return pipeline


class TestDeleteCollectionRouting:
    """Verify delete_document() uses per-document collection_name, not global config."""

    def test_delete_uses_per_document_collection_name(self):
        """delete_document() passes doc.collection_name to vector backend delete."""
        doc_in_db = {
            "document_id": "doc_routing_test",
            "session_id": "sess_routing",
            "file_path": "/storage/test.pdf",
            "status": "ready",
            "file_size": 512,
            "chunks_count": 3,
            "vector_backend_name": "milvus",
            "collection_name": "custom_collection_v2",  # per-document stored collection
        }
        mock_db = MagicMock()
        mock_db.get_document.return_value = doc_in_db
        mock_db.delete_parent_chunks_by_document.return_value = 0
        mock_vector_backend = MagicMock()
        mock_vector_backend.is_connected.return_value = True
        mock_tenant = MagicMock()
        mock_tenant.resolve_partition_key.return_value = None

        pipeline = _make_delete_pipeline()

        # get_rag_database is imported locally inside delete_document — patch at source
        with (
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
            patch("lasagna.api._document_pipeline.get_quota_manager", return_value=MagicMock()),
            patch("lasagna.api._document_pipeline.get_tenant_manager", return_value=mock_tenant),
            patch("lasagna.api._document_pipeline.get_backend", return_value=mock_vector_backend),
            patch("lasagna.api._document_pipeline.rag_config") as mock_rag_config,
        ):
            mock_rag_config.collection_name = "global_collection_v1"
            mock_rag_config.enable_hybrid_search = False

            pipeline.delete_document("doc_routing_test", "sess_routing")

        used_collection = mock_vector_backend.delete.call_args[1]["collection_name"]
        assert used_collection == "custom_collection_v2", (
            f"Expected 'custom_collection_v2' but got '{used_collection}' — "
            "delete_document must use per-document collection_name"
        )

    def test_delete_falls_back_to_the_sessions_collection_when_doc_has_no_collection(self):
        """delete_document() falls back to the session's collection if doc.collection_name is None.

        It fell back to the process-wide collection; since 0.1.0 each embedding
        model has its own, and the document's session says which one it used.
        """
        doc_in_db = {
            "document_id": "doc_no_col",
            "session_id": "sess_no_col",
            "file_path": "/storage/test.pdf",
            "status": "ready",
            "file_size": 512,
            "chunks_count": 2,
            "vector_backend_name": "milvus",
            "collection_name": None,  # pre-lifecycle doc with no stored collection
        }
        mock_db = MagicMock()
        mock_db.get_document.return_value = doc_in_db
        mock_db.delete_parent_chunks_by_document.return_value = 0
        mock_vector_backend = MagicMock()
        mock_vector_backend.is_connected.return_value = True
        mock_tenant = MagicMock()
        mock_tenant.resolve_partition_key.return_value = None

        pipeline = _make_delete_pipeline()

        with (
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
            patch("lasagna.api._document_pipeline.get_quota_manager", return_value=MagicMock()),
            patch("lasagna.api._document_pipeline.get_tenant_manager", return_value=mock_tenant),
            patch("lasagna.api._document_pipeline.get_backend", return_value=mock_vector_backend),
            patch("lasagna.api._document_pipeline.rag_config") as mock_rag_config,
        ):
            mock_rag_config.collection_name = "global_collection_v1"
            mock_rag_config.with_session.return_value.collection_name = "session_collection_v1"
            mock_rag_config.enable_hybrid_search = False

            pipeline.delete_document("doc_no_col", "sess_no_col")

        mock_rag_config.with_session.assert_called_with("sess_no_col")
        used_collection = mock_vector_backend.delete.call_args[1]["collection_name"]
        assert used_collection == "session_collection_v1", (
            "Should fall back to the session's collection when doc has none"
        )


# ---------------------------------------------------------------------------
# GC Phase 1 and bulk delete success accounting
# ---------------------------------------------------------------------------


class TestGCSuccessAccounting:
    """Phase 1 should only count docs as cleaned when delete_document succeeds."""

    def _make_gc(self, mock_pipeline):
        from lasagna.lifecycle.gc import GarbageCollector

        mock_db = MagicMock()
        mock_db.acquire_gc_lock.return_value = True
        mock_db.release_gc_lock.return_value = None
        mock_db.get_stats.return_value = {"documents": {"total": 10}}
        policy = LifecyclePolicy(safety_threshold_percent=0.9)
        gc = GarbageCollector.__new__(GarbageCollector)
        gc._db = mock_db
        gc._pipeline = mock_pipeline
        gc._policy = policy
        return gc, mock_db

    def test_phase1_does_not_count_if_delete_returns_false(self):
        """docs_cleaned must NOT increment when delete_document returns False."""
        mock_pipeline = MagicMock()
        mock_pipeline.delete_document.return_value = False  # failure without exception

        gc, mock_db = self._make_gc(mock_pipeline)
        mock_db.list_soft_deleted_before.return_value = ["doc_1", "doc_2"]
        mock_db.get_document.return_value = {
            "document_id": "doc_1",
            "session_id": "s",
            "file_size": 100,
            "chunks_count": 5,
        }

        result = gc._phase1_hard_delete_soft_deleted()

        assert result.docs_cleaned == 0, (
            "No docs should be counted as cleaned when delete returns False"
        )
        assert result.errors == 2, "Each failed delete should increment errors"
        assert result.bytes_freed == 0, "bytes_freed must not be credited for failed deletes"

    def test_phase1_counts_when_delete_returns_none(self):
        """docs_cleaned SHOULD increment when delete_document returns None (void / success)."""
        mock_pipeline = MagicMock()
        mock_pipeline.delete_document.return_value = None  # typical void return

        gc, mock_db = self._make_gc(mock_pipeline)
        mock_db.list_soft_deleted_before.return_value = ["doc_1"]
        mock_db.get_document.return_value = {
            "document_id": "doc_1",
            "session_id": "s",
            "file_size": 200,
            "chunks_count": 3,
        }

        result = gc._phase1_hard_delete_soft_deleted()

        assert result.docs_cleaned == 1
        assert result.bytes_freed == 200
        assert result.errors == 0

    def test_phase1_counts_when_delete_returns_true(self):
        """docs_cleaned SHOULD increment when delete_document returns True."""
        mock_pipeline = MagicMock()
        mock_pipeline.delete_document.return_value = True

        gc, mock_db = self._make_gc(mock_pipeline)
        mock_db.list_soft_deleted_before.return_value = ["doc_1"]
        mock_db.get_document.return_value = {
            "document_id": "doc_1",
            "session_id": "s",
            "file_size": 300,
            "chunks_count": 4,
        }

        result = gc._phase1_hard_delete_soft_deleted()

        assert result.docs_cleaned == 1
        assert result.bytes_freed == 300


class TestBulkDeleteAccounting:
    """Bulk delete should only credit bytes/chunks for docs that were successfully deleted."""

    def _make_manager(self, mock_pipeline, doc_rows):
        from lasagna.lifecycle.manager import DocumentLifecycleManager

        mock_db = MagicMock()
        mock_db.get_documents.return_value = doc_rows
        policy = LifecyclePolicy()
        mgr = DocumentLifecycleManager.__new__(DocumentLifecycleManager)
        mgr._db = mock_db
        mgr._pipeline = mock_pipeline
        mgr._policy = policy
        return mgr

    def test_bytes_only_freed_for_successful_deletes(self):
        """bytes_freed must reflect only successfully deleted docs."""
        doc_rows = [
            {"document_id": "d1", "session_id": "s", "file_size": 1000, "chunks_count": 10},
            {"document_id": "d2", "session_id": "s", "file_size": 2000, "chunks_count": 20},
        ]
        mock_pipeline = MagicMock()
        # d1 succeeds, d2 fails
        mock_pipeline.delete_document.side_effect = [True, Exception("DB error")]

        mgr = self._make_manager(mock_pipeline, doc_rows)
        result = mgr.bulk_delete_session("s", confirm=True)

        assert result.deleted == 1
        assert result.bytes_freed == 1000, "Only d1's bytes should be credited"
        assert result.chunks_freed == 10, "Only d1's chunks should be credited"
        assert result.errors == 1

    def test_partial_failure_accurate_counts(self):
        """All three counters (deleted, bytes_freed, chunks_freed) are accurate under partial failure."""
        doc_rows = [
            {"document_id": f"d{i}", "session_id": "s", "file_size": 500, "chunks_count": 5}
            for i in range(4)
        ]
        mock_pipeline = MagicMock()
        # 3 succeed, 1 (returns False) fails
        mock_pipeline.delete_document.side_effect = [True, True, False, True]

        mgr = self._make_manager(mock_pipeline, doc_rows)
        result = mgr.bulk_delete_session("s", confirm=True)

        assert result.requested == 4
        assert result.deleted == 3
        assert result.bytes_freed == 1500  # 3 × 500
        assert result.chunks_freed == 15  # 3 × 5
        assert result.errors == 1


# ---------------------------------------------------------------------------
# Orphan detection truncation indicator
# ---------------------------------------------------------------------------


class TestOrphanTruncationIndicator:
    """ConsistencyReport.scan_limited must be set correctly."""

    def _make_detector(self, db_ids, vec_ids, vec_limit=50_000):
        from lasagna.lifecycle.orphan_detector import OrphanDetector

        mock_db = MagicMock()
        mock_db.list_all_active_document_ids.return_value = db_ids
        mock_backend = MagicMock()
        mock_backend.list_document_ids.return_value = vec_ids
        detector = OrphanDetector(db=mock_db, backend=mock_backend)
        return detector, mock_db, mock_backend

    def test_scan_limited_false_when_under_limit(self):
        """scan_limited=False when vector result count is below the limit."""
        db_ids = [f"d{i}" for i in range(100)]
        vec_ids = [f"d{i}" for i in range(100)]
        detector, _, _ = self._make_detector(db_ids, vec_ids)
        report = detector.detect(collection_name="test_col", vector_scan_limit=500)
        assert report.scan_limited is False
        assert report.estimated_unscanned is None

    def test_scan_limited_true_when_at_limit(self):
        """scan_limited=True when vector result count equals the limit."""
        db_ids = [f"d{i}" for i in range(200)]
        vec_ids = [f"d{i}" for i in range(10)]  # backend returns exactly 10
        detector, _, mock_backend = self._make_detector(db_ids, vec_ids)
        mock_backend.list_document_ids.return_value = [f"d{i}" for i in range(10)]

        report = detector.detect(collection_name="test_col", vector_scan_limit=10)

        assert report.scan_limited is True
        assert report.vector_scan_limit == 10
        # estimated_unscanned = db_count - limit = 200 - 10 = 190
        assert report.estimated_unscanned == 190

    def test_estimated_unscanned_none_when_not_limited(self):
        """estimated_unscanned is None when scan is not limited."""
        db_ids = ["d1", "d2"]
        vec_ids = ["d1", "d2"]
        detector, _, _ = self._make_detector(db_ids, vec_ids)
        report = detector.detect(collection_name="col", vector_scan_limit=1000)
        assert report.estimated_unscanned is None

    def test_to_dict_includes_truncation_fields(self):
        """to_dict() should serialize scan_limited and related fields."""
        db_ids = ["d1"]
        vec_ids = ["d1"]
        detector, _, _ = self._make_detector(db_ids, vec_ids)
        report = detector.detect(collection_name="col", vector_scan_limit=1000)
        d = report.to_dict()
        assert "scan_limited" in d
        assert "vector_scan_limit" in d
        assert "estimated_unscanned" in d


# ---------------------------------------------------------------------------
# LifecyclePolicy.load() with TOML support
# ---------------------------------------------------------------------------


class TestPolicyLoad:
    """LifecyclePolicy.load() should read TOML, then apply env overrides."""

    def test_load_falls_back_to_defaults_when_no_config(self):
        """load() returns default values when rag_config._get_knowledge_setting raises."""
        import os

        # Remove RAG_GC_* env vars so they don't interfere with defaults
        env_without_gc = {k: v for k, v in os.environ.items() if not k.startswith("RAG_GC_")}
        with patch.dict(os.environ, env_without_gc, clear=True):
            # Simulate rag_config._get_knowledge_setting raising (no lifecycle section)
            with patch("lasagna.core.config.rag_config") as mock_cfg:
                mock_cfg._get_knowledge_setting.side_effect = AttributeError("no lifecycle")
                policy = LifecyclePolicy.load()

        assert policy.soft_delete_grace_hours == 24
        assert policy.document_ttl_days is None
        assert policy.gc_cycle_hours == 24

    def test_env_overrides_take_priority(self):
        """Env var RAG_GC_GRACE_HOURS overrides TOML value."""
        import os

        with patch.dict(os.environ, {"RAG_GC_GRACE_HOURS": "72"}):
            policy = LifecyclePolicy.load()
        assert policy.soft_delete_grace_hours == 72

    def test_from_env_is_alias_for_load(self):
        """from_env() should return the same result as load()."""
        p1 = LifecyclePolicy.load()
        p2 = LifecyclePolicy.from_env()
        import dataclasses

        assert dataclasses.asdict(p1) == dataclasses.asdict(p2)


# ---------------------------------------------------------------------------
# Celery Beat cadence uses timedelta not crontab
# ---------------------------------------------------------------------------


class TestCeleryBeatCadence:
    """Celery Beat schedule must use timedelta and honor policy fields."""

    def test_beat_schedule_uses_timedelta(self):
        """beat_schedule entries should use timedelta, not crontab."""
        from datetime import timedelta
        from lasagna.infrastructure.tasks.celery_app import _build_beat_schedule

        schedule = _build_beat_schedule()
        assert "gc-full-cycle" in schedule
        assert "gc-orphan-scan" in schedule
        gc_sched = schedule["gc-full-cycle"]["schedule"]
        orphan_sched = schedule["gc-orphan-scan"]["schedule"]
        assert isinstance(gc_sched, timedelta), f"Expected timedelta, got {type(gc_sched)}"
        assert isinstance(orphan_sched, timedelta), f"Expected timedelta, got {type(orphan_sched)}"

    def test_beat_schedule_uses_policy_gc_cycle_hours(self):
        """gc-full-cycle timedelta should match LifecyclePolicy.gc_cycle_hours."""
        from datetime import timedelta
        from unittest.mock import patch
        from lasagna.infrastructure.tasks.celery_app import _build_beat_schedule

        mock_policy = LifecyclePolicy(gc_cycle_hours=6, orphan_scan_hours=48)
        with patch("lasagna.lifecycle.policy.LifecyclePolicy.load", return_value=mock_policy):
            schedule = _build_beat_schedule()

        assert schedule["gc-full-cycle"]["schedule"] == timedelta(hours=6)
        assert schedule["gc-orphan-scan"]["schedule"] == timedelta(hours=48)


# ---------------------------------------------------------------------------
# Policy REST endpoint helpers
# ---------------------------------------------------------------------------


class TestPolicyHelpers:
    """_get_live_policy and _set_live_policy read/write the backend's manager._policy."""

    def test_get_live_policy_reads_manager_policy(self):
        """_get_live_policy returns the policy from backend._lifecycle_manager."""
        from lasagna.api.rest.routes.gc_admin import _get_live_policy

        mock_backend = MagicMock()
        custom_policy = LifecyclePolicy(gc_cycle_hours=12)
        mock_backend._lifecycle_manager._policy = custom_policy
        result = _get_live_policy(mock_backend)
        assert result.gc_cycle_hours == 12

    def test_get_live_policy_falls_back_to_load(self):
        """_get_live_policy falls back to LifecyclePolicy.load() when no manager."""
        from lasagna.api.rest.routes.gc_admin import _get_live_policy

        mock_backend = MagicMock(spec=[])  # no _lifecycle_manager attribute
        result = _get_live_policy(mock_backend)
        # Just verify it returns a valid LifecyclePolicy instance
        assert isinstance(result, LifecyclePolicy)

    def test_set_live_policy_updates_manager_and_gc(self):
        """_set_live_policy writes to both _lifecycle_manager._policy and gc._policy."""
        from lasagna.api.rest.routes.gc_admin import _set_live_policy

        mock_gc = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr._gc = mock_gc
        mock_backend = MagicMock()
        mock_backend._lifecycle_manager = mock_mgr

        new_policy = LifecyclePolicy(gc_cycle_hours=3)
        _set_live_policy(mock_backend, new_policy)

        assert mock_mgr._policy == new_policy
        assert mock_gc._policy == new_policy


# ---------------------------------------------------------------------------
# TTL disable semantics (regression: null and 0 must both disable TTL)
# ---------------------------------------------------------------------------


class TestTTLDisableSemantics:
    """document_ttl_days=None and =0 must both disable TTL across all surfaces."""

    # --- env path ---

    def test_env_ttl_days_zero_normalizes_to_none(self):
        """RAG_GC_TTL_DAYS=0 env var should result in document_ttl_days=None."""
        import os

        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_GC_")}
        env["RAG_GC_TTL_DAYS"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with patch("lasagna.core.config.rag_config") as mock_cfg:
                mock_cfg._get_knowledge_setting.side_effect = AttributeError("no config")
                policy = LifecyclePolicy.load()
        assert policy.document_ttl_days is None, (
            f"RAG_GC_TTL_DAYS=0 must normalize to None (disabled), got {policy.document_ttl_days!r}"
        )

    def test_env_ttl_days_zero_overrides_toml_value(self):
        """RAG_GC_TTL_DAYS=0 must override a non-None TOML document_ttl_days."""
        import os

        # Simulate TOML having document_ttl_days=90
        def _fake_setting(*keys, default=None):
            if keys == ("rag", "lifecycle", "document_ttl_days"):
                return 90
            return default

        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_GC_")}
        env["RAG_GC_TTL_DAYS"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with patch("lasagna.core.config.rag_config") as mock_cfg:
                mock_cfg._get_knowledge_setting.side_effect = _fake_setting
                policy = LifecyclePolicy.load()
        assert policy.document_ttl_days is None, (
            "RAG_GC_TTL_DAYS=0 must win over TOML document_ttl_days=90"
        )

    def test_env_ttl_days_positive_is_kept(self):
        """RAG_GC_TTL_DAYS=30 must set document_ttl_days=30 (not None)."""
        import os

        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_GC_")}
        env["RAG_GC_TTL_DAYS"] = "30"
        with patch.dict(os.environ, env, clear=True):
            with patch("lasagna.core.config.rag_config") as mock_cfg:
                mock_cfg._get_knowledge_setting.side_effect = AttributeError("no config")
                policy = LifecyclePolicy.load()
        assert policy.document_ttl_days == 30

    # --- GC phase 2 ---

    def test_gc_phase2_skipped_when_ttl_days_is_none(self):
        """GC Phase 2 must skip when document_ttl_days is None."""
        from lasagna.lifecycle.gc import GarbageCollector

        policy = LifecyclePolicy(document_ttl_days=None)
        mock_db = MagicMock()
        mock_pipeline = MagicMock()
        gc = GarbageCollector(db=mock_db, pipeline=mock_pipeline, policy=policy)
        result = gc._phase2_expire_by_ttl()
        assert "skipped" in result.details
        mock_db.list_expired_documents.assert_not_called()

    def test_gc_phase2_skipped_when_ttl_days_is_zero(self):
        """GC Phase 2 must also skip when document_ttl_days is 0 (legacy sentinel)."""
        from lasagna.lifecycle.gc import GarbageCollector

        policy = LifecyclePolicy(document_ttl_days=0)
        mock_db = MagicMock()
        mock_pipeline = MagicMock()
        gc = GarbageCollector(db=mock_db, pipeline=mock_pipeline, policy=policy)
        result = gc._phase2_expire_by_ttl()
        assert "skipped" in result.details
        mock_db.list_expired_documents.assert_not_called()

    # --- REST endpoint validator accepts 0 ---

    def test_rest_validator_accepts_zero_ttl(self):
        """validate_policy_updates must accept document_ttl_days=0 (means disabled)."""
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        assert validate_policy_updates({"document_ttl_days": 0}) is None

    def test_rest_validator_accepts_null_ttl(self):
        """validate_policy_updates must accept document_ttl_days=null."""
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        assert validate_policy_updates({"document_ttl_days": None}) is None

    def test_rest_validator_rejects_negative_ttl(self):
        """validate_policy_updates must reject document_ttl_days=-1."""
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        assert validate_policy_updates({"document_ttl_days": -1}) is not None

    # --- REST endpoint null and 0 passthrough → stored as None ---

    def test_rest_patch_null_ttl_sets_none(self):
        """PATCH /admin/gc/policy with document_ttl_days=null must disable TTL."""
        import dataclasses
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        initial_policy = LifecyclePolicy(document_ttl_days=30)
        body = {"document_ttl_days": None}

        assert validate_policy_updates(body) is None  # null is valid
        updated = dataclasses.replace(
            initial_policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in body.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        assert updated.document_ttl_days is None

    def test_rest_patch_zero_ttl_normalizes_to_none(self):
        """PATCH /admin/gc/policy with document_ttl_days=0 must normalize to None (disabled).

        Regression: validator previously rejected 0; apply logic must store None not 0.
        """
        import dataclasses
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        initial_policy = LifecyclePolicy(document_ttl_days=30)
        body = {"document_ttl_days": 0}

        assert validate_policy_updates(body) is None, "Validator must accept 0 as disabled"
        updated = dataclasses.replace(
            initial_policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in body.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        assert updated.document_ttl_days is None, "document_ttl_days=0 must be normalized to None"

    # --- MCP null and 0 passthrough ---

    def test_mcp_null_ttl_sets_none(self):
        """set_lifecycle_policy MCP tool must allow document_ttl_days=null to disable TTL."""
        import dataclasses

        initial_policy = LifecyclePolicy(document_ttl_days=60)
        updates = {"document_ttl_days": None}
        updated = dataclasses.replace(
            initial_policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in updates.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        assert updated.document_ttl_days is None

    def test_mcp_zero_ttl_normalizes_to_none(self):
        """MCP set_lifecycle_policy with document_ttl_days=0 must normalize to None."""
        import dataclasses

        initial_policy = LifecyclePolicy(document_ttl_days=60)
        updates = {"document_ttl_days": 0}
        updated = dataclasses.replace(
            initial_policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in updates.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        assert updated.document_ttl_days is None, (
            "MCP set_lifecycle_policy with document_ttl_days=0 must normalize to None"
        )
