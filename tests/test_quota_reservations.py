"""Quota reservation correctness tests.

Validates the reserve → finalize/release lifecycle, with emphasis on:
- Tagged reservation IDs for correct rollback order.
- No-op semantics for unknown/empty reservation IDs.
- Idempotent double-release.
- Async (Celery) dispatch path forwarding reservation params.
- Worker-path lifecycle (finalize on success, release on failure, hold on retry).
- Upload pipeline reservation ordering (reserve before I/O, rollback on failure).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lasagna.core.quota import (
    QuotaExceededError,
    QuotaManager,
    TenantQuota,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qm():
    """Enabled QuotaManager with tight limits for testing."""
    return QuotaManager(
        enabled=True,
        default_quota=TenantQuota(
            max_documents=10,
            max_storage_bytes=10_000,
            max_uploads_per_hour=5,
            max_searches_per_minute=10,
            max_chunks=500,
        ),
    )


# =========================================================================
# Reserve → release with correct ID
# =========================================================================


class TestReservationTagging:
    """Verify reservation_id tagging for correct rollback."""

    def test_reserve_a_b_release_a_keeps_b(self, qm):
        """Reserve A then B, release A → B's timestamp survives."""
        rid_a = qm.check_and_reserve_upload("s1", 100)
        rid_b = qm.check_and_reserve_upload("s1", 200)

        assert rid_a != rid_b
        usage = qm.get_usage("s1")
        assert usage["documents"] == 2
        assert usage["storage_bytes"] == 300

        qm.release_upload_reservation("s1", 100, reservation_id=rid_a)

        usage = qm.get_usage("s1")
        assert usage["documents"] == 1
        assert usage["storage_bytes"] == 200

        # Verify B's entry is still tracked
        with qm._lock:
            u = qm._get_usage_locked("s1")
            assert rid_b in u._upload_timestamps
            assert rid_a not in u._upload_timestamps

    def test_reserve_a_b_release_b_keeps_a(self, qm):
        """Reserve A then B, release B → A's timestamp survives."""
        rid_a = qm.check_and_reserve_upload("s1", 100)
        rid_b = qm.check_and_reserve_upload("s1", 200)

        qm.release_upload_reservation("s1", 200, reservation_id=rid_b)

        with qm._lock:
            u = qm._get_usage_locked("s1")
            assert rid_a in u._upload_timestamps
            assert rid_b not in u._upload_timestamps

    def test_finalize_records_chunks(self, qm):
        """Finalize adds chunks to usage after reserve."""
        _rid = qm.check_and_reserve_upload("s1", 500)
        qm.finalize_upload_reservation("s1", chunks=42)

        usage = qm.get_usage("s1")
        assert usage["documents"] == 1
        assert usage["chunks"] == 42


# =========================================================================
# No-op semantics for unknown/empty reservation IDs
# =========================================================================


class TestReleaseNoOp:
    """Unknown or empty reservation_id must not decrement counters."""

    def test_release_unknown_id_is_noop(self, qm):
        """Releasing an unknown reservation_id does NOT decrement counters."""
        _rid = qm.check_and_reserve_upload("s1", 500)

        # Release with a fake ID
        qm.release_upload_reservation("s1", 500, reservation_id="fake_id_123")

        usage = qm.get_usage("s1")
        assert usage["documents"] == 1, "documents should NOT be decremented"
        assert usage["storage_bytes"] == 500, "storage should NOT be decremented"

    def test_release_empty_id_is_noop(self, qm):
        """Releasing with empty reservation_id does NOT decrement counters."""
        _rid = qm.check_and_reserve_upload("s1", 500)

        qm.release_upload_reservation("s1", 500, reservation_id="")

        usage = qm.get_usage("s1")
        assert usage["documents"] == 1
        assert usage["storage_bytes"] == 500

    def test_double_release_same_id_is_idempotent(self, qm):
        """Releasing the same reservation_id twice: first succeeds, second is no-op."""
        rid = qm.check_and_reserve_upload("s1", 500)

        qm.release_upload_reservation("s1", 500, reservation_id=rid)
        usage = qm.get_usage("s1")
        assert usage["documents"] == 0
        assert usage["storage_bytes"] == 0

        # Second release — should be no-op, not underflow
        qm.release_upload_reservation("s1", 500, reservation_id=rid)
        usage = qm.get_usage("s1")
        assert usage["documents"] == 0, "documents should not go negative"
        assert usage["storage_bytes"] == 0, "storage should not go negative"

    def test_release_without_prior_reserve_is_noop(self, qm):
        """Releasing on a session with no reservations is a no-op."""
        qm.release_upload_reservation("s1", 500, reservation_id="nonexistent")
        usage = qm.get_usage("s1")
        assert usage["documents"] == 0
        assert usage["storage_bytes"] == 0


# =========================================================================
# Quota enforcement with reservation
# =========================================================================


class TestReservationQuotaEnforcement:
    """Reservations count toward quota limits."""

    def test_reservation_blocks_when_full(self):
        """Reserving up to max_documents, next reservation should fail."""
        qm = QuotaManager(
            enabled=True,
            default_quota=TenantQuota(
                max_documents=5,
                max_storage_bytes=100_000,
                max_uploads_per_hour=20,  # high enough to not interfere
            ),
        )
        rids = []
        for _ in range(5):  # max_documents=5
            rids.append(qm.check_and_reserve_upload("s1", 100))

        with pytest.raises(QuotaExceededError, match="max_documents"):
            qm.check_and_reserve_upload("s1", 100)

        # Release one, then reserve should succeed
        qm.release_upload_reservation("s1", 100, reservation_id=rids[0])
        rid_new = qm.check_and_reserve_upload("s1", 100)
        assert rid_new != ""

    def test_hourly_upload_limit_with_reservations(self, qm):
        """max_uploads_per_hour enforced via reservation timestamps."""
        rids = []
        for _ in range(5):  # max_uploads_per_hour=5
            rids.append(qm.check_and_reserve_upload("s1", 0))

        with pytest.raises(QuotaExceededError, match="max_uploads_per_hour"):
            qm.check_and_reserve_upload("s1", 0)


# =========================================================================
# Deque resize on quota update
# =========================================================================


class TestDequeResize:
    """set_tenant_quota() resizes existing search deques."""

    def test_resize_on_quota_increase(self, qm):
        # Trigger usage creation
        qm.record_search("s1")

        with qm._lock:
            u = qm._get_usage_locked("s1")
            _old_maxlen = u._search_timestamps.maxlen

        # Raise quota
        qm.set_tenant_quota("s1", TenantQuota(max_searches_per_minute=2000))

        with qm._lock:
            u = qm._get_usage_locked("s1")
            assert u._search_timestamps.maxlen == 4000
            # Existing data preserved
            assert len(u._search_timestamps) == 1


# =========================================================================
# Async dispatch integration (mocked Celery)
# =========================================================================


class TestAsyncDispatchReservation:
    """Verify reservation params are forwarded to Celery tasks."""

    def test_dispatch_forwards_reservation_params(self):
        """dispatch_processing passes reservation_id and reserved_file_size."""
        from lasagna.infrastructure.tasks import document_tasks

        mock_task = MagicMock()
        with (
            patch.object(document_tasks, "process_document_task", mock_task),
            patch("lasagna.core.config.rag_config") as mock_config,
        ):
            mock_config.celery_enabled = True

            result = document_tasks.dispatch_processing(
                "doc_123",
                "sess_abc",
                "/tmp/file.pdf",
                reservation_id="res_xyz",
                reserved_file_size=1024,
            )

            assert result is True
            mock_task.delay.assert_called_once_with(
                "doc_123",
                "sess_abc",
                "/tmp/file.pdf",
                reservation_id="res_xyz",
                reserved_file_size=1024,
            )

    def test_dispatch_failure_returns_false(self):
        """When Celery dispatch fails and async_processing disabled, returns False."""
        from lasagna.infrastructure.tasks import document_tasks

        mock_task = MagicMock()
        mock_task.delay.side_effect = ConnectionError("Celery down")
        with (
            patch.object(document_tasks, "process_document_task", mock_task),
            patch("lasagna.core.config.rag_config") as mock_config,
        ):
            mock_config.celery_enabled = True
            # Disable async processing so thread pool tier is skipped
            mock_config.async_processing = False

            result = document_tasks.dispatch_processing(
                "doc_123",
                "sess_abc",
                "/tmp/file.pdf",
                reservation_id="res_xyz",
                reserved_file_size=1024,
            )

            assert result is False


# =========================================================================
# Worker-path lifecycle tests (_process_document_impl)
# =========================================================================


class TestWorkerReservationLifecycle:
    """Verify _process_document_impl finalize/release/retry behavior.

    Tests invoke the real _process_document_impl() function with mocked
    dependencies — no simulation of individual method calls.
    """

    @staticmethod
    def _mock_backend(chunks_count=10, file_size=500, status_value="ready"):
        """Create a mock RAG backend that succeeds."""
        mock_doc = MagicMock()
        mock_doc.chunks_count = chunks_count
        mock_doc.file_size = file_size
        mock_doc.status.value = status_value
        mock_backend = MagicMock()
        mock_backend._get_document_metadata.return_value = mock_doc
        mock_backend.get_session_config.return_value = MagicMock(enable_search_cache=False)
        return mock_backend

    def test_success_with_reservation_calls_finalize(self):
        """On success with reservation_id, finalize is called (not record_upload)."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _process_document_impl,
        )

        mock_backend = self._mock_backend()
        mock_quota = MagicMock()

        with (
            patch("lasagna.api.rag_backend.get_rag_backend", return_value=mock_backend),
            patch("lasagna.core.quota.get_quota_manager", return_value=mock_quota),
        ):
            result = _process_document_impl(
                "doc_1",
                "sess_1",
                "/tmp/f.pdf",
                reservation_id="res_123",
                reserved_file_size=500,
            )

        assert result["document_id"] == "doc_1"
        assert result["chunks_count"] == 10
        mock_quota.finalize_upload_reservation.assert_called_once_with("sess_1", 10)
        mock_quota.record_upload.assert_not_called()

    def test_success_without_reservation_calls_record_upload(self):
        """On success without reservation_id, record_upload is called (legacy path)."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _process_document_impl,
        )

        mock_backend = self._mock_backend()
        mock_quota = MagicMock()

        with (
            patch("lasagna.api.rag_backend.get_rag_backend", return_value=mock_backend),
            patch("lasagna.core.quota.get_quota_manager", return_value=mock_quota),
        ):
            result = _process_document_impl(
                "doc_1",
                "sess_1",
                "/tmp/f.pdf",
                reservation_id="",
                reserved_file_size=0,
            )

        assert result["document_id"] == "doc_1"
        mock_quota.record_upload.assert_called_once_with(
            session_id="sess_1",
            file_size_bytes=500,
            chunks=10,
        )
        mock_quota.finalize_upload_reservation.assert_not_called()

    def test_terminal_failure_with_reservation_calls_release(self):
        """On terminal failure (retries exhausted), release is called with exact ID."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _process_document_impl,
            TaskRetryContext,
        )

        mock_backend = MagicMock()
        mock_backend._get_document_metadata.return_value = None  # triggers ValueError

        mock_quota = MagicMock()
        mock_db = MagicMock()

        # retries == max_retries → terminal failure
        ctx = TaskRetryContext(retries=2, max_retries=2, retry_fn=None)

        with (
            patch("lasagna.api.rag_backend.get_rag_backend", return_value=mock_backend),
            patch("lasagna.core.quota.get_quota_manager", return_value=mock_quota),
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
        ):
            with pytest.raises(ValueError, match="not found in database"):
                _process_document_impl(
                    "doc_1",
                    "sess_1",
                    "/tmp/f.pdf",
                    reservation_id="res_456",
                    reserved_file_size=1024,
                    retry_ctx=ctx,
                )

        mock_quota.release_upload_reservation.assert_called_once_with(
            "sess_1",
            1024,
            reservation_id="res_456",
        )
        mock_quota.finalize_upload_reservation.assert_not_called()

    def test_terminal_failure_without_reservation_skips_release(self):
        """On terminal failure without reservation_id, release is NOT called."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _process_document_impl,
            TaskRetryContext,
        )

        mock_backend = MagicMock()
        mock_backend._get_document_metadata.return_value = None

        mock_quota = MagicMock()
        mock_db = MagicMock()

        ctx = TaskRetryContext(retries=2, max_retries=2, retry_fn=None)

        with (
            patch("lasagna.api.rag_backend.get_rag_backend", return_value=mock_backend),
            patch("lasagna.core.quota.get_quota_manager", return_value=mock_quota),
            patch("lasagna.core.database.get_rag_database", return_value=mock_db),
        ):
            with pytest.raises(ValueError, match="not found in database"):
                _process_document_impl(
                    "doc_1",
                    "sess_1",
                    "/tmp/f.pdf",
                    reservation_id="",
                    reserved_file_size=0,
                    retry_ctx=ctx,
                )

        mock_quota.release_upload_reservation.assert_not_called()

    def test_retry_path_does_not_release(self):
        """When retries remain, reservation is NOT released (held for retry)."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _process_document_impl,
            TaskRetryContext,
            _RetryRequested,
        )

        mock_backend = MagicMock()
        mock_backend._get_document_metadata.return_value = None

        mock_quota = MagicMock()

        retry_exc = _RetryRequested("retry-signal")
        ctx = TaskRetryContext(
            retries=0,
            max_retries=2,
            retry_fn=lambda exc: retry_exc,
        )

        with (
            patch("lasagna.api.rag_backend.get_rag_backend", return_value=mock_backend),
            patch("lasagna.core.quota.get_quota_manager", return_value=mock_quota),
        ):
            with pytest.raises(_RetryRequested):
                _process_document_impl(
                    "doc_1",
                    "sess_1",
                    "/tmp/f.pdf",
                    reservation_id="res_789",
                    reserved_file_size=500,
                    retry_ctx=ctx,
                )

        # Reservation held — neither finalize nor release
        mock_quota.finalize_upload_reservation.assert_not_called()
        mock_quota.release_upload_reservation.assert_not_called()


# =========================================================================
# Upload pipeline reservation ordering tests
# =========================================================================


class TestUploadPipelineReservationOrdering:
    """Verify reservation ordering and rollback by calling upload_document()
    with mocked collaborators.
    """

    @staticmethod
    def _make_pipeline(store_file_effect=None, save_metadata_effect=None):
        """Create a DocumentPipeline with mocked collaborators."""
        from lasagna.api._document_pipeline import DocumentPipeline

        pipeline = DocumentPipeline(
            document_processor=MagicMock(),
            embedding_manager=MagicMock(),
            chunker=MagicMock(),
            hybrid_searcher=MagicMock(),
            collection_manager=MagicMock(),
            storage=MagicMock(),
        )
        # Mock internal methods so upload_document doesn't do real I/O
        pipeline._store_file = MagicMock(return_value="/stored/path.pdf")
        pipeline._save_document_metadata = MagicMock()
        pipeline.process_document = MagicMock()
        pipeline._get_document_metadata = MagicMock(
            return_value=MagicMock(
                status=MagicMock(value="ready"),
                chunks_count=5,
            )
        )
        pipeline._collection_manager.maybe_reload_embedding_manager = MagicMock()
        pipeline._collection_manager.resolve_backend.return_value = MagicMock(backend_name="milvus")

        if store_file_effect:
            pipeline._store_file.side_effect = store_file_effect
        if save_metadata_effect:
            pipeline._save_document_metadata.side_effect = save_metadata_effect

        return pipeline

    def test_reservation_failure_prevents_io(self):
        """When check_and_reserve_upload raises, _store_file is never called."""
        import tempfile
        import os

        pipeline = self._make_pipeline()

        mock_quota = MagicMock()
        mock_quota.check_and_reserve_upload.side_effect = QuotaExceededError(
            "max_documents",
            limit=10,
            current=10,
            session_id="sess_1",
        )

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            with (
                patch("lasagna.api._document_pipeline.get_quota_manager", return_value=mock_quota),
                patch("lasagna.api._document_pipeline.dispatch_processing", return_value=False),
            ):
                pipeline.document_processor.detect_file_type.return_value = ".txt"
                pipeline._get_session_config = MagicMock(
                    return_value=MagicMock(
                        max_file_size=10_000_000,
                        supported_extensions=[".txt"],
                        enable_search_cache=False,
                    )
                )

                with pytest.raises(QuotaExceededError):
                    pipeline.upload_document(tmp_path, "sess_1", filename="test.txt")

            pipeline._store_file.assert_not_called()
            pipeline._save_document_metadata.assert_not_called()
        finally:
            os.unlink(tmp_path)

    def test_store_file_failure_releases_reservation(self):
        """If _store_file throws after reservation, release is called."""
        import tempfile
        import os

        pipeline = self._make_pipeline(
            store_file_effect=IOError("disk full"),
        )

        mock_quota = MagicMock()
        mock_quota.check_and_reserve_upload.return_value = "res_abc"

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            with (
                patch("lasagna.api._document_pipeline.get_quota_manager", return_value=mock_quota),
                patch("lasagna.api._document_pipeline.dispatch_processing", return_value=False),
            ):
                pipeline.document_processor.detect_file_type.return_value = ".txt"
                pipeline._get_session_config = MagicMock(
                    return_value=MagicMock(
                        max_file_size=10_000_000,
                        supported_extensions=[".txt"],
                        enable_search_cache=False,
                    )
                )

                with pytest.raises(IOError, match="disk full"):
                    pipeline.upload_document(tmp_path, "sess_1", filename="test.txt")

            # Reservation must be released
            mock_quota.release_upload_reservation.assert_called_once()
            args, kwargs = mock_quota.release_upload_reservation.call_args
            assert kwargs.get("reservation_id") == "res_abc" or args[2:] == ("res_abc",)

            # Metadata save should not have been attempted
            pipeline._save_document_metadata.assert_not_called()
        finally:
            os.unlink(tmp_path)

    def test_metadata_failure_deletes_file_and_releases_reservation(self):
        """If _save_document_metadata throws, stored file is deleted and reservation released."""
        import tempfile
        import os

        pipeline = self._make_pipeline(
            save_metadata_effect=RuntimeError("DB down"),
        )

        mock_quota = MagicMock()
        mock_quota.check_and_reserve_upload.return_value = "res_def"

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            with (
                patch("lasagna.api._document_pipeline.get_quota_manager", return_value=mock_quota),
                patch("lasagna.api._document_pipeline.dispatch_processing", return_value=False),
            ):
                pipeline.document_processor.detect_file_type.return_value = ".txt"
                pipeline._get_session_config = MagicMock(
                    return_value=MagicMock(
                        max_file_size=10_000_000,
                        supported_extensions=[".txt"],
                        enable_search_cache=False,
                    )
                )

                with pytest.raises(RuntimeError, match="DB down"):
                    pipeline.upload_document(tmp_path, "sess_1", filename="test.txt")

            # Stored file must be cleaned up
            pipeline.storage.delete.assert_called_once_with("/stored/path.pdf")

            # Reservation must be released
            mock_quota.release_upload_reservation.assert_called_once()
            args, kwargs = mock_quota.release_upload_reservation.call_args
            assert kwargs.get("reservation_id") == "res_def" or args[2:] == ("res_def",)
        finally:
            os.unlink(tmp_path)

    def test_call_ordering_reserve_before_store_before_metadata(self):
        """Verify actual call ordering: reserve → store → metadata (by mock call order)."""
        import tempfile
        import os

        call_log = []

        pipeline = self._make_pipeline()
        pipeline._store_file.side_effect = lambda *a, **kw: (
            call_log.append("store_file"),
            "/stored/path.pdf",
        )[1]
        pipeline._save_document_metadata.side_effect = lambda *a, **kw: call_log.append(
            "save_metadata"
        )

        mock_quota = MagicMock()

        def _reserve(*a, **kw):
            call_log.append("reserve")
            return "res_order"

        mock_quota.check_and_reserve_upload.side_effect = _reserve

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            tmp_path = f.name

        try:
            with (
                patch("lasagna.api._document_pipeline.get_quota_manager", return_value=mock_quota),
                patch("lasagna.api._document_pipeline.dispatch_processing", return_value=False),
                patch("lasagna.api._document_pipeline.TRACER") as mock_tracer,
                patch("lasagna.api._document_pipeline.get_search_cache"),
            ):
                mock_tracer.span.return_value.__enter__ = MagicMock()
                mock_tracer.span.return_value.__exit__ = MagicMock(return_value=False)

                pipeline.document_processor.detect_file_type.return_value = ".txt"
                pipeline._get_session_config = MagicMock(
                    return_value=MagicMock(
                        max_file_size=10_000_000,
                        supported_extensions=[".txt"],
                        enable_search_cache=False,
                    )
                )

                pipeline.upload_document(tmp_path, "sess_1", filename="test.txt")

            assert call_log.index("reserve") < call_log.index("store_file"), (
                f"reserve must happen before store_file; got: {call_log}"
            )
            assert call_log.index("store_file") < call_log.index("save_metadata"), (
                f"store_file must happen before save_metadata; got: {call_log}"
            )
        finally:
            os.unlink(tmp_path)
