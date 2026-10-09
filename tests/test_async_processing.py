"""Unit tests for async document processing (ThreadPoolExecutor dispatch).

Tests the three-tier dispatch logic, backpressure, retry behaviour,
and structured JSON response contract.  All heavy dependencies are
mocked so these tests run without Milvus, sentence-transformers, etc.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Dispatch tiering
# ---------------------------------------------------------------------------


class TestDispatchTiering:
    """Verify the Celery → ThreadPool → Sync fallback order."""

    def test_celery_dispatch_when_enabled(self):
        """Tier 1: Celery dispatches when enabled and broker is reachable."""
        from lasagna.infrastructure.tasks.document_tasks import (
            dispatch_processing,
        )

        mock_task = MagicMock()
        mock_config = MagicMock()
        mock_config.celery_enabled = True
        mock_config.async_processing = True
        mock_config.max_concurrent_uploads = 5

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                mock_task,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
        ):
            result = dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is True
            mock_task.delay.assert_called_once()

    def test_celery_failure_falls_to_thread_pool(self):
        """Tier 1 fails → Tier 2: Thread pool picks up."""
        from lasagna.infrastructure.tasks.document_tasks import (
            dispatch_processing,
        )

        mock_task = MagicMock()
        mock_task.delay.side_effect = ConnectionError("Redis down")

        mock_config = MagicMock()
        mock_config.celery_enabled = True
        mock_config.async_processing = True
        mock_config.max_concurrent_uploads = 5

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                mock_task,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
            patch(
                "lasagna.infrastructure.tasks.document_tasks._thread_pool_wrapper",
            ) as _mock_wrapper,
        ):
            result = dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is True  # Thread pool should have accepted it

    def test_thread_pool_when_celery_disabled(self):
        """Tier 2: Thread pool dispatches when Celery is disabled."""
        from lasagna.infrastructure.tasks.document_tasks import (
            dispatch_processing,
        )

        mock_config = MagicMock()
        mock_config.celery_enabled = False
        mock_config.async_processing = True
        mock_config.max_concurrent_uploads = 5

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                None,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
            patch(
                "lasagna.infrastructure.tasks.document_tasks._thread_pool_wrapper",
            ),
        ):
            result = dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is True

    def test_sync_when_async_disabled(self):
        """Tier 3: Sync fallback when async_processing=false."""
        from lasagna.infrastructure.tasks.document_tasks import (
            dispatch_processing,
        )

        mock_config = MagicMock()
        mock_config.celery_enabled = False
        mock_config.async_processing = False

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                None,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
        ):
            result = dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is False


# ---------------------------------------------------------------------------
# 2. Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    """Verify semaphore-based backpressure rejects when at capacity."""

    def test_rejects_when_semaphore_full(self):
        """dispatch_processing returns False when thread pool is at capacity."""
        from lasagna.infrastructure.tasks import document_tasks

        mock_config = MagicMock()
        mock_config.celery_enabled = False
        mock_config.async_processing = True
        mock_config.max_concurrent_uploads = 2

        # Create a semaphore with 0 permits (fully acquired)
        full_sem = threading.Semaphore(0)

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                None,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
            patch.object(
                document_tasks,
                "_get_doc_semaphore",
                return_value=full_sem,
            ),
        ):
            result = document_tasks.dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is False

    def test_accepts_when_semaphore_available(self):
        """dispatch_processing returns True when semaphore has capacity."""
        from lasagna.infrastructure.tasks import document_tasks

        mock_config = MagicMock()
        mock_config.celery_enabled = False
        mock_config.async_processing = True
        mock_config.max_concurrent_uploads = 5

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks.process_document_task",
                None,
            ),
            patch(
                "lasagna.core.config.rag_config",
                mock_config,
            ),
            patch(
                "lasagna.infrastructure.tasks.document_tasks._thread_pool_wrapper",
            ),
        ):
            result = document_tasks.dispatch_processing("doc1", "sess1", "/tmp/test.txt")
            assert result is True


# ---------------------------------------------------------------------------
# 3. Retry in thread pool
# ---------------------------------------------------------------------------


class TestThreadPoolRetry:
    """Verify retry logic in _thread_pool_wrapper."""

    def test_retries_on_transient_failure(self):
        """_thread_pool_wrapper retries when _process_document_impl fails."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _thread_pool_wrapper,
        )

        call_count = 0

        def mock_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient error")
            return {"document_id": "doc1", "status": "ready", "chunks_count": 5}

        sem = threading.Semaphore(1)
        sem.acquire()  # Pre-acquired, wrapper should release in finally

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks._process_document_impl",
                side_effect=mock_impl,
            ),
            patch(
                "lasagna.infrastructure.tasks.document_tasks.time",
            ) as mock_time,
        ):
            mock_time.sleep = MagicMock()  # Skip actual sleep
            _thread_pool_wrapper("doc1", "sess1", "/tmp/t.txt", "", 0, sem)

        assert call_count == 2
        # Semaphore should have been released
        assert sem.acquire(blocking=False) is True

    def test_exhausts_retries_and_releases_semaphore(self):
        """After max retries, semaphore is released."""
        from lasagna.infrastructure.tasks.document_tasks import (
            _thread_pool_wrapper,
        )

        sem = threading.Semaphore(1)
        sem.acquire()

        with (
            patch(
                "lasagna.infrastructure.tasks.document_tasks._process_document_impl",
                side_effect=RuntimeError("permanent error"),
            ),
            patch(
                "lasagna.infrastructure.tasks.document_tasks.time",
            ) as mock_time,
        ):
            mock_time.sleep = MagicMock()
            # Should not raise — errors are handled internally
            _thread_pool_wrapper("doc1", "sess1", "/tmp/t.txt", "", 0, sem)

        # Semaphore must be released even after failure
        assert sem.acquire(blocking=False) is True


# ---------------------------------------------------------------------------
# 4. Structured JSON response contract
# ---------------------------------------------------------------------------


class TestUploadResponseContract:
    """Verify MCP upload_document returns valid JSON with expected keys."""

    def test_upload_returns_json_with_required_keys(self):
        """upload_document MCP tool returns JSON with all required fields."""
        from lasagna.mcp.server import upload_document

        mock_doc = MagicMock()
        mock_doc.document_id = "doc_test123"
        mock_doc.filename = "report.pdf"
        mock_doc.status.value = "ready"
        mock_doc.chunks_count = 42
        mock_doc.file_size = 128000

        mock_backend = MagicMock()
        mock_backend.upload_document.return_value = mock_doc

        with (
            patch(
                "lasagna.mcp.server._get_backend",
                return_value=mock_backend,
            ),
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
            patch("os.path.isfile", return_value=True),
            patch("os.path.realpath", return_value="/tmp/report.pdf"),
        ):
            result = upload_document("/tmp/report.pdf", "sess1")

        parsed = json.loads(result)
        assert parsed["document_id"] == "doc_test123"
        assert parsed["filename"] == "report.pdf"
        assert parsed["status"] == "ready"
        assert parsed["chunks_count"] == 42
        assert parsed["file_size_kb"] == 125

    def test_upload_error_returns_json(self):
        """upload_document returns JSON error on failure."""
        from lasagna.mcp.server import upload_document

        with (
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
            patch("os.path.isfile", return_value=False),
            patch("os.path.realpath", return_value="/tmp/missing.pdf"),
        ):
            result = upload_document("/tmp/missing.pdf", "sess1")

        parsed = json.loads(result)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# 5. get_document_status tool
# ---------------------------------------------------------------------------


class TestGetDocumentStatus:
    """Verify get_document_status MCP tool returns correct JSON."""

    def test_returns_status_json(self):
        """get_document_status returns JSON with status fields."""
        from lasagna.mcp.server import get_document_status

        mock_doc = MagicMock()
        mock_doc.document_id = "doc_abc"
        mock_doc.status.value = "processing"
        mock_doc.chunks_count = 0
        mock_doc.error_message = None

        mock_backend = MagicMock()
        mock_backend.get_document_by_id.return_value = mock_doc

        with (
            patch(
                "lasagna.mcp.server._get_backend",
                return_value=mock_backend,
            ),
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
        ):
            result = get_document_status("doc_abc", "sess1")

        parsed = json.loads(result)
        assert parsed["document_id"] == "doc_abc"
        assert parsed["status"] == "processing"
        assert parsed["chunks_count"] == 0
        assert parsed["error_message"] is None

    def test_returns_error_for_missing_document(self):
        """get_document_status returns JSON error when doc not found."""
        from lasagna.mcp.server import get_document_status

        mock_backend = MagicMock()
        mock_backend.get_document_by_id.return_value = None

        with (
            patch(
                "lasagna.mcp.server._get_backend",
                return_value=mock_backend,
            ),
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
        ):
            result = get_document_status("doc_missing", "sess1")

        parsed = json.loads(result)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# 6. Wait=true polling in upload_document
# ---------------------------------------------------------------------------


class TestUploadWaitPolling:
    """Verify wait=true blocks until document processing completes."""

    def test_wait_true_polls_until_ready(self):
        """upload_document with wait=true polls until status=ready."""
        from lasagna.mcp.server import upload_document

        # upload_document returns status=processing initially
        mock_doc_initial = MagicMock()
        mock_doc_initial.document_id = "doc_wait"
        mock_doc_initial.filename = "test.pdf"
        mock_doc_initial.status.value = "processing"
        mock_doc_initial.chunks_count = 0
        mock_doc_initial.file_size = 1024

        # After polling, returns status=ready
        mock_doc_ready = MagicMock()
        mock_doc_ready.document_id = "doc_wait"
        mock_doc_ready.filename = "test.pdf"
        mock_doc_ready.status.value = "ready"
        mock_doc_ready.chunks_count = 10
        mock_doc_ready.file_size = 1024

        mock_backend = MagicMock()
        mock_backend.upload_document.return_value = mock_doc_initial
        mock_backend.get_document_by_id.return_value = mock_doc_ready

        with (
            patch(
                "lasagna.mcp.server._get_backend",
                return_value=mock_backend,
            ),
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
            patch("os.path.isfile", return_value=True),
            patch("os.path.realpath", return_value="/tmp/test.pdf"),
            patch(
                "lasagna.mcp.server.time.sleep",
            ),
        ):
            result = upload_document("/tmp/test.pdf", "sess1", wait=True)

        parsed = json.loads(result)
        assert parsed["status"] == "ready"
        assert parsed["chunks_count"] == 10

    def test_wait_false_returns_immediately(self):
        """upload_document with wait=false returns processing status immediately."""
        from lasagna.mcp.server import upload_document

        mock_doc = MagicMock()
        mock_doc.document_id = "doc_nowait"
        mock_doc.filename = "test.pdf"
        mock_doc.status.value = "processing"
        mock_doc.chunks_count = 0
        mock_doc.file_size = 2048

        mock_backend = MagicMock()
        mock_backend.upload_document.return_value = mock_doc

        with (
            patch(
                "lasagna.mcp.server._get_backend",
                return_value=mock_backend,
            ),
            patch(
                "lasagna.mcp.server._resolve_session",
                return_value=("sess1", None),
            ),
            patch("os.path.isfile", return_value=True),
            patch("os.path.realpath", return_value="/tmp/test.pdf"),
        ):
            result = upload_document("/tmp/test.pdf", "sess1", wait=False)

        parsed = json.loads(result)
        assert parsed["status"] == "processing"
        # get_document_by_id should NOT have been called
        mock_backend.get_document_by_id.assert_not_called()
