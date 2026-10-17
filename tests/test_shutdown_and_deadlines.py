"""Shutdown, a full processing pool and a slow search end with an answer.

Before: shutdown left every document still in the processing pool in
"processing" for good (``processing_timeout`` was defined and never read); a
full pool made the REST upload process the document inside the request; a
search had no deadline, so a hung stage held the request until a proxy gave
up; and a backend that could not open a locked data directory logged a full
traceback over the one sentence the CLI prints.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from lasagna.infrastructure.tasks import document_tasks


@pytest.fixture
def fresh_pool(monkeypatch):
    """An empty processing pool of two threads, isolated from the suite's."""
    monkeypatch.setattr(document_tasks, "_doc_executor", None)
    monkeypatch.setattr(document_tasks, "_doc_semaphore", None)
    monkeypatch.setattr(document_tasks, "_inflight", {})
    import lasagna.core.config as config_module

    monkeypatch.setattr(
        config_module,
        "rag_config",
        SimpleNamespace(celery_enabled=False, async_processing=True, max_concurrent_uploads=2),
        raising=False,
    )
    yield
    if document_tasks._doc_executor is not None:
        document_tasks._doc_executor.shutdown(wait=True)


@pytest.fixture
def statuses(monkeypatch):
    """What the drain writes to the database, and what it releases."""
    import lasagna.core.database as database
    import lasagna.core.quota as quota

    written, released = [], []
    monkeypatch.setattr(
        database,
        "get_rag_database",
        lambda: SimpleNamespace(update_document_status=lambda **kw: written.append(kw)),
    )
    monkeypatch.setattr(
        quota,
        "get_quota_manager",
        lambda: SimpleNamespace(
            release_upload_reservation=lambda sid, size, reservation_id: released.append(
                reservation_id
            )
        ),
    )
    return written, released


def _blocking_processor(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(document_tasks, "_process_document_impl", lambda *a, **k: release.wait(10))
    return release


class TestShutdownDrainsThePool:
    def test_an_unfinished_document_is_failed_with_the_reason(
        self, fresh_pool, statuses, monkeypatch
    ):
        release = _blocking_processor(monkeypatch)
        assert document_tasks.dispatch_processing(
            "doc_slow", "s1", "/f", reservation_id="r1", reserved_file_size=10
        )
        try:
            failed = document_tasks.drain_document_processing(timeout=0.2)
        finally:
            release.set()
        written, released = statuses
        assert failed == ["doc_slow"]
        assert written[0]["document_id"] == "doc_slow"
        assert written[0]["status"] == "error"
        assert "shut down before processing finished" in written[0]["error_message"]
        assert "processing_timeout" in written[0]["error_message"]
        assert released == ["r1"]

    def test_a_document_that_finishes_in_time_is_left_alone(
        self, fresh_pool, statuses, monkeypatch
    ):
        monkeypatch.setattr(document_tasks, "_process_document_impl", lambda *a, **k: None)
        document_tasks.dispatch_processing("doc_fast", "s1", "/f")
        assert document_tasks.drain_document_processing(timeout=5) == []
        assert statuses == ([], [])

    def test_close_backend_drains_for_processing_timeout(self, monkeypatch):
        import lasagna.api.rag_backend as rag_backend

        waited = []
        monkeypatch.setattr(
            document_tasks, "drain_document_processing", lambda timeout: waited.append(timeout)
        )
        monkeypatch.setattr(rag_backend, "_instance", SimpleNamespace())
        rag_backend.close_backend()
        assert waited == [float(rag_backend.rag_config._config.get("processing_timeout", 300))]


class TestAFullPool:
    def test_a_server_caller_is_refused(self, fresh_pool, statuses, monkeypatch):
        release = _blocking_processor(monkeypatch)
        try:
            for i in range(2):
                assert document_tasks.dispatch_processing(f"d{i}", "s1", "/f")
            with pytest.raises(document_tasks.ProcessingAtCapacity, match="RAG_MAX_CONCURRENT"):
                document_tasks.dispatch_processing("d2", "s1", "/f", refuse_when_busy=True)
            # A script still gets inline processing, as before.
            assert document_tasks.dispatch_processing("d3", "s1", "/f") is False
        finally:
            release.set()

    def test_the_refused_upload_leaves_nothing_behind(self, monkeypatch, tmp_path):
        from lasagna.api import _document_pipeline as pipeline_mod
        import lasagna.core.database as database

        def busy(*a, **k):
            raise document_tasks.ProcessingAtCapacity("all threads busy")

        monkeypatch.setattr(pipeline_mod, "dispatch_processing", busy)
        deleted_rows, deleted_files = [], []
        monkeypatch.setattr(
            database,
            "get_rag_database",
            # get_document: an upload with a document_id first looks for the
            # document it would replace; there is none here.
            lambda: SimpleNamespace(
                delete_document=deleted_rows.append, get_document=lambda doc_id: None
            ),
        )
        source = tmp_path / "a.txt"
        source.write_text("tides")
        p = pipeline_mod.DocumentPipeline.__new__(pipeline_mod.DocumentPipeline)
        p._get_session_config = lambda sid: SimpleNamespace(
            max_file_size=10_000, supported_extensions=["txt"]
        )
        p._collection_manager = SimpleNamespace(
            embedder_for=lambda c: SimpleNamespace(model_name="m", dimension=8),
            resolve_backend=lambda sid: SimpleNamespace(backend_name="milvus"),
        )
        p.document_processor = SimpleNamespace(detect_file_type=lambda path: "txt")
        p._store_file = lambda src, doc_id, name: "/stored/a.txt"
        p._save_document_metadata = lambda *a, **k: None
        p._storage = SimpleNamespace(delete=deleted_files.append)

        with pytest.raises(document_tasks.ProcessingAtCapacity):
            p.upload_document(str(source), "s1", document_id="doc_1", refuse_when_busy=True)
        assert deleted_rows == ["doc_1"]
        assert deleted_files == ["/stored/a.txt"]

    def test_the_upload_route_answers_503_with_retry_after(self, monkeypatch):
        from lasagna.api.rest.app import create_app
        from lasagna.api.rest.dependencies import get_backend

        monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
        seen = {}

        def upload_document(**kwargs):
            seen.update(kwargs)
            raise document_tasks.ProcessingAtCapacity("All 5 document-processing threads are busy")

        app = create_app()
        app.dependency_overrides[get_backend] = lambda: SimpleNamespace(
            upload_document=upload_document
        )
        r = TestClient(app).post(
            "/api/v1/documents", files={"file": ("a.txt", b"tides", "text/plain")}
        )
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "5"
        assert "busy" in r.json()["detail"]
        assert seen["refuse_when_busy"] is True


class TestSearchesHaveADeadline:
    @pytest.fixture
    def client(self, monkeypatch):
        from lasagna.api.rest import limits
        from lasagna.api.rest.app import create_app
        from lasagna.api.rest.dependencies import get_backend

        monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
        monkeypatch.setattr(limits, "search_timeout", lambda: 0.05)

        async def hangs(**kwargs):
            await asyncio.sleep(5)

        app = create_app()
        app.dependency_overrides[get_backend] = lambda: SimpleNamespace(
            async_search=hangs,
            async_search_with_citations=hangs,
            async_agentic_search=hangs,
            _search_orchestrator=None,
        )
        return TestClient(app)

    @pytest.mark.parametrize(
        "path", ["/api/v1/search", "/api/v1/search/citations", "/api/v1/search/agentic"]
    )
    def test_a_search_past_the_deadline_is_a_504(self, client, path):
        r = client.post(path, json={"query": "tides"})
        assert r.status_code == 504
        assert "RAG_SEARCH_TIMEOUT" in r.json()["detail"]

    def test_the_deadline_is_read_from_the_environment(self, monkeypatch):
        from lasagna.api.rest import limits

        monkeypatch.setenv("RAG_SEARCH_TIMEOUT", "7")
        assert limits.search_timeout() == 7.0
        monkeypatch.delenv("RAG_SEARCH_TIMEOUT")
        assert limits.search_timeout() == 120.0


class TestALockedDataDirectoryIsNotATraceback:
    @pytest.fixture
    def failing_init(self, monkeypatch):
        import lasagna.api.rag_backend as rag_backend

        monkeypatch.setattr(rag_backend, "_instance", None)

        def fail_with(exc):
            def _do_init(self):
                raise exc

            monkeypatch.setattr(rag_backend.RAGBackend, "_do_init", _do_init)

        yield fail_with
        rag_backend._instance = None

    @pytest.fixture
    def records(self):
        seen = []
        sink = logger.add(lambda m: seen.append(m.record), level="DEBUG")
        yield seen
        logger.remove(sink)

    def test_it_is_logged_at_debug_without_a_traceback(self, failing_init, records):
        import lasagna.api.rag_backend as rag_backend

        DataDirLockedError = type("DataDirLockedError", (RuntimeError,), {})
        failing_init(DataDirLockedError("~/.lasagna is in use by another process"))
        with pytest.raises(DataDirLockedError):
            rag_backend.RAGBackend()
        mine = [r for r in records if "RAG backend not initialized" in r["message"]]
        assert mine and mine[0]["level"].name == "DEBUG" and mine[0]["exception"] is None
        assert not [r for r in records if r["level"].name == "ERROR"]

    def test_any_other_failure_is_still_an_error_with_its_traceback(self, failing_init, records):
        import lasagna.api.rag_backend as rag_backend

        failing_init(RuntimeError("model {weights} missing"))  # braces once broke loguru
        with pytest.raises(RuntimeError):
            rag_backend.RAGBackend()
        errors = [r for r in records if r["level"].name == "ERROR"]
        assert errors and "model {weights} missing" in errors[0]["message"]
        assert errors[0]["exception"] is not None
