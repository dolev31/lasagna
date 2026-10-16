"""A batch upload larger than the processing pool is processed in full.

Without Celery, documents are processed by a pool of RAG_MAX_CONCURRENT_UPLOADS
threads (5 by default), and the batch route refused a file whenever every
thread was busy. A batch may hold RAG_MAX_BATCH_FILES files (20 by default),
so files 6 to 20 of a full batch came back "threads are busy". A batch now
waits for a thread to free up for each file (at most ``processing_timeout``
per file); a single upload into a full pool is still refused with 503.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from lasagna.api import _document_pipeline as pipeline_mod
from lasagna.api.rest.app import create_app
from lasagna.api.rest.dependencies import get_backend
from lasagna.infrastructure.tasks import document_tasks

POOL = 5


@pytest.fixture
def pool(monkeypatch):
    """A fresh pool of five threads whose processing takes a moment."""
    import lasagna.core.config as config_module

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(document_tasks, "_doc_executor", None)
    monkeypatch.setattr(document_tasks, "_doc_semaphore", None)
    monkeypatch.setattr(document_tasks, "_inflight", {})
    real = config_module.rag_config
    settings = SimpleNamespace(
        celery_enabled=False,
        async_processing=True,
        max_concurrent_uploads=POOL,
        max_file_size=real.max_file_size,
        _config={**real._config, "processing_timeout": 30},
    )
    monkeypatch.setattr(config_module, "rag_config", settings, raising=False)
    processed, running, peak = [], [0], [0]
    lock = threading.Lock()

    def process(document_id, *args, **kwargs):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
            processed.append(document_id)

    monkeypatch.setattr(document_tasks, "_process_document_impl", process)
    yield SimpleNamespace(processed=processed, peak=peak)
    if document_tasks._doc_executor is not None:
        document_tasks._doc_executor.shutdown(wait=True)


@pytest.fixture
def client(monkeypatch, pool):
    """The REST app over a real DocumentPipeline whose storage and database are fakes."""
    monkeypatch.setattr(pipeline_mod, "get_quota_manager", lambda: MagicMock())
    pipeline = pipeline_mod.DocumentPipeline.__new__(pipeline_mod.DocumentPipeline)
    pipeline._get_session_config = lambda sid: SimpleNamespace(
        max_file_size=10_000, supported_extensions=["txt"]
    )
    pipeline._collection_manager = SimpleNamespace(
        embedder_for=lambda c: SimpleNamespace(model_name="m", dimension=8),
        resolve_backend=lambda sid: SimpleNamespace(backend_name="milvus"),
    )
    pipeline.document_processor = SimpleNamespace(detect_file_type=lambda path: "txt")
    pipeline._store_file = lambda src, doc_id, name: f"/stored/{doc_id}"
    pipeline._save_document_metadata = lambda *a, **k: None
    pipeline._storage = SimpleNamespace(delete=lambda path: None)
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: SimpleNamespace(
        upload_document=pipeline.upload_document
    )
    return TestClient(app)


def _files(n):
    return [("files", (f"note{i}.txt", f"note {i}".encode(), "text/plain")) for i in range(n)]


def test_a_batch_of_twenty_with_five_threads_processes_all_twenty(client, pool):
    reply = client.post("/api/v1/documents/batch", files=_files(20))

    assert reply.status_code == 201, reply.text
    body = reply.json()
    assert (body["succeeded"], body["failed"]) == (20, 0), [
        r["error"] for r in body["results"] if r["error"]
    ]
    assert document_tasks.drain_document_processing(timeout=30) == []
    assert len(pool.processed) == 20
    assert pool.peak[0] <= POOL


def test_a_single_upload_into_a_full_pool_is_still_refused(client, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(document_tasks, "_process_document_impl", lambda *a, **k: release.wait(10))
    try:
        for i in range(POOL):
            assert document_tasks.dispatch_processing(f"busy{i}", "s1", "/f")
        reply = client.post("/api/v1/documents", files={"file": ("a.txt", b"a", "text/plain")})
    finally:
        release.set()

    assert reply.status_code == 503, reply.text
    assert "RAG_MAX_CONCURRENT_UPLOADS" in reply.json()["detail"]
