"""A document whose worker died is reported failed, with the reason, not "processing" forever.

A Celery worker child killed by the 600 s hard time limit, the out-of-memory
killer or a crash never reaches the code that marks a document failed, and
with acks_late Celery acknowledges the message anyway: the document stayed
"processing" for good, and clients polling it waited forever. No GC phase
looked for it. A redelivered message (a worker that died as a whole) also
processed again a document that was already finished or failed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lasagna.core import database
from lasagna.core.database import RAGDatabase
from lasagna.core.schema import Document, DocumentStatus
from lasagna.lifecycle import gc as gc_module
from lasagna.lifecycle.gc import GarbageCollector
from lasagna.lifecycle.policy import LifecyclePolicy

SESSION = "s-stuck"


@pytest.fixture
def db(tmp_path, monkeypatch):
    store = RAGDatabase(str(tmp_path / "metadata.db"))
    monkeypatch.setattr(database, "_rag_database", store)
    return store


def _save(db, doc_id, status, age):
    db.save_document(
        {
            "document_id": doc_id,
            "session_id": SESSION,
            "filename": f"{doc_id}.pdf",
            "file_size": 10,
            "file_type": "pdf",
            "status": status,
            "upload_date": (datetime.now(timezone.utc) - age).isoformat(),
        },
        file_path=f"/tmp/{doc_id}.pdf",
    )


def _collector(db):
    return GarbageCollector(db=db, pipeline=SimpleNamespace(), policy=LifecyclePolicy())


def test_a_gc_cycle_fails_a_document_processing_far_longer_than_processing_timeout(db):
    _save(db, "stuck", "processing", timedelta(days=2))
    _save(db, "working", "processing", timedelta(minutes=5))
    _save(db, "done", "ready", timedelta(days=2))

    report = _collector(db).run_full_cycle()

    stuck = db.get_document("stuck")
    assert stuck["status"] == "error"
    assert "did not finish" in stuck["error_message"]
    assert "Upload the document again" in stuck["error_message"]
    assert db.get_document("working")["status"] == "processing"
    assert db.get_document("done")["status"] == "ready"
    phase = next(p for p in report.phases if p.phase == "phase0_fail_stale_processing")
    assert phase.docs_cleaned == 1


def test_the_cutoff_follows_processing_timeout_and_covers_the_celery_retries(monkeypatch):
    from lasagna.core.config import rag_config

    monkeypatch.setitem(rag_config._config, "processing_timeout", 300)
    assert gc_module.stale_processing_after() == timedelta(hours=1)
    monkeypatch.setitem(rag_config._config, "processing_timeout", 1800)
    assert gc_module.stale_processing_after() == timedelta(hours=6)
    monkeypatch.setitem(rag_config._config, "processing_timeout", 10)
    # Three Celery attempts of up to 600 s each must never be cut short.
    assert gc_module.stale_processing_after() >= timedelta(seconds=3 * 600)


def test_starting_the_lifecycle_scheduler_fails_documents_left_by_a_dead_process(db, monkeypatch):
    from lasagna.lifecycle import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "_celery_workers_running", lambda: False)
    monkeypatch.setattr(scheduler_module.LifecycleScheduler, "_start_apscheduler", lambda s: None)
    _save(db, "stuck", "processing", timedelta(days=2))

    scheduler_module.LifecycleScheduler(gc=_collector(db), policy=LifecyclePolicy()).start()

    assert db.get_document("stuck")["status"] == "error"


def test_only_unfinished_documents_are_failed(db):
    for status in ("uploading", "processing", "indexing", "ready", "error", "soft_deleted"):
        _save(db, status, status, timedelta(days=2))
    failed = db.fail_stale_processing_documents(datetime.now(timezone.utc), "stuck")
    assert sorted(failed) == ["indexing", "processing", "uploading"]
    assert db.get_document("ready")["status"] == "ready"
    assert db.get_document("soft_deleted")["status"] == "soft_deleted"


def _celery_task_module(monkeypatch):
    """``document_tasks`` as a worker loads it, with a stand-in for Celery."""
    import importlib.util
    import sys

    from lasagna.infrastructure.tasks import document_tasks

    class _Task:
        def __init__(self, fn, max_retries=0, **_):
            self.fn, self.max_retries = fn, max_retries
            self.request = SimpleNamespace(retries=0)

        def retry(self, exc=None):
            return exc

        def __call__(self, *args, **kwargs):
            return self.fn(self, *args, **kwargs)

    fake = SimpleNamespace(shared_task=lambda **opts: lambda fn: _Task(fn, **opts))
    monkeypatch.setitem(sys.modules, "celery", fake)
    spec = importlib.util.spec_from_file_location("_worker_tasks", document_tasks.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_worker_tasks", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status", [DocumentStatus.READY, DocumentStatus.ERROR])
def test_a_redelivered_task_does_not_process_a_finished_document_again(monkeypatch, status):
    from lasagna.api import rag_backend

    processed = []
    doc = Document(
        document_id="d1",
        session_id=SESSION,
        filename="d1.pdf",
        file_type="pdf",
        file_size=10,
        status=status,
        chunks_count=3,
        upload_date=datetime.now(timezone.utc),
    )
    backend = SimpleNamespace(
        _get_document_metadata=lambda doc_id: doc,
        get_session_config=lambda sid: None,
        _document_pipeline=SimpleNamespace(
            process_document=lambda *a, **k: processed.append(a[0].document_id)
        ),
    )
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    tasks = _celery_task_module(monkeypatch)

    out = tasks.process_document_task("d1", SESSION, "/tmp/d1.pdf")

    assert processed == []
    assert out["status"] == status.value


def test_the_thread_pool_still_retries_a_document_it_marked_failed(monkeypatch):
    """Its retries run after the failed attempt marked the document error."""
    from lasagna.api import rag_backend
    from lasagna.infrastructure.tasks import document_tasks

    processed = []
    doc = Document(
        document_id="d1",
        session_id=SESSION,
        filename="d1.pdf",
        file_type="pdf",
        file_size=10,
        status=DocumentStatus.ERROR,
        upload_date=datetime.now(timezone.utc),
    )

    def process(document, *a, **k):
        processed.append(document.document_id)
        raise ValueError("stop here")

    backend = SimpleNamespace(
        _get_document_metadata=lambda doc_id: doc,
        get_session_config=lambda sid: None,
        _document_pipeline=SimpleNamespace(process_document=process),
    )
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    with pytest.raises(ValueError):
        document_tasks._process_document_impl("d1", SESSION, "/tmp/d1.pdf")
    assert processed == ["d1"]


def test_celery_does_not_requeue_a_task_whose_worker_child_died(monkeypatch):
    """A child killed by memory or a crash would be killed again by the same file."""
    from lasagna.infrastructure.tasks import celery_app

    class _Conf(dict):
        def update(self, **kw):
            super().update(kw)

    class _Celery:
        def __init__(self, *a, **kw):
            self.conf = _Conf()

        def set_default(self):
            pass

    monkeypatch.setattr(celery_app, "Celery", _Celery)
    conf = celery_app.create_celery_app().conf
    assert conf["task_acks_late"] is True
    assert conf["task_reject_on_worker_lost"] is False
