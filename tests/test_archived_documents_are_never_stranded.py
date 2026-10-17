"""An archived document is never left in a collection its session no longer searches.

Profile switches and the documents check skipped archived documents, so a switch
to another model left them in the old model's collection, and restoring one put
it back as ready there, where no search of the session found it. Archive and
restore took no document lock, and a reindex that finished after an archive
wrote READY over it.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lasagna.api import _profile_switch
from lasagna.api._profile_guard import DocumentsLeftBehind
from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager

SMALL = {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384}


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _session(*, ready=(), archived=(), collection=None) -> str:
    sid = f"arch-{uuid.uuid4().hex[:8]}"
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid)
    where = collection or get_rag_config().with_session(sid).collection_name
    db = get_rag_database()
    for name in (*ready, *archived):
        db.save_document(
            {
                "document_id": f"{name}_{sid}",
                "session_id": sid,
                "filename": f"{name}.md",
                "file_size": 10,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
            },
            f"/nonexistent/{name}.md",
            collection_name=where,
        )
    for name in archived:
        db.soft_delete_document(f"{name}_{sid}")
    return sid


def _lifecycle():
    from lasagna.lifecycle.manager import DocumentLifecycleManager
    from lasagna.lifecycle.policy import LifecyclePolicy

    pipeline = SimpleNamespace(
        _invalidate_search_cache=lambda s: None,
        _remove_from_session_focus=lambda d, s: None,
        hybrid_searcher=SimpleNamespace(note_local_change=lambda s, g: None),
    )
    return DocumentLifecycleManager(
        db=get_rag_database(),
        pipeline=pipeline,
        backend=None,
        policy=LifecyclePolicy(),
        collection_name="unused",
    )


class _Backend:
    def __init__(self):
        self.reindexed = []

    def reindex_document(self, document_id, session_id, *, config=None):
        self.reindexed.append(document_id)


def test_a_switch_with_reindex_is_refused_while_archived_documents_would_stay_behind():
    sid = _session(ready=["a"], archived=["old"])
    backend = _Backend()
    with pytest.raises(DocumentsLeftBehind) as refused:
        _profile_switch.switch_profile(
            KnowledgeConfigManager(),
            "speed",
            session_id=sid,
            reindex=True,
            config=SMALL,
            backend=backend,
        )
    assert f"old_{sid}" in str(refused.value) and "Restore it first" in str(refused.value)
    assert backend.reindexed == []
    assert get_rag_database().get_profile_switch(sid) is None


def test_a_switch_without_reindex_is_refused_for_archived_documents_too():
    sid = _session(archived=["old"])
    with pytest.raises(DocumentsLeftBehind, match="archived document"):
        KnowledgeConfigManager().set_active_profile("speed", config=SMALL, session_id=sid)
    assert get_rag_config().with_session(sid).embedding_model == "all-MiniLM-L6-v2"


def test_a_restore_into_a_collection_the_session_does_not_search_is_refused():
    small = collection_name_for_settings(SMALL)
    sid = _session(archived=["old"], collection=small)
    with pytest.raises(DocumentsLeftBehind) as refused:
        _lifecycle().restore(f"old_{sid}", sid)
    assert "lasagna configure --profile" in str(refused.value)
    assert get_rag_database().get_document(f"old_{sid}")["status"] == "soft_deleted"


def test_archive_and_restore_wait_for_the_documents_lock(monkeypatch):
    from lasagna.core.store_locks import StoreLockTimeout, store_lock

    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    sid = _session(ready=["a"])
    doc = f"a_{sid}"
    taken, release = threading.Event(), threading.Event()

    def reindexing():
        with store_lock(f"document:{doc}"):
            taken.set()
            release.wait(30)

    holder = threading.Thread(target=reindexing, daemon=True)
    holder.start()
    assert taken.wait(10)
    try:
        with pytest.raises(StoreLockTimeout):
            _lifecycle().soft_delete(doc, sid)
        assert get_rag_database().get_document(doc)["status"] == "ready"
    finally:
        release.set()
        holder.join(10)
    _lifecycle().soft_delete(doc, sid)
    assert get_rag_database().get_document(doc)["status"] == "soft_deleted"


def test_a_reindex_that_finishes_after_an_archive_keeps_it_archived():
    from lasagna.api._document_pipeline import DocumentPipeline

    sid = _session(archived=["late"])
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._storage = SimpleNamespace(backend_name="local")
    document = SimpleNamespace(
        document_id=f"late_{sid}",
        filename="late.md",
        file_size=10,
        file_type="md",
        upload_date=datetime.now(timezone.utc),
    )
    pipeline._commit_replacement(
        document,
        "/nonexistent/late.md",
        SimpleNamespace(collection_name="c"),
        SimpleNamespace(backend_name="milvus"),
        chunk_count=3,
        processing_time=0.1,
        text_length=10,
        preview_text="",
    )
    assert get_rag_database().get_document(f"late_{sid}")["status"] == "soft_deleted"
