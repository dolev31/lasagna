"""A switch whose reindex lost a document's chunks moves it back, and says what is true.

A reindex removes the old chunks before it records the new copy. When that
record failed, the new chunks were removed too, so the document had none; the
switch's restore skipped it (its reindex had "failed", so it was taken to be
where it was), and the message still said "every document is searchable as
before".
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.api import _profile_switch
from lasagna.api._document_pipeline import ReplacementNotRecorded
from lasagna.core import model_downloads
from lasagna.core.config import get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager

SMALL = {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384}


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _session() -> str:
    sid = f"lost-{uuid.uuid4().hex[:8]}"
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid)
    collection = get_rag_config().with_session(sid).collection_name
    for name in ("a", "b"):
        get_rag_database().save_document(
            {
                "document_id": f"{name}_{sid}",
                "session_id": sid,
                "filename": f"{name}.md",
                "file_size": 10,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
                "upload_date": f"2026-01-0{1 if name == 'a' else 2}T00:00:00",
            },
            f"/nonexistent/{name}.md",
            collection_name=collection,
        )
    return sid


class _Backend:
    """b's new copy cannot be recorded; moving b back fails when ``back_fails``."""

    def __init__(self, back_fails=False):
        self.calls, self.back_fails = [], back_fails

    def reindex_document(self, document_id, session_id, *, config=None):
        self.calls.append((document_id, "new" if config is not None else "back"))
        if document_id.startswith("b_") and config is not None:
            raise ReplacementNotRecorded(
                f"Reindex of {document_id} failed. Its previous chunks were removed, and the "
                f"new copy could not be recorded (OperationalError: disk full). Reindex it again."
            )
        if document_id.startswith("b_") and self.back_fails:
            raise RuntimeError("the store is still full")


def _switch(sid, backend):
    with pytest.raises(_profile_switch.ReindexFailed) as failed:
        _profile_switch.switch_profile(
            KnowledgeConfigManager(),
            "speed",
            session_id=sid,
            reindex=True,
            config=SMALL,
            backend=backend,
        )
    return str(failed.value), failed.value.report


def test_the_document_that_lost_its_chunks_is_reindexed_back():
    sid = _session()
    backend = _Backend()
    message, report = _switch(sid, backend)
    assert (f"b_{sid}", "back") in backend.calls
    assert set(report.restored) == {f"a_{sid}", f"b_{sid}"}
    assert f"b_{sid} lost its chunks when its new copy could not be recorded" in message
    assert "every document is searchable as before" in message


def test_one_that_could_not_be_reindexed_back_is_named_and_nothing_else_is_claimed():
    sid = _session()
    message, report = _switch(sid, _Backend(back_fails=True))
    assert [d for d, _ in report.stranded] == [f"b_{sid}"]
    assert "every document is searchable as before" not in message
    assert f"b_{sid} could not be moved back" in message
    assert "has no chunks, so no search finds it" in message
