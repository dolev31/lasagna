"""A profile switch that fails answers with a plain reason, never the exception's text.

A failed reindex during ``PUT /api/config/knowledge/active?reindex`` (and
``/api/v1/config/profiles/{name}``, the CLI and MCP) put each document's raw
exception text in the 409: internal paths, library messages, whatever the
failure said. It now gives the reason REST's reindex route gives for that
failure, and the raw text goes to the log. A store that cannot be read is
named by its kind of error.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

RAW = "/srv/lasagna/secret/table.bin: bad header at 0x7f3a"


@pytest.fixture
def rest(monkeypatch):
    import lasagna.api.rag_backend as rag_backend
    from lasagna.api.rest.app import create_app
    from lasagna.core import local_server

    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(local_server, "local_store_held", lambda: False)
    holder = {}
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: holder["backend"])
    return TestClient(create_app()), holder


def _session():
    sid = f"plain-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
    get_rag_database().save_document(
        {
            "document_id": f"a_{sid}",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        collection_name=collection_name_for_settings(mgr.get_resolved_config(sid)),
    )
    return sid


class _Failing:
    def __init__(self, exc):
        self.exc = exc

    def reindex_document(self, document_id, session_id, *, config=None):
        if config is not None:
            raise self.exc


def test_an_internal_error_is_named_plainly(rest):
    client, holder = rest
    holder["backend"] = _Failing(RuntimeError(RAW))
    sid = _session()
    for answer in (
        client.put(
            "/api/config/knowledge/active",
            json={"active_profile": "standard", "reindex": True},
            headers={"X-Session-ID": sid},
        ),
        client.put("/api/v1/config/profiles/standard?reindex=true", headers={"X-Session-ID": sid}),
    ):
        assert answer.status_code == 409, answer.text
        detail = answer.json()["detail"]
        assert "was not applied" in detail
        assert f"Reindex of a_{sid} failed with an internal error" in detail
        assert RAW not in detail and "0x7f3a" not in detail


def test_a_reason_the_code_gives_itself_is_kept(rest):
    client, holder = rest
    holder["backend"] = _Failing(ValueError("Unsupported file type: .xyz"))
    sid = _session()
    answer = client.put(
        "/api/config/knowledge/active",
        json={"active_profile": "standard", "reindex": True},
        headers={"X-Session-ID": sid},
    )
    assert answer.status_code == 409
    assert "Unsupported file type: .xyz" in answer.json()["detail"]


def test_an_unreadable_store_is_named_by_its_kind(monkeypatch):
    from lasagna.api import _profile_guard

    def unreadable(collection, vector_backend=None):
        raise OSError(RAW)

    monkeypatch.setattr(_profile_guard, "stored_embedding_dim", unreadable)
    with pytest.raises(_profile_guard.EmbeddingSizeUnverified) as refused:
        _profile_guard.refuse_embedding_change(
            before={"embedding_dim": 384},
            after={"embedding_model": "BAAI/bge-m3", "embedding_dim": 1024},
            change="Profile 'standard'",
            kept="speed",
        )
    assert "OSError" in str(refused.value) and RAW not in str(refused.value)
