"""`lasagna configure` works while `lasagna serve` holds the local store.

With ``serve`` up, ``/api/v1/health/ready`` answered 503 telling the user to
run ``lasagna configure --profile speed -s default``, and running it failed:
"the vector store could not be read ... in use by another process ... To
switch anyway: --force", for a session with no documents. The command now goes
through the running server, as ``status`` and ``search`` do; and when another
process holds the store, a session the metadata database says has no
documents switches without reading it.
"""

from __future__ import annotations

import uuid

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.api import _profile_guard
from lasagna.api._profile_guard import EmbeddingSizeUnverified
from lasagna.core import local_server
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager


def _sid() -> str:
    return f"held-{uuid.uuid4().hex[:8]}"


def _active(sid: str) -> str:
    return get_knowledge_config_manager(session_id=sid).get_active_profile(session_id=sid)


class _Response:
    def __init__(self, status_code, data):
        self.status_code, self._data, self.text = status_code, data, str(data)

    def json(self):
        return self._data


def test_configure_goes_through_the_running_server(monkeypatch):
    import httpx

    sent = {}

    def put(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return _Response(200, {"active_profile": json["active_profile"]})

    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: "http://127.0.0.1:8765")
    monkeypatch.setattr(httpx, "put", put)
    sid = _sid()
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "speed", "-s", sid])
    assert result.exit_code == 0, result.output
    assert sent["url"] == "http://127.0.0.1:8765/api/config/knowledge/active"
    assert sent["body"] == {"active_profile": "speed", "force": False, "reindex": False}
    assert sent["headers"] == {"X-Session-ID": sid}
    assert "Applied profile: speed" in result.output


def test_a_refusal_from_the_server_names_the_ways_through(monkeypatch):
    import httpx

    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: "http://127.0.0.1:8765")
    monkeypatch.setattr(
        httpx, "put", lambda *a, **k: _Response(409, {"detail": "Nothing was changed."})
    )
    sid = _sid()
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "standard", "-s", sid])
    words = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "Nothing was changed." in words
    assert f"lasagna configure --profile standard -s {sid} --reindex" in words


def _store_unreadable(monkeypatch):
    def unreadable(collection, vector_backend=None):
        raise RuntimeError("in use by another process")

    monkeypatch.setattr(_profile_guard, "stored_embedding_dim", unreadable)


def test_a_session_without_documents_switches_when_the_store_is_held(monkeypatch):
    _store_unreadable(monkeypatch)
    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: None)
    sid = _sid()
    # From the default model (1024 dimensions) to speed's (384): a size check.
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "speed", "-s", sid])
    assert result.exit_code == 0, result.output
    assert _active(sid) == "speed"


def test_a_session_with_documents_still_needs_the_store(monkeypatch):
    _store_unreadable(monkeypatch)
    sid = _sid()
    mgr = get_knowledge_config_manager(session_id=sid)
    # A document already in speed's collection: only the size check is left.
    resolved, _ = mgr.preview_active_profile("speed", sid)
    get_rag_database().save_document(
        {
            "document_id": f"doc_{sid}",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        collection_name=collection_name_for_settings(resolved),
    )
    with pytest.raises(EmbeddingSizeUnverified):
        mgr.set_active_profile("speed", session_id=sid)
    assert _active(sid) == "default"
