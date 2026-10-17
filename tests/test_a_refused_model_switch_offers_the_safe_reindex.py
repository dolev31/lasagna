"""A model switch refused for a session's documents names a way through on every surface.

A session holding documents is refused a switch to another embedding model,
since none of its documents would be found, and the refusal said to switch
anyway and reindex. Only the CLI could switch anyway (``--force``), the CLI
could not reindex, and the message named the SDK and MCP reindex but not
REST's. Now REST and MCP take ``force`` too, ``configure --reindex`` reindexes
the session's documents and then switches (see
``test_a_failed_switch_with_reindex_leaves_the_session_searchable.py``), and the
message names the reindex in the SDK, REST and MCP.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lasagna import cli
from lasagna.api._profile_guard import EmbeddingSizeConflict
from lasagna.api.rest.app import create_app
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager
from lasagna.mcp import server


def _session_with_a_document() -> str:
    """A ``speed`` session holding one ready document in speed's collection."""
    sid = f"switch-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
    get_rag_database().save_document(
        {
            "document_id": f"doc_{sid}",
            "session_id": sid,
            "filename": "glaze.txt",
            "file_size": 10,
            "file_type": "txt",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/glaze.txt",
        collection_name=collection_name_for_settings(mgr.get_resolved_config(sid)),
    )
    return sid


def _active(sid):
    return get_knowledge_config_manager(session_id=sid).get_active_profile(session_id=sid)


@pytest.fixture(autouse=True)
def _models_load(monkeypatch):
    """A forced switch checks that the new profile's models load; here they do."""
    from lasagna.core import model_downloads

    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def test_the_refusal_names_the_reindex_on_every_surface():
    sid = _session_with_a_document()
    with pytest.raises(EmbeddingSizeConflict) as refused:
        get_knowledge_config_manager(session_id=sid).set_active_profile("standard", session_id=sid)
    message = " ".join(str(refused.value).split())
    for way in (
        "client.documents.reindex(document_id)",
        "POST /api/v1/documents/<id>/reindex",
        "POST /api/knowledge/reindex",
        "reindex_document",
        "force=true over REST and MCP",
        "lasagna configure --reindex",
        "reindex=true over REST and MCP",
        "keeps the session as it is if one fails",
    ):
        assert way in message, way


@pytest.fixture
def rest(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    return TestClient(create_app())


def test_rest_switches_with_force(rest):
    sid = _session_with_a_document()
    headers = {"X-Session-ID": sid}
    refused = rest.put("/api/v1/config/profiles/standard", headers=headers)
    assert refused.status_code == 409 and "POST /api/knowledge/reindex" in refused.json()["detail"]
    assert _active(sid) == "speed"
    assert (
        rest.put("/api/v1/config/profiles/standard?force=true", headers=headers).status_code == 200
    )
    assert _active(sid) == "standard"


def test_the_configurator_route_switches_with_force(rest):
    sid = _session_with_a_document()
    body = {"active_profile": "standard"}
    headers = {"X-Session-ID": sid}
    assert rest.put("/api/config/knowledge/active", json=body, headers=headers).status_code == 409
    forced = rest.put("/api/config/knowledge/active", json={**body, "force": True}, headers=headers)
    assert forced.status_code == 200, forced.text
    assert _active(sid) == "standard"


def test_mcp_switches_with_force():
    sid = _session_with_a_document()
    assert server.set_active_profile("standard", session_id=sid).startswith("Failed")
    assert "activated" in server.set_active_profile("standard", session_id=sid, force=True)
    assert _active(sid) == "standard"


def test_configure_offers_the_reindex():
    sid = _session_with_a_document()
    refused = CliRunner().invoke(cli.app, ["configure", "--profile", "standard", "-s", sid])
    words = " ".join(refused.output.split())
    assert refused.exit_code == 1, refused.output
    assert f"lasagna configure --profile standard -s {sid} --reindex" in words
    assert _active(sid) == "speed"
