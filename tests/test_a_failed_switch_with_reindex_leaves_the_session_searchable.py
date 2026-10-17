"""A profile switch with reindex changes the session only once every document is reindexed.

``lasagna configure --profile speed -s r1``, an upload of three files, then
``lasagna configure --profile standard -s r1 --reindex`` with BGE-M3 not
downloaded printed "Applied profile: standard ... Reindexed 0 of 3" and
exited 1; ``lasagna search -s r1`` then failed with "BAAI/bge-m3 is not
downloaded", although the help said each document keeps its old index until
its new one is stored, and ``doctor -s r1`` reported "0 blocking". The switch
happened first and the reindex after.

Now the new profile's models are checked before anything changes, every
document is reindexed with the new profile while the session stays on the old
one, and the session switches only when all of them are; after a failure the
documents already moved are moved back. The same holds over REST and MCP, and
``doctor`` reports documents in another model's collection as blocking.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

MINILM = "all-MiniLM-L6-v2"


def _session_with_documents(*document_ids: str) -> str:
    """A ``speed`` session holding ready documents in speed's collection."""
    sid = f"switch-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
    collection = collection_name_for_settings(mgr.get_resolved_config(sid))
    for document_id in document_ids:
        get_rag_database().save_document(
            {
                "document_id": f"{document_id}_{sid}",
                "session_id": sid,
                "filename": f"{document_id}.md",
                "file_size": 10,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
            },
            f"/nonexistent/{document_id}.md",
            collection_name=collection,
        )
    return sid


def _active(sid: str) -> str:
    return get_knowledge_config_manager(session_id=sid).get_active_profile(session_id=sid)


class _Backend:
    """Records each reindex: the document, the model it embeds with, the session's profile."""

    def __init__(self, fail: str = ""):
        self.fail, self.calls = fail, []

    @staticmethod
    def get_session_config(session_id):
        return get_rag_config().with_session(session_id)

    def reindex_document(self, document_id, session_id, *, config=None):
        searched = config or get_rag_config().with_session(session_id)
        self.calls.append(
            (document_id.split("_")[0], searched.embedding_model, _active(session_id))
        )
        if config is not None and document_id.startswith(self.fail or "\0"):
            raise RuntimeError("Reindex of the document failed, and it is unchanged: bad table")


@pytest.fixture
def models_load(monkeypatch):
    """Every model loads, so the reindex itself is what is tested."""
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


@pytest.fixture
def backend(monkeypatch):
    holder = {"backend": _Backend()}
    import lasagna.api.rag_backend as rag_backend

    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: holder["backend"])
    monkeypatch.setattr(rag_backend, "close_backend", lambda: None)
    from lasagna.core import local_server

    monkeypatch.setattr(local_server, "local_store_held", lambda: False)
    return holder


def _configure(*argv):
    result = CliRunner().invoke(cli.app, ["configure", *argv])
    return result, " ".join(result.output.split())


def test_a_model_that_cannot_load_changes_nothing(monkeypatch, backend):
    def refusal(model, kind="embedding model", filename="config.json"):
        if model == MINILM:
            return None
        return model_downloads.ModelNotDownloaded(model, offline=True, kind=kind)

    monkeypatch.setattr(model_downloads, "refusal", refusal)
    sid = _session_with_documents("a", "b", "c")
    result, out = _configure("--profile", "standard", "-s", sid, "--reindex")
    assert result.exit_code == 1, out
    assert "BAAI/bge-m3 is not downloaded" in out
    assert "Nothing was changed" in out and "stays on profile 'speed'" in out
    assert "hf download BAAI/bge-m3 --include" in out and "onnx" not in out
    assert "Applied profile" not in out
    assert _active(sid) == "speed"
    assert backend["backend"].calls == []


def test_every_document_is_reindexed_before_the_switch(models_load, backend):
    sid = _session_with_documents("a", "b")
    result, out = _configure("--profile", "standard", "-s", sid, "--reindex")
    assert result.exit_code == 0, out
    # Each embedded with the new profile's model while the session was still on speed.
    assert backend["backend"].calls == [
        ("a", "BAAI/bge-m3", "speed"),
        ("b", "BAAI/bge-m3", "speed"),
    ]
    assert _active(sid) == "standard"
    assert f"Reindexed 2 of 2 documents in session {sid} with BAAI/bge-m3." in out


def test_a_failed_reindex_keeps_the_old_profile_and_moves_the_others_back(models_load, backend):
    backend["backend"] = _Backend(fail="b")
    sid = _session_with_documents("a", "b", "c")
    result, out = _configure("--profile", "standard", "-s", sid, "--reindex")
    assert result.exit_code == 1, out
    assert _active(sid) == "speed"
    assert backend["backend"].calls == [
        ("a", "BAAI/bge-m3", "speed"),
        ("b", "BAAI/bge-m3", "speed"),
        ("a", MINILM, "speed"),  # moved back with the session's own settings
    ]
    assert "Profile 'standard' was not applied" in out and "stays on profile 'speed'" in out
    assert f"b_{sid}" in out and "bad table" in out
    assert "moved back" in out
    assert "Applied profile" not in out


def test_rest_switches_only_after_the_reindex(models_load, backend, monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    from lasagna.api.rest.app import create_app

    rest = TestClient(create_app())
    backend["backend"] = _Backend(fail="b")
    sid = _session_with_documents("a", "b")
    body = {"active_profile": "standard", "reindex": True}
    headers = {"X-Session-ID": sid}
    refused = rest.put("/api/config/knowledge/active", json=body, headers=headers)
    assert refused.status_code == 409, refused.text
    assert "was not applied" in refused.json()["detail"]
    assert _active(sid) == "speed"

    backend["backend"] = _Backend()
    moved = rest.put("/api/v1/config/profiles/standard?reindex=true", headers=headers)
    assert moved.status_code == 200, moved.text
    assert moved.json()["reindex"]["reindexed"] == [f"a_{sid}", f"b_{sid}"]
    assert _active(sid) == "standard"


def test_mcp_switches_only_after_the_reindex(models_load, backend):
    from lasagna.mcp import server

    backend["backend"] = _Backend(fail="a")
    sid = _session_with_documents("a")
    answer = server.set_active_profile("standard", session_id=sid, reindex=True)
    assert answer.startswith("Failed") and "was not applied" in answer
    assert _active(sid) == "speed"
    backend["backend"] = _Backend()
    answer = server.set_active_profile("standard", session_id=sid, reindex=True)
    assert "activated" in answer and "Reindexed 1 of 1 document" in answer
    assert _active(sid) == "standard"


def test_a_forced_switch_to_a_model_that_cannot_load_is_refused(monkeypatch):
    monkeypatch.setattr(
        model_downloads,
        "refusal",
        lambda model, kind="embedding model", filename="config.json": (
            None if model == MINILM else model_downloads.ModelNotDownloaded(model, True, kind)
        ),
    )
    sid = _session_with_documents("a")
    result, out = _configure("--profile", "standard", "-s", sid, "--force")
    assert result.exit_code == 1 and "BAAI/bge-m3 is not downloaded" in out
    assert _active(sid) == "speed"


def test_doctor_blocks_on_documents_another_model_indexed(models_load):
    sid = _session_with_documents("a", "b")
    get_knowledge_config_manager(session_id=sid).set_active_profile(
        "standard", session_id=sid, force=True
    )
    from lasagna.core.diagnostics import _check_session_documents

    check = _check_session_documents(sid)
    assert check.status == "error"
    assert f"2 of 2 documents in session {sid!r}" in check.detail
    assert "BAAI/bge-m3" in check.detail
    assert f"lasagna configure --profile standard -s {sid} --reindex" in check.advice
    assert f"lasagna configure --profile speed -s {sid}" in check.advice

    get_knowledge_config_manager(session_id=sid).set_active_profile("speed", session_id=sid)
    assert _check_session_documents(sid).status == "ok"


def test_pinned_settings_choose_the_collection_without_touching_the_session():
    sid = _session_with_documents("a")
    config = get_rag_config().with_session_settings(
        sid, {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384}
    )
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert "bge_small" in config.collection_name.replace("-", "_").lower()
    assert get_rag_config().with_session(sid).embedding_model == MINILM


def test_a_switch_report_says_when_there_was_nothing_to_reindex():
    from lasagna.api import _profile_switch

    report = _profile_switch.SwitchReport(
        profile="standard", previous="speed", session_id="s", reindex=True
    )
    assert _profile_switch.summary(report) == "Session s has no documents to reindex."
