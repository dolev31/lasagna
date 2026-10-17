"""A settings change the store does not save is refused on every surface, and nothing changes.

Both session stores logged a failed write as a warning and went on: every patch,
import and switch answered 200 with nothing stored, and within a second the
process read the old row back. A switch with reindex had already moved the
documents and deleted their old chunks when its write failed, so the session
searched an empty collection. A failed write now raises ``SettingsNotSaved``
(503 over REST), the process keeps the settings it had, and a switch moves the
documents back first.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager

import pytest

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._data_models import SettingsNotSaved
from tests._postgres_store import FakePostgres, manager_class_on

SMALL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _sid() -> str:
    return f"unsaved-{uuid.uuid4().hex[:8]}"


@contextmanager
def _sqlite_writes_fail(monkeypatch):
    db = get_rag_database()

    def full(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(db, "save_session", full)
    try:
        yield
    finally:
        monkeypatch.undo()


# -- each change, on each store --------------------------------------------------


def _changes(sid):
    return {
        "patch": lambda m: m.patch_config({"rerank_top_k": 13}, session_id=sid),
        "import": lambda m: m.import_config(
            {"schema_version": 2, "profiles": [], "session_overrides": {"rerank_top_k": 13}},
            session_id=sid,
        ),
        "switch": lambda m: m.set_active_profile("balanced", session_id=sid),
    }


@pytest.mark.parametrize("change", ["patch", "import", "switch"])
def test_on_sqlite_a_failed_write_raises_and_the_session_keeps_its_settings(monkeypatch, change):
    sid = _sid()
    manager = KnowledgeConfigManager()
    manager.set_active_profile("speed", session_id=sid)
    with _sqlite_writes_fail(monkeypatch):
        with pytest.raises(SettingsNotSaved) as refused:
            _changes(sid)[change](manager)
    message = str(refused.value)
    assert f"session {sid!r}" in message and "nothing was changed" in message
    assert "database or disk is full" in message
    for m in (manager, KnowledgeConfigManager()):  # this process, and the store
        assert m.get_active_profile(sid) == "speed"
        assert "rerank_top_k" not in m.get_session_state(sid).session_overrides


@pytest.mark.parametrize("change", ["patch", "import", "switch"])
def test_on_postgres_a_failed_write_raises_and_the_session_keeps_its_settings(monkeypatch, change):
    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    sid = _sid()
    manager = on_postgres()
    manager.set_active_profile("speed", session_id=sid)
    store.fail = True
    with pytest.raises(SettingsNotSaved) as refused:
        _changes(sid)[change](manager)
    assert "Connection refused" in str(refused.value)
    store.fail = False
    for m in (manager, on_postgres()):
        assert m.get_active_profile(sid) == "speed"
        assert "rerank_top_k" not in m.get_session_state(sid).session_overrides
    assert store.sessions[sid]["active_profile"] == "speed"


def test_rest_answers_503_with_the_reason(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lasagna.api.rest.app import _answer_switch_in_progress
    from lasagna.api.rest.routes.knowledge_config import router

    app = FastAPI()
    _answer_switch_in_progress(app)
    app.include_router(router)
    client = TestClient(app)
    sid = _sid()
    headers = {"X-Session-ID": sid}
    with _sqlite_writes_fail(monkeypatch):
        put = client.put(
            "/api/config/knowledge/active", json={"active_profile": "speed"}, headers=headers
        )
        imported = client.post(
            "/api/config/knowledge/import",
            json={"schema_version": 2, "profiles": [], "session_overrides": {"rerank_top_k": 4}},
            headers=headers,
        )
    for response in (put, imported):
        assert response.status_code == 503, response.text
        assert "could not be saved" in response.json()["detail"]


def test_a_switch_whose_write_fails_moves_the_documents_back(monkeypatch):
    from lasagna.api import _profile_switch

    sid = _sid()
    manager = KnowledgeConfigManager()
    manager.set_active_profile("speed", session_id=sid)
    speed = get_rag_config().with_session(sid).collection_name
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
            },
            f"/nonexistent/{name}.md",
            collection_name=speed,
        )

    class _Backend:
        def __init__(self):
            self.moves = []

        def reindex_document(self, document_id, session_id, *, config=None):
            config = config or get_rag_config().with_session(session_id)
            self.moves.append((document_id, config.embedding_model))
            get_rag_database().update_document_fields(
                document_id, {"collection_name": config.collection_name}
            )

    backend = _Backend()
    with _sqlite_writes_fail(monkeypatch):
        with pytest.raises(SettingsNotSaved) as refused:
            _profile_switch.switch_profile(
                manager,
                "speed",
                session_id=sid,
                reindex=True,
                config={"embedding_model": SMALL, "embedding_dim": 384},
                backend=backend,
            )
    assert "moved back" in str(refused.value)
    assert [m for _, m in backend.moves] == [SMALL, SMALL, "all-MiniLM-L6-v2", "all-MiniLM-L6-v2"]
    assert {r["collection_name"] for r in get_rag_database().get_documents(sid)} == {speed}
    assert get_rag_config().with_session(sid).collection_name == speed
    assert get_rag_database().get_profile_switch(sid) is None  # the journal is closed
    assert collection_name_for_settings({"embedding_model": SMALL, "embedding_dim": 384}) != speed


def test_the_cli_says_so_in_one_message(monkeypatch, capsys):
    from lasagna import cli

    monkeypatch.setattr("sys.argv", ["lasagna", "configure", "--profile", "speed", "-s", _sid()])
    with _sqlite_writes_fail(monkeypatch):
        with pytest.raises(SystemExit) as exited:
            cli.main()
    assert exited.value.code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "could not be saved (OperationalError: database or disk is full)" in out
    assert "Traceback" not in out
