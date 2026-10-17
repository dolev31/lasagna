"""An SDK client's settings are refused when they would hide its session's documents.

``KnowledgeClient(session_id="r1", config=config_for_speed())`` on a session
whose documents BGE-M3 embedded wrote speed's model as r1's settings: every
surface then searched r1 in all-MiniLM-L6-v2's empty collection, while the CLI,
REST and MCP refuse that change. A later ``set_config(embedding_model=...)``
did the same, and the SDK docs said it had no effect. The SDK's writes were
``trusted``, which skipped the documents check; only the evaluation harness is
exempt now, for its own sessions (``keep_in_process``).

The advice for documents in another model's collection is generated in one
place and does what it says, and a refusal names the profile's model and the
session's override apart, comparing the model and size on both sides.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

MINILM, BGE_M3, SMALL = "all-MiniLM-L6-v2", "BAAI/bge-m3", "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _mgr(sid):
    return get_knowledge_config_manager(session_id=sid)


def _session(profile: str, *names: str, collection: str = "") -> str:
    """A session on ``profile`` holding ready documents ``names`` (one, "a", by default)."""
    sid = f"sdk-{uuid.uuid4().hex[:8]}"
    _mgr(sid).set_active_profile(profile, session_id=sid, force=True)
    collection = collection or collection_name_for_settings(_mgr(sid).get_resolved_config(sid))
    for name in names or ("a",):
        if not name:
            continue
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
            collection_name=collection,
        )
    return sid


def _searches_with(sid: str) -> str:
    return get_rag_config().with_session(sid).embedding_model


class _Backend:
    """Records the model each reindex embeds with."""

    def __init__(self):
        self.models = []

    def reindex_document(self, document_id, session_id, *, config=None):
        self.models.append((config or get_rag_config().with_session(session_id)).embedding_model)
        with get_rag_database().transaction() as conn:
            conn.execute(
                "UPDATE rag_documents SET collection_name = ? WHERE document_id = ?",
                (
                    (config or get_rag_config().with_session(session_id)).collection_name,
                    document_id,
                ),
            )


@pytest.fixture
def client_for(monkeypatch):
    import copy

    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.config import rag_config
    from lasagna.sdk import KnowledgeClient

    backend = _Backend()
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    # A client's init-time fields go where environment variables do, for the
    # whole process: this test's must not reach the next one's.
    monkeypatch.setattr(rag_config, "_env_overrides", copy.deepcopy(rag_config._env_overrides))

    def make(sid, config):
        client = KnowledgeClient(session_id=sid, config=config)
        return client, backend

    return make


def test_config_for_speed_on_a_session_with_bge_m3_documents_is_refused(client_for):
    from lasagna.api._profile_guard import DocumentsLeftBehind
    from lasagna.sdk import config_for_speed

    sid = _session("standard", "a", "b")
    client, _ = client_for(sid, config_for_speed())

    for _ in range(2):  # every operation, not only the first
        with pytest.raises(DocumentsLeftBehind) as refused:
            asyncio.run(client._get_backend())
        message = str(refused.value)
        assert f"embedded with {BGE_M3}" in message and f"search with {MINILM}" in message
        assert "await client.set_config(reindex=True)" in message
        assert "await client.set_config(force=True)" in message
    assert _searches_with(sid) == BGE_M3
    assert _mgr(sid).get_active_profile(sid) == "standard"


def test_a_later_set_config_of_the_embedding_model_is_refused_too(client_for):
    from lasagna.api._profile_guard import DocumentsLeftBehind
    from lasagna.sdk import config_for_speed

    sid = _session("speed")
    client, _ = client_for(sid, config_for_speed())
    asyncio.run(client._get_backend())  # the session has these settings already

    with pytest.raises(DocumentsLeftBehind):
        asyncio.run(client.set_config(embedding_model=SMALL, embedding_dim=384))
    assert _searches_with(sid) == MINILM

    asyncio.run(client.set_config(force=True))
    assert _searches_with(sid) == SMALL


def test_reindex_moves_every_document_then_applies_the_settings(client_for):
    from lasagna.sdk import config_for_speed

    sid = _session("speed", "a", "b")
    client, backend = client_for(sid, config_for_speed())
    asyncio.run(client._get_backend())

    asyncio.run(client.set_config(embedding_model=SMALL, embedding_dim=384, reindex=True))
    assert backend.models == [SMALL, SMALL]
    assert _searches_with(sid) == SMALL
    small = get_rag_config().with_session(sid).collection_name
    assert {r["collection_name"] for r in get_rag_database().get_documents(sid)} == {small}


def test_settings_the_session_has_already_are_not_written(client_for, monkeypatch):
    from lasagna.core.session_config import KnowledgeConfigManager
    from lasagna.sdk import config_for_speed

    sid = _session("speed")
    writes = []
    original = KnowledgeConfigManager.patch_config
    monkeypatch.setattr(
        KnowledgeConfigManager,
        "patch_config",
        lambda self, patch, *a, **k: writes.append(patch) or original(self, patch, *a, **k),
    )
    client, _ = client_for(sid, config_for_speed())
    asyncio.run(client._get_backend())
    assert writes == []


def test_only_the_harness_sessions_are_exempt():
    from lasagna.api._profile_guard import DocumentsLeftBehind
    from lasagna.core.session_config import keep_in_process

    move = {"embedding_model": SMALL, "embedding_dim": 384}
    sid = _session("speed")
    with pytest.raises(DocumentsLeftBehind):
        _mgr(sid).patch_config(move, session_id=sid, trusted=True)

    harness = _session("speed")
    keep_in_process(harness)
    _mgr(harness).patch_config(move, session_id=harness, trusted=True)  # not refused
    assert _searches_with(harness) == SMALL


def test_the_offered_reindex_reindexes_with_the_sessions_own_model():
    """A session on speed whose override embeds with bge-small, documents in MiniLM's collection."""
    from lasagna.api import _profile_switch
    from lasagna.api._profile_guard import documents_elsewhere, forget_documents_elsewhere
    from lasagna.core.diagnostics import _check_session_documents

    sid = _session("speed", "a")
    _mgr(sid).patch_config(
        {"embedding_model": SMALL, "embedding_dim": 384}, session_id=sid, force=True
    )
    config = get_rag_config().with_session(sid)
    warning = documents_elsewhere(sid, config.collection_name, config.embedding_model)
    command = f"lasagna configure --profile speed -s {sid} --reindex"
    assert command in warning
    assert command in _check_session_documents(sid).advice  # doctor offers the same
    # A switch back to speed alone keeps the override, so it is not what is offered.
    assert f"(lasagna configure --profile speed -s {sid})" not in warning
    assert f"lasagna configure --profile speed --reset -s {sid}" in warning

    backend = _Backend()
    _profile_switch.switch_profile(
        _mgr(sid), "speed", session_id=sid, reindex=True, backend=backend
    )
    assert backend.models == [SMALL]
    forget_documents_elsewhere(sid)
    assert documents_elsewhere(sid, config.collection_name, SMALL) is None


def test_the_offered_reset_makes_the_session_search_where_its_documents_are():
    from lasagna.api._profile_guard import documents_elsewhere, forget_documents_elsewhere

    sid = _session("speed", "a")
    _mgr(sid).patch_config(
        {"embedding_model": SMALL, "embedding_dim": 384}, session_id=sid, force=True
    )
    _mgr(sid).reset_config(session_id=sid)
    forget_documents_elsewhere(sid)
    config = get_rag_config().with_session(sid)
    assert config.embedding_model == MINILM
    assert documents_elsewhere(sid, config.collection_name, MINILM) is None


def test_without_an_override_the_offered_switch_back_works():
    from lasagna.api._profile_guard import documents_elsewhere, forget_documents_elsewhere

    speed_collection = collection_name_for_settings(
        {"embedding_model": MINILM, "embedding_dim": 384}
    )
    sid = _session("standard", "a", collection=speed_collection)
    config = get_rag_config().with_session(sid)
    warning = documents_elsewhere(sid, config.collection_name, config.embedding_model)
    assert f"(lasagna configure --profile speed -s {sid})" in warning

    _mgr(sid).set_active_profile("speed", session_id=sid)
    forget_documents_elsewhere(sid)
    config = get_rag_config().with_session(sid)
    assert documents_elsewhere(sid, config.collection_name, config.embedding_model) is None


def test_a_refusal_names_the_profiles_model_and_the_override_apart():
    from lasagna.api._profile_guard import DocumentsLeftBehind

    sid = _session("speed", "")  # no documents yet
    _mgr(sid).patch_config({"embedding_model": SMALL}, session_id=sid, force=True)
    here = get_rag_config().with_session(sid).collection_name
    get_rag_database().save_document(
        {
            "document_id": f"s_{sid}",
            "session_id": sid,
            "filename": "s.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/s.md",
        collection_name=here,
    )
    with pytest.raises(DocumentsLeftBehind) as refused:
        _mgr(sid).set_active_profile("standard", session_id=sid)
    message = str(refused.value)
    assert f"would search with {SMALL} at 1024 dimensions" in message
    assert f"the profile embeds with {BGE_M3}" in message
    assert f"override embedding_model={SMALL}" in message
    assert f"embedded with {SMALL} at 384 dimensions" in message
    assert "switch with --reset (lasagna configure --reset" in message


def test_get_active_profile_names_the_profiles_model_and_the_override():
    import json

    from lasagna.mcp import server

    sid = _session("speed")
    _mgr(sid).patch_config(
        {"embedding_model": SMALL, "embedding_dim": 384}, session_id=sid, force=True
    )
    answer = json.loads(server.get_active_profile(session_id=sid))
    assert answer["active_profile_name"] == "speed"
    assert answer["profile_embedding_model"] == MINILM
    assert answer["session_overrides"]["embedding_model"] == SMALL
    assert answer["embeds_with"]["embedding_model"] == SMALL
