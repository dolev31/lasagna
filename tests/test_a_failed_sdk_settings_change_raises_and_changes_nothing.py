"""An SDK settings change that cannot apply raises, and the client reports what is in force.

``await client.set_config(embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768,
reindex=True)`` on a session with three MiniLM documents, offline, returned
normally: the switch refused (the model is not downloaded) and the client turned
every failure but a documents refusal into a log line, so ``get_config_dict()``
named bge-base while the session searched with MiniLM. A store that did not save
the settings was logged the same way. With ``force=True`` the model was not
checked at all, as ``lasagna configure --force`` checks it, and the session was
left on a model that could not load.
"""

from __future__ import annotations

import asyncio
import copy
import uuid

import pytest

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

MINILM, BASE = "all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"


@pytest.fixture(autouse=True)
def base_is_not_downloaded(monkeypatch):
    def refusal(model, kind="embedding model", *args, **kwargs):
        if model == BASE:
            return model_downloads.ModelNotDownloaded(model, offline=True, kind=kind)
        return None

    monkeypatch.setattr(model_downloads, "refusal", refusal)


def _mgr(sid):
    return get_knowledge_config_manager(session_id=sid)


def _session_with_three_documents() -> str:
    sid = f"sdkfail-{uuid.uuid4().hex[:8]}"
    _mgr(sid).set_active_profile("speed", session_id=sid, force=True)
    collection = collection_name_for_settings(_mgr(sid).get_resolved_config(sid))
    for name in ("a", "b", "c"):
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


class _Backend:
    """Records each reindex; none should happen."""

    def __init__(self):
        self.reindexed = []

    def reindex_document(self, document_id, session_id, *, config=None):
        self.reindexed.append(document_id)


@pytest.fixture
def client(monkeypatch):
    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.config import rag_config
    from lasagna.sdk import KnowledgeClient, config_for_speed

    backend = _Backend()
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    monkeypatch.setattr(rag_config, "_env_overrides", copy.deepcopy(rag_config._env_overrides))
    sid = _session_with_three_documents()
    made = KnowledgeClient(session_id=sid, config=config_for_speed())
    made._test_backend = backend
    return made


def _in_force(client) -> str:
    return get_rag_config().with_session(client.session_id).embedding_model


@pytest.mark.parametrize("how", [{"reindex": True}, {"force": True}])
def test_a_model_that_cannot_load_is_refused_and_nothing_changes(client, how):
    from lasagna.sdk import ProfileCannotLoad

    with pytest.raises(ProfileCannotLoad) as refused:
        asyncio.run(client.set_config(embedding_model=BASE, embedding_dim=768, **how))
    message = str(refused.value)
    assert f"{BASE} is not downloaded" in message and "Nothing was changed" in message
    # True of the documents, as they are: all three in the model's collection it searches.
    assert "Its documents stay searchable." in message
    assert _in_force(client) == MINILM
    assert client.get_config_dict()["embedding_model"] == MINILM
    assert client._test_backend.reindexed == []


def test_a_store_that_does_not_save_the_settings_fails_the_change(client, monkeypatch):
    from lasagna.core.session_config import KnowledgeConfigManager

    asyncio.run(client._get_backend())  # the session has the client's settings

    def unsaved(self, *args, **kwargs):
        raise RuntimeError("database or disk is full")

    monkeypatch.setattr(KnowledgeConfigManager, "patch_config", unsaved)
    before = get_rag_config().with_session(client.session_id).rerank_top_k
    with pytest.raises(RuntimeError, match="disk is full"):
        asyncio.run(client.set_config(rerank_top_k=before + 3))
    assert client.get_config_dict()["rerank_top_k"] == before
    # Every later operation tries again, and says so while it fails.
    with pytest.raises(RuntimeError, match="disk is full"):
        asyncio.run(client._get_backend())

    monkeypatch.undo()
    asyncio.run(client.set_config(rerank_top_k=before + 3))
    assert get_rag_config().with_session(client.session_id).rerank_top_k == before + 3


def test_settings_refused_when_the_client_starts_are_reported_as_in_force(monkeypatch):
    """The client's own configuration, refused for its session: what is in force is shown."""
    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.config import rag_config
    from lasagna.sdk import DocumentsLeftBehind, KnowledgeClient, KnowledgeConfig

    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: _Backend())
    monkeypatch.setattr(rag_config, "_env_overrides", copy.deepcopy(rag_config._env_overrides))
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)
    sid = _session_with_three_documents()
    small = KnowledgeConfig(embedding_model="BAAI/bge-small-en-v1.5", embedding_dim=384)
    client = KnowledgeClient(session_id=sid, config=small)
    with pytest.raises(DocumentsLeftBehind):
        asyncio.run(client._get_backend())
    assert client.get_config_dict()["embedding_model"] == MINILM
