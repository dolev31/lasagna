"""A session's settings beat the environment only where the session chose them.

Search reads a session's settings above the environment's. What it was handed
as "the session's settings" was the active profile's whole configuration: for
``default`` the packaged settings file, for a built-in profile that file plus
the profile's own few settings. So with the persistent session backend (the
default of ``lasagna``, ``lasagna-mcp`` and now the compose stack) every
session searched with the settings file's ``vector_backend = "milvus"`` and
``embedding_model = "BAAI/bge-m3"`` whatever ``RAG_VECTOR_BACKEND`` and
``RAG_EMBEDDING_MODEL`` said. The compose stack configured for Qdrant opened a
Milvus Lite file per process, and its two api workers then refused each
other's lock. The in-memory backend did the same once a profile was applied.

Found running ``tests/live/test_deployment.py`` against that stack.
"""

from __future__ import annotations

import pytest

from lasagna.core.config import RAGConfig
from lasagna.core.session_config import get_knowledge_config_manager
from lasagna.core.session_config._profiles import ProfileMixin


@pytest.fixture(params=["memory", "persistent"])
def sessions(request, monkeypatch, tmp_path):
    """Fresh providers on each session backend, with the environment a deployment sets."""
    import lasagna.providers as providers
    from lasagna.core.config import reset_rag_config

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_SESSION_BACKEND", request.param)
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "384")
    providers.reset_providers()
    reset_rag_config()
    yield request.param
    providers.reset_providers()
    reset_rag_config()


def _session(session_id):
    return RAGConfig().with_session(session_id)


def _new_id():
    import uuid

    return f"chosen-{uuid.uuid4().hex[:8]}"


def test_a_session_that_chose_nothing_uses_the_environment(sessions):
    config = _session(_new_id())

    assert config.vector_backend == "qdrant"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding_dim == 384


def test_a_profile_sets_its_own_settings_and_nothing_else(sessions):
    sid = _new_id()
    get_knowledge_config_manager(session_id=sid).set_active_profile("speed", session_id=sid)
    config = _session(sid)
    speed = ProfileMixin._PROFILE_SPECS["speed"]

    assert config.embedding_model == speed["embedding_model"]
    assert config.enable_reranking is speed["enable_reranking"]
    assert config.vector_backend == "qdrant", "the settings file's backend beat the environment"


def test_switching_back_to_default_drops_the_profile_settings(sessions):
    sid = _new_id()
    manager = get_knowledge_config_manager(session_id=sid)
    manager.set_active_profile("speed", session_id=sid)
    manager.set_active_profile("default", session_id=sid)

    assert _session(sid).embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_an_explicit_override_still_wins(sessions):
    sid = _new_id()
    manager = get_knowledge_config_manager(session_id=sid)
    manager.patch_config({"rerank_top_k": 17}, session_id=sid)

    assert _session(sid).rerank_top_k == 17
    assert _session(sid).vector_backend == "qdrant"
