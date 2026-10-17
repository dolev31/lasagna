"""A forced switch checks the models search will run with, not the settings file's.

A forced switch (or one with reindex) first checks that the new profile's
models load. It checked the profile's whole configuration, which carries the
packaged settings file's models: on the Compose stack, started with
``RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`` and an image
holding only that reranker, ``PUT /api/config/knowledge/active`` with
``force`` was refused for mixedbread-ai/mxbai-rerank-large-v1, a model the
server never loads. It now checks the session's choice over the server's own
settings, as search reads them.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.api import _profile_switch
from lasagna.core import model_downloads
from lasagna.core.config import get_rag_config
from lasagna.core.session_config import get_knowledge_config_manager

SERVED = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PACKAGED = "mixedbread-ai/mxbai-rerank-large-v1"


@pytest.fixture
def served_reranker(monkeypatch):
    """The server runs SERVED (its environment) and has only that one downloaded."""
    monkeypatch.setitem(get_rag_config()._env_overrides, "reranker_model", SERVED)
    monkeypatch.setattr(
        model_downloads,
        "refusal",
        lambda model, kind="embedding model", filename="config.json": (
            model_downloads.ModelNotDownloaded(model, offline=False, kind=kind)
            if model == PACKAGED
            else None
        ),
    )


def test_a_forced_switch_is_not_refused_for_a_reranker_the_server_never_loads(served_reranker):
    sid = f"forced-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    report = _profile_switch.switch_profile(
        mgr,
        "default",
        session_id=sid,
        force=True,
        config={"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384},
    )
    assert report.embedding_model == "BAAI/bge-small-en-v1.5"
    assert get_rag_config().with_session(sid).reranker_model == SERVED


def test_a_profile_whose_own_reranker_is_missing_is_switched_to_with_the_download(
    served_reranker,
):
    """A search skips a reranker that is not downloaded and says so; the switch says it first.

    It was refused, while a search on that profile went through without the
    reranker (``degraded_stages`` ``reranker``, results in retrieval order).
    """
    sid = f"forced-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    report = _profile_switch.switch_profile(mgr, "standard", session_id=sid, force=True)
    assert mgr.get_active_profile(sid) == "standard"
    (warning,) = report.warnings
    assert PACKAGED in warning and f"hf download {PACKAGED}" in warning
    assert "retrieval order" in warning
    assert warning in _profile_switch.summary(report)


def test_a_profile_whose_embedder_is_missing_is_still_refused(monkeypatch):
    missing = "BAAI/bge-m3"
    monkeypatch.setattr(
        model_downloads,
        "refusal",
        lambda model, kind="embedding model", filename="config.json": (
            model_downloads.ModelNotDownloaded(model, offline=True, kind=kind)
            if model == missing
            else None
        ),
    )
    sid = f"forced-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    with pytest.raises(_profile_switch.ProfileCannotLoad, match=missing):
        _profile_switch.switch_profile(mgr, "standard", session_id=sid, force=True)
    assert mgr.get_active_profile(sid) == "default"


def test_doctor_says_the_same_of_a_reranker_that_is_not_downloaded(monkeypatch):
    from lasagna.core import diagnostics

    sid = f"forced-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("standard", session_id=sid, force=True)
    reranker = get_rag_config().with_session(sid).reranker_model
    monkeypatch.setattr(model_downloads, "offline", lambda: True)
    monkeypatch.setattr(model_downloads, "is_cached", lambda model, *a, **k: model != reranker)
    check = diagnostics._check_model_cache(sid)
    assert check.status == "warning"
    assert "skip the reranker and return results in retrieval order" in check.detail
    assert reranker in check.detail
