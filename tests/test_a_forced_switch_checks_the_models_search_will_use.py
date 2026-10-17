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


def test_a_profile_whose_own_reranker_is_missing_is_still_refused(served_reranker):
    sid = f"forced-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    with pytest.raises(_profile_switch.ProfileCannotLoad, match=PACKAGED):
        _profile_switch.switch_profile(mgr, "standard", session_id=sid, force=True)
