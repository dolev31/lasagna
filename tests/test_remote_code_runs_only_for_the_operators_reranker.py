"""``reranker_trust_remote_code`` trusts the operator's reranker, not a session's.

The setting is the server's (a session may not set it), but ``reranker_model``
is a session setting, and the reranker read the trust flag from the config of
each call. With trust on for, say, jina-reranker-v2, any session could name any
Hugging Face repository as its reranker and the server ran that repository's
modelling code. Trust now applies to the model names the operator gave with
it: the process settings, or a sweep arm's own overrides; a model a session
chooses loads with trust off.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.config import get_rag_config
from lasagna.providers import get_session_provider
from lasagna.retrieval.search import ranker

OPERATOR = "jinaai/jina-reranker-v2-base-multilingual"
SESSION_CHOICE = "someone/arbitrary-repo"


class _CrossEncoder:
    loaded: list = []

    def __init__(self, name, device=None, **kwargs):
        _CrossEncoder.loaded.append((name, kwargs.get("trust_remote_code", False)))

    def predict(self, pairs, **kw):
        return [float(len(p[1])) for p in pairs]


#: The server's settings (the settings file or the environment): trust on, for
#: the operator's reranker.
SERVER = {"reranker_trust_remote_code": True, "reranker_model": OPERATOR, "enable_reranking": True}


@pytest.fixture
def server(monkeypatch):
    _CrossEncoder.loaded = []
    monkeypatch.setattr(ranker, "CrossEncoder", _CrossEncoder, raising=False)
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    monkeypatch.setattr(ranker.model_downloads, "check", lambda *a, **k: None)
    monkeypatch.setattr(ranker, "_refuse_unscorable_reranker", lambda *a: None)
    settings = get_rag_config()
    monkeypatch.setattr(settings, "_env_overrides", {**(settings._env_overrides or {}), **SERVER})
    assert settings.reranker_trust_remote_code is True and settings.reranker_model == OPERATOR
    monkeypatch.setattr(ranker, "rag_config", settings)
    return settings


def _session(**settings):
    """A session's config, with what it set through the session store."""
    session_id = f"trust-{uuid.uuid4().hex[:8]}"
    get_session_provider().set_config_overrides(session_id, settings)
    return get_rag_config().with_session(session_id)


RESULTS = [{"content": "short", "score": 0.5}, {"content": "a longer passage", "score": 0.4}]


def _rerank(config):
    ranker.CrossEncoderReranker().rerank("q", [dict(r) for r in RESULTS], config=config)


def test_a_model_a_session_names_loads_without_remote_code(server):
    session = _session(reranker_model=SESSION_CHOICE)
    assert session.reranker_model == SESSION_CHOICE
    assert session.reranker_trust_remote_code is True  # the server's flag, in force
    _rerank(session)
    assert _CrossEncoder.loaded == [(SESSION_CHOICE, False)]


def test_the_operators_model_loads_with_remote_code(server):
    _rerank(server)
    assert _CrossEncoder.loaded == [(OPERATOR, True)]


def test_a_session_on_the_operators_model_gets_its_trust(server):
    _rerank(_session(rerank_top_k=1))
    assert _CrossEncoder.loaded == [(OPERATOR, True)]


def test_an_ensemble_member_a_session_names_loads_without_remote_code(server):
    session = _session(ensemble_reranker_models=[OPERATOR, SESSION_CHOICE])
    ranker.EnsembleReranker().rerank("q", [dict(r) for r in RESULTS], config=session)
    assert SESSION_CHOICE in {name for name, _ in _CrossEncoder.loaded}
    assert {name for name, trust in _CrossEncoder.loaded if trust} <= {OPERATOR}


def test_a_sweep_arm_that_names_its_model_with_trust_gets_it(server):
    """Per-query overrides are set by code in this process, never by a session."""
    arm = get_rag_config().with_query_overrides(
        {"reranker_model": "lab/own-reranker", "reranker_trust_remote_code": True}
    )
    _rerank(arm)
    assert _CrossEncoder.loaded == [("lab/own-reranker", True)]


def test_with_trust_off_nothing_loads_with_remote_code(server, monkeypatch):
    monkeypatch.setattr(
        server, "_env_overrides", {**server._env_overrides, "reranker_trust_remote_code": False}
    )
    _rerank(server)
    assert _CrossEncoder.loaded == [(OPERATOR, False)]
