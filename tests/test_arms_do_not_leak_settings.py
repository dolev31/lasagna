"""A sweep arm must not inherit the previous arm's settings.

Arms share one corpus session. A setting the profile does not name resolved
from that session, which still held the previous arm's value: a control ran
with the treatment's rrf_k and the sweep reported no effect.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from lasagna.core.config import rag_config
from lasagna.knowledge_evaluation_sdk.experiment import (
    _apply_overrides_to_pipeline,
    _complete_stage_settings,
)
from lasagna.knowledge_evaluation_sdk.stages import stage_of


def test_every_stage_setting_gets_an_explicit_value():
    config = {"enable_reranking": True}
    _complete_stage_settings(config)
    assert config["enable_reranking"] is True  # what the arm set is kept
    assert config["rrf_k"] == rag_config.rrf_k
    assert all(stage_of(k) is not None for k in config), "operational keys copied"
    assert "database_dsn" not in config and "redis_url" not in config


def _pipeline(profile_config):
    p = MagicMock()
    p._config = dict(profile_config)
    return p


def test_the_next_arm_resolves_its_own_rrf_k(monkeypatch):
    from lasagna.providers.session_provider import InMemorySessionProvider
    import lasagna.providers as providers

    provider = InMemorySessionProvider()
    monkeypatch.setattr(providers, "get_session_provider", lambda: provider)
    sid = "bench_corpus_leak_test"
    profile = {"enable_hybrid_search": True}

    treatment = _pipeline(profile)
    _apply_overrides_to_pipeline(treatment, {"rrf_k": 1}, sid)
    control = _pipeline(profile)
    _apply_overrides_to_pipeline(control, {}, sid)

    task_config = dict(control._config)
    assert task_config["rrf_k"] == rag_config.rrf_k
    stored = provider._sessions[sid]["overrides"]
    assert stored["rrf_k"] == rag_config.rrf_k, "the session still holds the treatment's rrf_k"
