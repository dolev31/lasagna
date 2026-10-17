"""A session's settings name components, models and services the server allows.

The session allowlist refused server settings and paths that start with "/",
"~" or "./", and let through: a component slot holding an import reference
(``reranker: "os:system"``) or any unregistered name; ``search_params`` with
its own ``metric_type``, a server setting; a relative path such as
``models/x`` for a model, which loads that folder when it exists on the
server; and the settings that send documents or queries to a third party on
the server's API key (``embedding_provider``, ``crag_web_search_fallback``,
and the API-backed embedders and rerankers).
"""

from __future__ import annotations

import pytest

from lasagna.core.session_config._session_keys import (
    SessionSettingRefused,
    check_session_settings,
)


def _refused(settings):
    with pytest.raises(SessionSettingRefused) as refused:
        check_session_settings(settings)
    return str(refused.value)


@pytest.mark.parametrize(
    "settings",
    [
        {"reranker": "os:system"},
        {"reranker": "my_module:MyReranker"},
        {"chunker": "no_such_chunker"},
        {"generator": ["concise"]},
    ],
)
def test_a_component_slot_takes_registered_names_only(settings):
    assert "registered" in _refused(settings)


def test_a_registered_component_is_kept():
    kept = {"reranker": "cross_encoder", "chunker": "semantic", "generator": "concise"}
    assert check_session_settings(kept) == kept


@pytest.mark.parametrize(
    "settings",
    [
        {"embedding_provider": "openai"},
        {"crag_web_search_fallback": True},
        {"embedder": "openai"},
        {"embedder": "voyage"},
        {"reranker": "cohere"},
    ],
)
def test_third_party_services_are_the_operators_to_choose(settings):
    assert "third-party service on the server's API key" in _refused(settings)


def test_search_params_cannot_carry_the_metric():
    assert "metric_type" in _refused({"search_params": {"ef": 64, "metric_type": "L2"}})
    assert check_session_settings({"search_params": {"ef": 64}}) == {"search_params": {"ef": 64}}


def test_a_relative_path_to_a_model_on_the_server_is_a_path(tmp_path, monkeypatch):
    (tmp_path / "models" / "mine").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert "path or URL" in _refused({"embedding_model": "models/mine"})
    assert "path or URL" in _refused({"chunk_tokenizer": "models/mine"})
    assert "path or URL" in _refused({"ensemble_reranker_models": ["BAAI/x", "models/mine"]})
    assert "path or URL" in _refused({"reranker_model": "models\\mine"})
    assert "path or URL" in _refused({"splade_model": "models/.."})


def test_model_names_still_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kept = {
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_tokenizer": "tiktoken:cl100k_base",
    }
    assert check_session_settings(kept) == kept
