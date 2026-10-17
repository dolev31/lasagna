"""Every config value is checked by the store that keeps it, on every surface.

`import_config` stored ``session_overrides`` as they came: on the auth-on
server ``POST /api/config/knowledge/import`` with ``{"chunk_size": "lots",
"embedding_dim": -5}`` answered 200, after which ``GET /api/v1/config`` was a
500 (with the connection reset) until someone guessed to reset, and every
upload and search failed on a -5-dimension "mismatch". PATCH validated the
same values; import, which writes the same store, did not.
"""

from __future__ import annotations

import copy

import pytest

from tests._isolated_config import isolated_manager, rest_client


@pytest.fixture
def manager(tmp_path, monkeypatch):
    return isolated_manager(tmp_path, monkeypatch)


def _state(mgr, sid):
    from lasagna.providers import get_session_provider

    state = mgr.get_session_state(sid)
    return copy.deepcopy(
        (
            state.active_profile,
            state.session_overrides,
            [p.to_dict() for p in state.profiles],
            get_session_provider().get_resolved_config(sid),
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": "lots"},
        {"embedding_dim": -5},
        {"default_top_k": 1_000_000_000},
        {"chunk_size": 800.5},
        {"enable_reranking": 1},
        {"rrf_k": "sixty"},
        {"reranker_model": 7},
        {"document_awareness": {"first_tokens_count": "many"}},
    ],
)
def test_an_import_with_invalid_values_is_refused_and_stores_nothing(manager, overrides):
    sid = "import-values"
    before = _state(manager, sid)
    data = {"schema_version": 2, "profiles": [], "session_overrides": overrides}
    # A number past a session's maximum is refused before validation runs.
    with pytest.raises(ValueError, match="Invalid configuration|is at most"):
        manager.import_config(data, session_id=sid)
    assert _state(manager, sid) == before


def test_a_valid_import_reaches_search(manager):
    from lasagna.core.config import rag_config

    sid = "import-applies"
    manager.import_config(
        {"schema_version": 2, "profiles": [], "session_overrides": {"rerank_top_k": 9}},
        session_id=sid,
    )
    assert rag_config.with_session(sid).rerank_top_k == 9


@pytest.mark.parametrize("value", [True, 800.5, "800"])
def test_an_integer_setting_takes_only_an_integer(value):
    from lasagna.configurator.validation import validate_config

    assert not validate_config({"chunk_size": value}).valid


def test_every_built_in_profile_still_validates(manager):
    from lasagna.configurator.validation import validate_config

    for profile in manager._default_profiles:
        result = validate_config(profile.config)
        assert result.valid, (profile.name, result.errors)


def test_an_import_with_an_invalid_value_is_a_422_and_the_config_still_reads(manager):
    client = rest_client()
    sid = {"X-Session-ID": "rest-import-values"}
    body = {"schema_version": 2, "profiles": [], "session_overrides": {"chunk_size": "lots"}}
    refused = client.post("/api/config/knowledge/import", json=body, headers=sid)
    assert refused.status_code == 422, refused.text
    assert "chunk_size" in refused.json()["detail"]
    assert client.get("/api/v1/config", headers=sid).status_code == 200


def test_a_config_stored_before_validation_reads_as_a_422_naming_the_key(manager):
    from lasagna.providers import get_session_provider

    get_session_provider().set_config_overrides("rest-poisoned", {"chunk_size": "lots"})
    response = rest_client().get("/api/v1/config", headers={"X-Session-ID": "rest-poisoned"})
    assert response.status_code == 422
    assert "chunk_size" in response.json()["detail"]
    assert "reset" in response.json()["detail"]
