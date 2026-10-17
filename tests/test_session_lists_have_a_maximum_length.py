"""Every list a session may set has a maximum length, and names nothing twice.

``SESSION_MAXIMUMS`` bounded numbers only. A REST patch was accepted with
``reranker: ensemble`` and 5000 ``ensemble_reranker_models``, and the ensemble
reranker scores every member on every search: one request made each later
search of that session run 5000 cross-encoders. A list-valued session setting
now has a maximum length in the same table (eight ensemble members), a list
of names may not repeat one, and a list for a key the table does not name is
bounded too. Session writes only: the harness and the server's own settings
are not.
"""

from __future__ import annotations

import pytest

from lasagna.core.session_config._session_keys import (
    SESSION_MAXIMUMS,
    SessionSettingRefused,
    check_session_settings,
)
from tests.test_session_settings_have_maximums import _shipped


def test_every_list_a_session_may_set_has_a_maximum_length():
    from lasagna.core.session_config._session_keys import list_session_keys

    missing = sorted(set(list_session_keys()) - set(SESSION_MAXIMUMS))
    assert not missing, f"no maximum length in SESSION_MAXIMUMS for: {missing}"
    assert SESSION_MAXIMUMS["ensemble_reranker_models"] == 8


def test_five_thousand_ensemble_members_are_refused():
    members = [f"cross-encoder/model-{i}" for i in range(5000)]
    with pytest.raises(SessionSettingRefused, match="ensemble_reranker_models has at most 8"):
        check_session_settings({"reranker": "ensemble", "ensemble_reranker_models": members})


def test_a_member_named_twice_is_refused():
    twice = ["cross-encoder/ms-marco-MiniLM-L-6-v2"] * 2
    with pytest.raises(
        SessionSettingRefused, match="names cross-encoder/ms-marco-MiniLM-L-6-v2 twice"
    ):
        check_session_settings({"ensemble_reranker_models": twice})


def test_equal_weights_are_not_a_repeated_name():
    assert check_session_settings({"ensemble_reranker_weights": [0.5, 0.5]}) == {
        "ensemble_reranker_weights": [0.5, 0.5]
    }


def test_a_list_for_a_key_the_table_does_not_name_is_bounded_too():
    with pytest.raises(SessionSettingRefused, match="has at most"):
        check_session_settings({"reranker_model": [f"m{i}" for i in range(1000)]})


def _lists(node):
    """(key, list) for every list under a mapping."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, list):
                yield str(key), value
            yield from _lists(value)
    elif isinstance(node, list):
        for item in node:
            yield from _lists(item)


@pytest.mark.parametrize(
    "where,config", list(_shipped()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_nothing_shipped_exceeds_a_maximum_length(where, config):
    over = sorted(
        f"{key} has {len(value)} > {SESSION_MAXIMUMS[key]}"
        for key, value in _lists(config)
        if key in SESSION_MAXIMUMS and len(value) > SESSION_MAXIMUMS[key]
    )
    assert not over, f"{where}: {over}"


def test_the_harness_sets_any_list(tmp_path, monkeypatch):
    from tests._isolated_config import isolated_manager

    manager = isolated_manager(tmp_path, monkeypatch)
    members = [f"cross-encoder/m{i}" for i in range(12)]
    manager.patch_config({"ensemble_reranker_models": members}, session_id="sweep", trusted=True)
    assert manager.get_resolved_config("sweep")["ensemble_reranker_models"] == members
    with pytest.raises(SessionSettingRefused):
        manager.patch_config({"ensemble_reranker_models": members}, session_id="client")
