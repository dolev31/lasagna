"""Every number a session may set has a maximum, and nothing shipped exceeds it.

A session's settings come from whoever holds the session (a REST caller, an MCP
client). Numeric ones had no upper bound: ``ircot_max_steps``,
``speculative_n_drafts``, ``rag_fusion_n_queries``, ``hybrid_fusion_depth`` and
``max_content_length`` accepted 10^6, so one request could run a million LLM
calls or fetch a million candidates. The maximums apply to session writes only:
the evaluation harness sets its arms' values in-process, and a researcher's
sweep is never refused by them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lasagna.core.session_config._session_keys import (
    SESSION_MAXIMUMS,
    SessionSettingRefused,
    check_session_settings,
    list_session_keys,
    numeric_session_keys,
)
from tests._paths import PACKAGE_DIR

ROOT = PACKAGE_DIR.parents[1]


def test_every_numeric_session_setting_has_a_maximum():
    missing = sorted(set(numeric_session_keys()) - set(SESSION_MAXIMUMS))
    assert not missing, f"no maximum in SESSION_MAXIMUMS for: {missing}"


def test_the_table_names_only_session_settings():
    # It holds the longest list of each list setting too (see
    # test_session_lists_have_a_maximum_length.py).
    stale = sorted(set(SESSION_MAXIMUMS) - set(numeric_session_keys()) - set(list_session_keys()))
    assert not stale, f"not numeric or list session settings: {stale}"


def _numbers(node, path=()):
    """(key, value) for every number under a mapping, with nested keys as ``a.b``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                yield ".".join((*path, str(key))) if path else str(key), value
                yield str(key), value
            else:
                yield from _numbers(value, (*path, str(key)) if isinstance(value, dict) else ())
    elif isinstance(node, list):
        for item in node:
            yield from _numbers(item)


def _shipped():
    from lasagna.core.config import get_rag_config
    from lasagna.core.session_config._profiles import ProfileMixin
    from lasagna.sdk import config as sdk_config

    yield "defaults", get_rag_config()._config
    for name, spec in ProfileMixin._PROFILE_SPECS.items():
        yield f"profile {name}", spec
    from lasagna.sdk._config_bridge import get_runtime_overrides

    for name in sorted(dir(sdk_config)):
        if name.startswith("config_for_"):
            yield f"preset {name}", get_runtime_overrides(getattr(sdk_config, name)())
    yamls = [
        *ROOT.glob("examples/**/*.yaml"),
        *ROOT.glob("docs/**/*.yaml"),
        *PACKAGE_DIR.glob("knowledge_evaluation_sdk/**/*.yaml"),
    ]
    for path in sorted(yamls):
        yield str(path.relative_to(ROOT)), yaml.safe_load(Path(path).read_text())


@pytest.mark.parametrize(
    "where,config", list(_shipped()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_nothing_shipped_exceeds_a_maximum(where, config):
    over = sorted(
        f"{key}={value} > {SESSION_MAXIMUMS[key]}"
        for key, value in _numbers(config)
        if key in SESSION_MAXIMUMS and value > SESSION_MAXIMUMS[key]
    )
    assert not over, f"{where}: {over}"


@pytest.mark.parametrize(
    "key",
    [
        "ircot_max_steps",
        "speculative_n_drafts",
        "rag_fusion_n_queries",
        "hybrid_fusion_depth",
        "max_content_length",
    ],
)
def test_a_session_cannot_set_a_number_past_its_maximum(key):
    with pytest.raises(SessionSettingRefused, match=f"{key} is at most {SESSION_MAXIMUMS[key]:g}"):
        check_session_settings({key: 10**6})
    assert check_session_settings({key: SESSION_MAXIMUMS[key]}) == {key: SESSION_MAXIMUMS[key]}


def test_a_nested_number_is_bounded_too():
    with pytest.raises(SessionSettingRefused, match="chunk_cite_retrieval.max_chunks is at most"):
        check_session_settings({"chunk_cite_retrieval": {"strategy": "fixed", "max_chunks": 10**6}})


def test_the_harness_sets_any_value(tmp_path, monkeypatch):
    """A sweep's arm writes through the session provider, which the maximums do not apply to."""
    from tests._isolated_config import isolated_manager

    manager = isolated_manager(tmp_path, monkeypatch)
    manager.patch_config({"rag_fusion_n_queries": 500}, session_id="sweep", trusted=True)
    assert manager.get_resolved_config("sweep")["rag_fusion_n_queries"] == 500
    with pytest.raises(SessionSettingRefused):
        manager.patch_config({"rag_fusion_n_queries": 500}, session_id="client")
