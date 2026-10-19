"""Settings are finite numbers, counts are not negative, and a session's dict settings have a size.

``hybrid_alpha`` and ``similarity_threshold`` accepted NaN (every comparison with
it is false, so the maximum let it through), ``hybrid_fusion_depth`` and
``search_params.ef`` accepted negative counts, and a dict setting had no size
limit: 5,000 ``agentic_prompts`` of 1 KB each were stored, and patches kept
merging into the stored row. On PostgreSQL a NaN made the JSONB write fail.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._session_keys import (
    DICT_MAXIMUM_BYTES,
    SessionSettingRefused,
    number_problems,
    session_key_problem,
)


@pytest.mark.parametrize(
    "key, value",
    [
        ("hybrid_alpha", float("nan")),
        ("similarity_threshold", float("nan")),
        ("similarity_threshold", float("inf")),
        ("mmr_lambda", float("-inf")),
        ("hybrid_fusion_depth", -1),
        ("default_top_k", -5),
        ("search_params", {"ef": -8}),
        ("search_params", {"ef": float("nan")}),
        ("chunk_cite_retrieval", {"max_chunks": -1}),
    ],
)
def test_a_value_that_is_not_a_finite_number_or_a_negative_count_is_refused(key, value):
    manager = KnowledgeConfigManager()
    sid = f"num-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError) as refused:
        manager.patch_config({key: value}, session_id=sid)
    # The configurator's range, or the session maximum, says so where there is one.
    why = str(refused.value)
    assert any(w in why for w in ("finite", "negative", "below minimum", "at most")), why
    assert key not in manager.get_session_state(sid).session_overrides


@pytest.mark.parametrize("key, value", [("hybrid_alpha", 0.0), ("default_top_k", 1)])
def test_zero_and_small_values_are_still_values(key, value):
    assert number_problems({key: value}) == []


def test_a_dict_setting_has_a_size():
    prompts = {f"prompt_{i}": "x" * 1024 for i in range(5_000)}
    problem = session_key_problem("agentic_prompts", prompts)
    assert problem is not None and "at most" in problem


def test_patches_cannot_grow_a_dict_setting_past_its_size():
    manager = KnowledgeConfigManager()
    sid = f"dict-{uuid.uuid4().hex[:8]}"
    chunk = DICT_MAXIMUM_BYTES // 4
    manager.patch_config({"agentic_prompts": {"a": "x" * chunk}}, session_id=sid)
    manager.patch_config({"agentic_prompts": {"b": "x" * chunk}}, session_id=sid)
    with pytest.raises(SessionSettingRefused, match="agentic_prompts"):
        for name in "cdefgh":
            manager.patch_config({"agentic_prompts": {name: "x" * chunk}}, session_id=sid)
    stored = manager.get_session_state(sid).session_overrides["agentic_prompts"]
    assert len(str(stored)) <= DICT_MAXIMUM_BYTES + 100


def test_an_import_or_a_trusted_write_with_nan_is_refused_too():
    manager = KnowledgeConfigManager()
    sid = f"nan-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError, match="hybrid_alpha: is nan"):
        manager.import_config(
            {
                "schema_version": 2,
                "profiles": [],
                "session_overrides": {"hybrid_alpha": float("nan")},
            },
            session_id=sid,
        )
    with pytest.raises(ValueError, match="hybrid_alpha: is nan"):
        manager.patch_config({"hybrid_alpha": float("nan")}, session_id=sid, trusted=True)
    assert "hybrid_alpha" not in manager.get_session_state(sid).session_overrides


@pytest.mark.parametrize(
    "key, value, named",
    [
        ("ensemble_reranker_weights", [float("nan"), 1.0], "ensemble_reranker_weights[0]"),
        ("ensemble_reranker_weights", [0.5, float("inf")], "ensemble_reranker_weights[1]"),
        ("search_params", {"nested": {"ef": float("nan")}}, "search_params.nested.ef"),
        ("agentic_prompts", {"a": [1.0, float("-inf")]}, "agentic_prompts.a[1]"),
    ],
)
def test_a_nan_inside_a_list_or_a_nested_dict_is_refused_too(key, value, named):
    """The check looked one dict level deep and skipped lists: ``[NaN, 1]`` was stored."""
    assert [name for name, _ in number_problems({key: value})] == [named]
    manager = KnowledgeConfigManager()
    sid = f"deep-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError, match="finite"):
        manager.patch_config({key: value}, session_id=sid)
    assert key not in manager.get_session_state(sid).session_overrides


def test_a_list_of_finite_numbers_is_a_value():
    assert number_problems({"ensemble_reranker_weights": [0.0, -0.5, 1]}) == []
