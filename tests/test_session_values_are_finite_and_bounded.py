"""A session's settings are finite numbers, counts are not negative, and dict settings have a size.

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
    problem = session_key_problem(key, value)
    assert problem is not None and ("finite" in problem or "negative" in problem), problem


@pytest.mark.parametrize("key, value", [("hybrid_alpha", 0.0), ("default_top_k", 0)])
def test_zero_is_still_a_value(key, value):
    assert session_key_problem(key, value) is None


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


def test_a_patch_with_nan_is_refused_before_anything_is_stored():
    manager = KnowledgeConfigManager()
    sid = f"nan-{uuid.uuid4().hex[:8]}"
    with pytest.raises(SessionSettingRefused, match="hybrid_alpha"):
        manager.patch_config({"hybrid_alpha": float("nan")}, session_id=sid)
    assert "hybrid_alpha" not in manager.get_session_state(sid).session_overrides
