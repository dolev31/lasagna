"""A sweep arm must measure the component it names, or refuse to run.

An arm naming an unregistered component used to log one warning and run the
profile's own component, reporting that number under the arm's name; component
lists given beside config_overrides were dropped without a word; and a reranker
arm under a profile with reranking off never called the reranker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import (
    PipelineSweepSpec,
    _apply_overrides_to_pipeline,
    _check_components_exist,
    _generate_pipeline_configs,
)


def test_an_unregistered_component_is_refused_before_anything_runs():
    configs = [{"name": "mine", "overrides": {"reranker": "not_installed_anywhere"}}]
    with pytest.raises(ValueError, match="not registered"):
        _check_components_exist(configs)


def test_a_registered_component_passes():
    _check_components_exist([{"name": "ce", "overrides": {"reranker": "cross_encoder"}}])


def test_component_lists_beside_config_overrides_are_refused():
    spec = PipelineSweepSpec(
        rerankers=["cross_encoder"], config_overrides=[{"name": "x", "enable_hyde": True}]
    )
    with pytest.raises(ValueError, match="would be ignored"):
        _generate_pipeline_configs(spec)


def test_naming_a_reranker_turns_reranking_on():
    pipeline = MagicMock()
    pipeline._config = {}
    _apply_overrides_to_pipeline(pipeline, {"reranker": "cross_encoder"}, "session")
    assert pipeline._config.get("enable_reranking") is True
