"""`tracks:` lets a component sweep score retrieval alone.

The documented reranker comparison ran on `smoke`, whose queries carry both
qrels and answers. On a core install there is no RAGAS and often no LLM, so
every generation failed and, now that failures are not scores, every task
failed too, taking its valid retrieval numbers with it.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _parse_tracks
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask


def test_tracks_are_checked():
    assert _parse_tracks(None) == ["retrieval", "generation"]
    assert _parse_tracks("retrieval") == ["retrieval"]
    with pytest.raises(ValueError, match="tracks"):
        _parse_tracks(["retreival"])
    with pytest.raises(ValueError, match="tracks"):
        _parse_tracks([])


def test_the_default_leaves_task_keys_unchanged():
    base = dict(benchmark_name="b", config_name="c", config_overrides={}, run_index=0)
    assert BenchmarkTask(**base).config_hash == BenchmarkTask(**base, tracks=None).config_hash
    assert (
        BenchmarkTask(**base).config_hash
        != BenchmarkTask(**base, tracks=("retrieval",)).config_hash
    )


def test_a_retrieval_only_sweep_carries_the_choice_to_every_task():
    exp = Experiment.from_dict(
        {
            "name": "ir",
            "benchmarks": [{"name": "smoke"}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed", "rerankers": ["cross_encoder_adapted"]},
        }
    )
    tasks = exp.generate_tasks()
    assert tasks and all(t.tracks == ("retrieval",) for t in tasks)
