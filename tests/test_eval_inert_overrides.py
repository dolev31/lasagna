"""A mode that cannot measure a component must refuse, not answer.

`_eval_offline` scores gold context with an LLM judge; it never constructs the
retrieval pipeline. So `run_benchmark(..., reranker="mine")` — the single most
likely thing a researcher types — could not possibly be measuring `mine`.

It used to return a number anyway. The warning that was supposed to catch this
listed 14 retrieval feature flags and omitted every component key, so the one
override that matters was the one that passed silently. A plausible, precise,
unrelated number is the worst thing a benchmark harness can produce, because
nothing downstream looks wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import (
    _COMPONENT_KEYS,
    _HARNESS_KEYS,
    _eval_offline,
)
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask


def _task(**overrides) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark_name="unit",
        config_name="cfg",
        run_index=0,
        mode="offline",
        config_overrides=overrides,
    )


def _samples():
    """Samples that offline mode *can* score, so a refusal is about the
    overrides and not about the data."""
    return [
        BenchmarkSample(
            query_id="q1",
            question="what colour is the sky?",
            ground_truth="blue",
            context="The sky is blue.",
        )
    ]


class TestInertOverridesRefuse:
    # Every stage that acts on retrieval. A generator writes the answer from
    # the gold context, so offline mode does measure it.
    @pytest.mark.parametrize("key", sorted(_COMPONENT_KEYS - {"generator"}))
    def test_every_component_swap_is_refused(self, key):
        """This is the parametrisation that would have caught the original bug:
        it fails for any component key missing from the inert set."""
        result = asyncio.run(_eval_offline(_task(**{key: "whatever"}), _samples()))

        assert result.error, (
            f"offline mode accepted a {key!r} override and returned metrics "
            f"{result.metrics!r}. That number cannot reflect {key!r} — this mode "
            f"never runs the pipeline."
        )
        assert key in result.error
        assert "live" in result.error, "the error must name the mode that can measure it"
        assert not result.metrics, "a refused task must not also report metrics"

    def test_retrieval_feature_flags_are_still_refused(self):
        """The keys the original warning did cover must not regress to warnings."""
        result = asyncio.run(_eval_offline(_task(enable_reranking=True), _samples()))
        assert result.error and "enable_reranking" in result.error

    def test_escape_hatch_permits_scoring_gold_context(self):
        """Someone who genuinely wants the gold-context number can still have it,
        but has to say so — the default cannot be the silent one."""
        task = _task(reranker="mine", allow_inert_overrides=True)
        result = asyncio.run(_eval_offline(task, _samples()))
        # It proceeds past the guard. Whether it then produces metrics depends on
        # RAGAS being installed, which is not what this test is about.
        assert result.error is None or "have no effect" not in (result.error or "")

    def test_clean_run_is_untouched(self):
        """No overrides, no refusal — the guard must not break ordinary use."""
        result = asyncio.run(_eval_offline(_task(), _samples()))
        assert result.error is None or "have no effect" not in result.error


class TestHarnessKeysNeverReachThePipeline:
    def test_allow_inert_overrides_is_declared_a_harness_key(self):
        """Unknown override keys are merged straight into `pipeline._config`.
        A harness flag landing there would be applied as a retrieval setting."""
        assert "allow_inert_overrides" in _HARNESS_KEYS

    def test_harness_keys_are_stripped_before_pipeline_application(self):
        from tests.test_event_loop_safety import executable_source

        from lasagna.knowledge_evaluation_sdk import experiment

        src = executable_source(experiment._apply_overrides_to_pipeline)
        assert "_HARNESS_KEYS" in src, (
            "_apply_overrides_to_pipeline must skip harness keys, or they are set "
            "as pipeline config on every live run"
        )
