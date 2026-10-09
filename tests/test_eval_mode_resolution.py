"""Mode selection must not be a trap the caller falls into.

`mode` decides whether the retrieval pipeline runs at all, and the two settings
measure different things. The old default was `offline`, which does not run the
pipeline — so the documented quickstart (`lasagna benchmark run beir/nfcorpus`)
paired a retrieval-only benchmark with the one mode that cannot score
retrieval, and always failed. Anyone who reached for a component override got a
number scored from gold context instead.

`auto` resolves from what the benchmark can actually support, and records what
it chose and why. The old name is kept as an alias but is a poor description:
"offline" implied no network, and it calls a judge model.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import (
    _MODE_ALIASES,
    _VALID_MODES,
    resolve_mode,
)


class TestAutoPicksWhatTheBenchmarkSupports:
    def test_benchmark_with_a_corpus_resolves_to_live(self):
        mode, reason = resolve_mode("auto", "smoke")
        assert mode == "live"
        assert "corpus" in reason

    def test_retrieval_benchmark_resolves_to_live(self):
        """The exact case the documented quickstart got wrong: BEIR has qrels
        and no gold answers, so generation_only can only fail on it."""
        mode, _ = resolve_mode("auto", "beir/nfcorpus")
        assert mode == "live"

    def test_a_benchmark_with_inline_gold_context_resolves_to_live_and_says_why(self):
        """SQuAD v2 used to be generation-only, because it ships no corpus.

        It ships each question's source paragraph instead, and those pool into
        one. The reason string has to carry the caveat with it: the pooled
        corpus has no hard negatives, so it is an easier retrieval task than
        the published benchmark, and a number produced against it means
        something different.
        """
        mode, reason = resolve_mode("auto", "squad_v2")
        assert mode == "live"
        assert "pooled" in reason and "easier" in reason

    def test_a_benchmark_with_nothing_to_retrieve_still_resolves_to_generation_only(self):
        """TriviaQA ships a question and an answer string. There is nothing to
        pool, and an empty corpus would score zero and read as a broken
        retriever."""
        mode, reason = resolve_mode("auto", "triviaqa")
        assert mode == "generation_only"
        assert "no corpus" in reason

    def test_every_resolution_states_a_reason(self):
        for benchmark in ("smoke", "beir/nfcorpus", "squad_v2", "triviaqa"):
            _, reason = resolve_mode("auto", benchmark)
            assert reason and len(reason) > 10, f"{benchmark} resolved without an explanation"

    def test_unknown_benchmark_does_not_crash_resolution(self):
        """Resolution runs while planning, before the task fails properly."""
        mode, reason = resolve_mode("auto", "no_such_benchmark")
        assert mode in _VALID_MODES
        assert "could not inspect" in reason


class TestExplicitModesAreRespected:
    @pytest.mark.parametrize("mode", ["live", "generation_only"])
    def test_explicit_mode_is_never_overridden(self, mode):
        got, reason = resolve_mode(mode, "smoke")
        assert got == mode
        assert "explicit" in reason

    def test_offline_is_an_alias_for_generation_only(self):
        assert _MODE_ALIASES["offline"] == "generation_only"
        assert resolve_mode("offline", "smoke")[0] == "generation_only"

    def test_live_is_honoured_even_when_auto_would_agree(self):
        assert resolve_mode("live", "smoke")[0] == "live"


class TestResolutionIsCheap:
    def test_corpus_support_is_detected_without_downloading(self):
        """`auto` runs while planning a sweep, so it must not fetch corpora.

        Detection compares the adapter's `load_corpus` against the base
        implementation rather than calling it.
        """
        from lasagna.knowledge_evaluation_sdk.experiment import _adapter_has_corpus
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
        from lasagna.knowledge_evaluation_sdk.benchmarks._base import BaseBenchmarkAdapter

        class NeverCallMe(BaseBenchmarkAdapter):
            def info(self):  # pragma: no cover - not exercised
                raise AssertionError("info() should not be called during resolution")

            def load(self, **kwargs):  # pragma: no cover
                raise AssertionError("load() should not be called during resolution")

            def load_corpus(self, cache_dir=None, max_docs=100_000):
                raise AssertionError("load_corpus() must not be called during resolution")

        assert _adapter_has_corpus(NeverCallMe()) is True
        assert _adapter_has_corpus(benchmark_registry.get("smoke")) is True


class TestDefaultsChanged:
    def test_experiment_config_defaults_to_auto(self):
        from lasagna.knowledge_evaluation_sdk.experiment import ExperimentConfig

        assert ExperimentConfig(name="x").mode == "auto"

    def test_task_defaults_to_auto(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        task = BenchmarkTask(
            benchmark_name="smoke", config_name="c", config_overrides={}, run_index=0
        )
        assert task.mode == "auto"

    def test_task_carries_the_resolution_for_the_record(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        task = BenchmarkTask(
            benchmark_name="smoke", config_name="c", config_overrides={}, run_index=0
        )
        assert hasattr(task, "resolved_mode")
        assert hasattr(task, "mode_resolution_reason")
