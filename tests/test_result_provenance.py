"""A result must carry the identity of the arm that produced it.

`config_name` is a label a human chose. `config_overrides` is what was actually
varied, and it lived only on `BenchmarkTask` — the input — while every consumer
downstream reads `TaskResult.to_dict()` rows out of a results file.

The consequence was not a missing field on a report. `bundle.varied_dimensions`
read `row["config_overrides"]` from a row that never had the key, so it returned
`[]` for every configuration of every run ever produced, and the leaderboard
rendered every row — baseline and treatment alike — as `_baseline_`. The feature
was unit-tested against hand-built dicts that supplied the key, so the tests
passed while nothing worked.

These tests go through the runner, because the gap was in the copy, not in the
functions on either side of it.
"""

from __future__ import annotations

import asyncio

from lasagna.knowledge_evaluation_sdk.runner import BenchmarkRunner, BenchmarkTask, TaskResult


def _task(**kw) -> BenchmarkTask:
    base = dict(
        benchmark_name="smoke",
        config_name="hyde_on",
        run_index=0,
        n_samples=2,
        profile="standard",
        config_overrides={"enable_hyde": True},
        mode="generation_only",
    )
    base.update(kw)
    return BenchmarkTask(**base)


def _run(task: BenchmarkTask, eval_fn) -> TaskResult:
    runner = BenchmarkRunner(max_concurrency=1)
    return asyncio.run(runner.run_tasks(tasks=[task], eval_fn=eval_fn))[0]


async def _ok(task, samples):
    return TaskResult(
        task_key=task.task_key,
        benchmark_name=task.benchmark_name,
        config_name=task.config_name,
        run_index=task.run_index,
        metrics={"ndcg@10": 0.5},
    )


async def _boom(task, samples):
    raise RuntimeError("pipeline exploded")


class TestTheArmIsCarriedOntoItsResult:
    def test_overrides_reach_the_result(self):
        assert _run(_task(), _ok).config_overrides == {"enable_hyde": True}

    def test_overrides_survive_serialisation(self):
        assert _run(_task(), _ok).to_dict()["config_overrides"] == {"enable_hyde": True}

    def test_the_base_profile_is_recorded(self):
        """Identical overrides on different profiles are different systems."""
        assert _run(_task(profile="max_quality"), _ok).profile == "max_quality"

    def test_a_baseline_records_an_empty_dict_not_a_missing_key(self):
        """Absent and empty must be distinguishable.

        "this arm varied nothing" and "nobody recorded what this arm varied"
        are opposite claims, and before this they were the same value.
        """
        row = _run(_task(config_overrides={}), _ok).to_dict()
        assert "config_overrides" in row and row["config_overrides"] == {}

    def test_a_failed_task_still_names_its_arm(self):
        """ "config_3 crashed" is not actionable; the overrides are."""
        result = _run(_task(), _boom)
        assert result.error
        assert result.config_overrides == {"enable_hyde": True}

    def test_the_result_does_not_alias_the_task(self):
        """A shared dict would let one arm's mutation rewrite another's record."""
        task = _task()
        result = _run(task, _ok)
        result.config_overrides["enable_hyde"] = "tampered"
        assert task.config_overrides == {"enable_hyde": True}


class TestTheSampleDigestCertifiesWhatItClaims:
    """`sample_id_digest` is in the comparability key and `bundle.py` documents
    it as proof two bundles cover the same queries.

    It hashed `query_id` alone, and 14 of 16 adapters assign `query_id=str(i)`.
    So it was a pure function of `n_samples` — `beir/nfcorpus` and `squad_v2` at
    n=50 both produced `770b707572088fe8`. Two runs over entirely different
    questions certified as covering the same ones.
    """

    def _samples(self, prefix, n=50):
        from lasagna.knowledge_evaluation_sdk.benchmarks import BenchmarkSample

        return [
            BenchmarkSample(query_id=str(i), question=f"{prefix} question {i}") for i in range(n)
        ]

    def test_different_questions_at_the_same_n_differ(self):
        from lasagna.knowledge_evaluation_sdk.runner import _sample_id_digest

        assert _sample_id_digest(self._samples("nfcorpus")) != _sample_id_digest(
            self._samples("squad")
        )

    def test_the_same_questions_still_match(self):
        """Otherwise it cannot do the job it exists for."""
        from lasagna.knowledge_evaluation_sdk.runner import _sample_id_digest

        assert _sample_id_digest(self._samples("nfcorpus")) == _sample_id_digest(
            self._samples("nfcorpus")
        )

    def test_order_does_not_matter(self):
        from lasagna.knowledge_evaluation_sdk.runner import _sample_id_digest

        forward = self._samples("x", n=10)
        assert _sample_id_digest(forward) == _sample_id_digest(list(reversed(forward)))

    def test_it_ships_no_question_text(self):
        """A digest, not a corpus — several benchmarks forbid redistribution."""
        from lasagna.knowledge_evaluation_sdk.runner import _sample_id_digest

        digest = _sample_id_digest(self._samples("a-very-distinctive-phrase"))
        assert "distinctive" not in digest
        assert len(digest) == 16

    def test_beir_carries_its_upstream_query_id(self):
        """The loader had the real id and dropped it on the floor."""
        import inspect

        from lasagna.knowledge_evaluation_sdk import datasets as ds

        source = inspect.getsource(ds._convert_beir_with_qrels)
        assert '"query_id": qid' in source, (
            "the BEIR converter must carry the upstream query-id into the sample"
        )


class TestOldResultFilesStillLoad:
    def test_from_dict_tolerates_a_row_without_the_new_fields(self):
        row = {
            "task_key": "k",
            "benchmark_name": "smoke",
            "config_name": "c",
            "run_index": 0,
            "metrics": {"ndcg@10": 0.5},
        }
        restored = TaskResult.from_dict(row)
        assert restored.config_overrides == {}
        assert restored.profile == ""

    def test_two_restored_rows_do_not_share_a_default(self):
        """`field(default_factory=dict)`, not a mutable class attribute."""
        base = {"benchmark_name": "b", "config_name": "c", "run_index": 0}
        a = TaskResult.from_dict({"task_key": "a", **base})
        b = TaskResult.from_dict({"task_key": "b", **base})
        a.config_overrides["x"] = 1
        assert b.config_overrides == {}


class TestTheConsumersActuallyLightUp:
    """The point of the field. Asserted on the real path, not on fixtures."""

    def test_varied_dimensions_is_no_longer_empty_for_a_real_run(self):
        from lasagna.knowledge_evaluation_sdk.bundle import varied_dimensions

        row = _run(_task(), _ok).to_dict()
        assert varied_dimensions(row["config_overrides"]) == ["enable_hyde"]

    def test_the_leaderboard_no_longer_calls_a_treatment_arm_baseline(self):
        from lasagna.knowledge_evaluation_sdk.bundle import varied_dimensions

        treatment = varied_dimensions(_run(_task(), _ok).to_dict()["config_overrides"])
        baseline = varied_dimensions(
            _run(_task(config_overrides={}), _ok).to_dict()["config_overrides"]
        )
        assert treatment != baseline, (
            "a treatment arm and a baseline must not render identically in the "
            "leaderboard's Varied column"
        )
