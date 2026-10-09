"""Two ways a run can quietly mean less than it appears to.

**Power.** A non-significant result has two readings — the techniques are the
same, or the run was too small to tell — and they call for opposite next steps.
A p-value does not distinguish them, so "we found no difference" gets published
where "we could not have found one" is the truth.

**Revisions.** A HuggingFace dataset name is a moving reference. Nothing here
pinned one and nothing recorded one, so `hf_revision` was empty on every result
ever produced and `UNPINNED_DATASET` fired unconditionally — a warning nobody
could act on, which is a warning nobody reads. Two people reporting numbers on
"the same benchmark" may have measured different data.
"""

from __future__ import annotations

import math

import pytest

from lasagna.knowledge_evaluation_sdk.results import (
    BenchmarkResults,
    _normal_quantile,
    _samples_for_effect,
)
from lasagna.knowledge_evaluation_sdk.runner import TaskResult


def _paired(a_scores, b_scores, benchmark="scifact", metric="ndcg@10"):
    return BenchmarkResults(
        [
            TaskResult(
                task_key=name,
                benchmark_name=benchmark,
                config_name=name,
                run_index=0,
                per_sample_scores=[
                    {"_sample_id": f"q{i}", metric: s} for i, s in enumerate(scores)
                ],
            )
            for name, scores in (("baseline", a_scores), ("candidate", b_scores))
        ]
    )


class TestTheNormalQuantile:
    """Written out rather than pulled from scipy, so it is checked."""

    @pytest.mark.parametrize(
        "p,expected",
        [(0.975, 1.959964), (0.80, 0.841621), (0.95, 1.644854), (0.5, 0.0), (0.025, -1.959964)],
    )
    def test_known_values(self, p, expected):
        assert _normal_quantile(p) == pytest.approx(expected, abs=1e-5)

    def test_it_is_symmetric(self):
        assert _normal_quantile(0.3) == pytest.approx(-_normal_quantile(0.7), abs=1e-9)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_it_refuses_impossible_probabilities(self, p):
        with pytest.raises(ValueError):
            _normal_quantile(p)

    def test_the_tails_are_still_accurate(self):
        """The approximation switches branches at p≈0.024; a seam there would
        show up as a discontinuity."""
        for p in (0.0243, 0.0242, 0.9757, 0.9758):
            assert abs(_normal_quantile(p)) == pytest.approx(1.97, abs=0.02)


class TestMinimumDetectableEffect:
    def test_a_tight_large_run_detects_small_differences(self):
        n = 400
        base = [0.5 + ((i % 7) - 3) / 1000.0 for i in range(n)]
        report = _paired(base, [s + 0.01 for s in base]).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["minimum_detectable_effect"] < 0.005

    def test_a_small_noisy_run_detects_almost_nothing(self):
        base = [0.1, 0.9, 0.2, 0.8, 0.5]
        report = _paired(base, [0.9, 0.1, 0.8, 0.2, 0.5]).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["minimum_detectable_effect"] > 0.4

    def test_more_queries_lower_the_bar(self):
        """MDE scales as 1/sqrt(n); the point of running more queries."""

        def _mde(n):
            base = [0.5 + ((i % 11) - 5) / 100.0 for i in range(n)]
            cand = [s + 0.01 * ((i % 5) - 2) for i, s in enumerate(base)]
            return _paired(base, cand).minimum_detectable_effect(
                "scifact", "baseline", "candidate", "ndcg@10"
            )["minimum_detectable_effect"]

        assert _mde(400) < _mde(100) < _mde(25)

    def test_the_formula_is_the_stated_one(self):
        base = [0.1, 0.4, 0.35, 0.8, 0.2, 0.55]
        cand = [0.2, 0.3, 0.50, 0.7, 0.4, 0.60]
        report = _paired(base, cand).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        z = _normal_quantile(0.975) + _normal_quantile(0.80)
        expected = z * report["sd_of_differences"] / math.sqrt(report["n_paired"])
        assert report["minimum_detectable_effect"] == pytest.approx(expected)


class TestTheSignIsTheOneTheHeaderClaims:
    """The report said "candidate vs baseline" and computed baseline minus
    candidate, so a +0.10 improvement printed as -0.1001.

    `underpowered` uses `abs()`, so the flag stayed right and the report read as
    internally consistent while its headline number was backwards — the shape
    that gets believed. `results interaction`, added in the same release,
    renders candidate-minus-baseline, so the two new commands disagreed.

    Nothing asserted the sign: the tests below check magnitudes and booleans, so
    breaking it again would leave the suite green.
    """

    def test_an_improvement_reads_positive(self):
        report = _paired([0.40] * 8, [0.50] * 8).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["observed_difference"] == pytest.approx(0.10)

    def test_a_regression_reads_negative(self):
        report = _paired([0.50] * 8, [0.40] * 8).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["observed_difference"] == pytest.approx(-0.10)

    def test_it_agrees_with_the_interaction_command(self):
        """Two commands shipped together must not disagree on which way is up."""
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

        base, cand = [0.40] * 8, [0.50] * 8
        mde = _paired(base, cand).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        # `results interaction` renders `mean(candidate) - mean(neither)`.
        rows = BenchmarkResults(
            [
                TaskResult(
                    task_key=n,
                    benchmark_name="scifact",
                    config_name=n,
                    run_index=0,
                    per_sample_scores=[
                        {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(v)
                    ],
                )
                for n, v in (("baseline", base), ("candidate", cand))
            ]
        )
        _, scores = rows.aligned_scores("scifact", ["baseline", "candidate"], "ndcg@10")
        interaction_style = sum(scores["candidate"]) / 8 - sum(scores["baseline"]) / 8
        assert mde["observed_difference"] == pytest.approx(interaction_style)


class TestItSeparatesInconclusiveFromNull:
    """The distinction the whole thing exists for."""

    def test_a_tiny_effect_in_a_noisy_run_is_flagged_underpowered(self):
        base = [0.1, 0.9, 0.2, 0.8, 0.5, 0.3]
        cand = [0.11, 0.89, 0.21, 0.79, 0.51, 0.31]
        report = _paired(base, cand).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["underpowered"]
        assert "inconclusive" in report["interpretation"]

    def test_a_clear_effect_in_a_tight_run_is_not(self):
        n = 200
        base = [0.5 + ((i % 7) - 3) / 1000.0 for i in range(n)]
        report = _paired(base, [s + 0.2 for s in base]).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert not report["underpowered"]

    def test_it_says_how_many_queries_would_be_needed(self):
        """A diagnosis without a next step is half an answer."""
        base = [0.1, 0.9, 0.2, 0.8, 0.5, 0.3]
        cand = [0.15, 0.88, 0.25, 0.77, 0.55, 0.32]
        report = _paired(base, cand).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["n_needed_for_observed_effect"] > report["n_paired"]

    def test_zero_variance_is_reported_as_such_not_as_a_zero_mde(self):
        """ "MDE = 0" reads as infinite sensitivity. It is a degenerate run."""
        report = _paired([0.5] * 6, [0.7] * 6).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert report["zero_variance"]
        assert not report["underpowered"]

    def test_a_significant_pair_is_reported_as_significant(self):
        """`results power` said "a real effect of this size would have been
        missed -- inconclusive" about a pair `results significance` had just
        called significant (p=0.035). A significant difference below the MDE
        was detected; what the MDE says is that a run this size detects one
        that small only some of the time."""
        import random

        rng = random.Random(4)
        base = [rng.uniform(0.2, 0.8) for _ in range(50)]
        cand = [b + 0.04 + rng.gauss(0, 0.12) for b in base]
        res = _paired(base, cand)
        report = res.minimum_detectable_effect("scifact", "baseline", "candidate", "ndcg@10")
        sig = res.significance_test("baseline", "candidate", "ndcg@10", "scifact")

        assert sig["significant"], "the premise: this pair is significant"
        assert report["underpowered"], "the premise: its difference is below the MDE"
        assert report["significant"] and report["p_value"] == pytest.approx(sig["p_value"])
        text = report["interpretation"]
        assert "significant" in text and "not significant" not in text
        assert "would have been missed" not in text and "inconclusive" not in text
        assert "80% of the time" in text, "power is the chance of detecting the MDE"

    def test_a_non_significant_pair_below_the_mde_is_inconclusive(self):
        base = [0.1, 0.9, 0.2, 0.8, 0.5, 0.3]
        cand = [0.11, 0.89, 0.21, 0.79, 0.51, 0.31]
        report = _paired(base, cand).minimum_detectable_effect(
            "scifact", "baseline", "candidate", "ndcg@10"
        )
        assert not report["significant"]
        assert "not significant" in report["interpretation"]
        assert "inconclusive" in report["interpretation"]

    def test_one_shared_sample_is_refused(self):
        with pytest.raises(ValueError, match="fewer than two"):
            _paired([0.5], [0.6]).minimum_detectable_effect(
                "scifact", "baseline", "candidate", "ndcg@10"
            )


class TestSampleSizeHelper:
    def test_a_smaller_effect_needs_more_queries(self):
        assert _samples_for_effect(0.01, 0.1, 0.05, 0.8) > _samples_for_effect(0.05, 0.1, 0.05, 0.8)

    def test_noisier_data_needs_more_queries(self):
        assert _samples_for_effect(0.05, 0.3, 0.05, 0.8) > _samples_for_effect(0.05, 0.1, 0.05, 0.8)

    def test_an_impossible_request_returns_none_rather_than_infinity(self):
        assert _samples_for_effect(0.0, 0.1, 0.05, 0.8) is None
        assert _samples_for_effect(0.05, 0.0, 0.05, 0.8) is None


class TestDatasetRevisions:
    def test_nothing_is_pinned_by_default(self):
        """Deliberate. A SHA written in without being resolved against the Hub
        would be a fabricated provenance record -- it would satisfy `--strict`
        while proving nothing, which is worse than an honest blank."""
        from lasagna.knowledge_evaluation_sdk.datasets import resolve_dataset_revision

        assert resolve_dataset_revision("beir/scifact") == ""

    def test_a_pin_is_honoured(self):
        from lasagna.knowledge_evaluation_sdk.datasets import (
            _PINNED_REVISIONS,
            pin_dataset_revision,
            resolve_dataset_revision,
        )

        try:
            pin_dataset_revision("beir/scifact", "abc123")
            assert resolve_dataset_revision("beir/scifact") == "abc123"
        finally:
            _PINNED_REVISIONS.pop("beir/scifact", None)

    def test_the_environment_can_pin_without_editing_code(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk.datasets import resolve_dataset_revision

        monkeypatch.setenv("LASAGNA_DATASET_REVISIONS", '{"beir/fiqa": "deadbeef"}')
        assert resolve_dataset_revision("beir/fiqa") == "deadbeef"

    def test_a_programmatic_pin_beats_the_environment(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk.datasets import (
            _PINNED_REVISIONS,
            pin_dataset_revision,
            resolve_dataset_revision,
        )

        monkeypatch.setenv("LASAGNA_DATASET_REVISIONS", '{"beir/fiqa": "from-env"}')
        try:
            pin_dataset_revision("beir/fiqa", "from-code")
            assert resolve_dataset_revision("beir/fiqa") == "from-code"
        finally:
            _PINNED_REVISIONS.pop("beir/fiqa", None)

    def test_a_malformed_pin_is_loud_and_does_not_crash_the_run(self, monkeypatch):
        """Silently loading whatever is current is the failure the pin prevents,
        but a bad environment variable must not take down a run either."""
        from loguru import logger

        from lasagna.knowledge_evaluation_sdk.datasets import resolve_dataset_revision

        monkeypatch.setenv("LASAGNA_DATASET_REVISIONS", "not json at all")
        captured: list = []
        sink = logger.add(captured.append, level="WARNING")
        try:
            assert resolve_dataset_revision("beir/fiqa") == ""
        finally:
            logger.remove(sink)
        assert any("not valid JSON" in m for m in captured)

    def test_the_revision_reaches_the_result(self):
        """It was declared on both the task and the result and set by neither,
        so `UNPINNED_DATASET` fired on every run that had ever been produced."""
        from lasagna.knowledge_evaluation_sdk.runner import _resolved_revision

        class _Adapter:
            hf_revision = "pinned-by-the-adapter"

        assert _resolved_revision(_Adapter(), "beir/scifact") == "pinned-by-the-adapter"

    def test_the_registry_is_the_fallback(self):
        from lasagna.knowledge_evaluation_sdk.datasets import (
            _PINNED_REVISIONS,
            pin_dataset_revision,
        )
        from lasagna.knowledge_evaluation_sdk.runner import _resolved_revision

        class _Adapter:
            hf_revision = ""

        try:
            pin_dataset_revision("beir/scifact", "from-registry")
            assert _resolved_revision(_Adapter(), "beir/scifact") == "from-registry"
        finally:
            _PINNED_REVISIONS.pop("beir/scifact", None)

    def test_resolution_never_fails_a_run(self):
        from lasagna.knowledge_evaluation_sdk.runner import _resolved_revision

        class _Hostile:
            @property
            def hf_revision(self):
                raise RuntimeError("boom")

        assert _resolved_revision(_Hostile(), "no_such_benchmark") == ""


class TestThePowerCommandAgreesWithSignificance:
    def test_a_significant_pair_prints_as_significant(self, tmp_path):
        import json
        import random

        typer_testing = pytest.importorskip("typer.testing")
        from lasagna.cli import app

        rng = random.Random(4)
        base = [rng.uniform(0.2, 0.8) for _ in range(50)]
        cand = [b + 0.04 + rng.gauss(0, 0.12) for b in base]
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_key": name,
                            "run_index": 0,
                            "benchmark_name": "scifact",
                            "config_name": name,
                            "metrics": {},
                            "per_sample_scores": [
                                {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(v)
                            ],
                        }
                        for name, v in (("baseline", base), ("candidate", cand))
                    ]
                }
            )
        )
        out = typer_testing.CliRunner().invoke(
            app, ["results", "power", str(path), "baseline", "candidate"]
        )
        assert out.exit_code == 0, out.output
        assert "is significant" in out.output
        assert "would have been missed" not in out.output
