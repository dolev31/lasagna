"""The two significance tests must agree on what they are reporting.

They did not. `significance_test` returned Cohen's `d_z` — the mean of the
paired differences over their standard deviation. `domain_stratified_
significance_test` returned the observed difference over the standard deviation
of the *bootstrap distribution*, which is a z-like statistic that shrinks as
sqrt(n) grows. Both were returned under the key `effect_size`, into the same
reports, and the second one made an effect look stronger merely for having been
measured on more queries.

The p-values disagreed too: the ablation path applied a continuity correction
and this one did not, so `p = 0.0` was reachable here — a value no finite
resample can support.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults  # noqa: E402


class TestEffectSizeDoesNotGrowWithSampleSize:
    """The property that separates an effect size from a test statistic."""

    def test_dz_is_stable_as_n_increases(self):
        rng = np.random.RandomState(0)
        small = rng.normal(0.05, 0.1, size=50)
        large = np.concatenate([small, rng.normal(0.05, 0.1, size=4950)])

        d_small = BenchmarkResults._cohens_dz(small)
        d_large = BenchmarkResults._cohens_dz(large)

        assert abs(d_small - d_large) < 0.25, (
            f"d_z moved from {d_small:.3f} to {d_large:.3f} on 100x the data; "
            f"an effect size must not track sample size"
        )

    def test_a_bootstrap_sd_ratio_would_have_failed_that(self):
        """Documents why the old stratified formula was wrong, not a tautology."""
        rng = np.random.RandomState(0)
        diffs = rng.normal(0.05, 0.1, size=50)

        def z_like(d):
            # observed / sd(bootstrap of the mean) == the old formula
            return float(np.mean(d)) / (float(np.std(d, ddof=1)) / np.sqrt(len(d)))

        small = z_like(diffs)
        large = z_like(np.concatenate([diffs] * 100))
        assert large > small * 5, (
            "the old formula should grow sharply with n — if it does not, this "
            "test is not demonstrating the bug it claims to"
        )

    def test_zero_variance_does_not_divide_by_zero(self):
        """Still the point of this test -- no ZeroDivisionError, no NaN.

        The expected value changed from 0.0 to infinity. Every query moving by
        the same amount is the most consistent effect there is, and 0.0 is what
        a genuinely null result returns, so the two opposite conclusions were
        indistinguishable. 0.0 was a sentinel chosen to avoid the division; it
        was never the answer.
        """
        import math

        assert BenchmarkResults._cohens_dz([0.3, 0.3, 0.3]) == math.inf
        assert BenchmarkResults._cohens_dz([-0.3, -0.3, -0.3]) == -math.inf

    def test_no_difference_at_all_is_still_zero(self):
        """The case 0.0 is right for, kept adjacent to the one above so the
        distinction cannot be collapsed again by accident."""
        assert BenchmarkResults._cohens_dz([0.0, 0.0, 0.0]) == 0.0

    def test_differences_equal_only_to_floating_point_noise_are_treated_as_equal(self):
        """0.9-0.4 and 0.7-0.2 are both 0.5 to a reader and differ in the last
        bit to a float, leaving sd around 2.5e-17 -- positive, so a bare
        ``sd > 0`` guard divides by it and reports d_z = 1.99e16.

        That is not a large effect size, it is the representation error being
        printed as a measurement, and it is not stable: reorder the inputs and
        the sixteen digits change.
        """
        import math

        noisy_constant = [0.9 - 0.4, 0.8 - 0.3, 0.7 - 0.2, 0.6 - 0.1, 0.5 - 0.0] * 10
        assert len(set(noisy_constant)) > 1, "the premise: these are not bit-identical"
        assert BenchmarkResults._cohens_dz(noisy_constant) == math.inf

    def test_a_real_but_small_spread_is_not_swallowed(self):
        """Guards the guard. A threshold set too loosely would report ∞ for
        genuine measurements, which is the same failure in the other
        direction."""
        import math

        values = [0.30, 0.302, 0.298, 0.301, 0.299] * 10
        result = BenchmarkResults._cohens_dz(values)
        assert math.isfinite(result)
        assert result > 100  # consistent, but measurably so

    def test_a_single_observation_has_no_effect_size(self):
        assert BenchmarkResults._cohens_dz([0.3]) == 0.0

    def test_sign_follows_the_direction_of_the_difference(self):
        assert BenchmarkResults._cohens_dz([0.1, 0.2, 0.15]) > 0
        assert BenchmarkResults._cohens_dz([-0.1, -0.2, -0.15]) < 0


class TestPValueIsNeverExactlyZero:
    """`p = 0.0` claims certainty a finite resample cannot provide."""

    def test_a_wildly_separated_effect_still_reports_a_floor(self):
        from lasagna.knowledge_evaluation_sdk.results import _sign_flip_p_value

        p, method = _sign_flip_p_value([5.0] * 200, n_resamples=10_000)
        assert method == "monte_carlo"
        assert p == pytest.approx(1 / 10_001, rel=1e-6), (
            "the floor should be 1/(n+1), i.e. 'smaller than one resample in n'"
        )

    def test_an_enumerated_effect_counts_the_observed_pattern(self):
        """Exact: the observed signs and their mirror image, 2 of 2**10."""
        from lasagna.knowledge_evaluation_sdk.results import _sign_flip_p_value

        assert _sign_flip_p_value([5.0] * 10)[0] == pytest.approx(2 / 1024)

    def test_a_null_effect_reports_a_large_p(self):
        from lasagna.knowledge_evaluation_sdk.results import _sign_flip_p_value

        rng = np.random.RandomState(1)
        diffs = rng.normal(0.0, 1.0, size=40)
        assert _sign_flip_p_value(diffs - diffs.mean())[0] > 0.5

    def test_p_is_a_probability(self):
        from lasagna.knowledge_evaluation_sdk.results import _sign_flip_p_value

        rng = np.random.RandomState(2)
        for shift in (-2.0, -0.1, 0.0, 0.1, 2.0):
            for n in (8, 60):
                p, _ = _sign_flip_p_value(rng.normal(shift, 1, n), n_resamples=2_000)
                assert 0.0 < p <= 1.0


class TestBothTestsUseTheSameConstruction:
    """A shared helper is the point; assert the seam rather than the copy."""

    def test_significance_test_uses_the_shared_p_value(self):
        import inspect

        src = inspect.getsource(BenchmarkResults.significance_test)
        assert "_sign_flip_p_value" in src
        assert "np.mean(np.abs(" not in src, "an inline p-value re-appeared"

    def test_stratified_test_uses_the_shared_helpers(self):
        import inspect

        src = inspect.getsource(BenchmarkResults.domain_stratified_significance_test)
        assert "_sign_flip_p_value" in src
        assert "_cohens_dz" in src

    def test_stratified_effect_size_is_computed_over_domains(self):
        """Not over the bootstrap distribution, which was the bug."""
        import inspect

        src = inspect.getsource(BenchmarkResults.domain_stratified_significance_test)
        assert "domain_diffs" in src
        assert "np.std(boot_arr" not in src, (
            "the effect size is being standardised by the bootstrap spread again"
        )


def _task(config, run_index, scores, benchmark="beir/scifact", metric="ndcg@10"):
    """A TaskResult carrying per-sample scores for one run."""
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    return TaskResult(
        task_key=f"{benchmark}|{config}|{run_index}",
        benchmark_name=benchmark,
        config_name=config,
        run_index=run_index,
        metrics={metric: sum(scores.values()) / len(scores)},
        per_sample_scores=[
            {"_sample_id": sid, metric: v, "_source_benchmark": benchmark}
            for sid, v in scores.items()
        ],
    )


class TestEveryRunContributes:
    """`n_runs` was collected, paid for, and then discarded by the test.

    `_paired_scores` read `tasks[0]`, so a 3-run experiment used one run — two
    thirds of the compute wasted, and the run-to-run variance that
    `variance_source="run"` exists to capture never reached the statistics.
    """

    def _results(self, n_runs):
        tasks = []
        for run in range(n_runs):
            # Config A drifts across runs; averaging must see all of it.
            tasks.append(_task("a", run, {"q1": 0.5 + 0.1 * run, "q2": 0.6 + 0.1 * run}))
            tasks.append(_task("b", run, {"q1": 0.4, "q2": 0.5}))
        return BenchmarkResults(tasks, "exp")

    def test_scores_are_averaged_across_runs(self):
        res = self._results(3)
        ids, a, b, *_ = res._paired_scores("beir/scifact", "a", "b", "ndcg@10")
        assert ids == ["q1", "q2"]
        # q1 across runs: 0.5, 0.6, 0.7 -> 0.6
        assert a[0] == pytest.approx(0.6)
        assert b[0] == pytest.approx(0.4)

    def test_run_zero_alone_would_give_a_different_answer(self):
        """Guards against the fix silently reverting to tasks[0]."""
        res = self._results(3)
        _, a, _, *_ = res._paired_scores("beir/scifact", "a", "b", "ndcg@10")
        assert a[0] != pytest.approx(0.5), "only run 0 was used"

    def test_one_run_still_works(self):
        res = self._results(1)
        _, a, b, *_ = res._paired_scores("beir/scifact", "a", "b", "ndcg@10")
        assert a[0] == pytest.approx(0.5)

    def test_a_failed_run_is_excluded_not_averaged_in(self):
        tasks = [
            _task("a", 0, {"q1": 0.9}),
            _task("b", 0, {"q1": 0.4}),
        ]
        broken = _task("a", 1, {"q1": 0.0})
        broken.error = "boom"
        tasks.append(broken)
        res = BenchmarkResults(tasks, "exp")
        _, a, _, *_ = res._paired_scores("beir/scifact", "a", "b", "ndcg@10")
        assert a[0] == pytest.approx(0.9), "a failed run was averaged in as a real score"

    def test_significance_test_consumes_all_runs(self):
        res = self._results(3)
        out = res.significance_test("a", "b", "ndcg@10", "beir/scifact", n_bootstrap=200)
        assert out["n_samples"] == 2
        assert out["mean_a"] == pytest.approx(0.65)  # (0.6 + 0.7) / 2

    def test_stratified_test_consumes_all_runs(self):
        res = self._results(3)
        out = res.domain_stratified_significance_test(
            "a", "b", "ndcg@10", "beir/scifact", n_bootstrap=200
        )
        assert out["mean_a"] == pytest.approx(0.65)
