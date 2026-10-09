"""A paired test at alpha=0.05 must reject a true null about 5% of the time.

The paired bootstrap did not at the sample sizes RAG papers use. Simulated under
the null, it rejected 9.7% of the time at n=10, 9.1% at n=12 (the smoke
benchmark) and 7.7% at n=20, and its 95% percentile interval covered the true
difference 91% of the time at n=15. LLM-judged generation metrics are routinely
run at n=20-50, so "significant" there meant roughly 1.5 times the error rate it
claimed. The interaction test inherited the same defect.

These simulations are small enough to run on every commit. They are seeded, so
a pass is reproducible; the thresholds sit several standard errors above 0.05,
so the exact test passes with room and the bootstrap it replaced fails.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from lasagna.knowledge_evaluation_sdk import results as results_module  # noqa: E402
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults  # noqa: E402
from lasagna.knowledge_evaluation_sdk.runner import TaskResult  # noqa: E402


def _results(cells):
    """BenchmarkResults with one task per cell, scored on the same queries."""
    return BenchmarkResults(
        [
            TaskResult(
                task_key=name,
                benchmark_name="b",
                config_name=name,
                run_index=0,
                per_sample_scores=[
                    {"_sample_id": f"q{i:03d}", "m": float(v)} for i, v in enumerate(scores)
                ],
            )
            for name, scores in cells.items()
        ]
    )


def _null_pair(rng, n, zero_inflated):
    """Two arms that are exchangeable within each query: the null is true."""
    quality = rng.uniform(0.2, 0.8, size=n)
    a = quality + rng.normal(0, 0.1, size=n)
    b = quality + rng.normal(0, 0.1, size=n)
    if zero_inflated:
        # NDCG-like: many queries score 0 in one arm or both.
        a = np.where(rng.random(n) < 0.3, 0.0, np.clip(a, 0, 1))
        b = np.where(rng.random(n) < 0.3, 0.0, np.clip(b, 0, 1))
    return a, b


def _rejection_rate(n, reps, seed, zero_inflated=False, n_bootstrap=2_000):
    rng = np.random.default_rng(seed)
    rejected = 0
    for _ in range(reps):
        a, b = _null_pair(rng, n, zero_inflated)
        out = _results({"a": a, "b": b}).significance_test(
            "a", "b", "m", "b", n_bootstrap=n_bootstrap
        )
        rejected += out["significant"]
    return rejected / reps


class TestTheFalsePositiveRateIsNominal:
    @pytest.mark.parametrize("zero_inflated", [False, True])
    def test_at_ten_to_twenty_queries(self, zero_inflated):
        """The bootstrap's pooled rate over these sizes was about 0.086."""
        rates = [
            _rejection_rate(n, 500, seed=n, zero_inflated=zero_inflated) for n in (10, 12, 15, 20)
        ]
        pooled = sum(rates) / len(rates)
        assert pooled <= 0.07, f"false positive rates {rates} at n=10,12,15,20"

    def test_at_thirty_and_fifty_queries(self):
        """Exact at 30, sampled sign flips at 50; both near 0.05."""
        for n in (30, 50):
            rate = _rejection_rate(n, 400, seed=100 + n)
            assert 0.02 <= rate <= 0.085, f"n={n}: false positive rate {rate}"

    def test_the_interaction_test_holds_its_level(self):
        """The bootstrap interaction test rejected 8.3% of true nulls at n=20."""
        rng = np.random.default_rng(7)
        rejected, reps, n = 0, 1_500, 20
        for _ in range(reps):
            quality = rng.uniform(0.2, 0.8, size=n)
            cells = {
                name: quality + rng.normal(0, 0.1, size=n) for name in ("neither", "a", "b", "a+b")
            }
            out = _results(cells).interaction_test(
                "b", "m", neither="neither", only_a="a", only_b="b", both="a+b", n_bootstrap=500
            )
            rejected += out["significant"]
        assert rejected / reps <= 0.065, f"interaction false positive rate {rejected / reps}"


class TestTheIntervalCoversAtItsLevel:
    def test_coverage_at_fifteen_queries(self):
        """The percentile bootstrap covered 91% at n=15 for a nominal 95%."""
        rng = np.random.default_rng(15)
        covered, reps = 0, 1_000
        for _ in range(reps):
            a, b = _null_pair(rng, 15, zero_inflated=False)
            lo, hi = (
                _results({"a": a + 0.05, "b": b})
                .significance_test("a", "b", "m", "b", n_bootstrap=200)
                .get("confidence_interval")
            )
            covered += lo <= 0.05 <= hi
        assert covered / reps >= 0.93, f"coverage {covered / reps}"

    def test_the_interval_is_at_the_requested_level(self):
        """alpha=0.2 is an 80% interval, narrower than the 95% one."""
        rng = np.random.default_rng(3)
        a, b = _null_pair(rng, 40, zero_inflated=False)
        res = _results({"a": a, "b": b})
        wide = res.significance_test("a", "b", "m", "b", alpha=0.05)
        narrow = res.significance_test("a", "b", "m", "b", alpha=0.2)
        assert narrow["confidence_level"] == pytest.approx(0.8)
        assert (narrow["confidence_interval"][1] - narrow["confidence_interval"][0]) < (
            wide["confidence_interval"][1] - wide["confidence_interval"][0]
        )


class TestTheSignFlipTest:
    def test_small_samples_are_enumerated_exactly(self):
        """Every sign pattern of three differences, counted by hand."""
        import itertools

        diffs = [0.3, -0.1, 0.2]
        observed = abs(sum(diffs))
        by_hand = (
            sum(
                abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=3)
            )
            / 8
        )
        p, method = results_module._sign_flip_p_value(diffs)
        assert method == "exact"
        assert p == pytest.approx(by_hand)

    def test_it_agrees_with_scipy_where_scipy_enumerates(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        diffs = rng.normal(0.2, 1.0, size=18)
        p, method = results_module._sign_flip_p_value(diffs)
        ref = stats.permutation_test(
            (diffs,), np.mean, permutation_type="samples", n_resamples=np.inf
        ).pvalue
        assert method == "exact"
        assert p == pytest.approx(ref, rel=1e-9)

    def test_large_samples_are_sampled_and_never_report_zero(self):
        p, method = results_module._sign_flip_p_value([1.0] * 200, n_resamples=999)
        assert method == "monte_carlo"
        assert p == pytest.approx(1 / 1000)

    def test_no_difference_at_all_is_p_one(self):
        assert results_module._sign_flip_p_value([0.0, 0.0, 0.0])[0] == 1.0

    def test_zero_differences_do_not_change_the_p_value(self):
        """They cannot change sign, so they carry no evidence either way."""
        with_zeros = results_module._sign_flip_p_value([0.3, -0.1, 0.2, 0.0, 0.0])[0]
        without = results_module._sign_flip_p_value([0.3, -0.1, 0.2])[0]
        assert with_zeros == pytest.approx(without)

    def test_the_result_says_which_test_ran(self):
        rng = np.random.default_rng(1)
        a, b = _null_pair(rng, 12, zero_inflated=False)
        out = _results({"a": a, "b": b}).significance_test("a", "b", "m", "b")
        assert out["test"] == "paired sign-flip (exact)"
        assert out["ci_method"] == "t"


class TestTheStudentT:
    """Written out because scipy is an optional extra; checked against it here."""

    @pytest.mark.parametrize("df", [1, 2, 4.5, 9, 14, 29, 99, 10_000])
    def test_quantiles_match_scipy(self, df):
        stats = pytest.importorskip("scipy.stats")
        for p in (0.6, 0.9, 0.975, 0.995, 0.9999):
            assert results_module._t_quantile(p, df) == pytest.approx(stats.t.ppf(p, df), rel=1e-8)

    @pytest.mark.parametrize("df", [1, 3, 11, 50])
    def test_tail_probabilities_match_scipy(self, df):
        stats = pytest.importorskip("scipy.stats")
        for t in (0.0, 0.5, 2.0, 4.0, 12.0):
            assert results_module._t_two_sided_p(t, df) == pytest.approx(
                2 * stats.t.sf(t, df), rel=1e-8
            )

    def test_the_interval_is_infinite_without_a_spread(self):
        lo, hi = results_module._t_interval([0.4], 0.05)
        assert lo == -math.inf and hi == math.inf


class TestTheAblationPathRunsTheSameTest:
    """`AblationResults.significance_test` had its own bootstrap and effect size."""

    def _ablation(self, a, b):
        from lasagna.knowledge_evaluation_sdk.ablation import AblationResults

        return AblationResults(
            [{"variant": "a", "metrics": {}}, {"variant": "b", "metrics": {}}],
            per_sample_scores={
                "a": [{"m": float(v)} for v in a],
                "b": [{"m": float(v)} for v in b],
            },
        )

    def test_it_matches_the_benchmark_results_test(self):
        rng = np.random.default_rng(5)
        a, b = _null_pair(rng, 14, zero_inflated=False)
        ours = self._ablation(a, b).significance_test("a", "b", "m")
        theirs = _results({"a": a, "b": b}).significance_test("a", "b", "m", "b")
        assert ours["p_value"] == pytest.approx(theirs["p_value"])
        assert ours["confidence_interval"] == pytest.approx(theirs["confidence_interval"])
        assert ours["effect_size"] == pytest.approx(theirs["effect_size"])

    def test_a_missing_score_is_not_read_as_zero(self):
        ablation = self._ablation([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
        del ablation._per_sample_scores["b"][2]["m"]
        out = ablation.significance_test("a", "b", "m")
        assert out["n_samples"] == 2
        assert out["mean_b"] == pytest.approx(0.45)

    def test_a_consistent_effect_is_not_reported_as_zero(self):
        out = self._ablation([0.6] * 5, [0.4] * 5).significance_test("a", "b", "m")
        assert out["effect_size"] == math.inf


class TestTheStratifiedTest:
    def test_one_domain_reduces_to_the_paired_test(self):
        """Same weights, same variance, n - 1 degrees of freedom."""
        rng = np.random.default_rng(9)
        a, b = _null_pair(rng, 16, zero_inflated=False)
        res = _results({"a": a + 0.03, "b": b})
        paired = res.significance_test("a", "b", "m", "b")
        stratified = res.domain_stratified_significance_test("a", "b", "m", "b")
        assert stratified["p_value"] == pytest.approx(paired["p_value"])
        assert stratified["confidence_interval"] == pytest.approx(paired["confidence_interval"])

    def test_a_small_domain_counts_as_much_as_a_large_one(self):
        """The macro average weights domains equally, whatever their size: a
        +0.3 gain on 4 queries and none on 40 is a +0.15 macro difference."""
        rng = np.random.default_rng(11)
        noise = {f"big{i}": rng.normal(0, 0.02) for i in range(40)}
        noise.update({f"small{i}": rng.normal(0, 0.02) for i in range(4)})

        def rows(gain):
            out = [
                {"_sample_id": sid, "m": 0.5 + noise[sid], "_source_benchmark": "big"}
                for sid in noise
                if sid.startswith("big")
            ]
            out += [
                {"_sample_id": sid, "m": 0.5 + gain + noise[sid], "_source_benchmark": "small"}
                for sid in noise
                if sid.startswith("small")
            ]
            return out

        rng_b = np.random.default_rng(12)
        tasks = [
            TaskResult(
                task_key="a",
                benchmark_name="b",
                config_name="a",
                run_index=0,
                per_sample_scores=rows(0.3),
            ),
            TaskResult(
                task_key="b",
                benchmark_name="b",
                config_name="b",
                run_index=0,
                per_sample_scores=[dict(r, m=r["m"] + rng_b.normal(0, 0.02)) for r in rows(0.0)],
            ),
        ]
        out = BenchmarkResults(tasks).domain_stratified_significance_test("a", "b", "m", "b")
        assert out["difference"] == pytest.approx(0.15, abs=0.02)
        lo, hi = out["confidence_interval"]
        assert lo < out["difference"] < hi and hi - lo < 0.2
