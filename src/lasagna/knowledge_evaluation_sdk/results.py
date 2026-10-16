"""Multi-dimensional benchmark results aggregation and statistical analysis.

Provides :class:`BenchmarkResults` — the primary output of an experiment run.
Supports per-benchmark aggregation, cross-config comparison, significance
testing, and export to multiple formats.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .runner import TaskResult


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF.

    Written out rather than taken from scipy, which is not a dependency here
    and would be a heavy one to add for two constants. Acklam's rational
    approximation; accurate to about 1e-9 across the range, far tighter than
    anything a power calculation needs.
    """
    import math

    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile needs 0 < p < 1, got {p}")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _samples_for_effect(effect: float, sd: float, alpha: float, power: float) -> Optional[int]:
    """Paired samples needed to detect ``effect`` at ``power``.

    The actionable half of a power report: "you would need about this many
    queries" is a decision, where "underpowered" alone is only a diagnosis.
    """
    import math

    if effect <= 0 or sd <= 0:
        return None
    z = _normal_quantile(1 - alpha / 2) + _normal_quantile(power)
    return int(math.ceil((z * sd / effect) ** 2))


def _betainc(a: float, b: float, x: float, x_complement: Optional[float] = None) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``.

    Lentz's continued fraction (Numerical Recipes, ``betacf``), for the same
    reason ``_normal_quantile`` exists: scipy is an optional extra, and the
    Student-t distribution below is all it would be needed for.

    ``x_complement`` is ``1 - x`` computed by the caller without cancellation,
    which matters when ``x`` is within rounding of 1.
    """
    import math

    y = (1.0 - x) if x_complement is None else x_complement
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0

    def _cf(a: float, b: float, x: float) -> float:
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > tiny else tiny)
        h = d
        for m in range(1, 1000):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            step = d * c
            h *= step
            if abs(step - 1.0) < 1e-15:
                break
        return h

    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    )
    front = math.exp(log_front)
    # The fraction converges fast only on one side of the mode; use symmetry
    # on the other.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _cf(a, b, x) / a
    return 1.0 - front * _cf(b, a, y) / b


def _t_two_sided_p(t: float, df: float) -> float:
    """``P(|T| >= |t|)`` for Student's t with ``df`` degrees of freedom."""
    import math

    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    t2 = t * t
    return _betainc(df / 2.0, 0.5, df / (df + t2), t2 / (df + t2))


def _t_quantile(p: float, df: float) -> float:
    """Inverse Student-t CDF, by bisection on :func:`_t_two_sided_p`.

    Bisection rather than a closed-form approximation because the common
    approximations lose accuracy at the small ``df`` this exists for: a paired
    comparison over 10 queries has 9 degrees of freedom.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile needs 0 < p < 1, got {p}")
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_t_quantile(1.0 - p, df)

    target = 2.0 * (1.0 - p)  # two-sided tail mass beyond the quantile
    lo, hi = 0.0, 1.0
    while _t_two_sided_p(hi, df) > target:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


#: Largest number of non-zero differences the sign-flip test enumerates in
#: full. Splitting the 2**n sign vectors into two halves of 2**(n/2) keeps this
#: to a few milliseconds per comparison; above it the test samples sign
#: vectors instead.
_EXACT_SIGN_FLIP_MAX_N = 30


def _sign_flip_p_value(terms, *, n_resamples: int = 10_000, seed: int = 42) -> Tuple[float, str]:
    """Two-sided paired sign-flip randomization test that ``sum(terms)`` is 0.

    ``terms`` are the paired differences (or weighted differences, for a
    stratified mean). Under the null that each difference is as likely to have
    come out with either sign, every one of the 2**n sign patterns is equally
    likely, so the p-value is the share of patterns whose sum is at least as
    far from zero as the observed one. That holds at every n, which is the
    reason this replaced the paired bootstrap: the bootstrap's plug-in variance
    is too narrow at small n, and at 12 to 20 queries it rejected a true null
    8 to 9 percent of the time at alpha 0.05.

    Exact when at most :data:`_EXACT_SIGN_FLIP_MAX_N` differences are non-zero
    (zeros cannot change sign, so they drop out without changing the
    p-value). Above that, ``n_resamples`` random sign patterns are drawn and
    ``(count + 1) / (n_resamples + 1)`` is returned, which counts the observed
    pattern and so is never exactly 0.

    Returns ``(p_value, method)`` where ``method`` is ``"exact"`` or
    ``"monte_carlo"``.
    """
    import numpy as np

    x = np.asarray(terms, dtype=float)
    x = x[x != 0.0]
    observed = abs(float(np.sum(x)))
    if x.size == 0 or observed == 0.0:
        return 1.0, "exact"

    # Sign patterns whose sum equals the observed one in exact arithmetic can
    # land a rounding error either side of it; count them as reaching it.
    threshold = observed - 1e-9 * float(np.sum(np.abs(x)))

    if x.size <= _EXACT_SIGN_FLIP_MAX_N:

        def _all_sums(values) -> Any:
            sums = np.zeros(1)
            for v in values:
                sums = np.concatenate([sums + v, sums - v])
            return sums

        half = x.size // 2
        left = _all_sums(x[:half])
        right = np.sort(_all_sums(x[half:]))
        # Pairs (l, r) with l + r >= threshold, plus pairs with l + r <= -threshold.
        upper = right.size - np.searchsorted(right, threshold - left, side="left")
        lower = np.searchsorted(right, -threshold - left, side="right")
        count = int(np.sum(upper) + np.sum(lower))
        return count / float(2**x.size), "exact"

    rng = np.random.default_rng(seed)
    total = float(np.sum(x))
    rows = max(1, min(n_resamples, 2_000_000 // x.size))
    count, remaining = 0, n_resamples
    while remaining > 0:
        take = min(rows, remaining)
        bits = rng.integers(0, 2, size=(take, x.size), dtype=np.int8)
        # sum(s * x) with s in {-1, +1} is 2 * sum(bits * x) - sum(x).
        sums = 2.0 * (bits @ x) - total
        count += int(np.count_nonzero(np.abs(sums) >= threshold))
        remaining -= take
    return (count + 1) / (n_resamples + 1), "monte_carlo"


def _t_interval(diffs, alpha: float) -> Tuple[float, float]:
    """Two-sided ``1 - alpha`` interval for the mean of paired differences.

    ``mean +/- t(1 - alpha/2, n - 1) * sd / sqrt(n)``. The percentile bootstrap
    it replaces covered the true difference 91 percent of the time at n=15 for
    a nominal 95, for the same reason its p-value was too small. Infinite when
    n < 2, since one difference says nothing about spread.
    """
    import math

    import numpy as np

    arr = np.asarray(diffs, dtype=float)
    n = arr.size
    mean = float(np.mean(arr)) if n else float("nan")
    if n < 2:
        return (-math.inf, math.inf)
    se = float(np.std(arr, ddof=1)) / math.sqrt(n)
    half = _t_quantile(1.0 - alpha / 2.0, n - 1) * se
    return (mean - half, mean + half)


def _mean_by_sample(tasks: List[TaskResult], metric: str) -> Dict[Any, float]:
    """Each sample's score for ``metric``, averaged across every run.

    Averaging within a sample before anything resamples is what keeps the
    resampling unit independent: treating every (sample, run) pair as its own
    observation counts the same query several times and narrows the interval
    accordingly.

    Module-level rather than a closure because the factorial estimator needs
    the same index for four configs at once, and three pairwise calls produce
    three different id sets.
    """
    collected: Dict[Any, List[float]] = defaultdict(list)
    for task in tasks:
        if task.error:
            continue
        for i, s in enumerate(task.per_sample_scores or []):
            if metric in s:
                collected[s.get("_sample_id", i)].append(float(s[metric]))
    return {sid: sum(vals) / len(vals) for sid, vals in collected.items() if vals}


#: Metrics where the smaller value is the better result: costs, failure counts,
#: and RAGAS noise sensitivity (the share of claims the retrieved noise led the
#: generator to get wrong). Every other metric ranks higher-is-better.
#:
#: ``llm_cache_hits`` and ``llm_cache_misses`` are deliberately absent: they
#: say what an earlier run left in the cache, not how good or costly the arm
#: is, so neither direction ranks anything.
LOWER_IS_BETTER = frozenset(
    {
        "noise_sensitivity",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "llm_calls",
        "input_tokens_per_query",
        "output_tokens_per_query",
        "llm_calls_per_query",
        "generations_failed",
        "short_lists",
        "agentic_fallback_count",
        "agentic_fallback_rate",
        "cost_usd_estimate",
    }
)


def lower_is_better(metric: str) -> bool:
    """Whether a smaller ``metric`` ranks first (``noise_sensitivity_regen`` too)."""
    for suffix in ("_regen", "_agentic"):
        if metric.endswith(suffix):
            metric = metric[: -len(suffix)]
    return metric in LOWER_IS_BETTER or metric.endswith("latency_ms")


@dataclass
class AggregatedMetrics:
    """Mean +/- std for a metric across multiple runs."""

    mean: float
    std: float
    n_runs: int
    values: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        if self.n_runs < 2:
            return f"{self.mean:.4f} (n={self.n_runs})"
        return f"{self.mean:.4f} ± {self.std:.4f} (n={self.n_runs})"


class BenchmarkResults:
    """Multi-dimensional results: benchmarks x pipeline_configs x runs.

    Example::

        results = await experiment.run()

        # Per-benchmark summary
        for bench, metrics in results.aggregate_by_benchmark().items():
            print(f"{bench}: {metrics}")

        # Leaderboard
        print(results.leaderboard(metric="ndcg@10"))

        # Significance
        sig = results.significance_test("config_a", "config_b", "ndcg@10", "beir/nfcorpus")
        print(f"p={sig['p_value']:.4f}")

        # Export
        results.save("results.json")
        print(results.to_markdown())
    """

    def __init__(
        self,
        task_results: List[TaskResult],
        experiment_name: str = "",
        experiment_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._results = task_results
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config or {}

        # Index results for fast lookup
        self._by_benchmark: Dict[str, List[TaskResult]] = defaultdict(list)
        self._by_config: Dict[str, List[TaskResult]] = defaultdict(list)
        self._by_bench_config: Dict[Tuple[str, str], List[TaskResult]] = defaultdict(list)

        # Sorted by run_index, not arrival order. The runner collects tasks with
        # `asyncio.as_completed`, so without this every `tasks[0]` in this class
        # means "whichever run finished first" -- which varies between identical
        # invocations and silently makes results order-dependent.
        for r in sorted(self._results, key=lambda t: t.run_index):
            self._by_benchmark[r.benchmark_name].append(r)
            self._by_config[r.config_name].append(r)
            self._by_bench_config[(r.benchmark_name, r.config_name)].append(r)

    @property
    def benchmarks(self) -> List[str]:
        return sorted(self._by_benchmark.keys())

    @property
    def configs(self) -> List[str]:
        return sorted(self._by_config.keys())

    @property
    def total_tasks(self) -> int:
        return len(self._results)

    @property
    def failed_tasks(self) -> List[TaskResult]:
        return [r for r in self._results if r.error]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_benchmark(
        self,
        metric: str = "ndcg@10",
    ) -> Dict[str, Dict[str, AggregatedMetrics]]:
        """Per-benchmark, per-config aggregation.

        Returns:
            ``{benchmark: {config: AggregatedMetrics}}``
        """
        result: Dict[str, Dict[str, AggregatedMetrics]] = {}
        for bench in self.benchmarks:
            result[bench] = {}
            for config in self.configs:
                tasks = self._by_bench_config.get((bench, config), [])
                # Only include tasks that have this metric — don't default to 0.0
                values = [t.metrics[metric] for t in tasks if not t.error and metric in t.metrics]
                if values:
                    result[bench][config] = _aggregate(values)
        return result

    def aggregate_by_pipeline(
        self,
        metric: str = "ndcg@10",
    ) -> Dict[str, AggregatedMetrics]:
        """Per-config aggregation across all benchmarks.

        Returns:
            ``{config: AggregatedMetrics}``
        """
        result: Dict[str, AggregatedMetrics] = {}
        for config in self.configs:
            tasks = [t for t in self._by_config[config] if not t.error]
            # Only include tasks that have this metric
            values = [t.metrics[metric] for t in tasks if metric in t.metrics]
            if values:
                result[config] = _aggregate(values)
        return result

    def get_metric_names(self) -> List[str]:
        """Return all unique metric names across results."""
        names: set = set()
        for r in self._results:
            names.update(r.metrics.keys())
        return sorted(names)

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_dz(diffs) -> float:
        """Cohen's ``d_z`` for a paired design: mean(diffs) / sd(diffs).

        Standardised by the spread of the *differences*, so it does not grow
        with sample size. Dividing the observed difference by the standard
        deviation of the bootstrap distribution -- which the stratified test
        used to do -- yields a z-like statistic that grows as sqrt(n), so the
        same effect looked stronger merely for having been measured on more
        queries. Both were reported under the key ``effect_size``.

        Returns ``±inf`` when every query moved by the same non-zero amount.
        That is a perfectly consistent effect, not the absence of one, and the
        ``sd > 0`` guard used to return ``0.0`` for it -- the same value a
        genuinely null result gives, so the strongest possible finding and no
        finding at all were indistinguishable. `minimum_detectable_effect`
        already flags this case as ``degenerate``; this did not.

        The threshold is relative rather than ``sd == 0`` because exact
        equality almost never holds. Scores like ``0.50 + 0.001*i`` against
        ``0.80 + 0.001*i`` differ by 0.3 everywhere, but in binary the
        differences vary in the last bit or two, leaving ``sd`` around 1e-17 --
        positive, so the old guard passed, and ``d_z`` came out at 5.1e15. A
        number of that size is floating-point noise being reported to two
        decimal places, which is worse than either 0 or infinity because it
        looks like a measurement.
        """
        import math

        import numpy as np

        arr = np.asarray(diffs, dtype=float)
        if arr.size < 2:
            return 0.0
        sd = float(np.std(arr, ddof=1))
        mean = float(np.mean(arr))

        # Constant to within the precision the differences are carried at.
        scale = max(abs(mean), float(np.max(np.abs(arr))), 1.0)
        if sd <= scale * 1e-12:
            if mean == 0.0:
                return 0.0
            return math.inf if mean > 0 else -math.inf

        return mean / sd

    def significance_test(
        self,
        config_a: str,
        config_b: str,
        metric: str,
        benchmark: str,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Paired significance test between two configs on the SAME benchmark.

        A sign-flip randomization test of the mean paired difference, exact
        when at most 30 differences are non-zero and sampled above (see
        :func:`_sign_flip_p_value`), with a t interval around the difference.
        Both hold their nominal level at the 10 to 50 queries RAG evaluations
        use; the paired bootstrap this replaced did not.

        Uses per-query scores from **every** run of each config, averaged per
        query first. The ``benchmark`` parameter is REQUIRED: cross-benchmark
        significance is not valid (different query sets).

        Args:
            config_a: First pipeline config name.
            config_b: Second pipeline config name.
            metric: Metric name to compare.
            benchmark: Benchmark name (required for valid pairing).
            n_bootstrap: Random sign patterns drawn when the sample is too
                large to enumerate. The name is kept for compatibility.
            alpha: Significance level; the interval is at ``1 - alpha``.
            seed: Random seed for the sampled patterns.

        Returns:
            Dict with ``p_value``, ``significant``, ``confidence_interval``
            (at ``confidence_level``), ``effect_size``, ``mean_a``, ``mean_b``,
            ``difference``, ``n_paired``, ``n_dropped_a``, ``n_dropped_b`` and
            ``test``. ``confidence_interval_95`` is the same interval under its
            old name, which was wrong for any alpha other than 0.05.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required: pip install lasagna-rag[eval]")

        # Joined on _sample_id, so position i of each array is the same query.
        sample_ids, scores_a, scores_b, dropped_a, dropped_b = self._paired_scores(
            benchmark, config_a, config_b, metric
        )

        if not scores_a:
            raise ValueError(
                f"No samples scored '{metric}' in both '{config_a}' and '{config_b}' "
                f"on benchmark '{benchmark}'. A paired test needs the same queries "
                f"on both sides."
            )

        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        n = len(a)
        diffs = a - b
        observed_diff = float(np.mean(diffs))

        p_value, method = _sign_flip_p_value(diffs, n_resamples=n_bootstrap, seed=seed)
        interval = _t_interval(diffs, alpha)
        effect_size = self._cohens_dz(diffs)

        return {
            "p_value": p_value,
            "significant": p_value < alpha,
            "confidence_interval": interval,
            "confidence_level": 1.0 - alpha,
            "confidence_interval_95": interval,
            "ci_method": "t",
            "test": f"paired sign-flip ({method.replace('_', ' ')})",
            "effect_size": effect_size,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "difference": observed_diff,
            "n_samples": n,
            # How much of each side was usable. A large drop means the two
            # configs scored different subsets, which is worth seeing next to
            # the p-value rather than discovering later.
            "n_paired": n,
            "n_dropped_a": dropped_a,
            "n_dropped_b": dropped_b,
            "bootstrap_seed": seed,
            "n_bootstrap": n_bootstrap,
        }

    def domain_stratified_significance_test(
        self,
        config_a: str,
        config_b: str,
        metric: str,
        benchmark: str,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Domain-stratified paired test for macro-averaged metrics.

        The statistic is the macro average of per-domain mean differences, so a
        large domain (e.g., nfcorpus) cannot dominate it. The p-value is a
        sign-flip test of that statistic, each query weighted by one over its
        domain's size times the number of domains, and the interval is a
        stratified t interval with Welch-Satterthwaite degrees of freedom. Both
        replaced a within-domain bootstrap, which had the same small-sample
        defect as the paired one.

        Requires per-sample scores to include ``_source_benchmark`` field
        (used as domain proxy).
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required: pip install lasagna-rag[eval]")

        # Get ALL per-sample scores (including _source_benchmark metadata)
        tasks_a = self._by_bench_config.get((benchmark, config_a), [])
        tasks_b = self._by_bench_config.get((benchmark, config_b), [])
        if not tasks_a or not tasks_b:
            raise ValueError(f"No tasks for ({benchmark}, {config_a}) or ({benchmark}, {config_b})")

        # Every run contributes, averaged per sample -- see _paired_scores for
        # why the average happens before resampling. This previously read
        # tasks_a[0], discarding every run but the first.
        def _by_sample(tasks):
            scores: Dict[Any, List[float]] = defaultdict(list)
            domain_of: Dict[Any, str] = {}
            for task in tasks:
                if task.error:
                    continue
                for i, s in enumerate(task.per_sample_scores or []):
                    if metric not in s:
                        continue
                    sid = s.get("_sample_id", i)
                    scores[sid].append(float(s[metric]))
                    domain_of.setdefault(sid, s.get("_source_benchmark", "unknown"))
            return {sid: sum(v) / len(v) for sid, v in scores.items()}, domain_of

        mean_a, domain_of = _by_sample(tasks_a)
        mean_b, _ = _by_sample(tasks_b)

        shared_ids = sorted(set(mean_a) & set(mean_b), key=str)
        if not shared_ids:
            raise ValueError(f"No shared sample IDs between configs '{config_a}' and '{config_b}'")

        domains_a: Dict[str, List[float]] = defaultdict(list)
        domains_b: Dict[str, List[float]] = defaultdict(list)
        for sid in shared_ids:
            domain = domain_of.get(sid, "unknown")
            domains_a[domain].append(mean_a[sid])
            domains_b[domain].append(mean_b[sid])

        all_domains = sorted(set(domains_a) & set(domains_b))
        if not all_domains:
            raise ValueError(f"No shared domains with metric '{metric}'")

        import math

        observed_macro_a = np.mean([np.mean(domains_a[d]) for d in all_domains])
        observed_macro_b = np.mean([np.mean(domains_b[d]) for d in all_domains])
        observed_diff = float(observed_macro_a - observed_macro_b)

        n_domains = len(all_domains)
        terms: List[float] = []
        variance, df_denominator = 0.0, 0.0
        for d in all_domains:
            diffs_d = np.asarray(domains_a[d], dtype=float) - np.asarray(domains_b[d], dtype=float)
            n_d = diffs_d.size
            # Each query's share of the macro average.
            terms.extend(diffs_d / (n_domains * n_d))
            if n_d >= 2:
                v_d = float(np.var(diffs_d, ddof=1)) / (n_d * n_domains**2)
                variance += v_d
                df_denominator += v_d**2 / (n_d - 1)

        p_value, method = _sign_flip_p_value(terms, n_resamples=n_bootstrap, seed=seed)
        if any(len(domains_a[d]) < 2 for d in all_domains):
            # A one-query domain has no within-domain spread to estimate.
            ci_lower, ci_upper = -math.inf, math.inf
        elif variance == 0.0:
            ci_lower = ci_upper = observed_diff
        else:
            df = variance**2 / df_denominator
            half = _t_quantile(1.0 - alpha / 2.0, df) * math.sqrt(variance)
            ci_lower, ci_upper = observed_diff - half, observed_diff + half

        # d_z over the PER-DOMAIN differences, which is the level this statistic
        # actually operates on -- the macro-average treats each domain as one
        # observation. Previously this divided by the standard deviation of the
        # bootstrap distribution, which is a z-statistic growing with sqrt(n),
        # reported under the same key as the other test's Cohen's d_z.
        domain_diffs = [float(np.mean(domains_a[d]) - np.mean(domains_b[d])) for d in all_domains]
        effect_size = self._cohens_dz(domain_diffs)

        return {
            "p_value": p_value,
            "significant": p_value < alpha,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": 1.0 - alpha,
            "confidence_interval_95": (ci_lower, ci_upper),
            "ci_method": "stratified t",
            "test": f"stratified sign-flip ({method.replace('_', ' ')})",
            "effect_size": effect_size,
            "mean_a": float(observed_macro_a),
            "mean_b": float(observed_macro_b),
            "difference": observed_diff,
            "n_domains": len(all_domains),
            "domains": all_domains,
            "n_samples_per_domain": {d: len(domains_a[d]) for d in all_domains},
        }

    def pairwise_significance(
        self,
        metric: str,
        benchmark: str,
        correction: str = "holm",
        alpha: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """All-pairs significance tests with multiple comparison correction.

        Every pair asked about stays in the family. A pair that cannot be tested
        (no query scored on both sides) is returned with ``p_value`` None and
        the reason in ``untestable``, and is counted in the Holm family as if
        its p were 1. Dropping it, as this used to, shrank the family and made
        the correction less strict without saying so.

        Sampled sign-flip p-values cannot go below ``1 / (resamples + 1)``, so
        the resample count grows with the family until that floor is a tenth of
        the smallest Holm threshold, ``alpha / m``. At the old fixed 10,000, a
        grid of 34 configs (561 pairs) could not reject anything.

        Args:
            metric: Metric to compare.
            benchmark: Benchmark (required).
            correction: ``"holm"`` (Holm-Bonferroni) or ``"none"``.
            alpha: Family-wise significance level.

        Returns:
            List of dicts with ``config_a``, ``config_b``, ``p_value`` (raw),
            ``p_adjusted`` (Holm-adjusted under ``"holm"``, else raw),
            ``significant``, ``corrected_alpha``, ``n_paired``, ``n_dropped_a``,
            ``n_dropped_b`` and ``test``; tested pairs first, by p.
        """
        import math

        configs = [c for c in self.configs if self._by_bench_config.get((benchmark, c))]
        pairs = [
            (configs[i], configs[j])
            for i in range(len(configs))
            for j in range(i + 1, len(configs))
        ]
        if not pairs:
            return []

        m = len(pairs)
        resamples = max(10_000, math.ceil(10 * m / alpha))
        if resamples > 10_000:
            logger.info(
                f"{m} comparisons: drawing {resamples:,} sign patterns per sampled "
                f"test so the smallest p-value ({1 / (resamples + 1):.1e}) is below "
                f"the smallest Holm threshold ({alpha / m:.1e})."
            )

        raw_results = []
        untestable = []
        for ca, cb in pairs:
            try:
                sig = self.significance_test(ca, cb, metric, benchmark, n_bootstrap=resamples)
            except ValueError as exc:
                untestable.append(
                    {
                        "config_a": ca,
                        "config_b": cb,
                        "p_value": None,
                        "p_adjusted": None,
                        "effect_size": None,
                        "difference": None,
                        "significant": False,
                        "corrected_alpha": None,
                        "n_paired": 0,
                        "untestable": str(exc),
                    }
                )
                continue
            raw_results.append(
                {
                    "config_a": ca,
                    "config_b": cb,
                    "p_value": sig["p_value"],
                    "effect_size": sig["effect_size"],
                    "difference": sig["difference"],
                    "n_paired": sig.get("n_paired"),
                    "n_dropped_a": sig.get("n_dropped_a", 0),
                    "n_dropped_b": sig.get("n_dropped_b", 0),
                    "test": sig.get("test"),
                }
            )

        if not raw_results:
            return untestable

        # Apply Holm-Bonferroni step-down correction:
        # Sort by p-value ascending. Test i has threshold alpha/(m-i).
        # Once a test fails to reject, all subsequent tests also fail.
        # m counts the untestable pairs too; they sort last, as p = 1.
        if correction == "holm":
            raw_results.sort(key=lambda x: x["p_value"])
            rejected_so_far = True
            running_max = 0.0
            for i, r in enumerate(raw_results):
                corrected_alpha = alpha / (m - i)
                r["corrected_alpha"] = corrected_alpha
                # The value a reader compares with alpha. Only the raw p was
                # returned, and printed under a Holm header.
                running_max = max(running_max, min(1.0, (m - i) * r["p_value"]))
                r["p_adjusted"] = running_max
                if rejected_so_far and r["p_value"] <= corrected_alpha:
                    r["significant"] = True
                else:
                    rejected_so_far = False
                    r["significant"] = False
        else:
            for r in raw_results:
                r["corrected_alpha"] = alpha
                r["p_adjusted"] = r["p_value"]
                r["significant"] = r["p_value"] < alpha

        return raw_results + untestable

    def _get_per_sample_scores(
        self,
        benchmark: str,
        config: str,
        metric: str,
    ) -> List[float]:
        """Extract per-sample scores for a (benchmark, config) pair.

        Unpaired — for single-config summaries only. **Do not use this to build
        the two arrays of a paired test**: filtering each config independently
        can yield equal-length arrays whose positions refer to different
        queries. Use :meth:`_paired_scores`.
        """
        tasks = self._by_bench_config.get((benchmark, config), [])
        for task in tasks:
            if task.per_sample_scores:
                return [s[metric] for s in task.per_sample_scores if metric in s]
        return []

    def _paired_scores(
        self,
        benchmark: str,
        config_a: str,
        config_b: str,
        metric: str,
    ) -> Tuple[List[Any], List[float], List[float], int, int]:
        """Per-sample scores for two configs, aligned by sample identity.

        Returns ``(sample_ids, a, b, n_dropped_a, n_dropped_b)``.

        A paired test subtracts these arrays element-wise, so position *i* of
        each must be the same query. Selecting each config's scores
        independently does not guarantee that: if the metric is missing on
        different samples in each config, both arrays can come out the same
        length — passing a length check — while comparing query 2 against
        query 3.

        This joins on ``_sample_id``, which every eval path already populates,
        and takes only samples where *both* configs scored the metric.

        **Every run contributes.** This used to read ``tasks[0]`` -- run 0 only
        -- so a 3-run experiment threw away two thirds of its compute *and* the
        run-to-run variance that ``variance_source="run"`` exists to capture.
        Each sample's score is averaged across the runs that scored it, and the
        bootstrap then resamples *samples*. Averaging within a sample first is
        what keeps the resampling unit independent: treating every (sample, run)
        pair as its own observation would count the same query several times and
        narrow the interval accordingly.
        """
        tasks_a = self._by_bench_config.get((benchmark, config_a), [])
        tasks_b = self._by_bench_config.get((benchmark, config_b), [])
        if not tasks_a or not tasks_b:
            raise ValueError(f"No tasks for ({benchmark}, {config_a}) or ({benchmark}, {config_b})")

        idx_a = _mean_by_sample(tasks_a, metric)
        idx_b = _mean_by_sample(tasks_b, metric)

        shared = sorted(set(idx_a) & set(idx_b), key=str)
        ids: List[Any] = list(shared)
        a: List[float] = [idx_a[sid] for sid in shared]
        b: List[float] = [idx_b[sid] for sid in shared]

        # Reported rather than swallowed: silent dropping is what let the
        # positional version look healthy.
        n_dropped_a = len(idx_a) - len(a)
        n_dropped_b = len(idx_b) - len(b)
        return ids, a, b, max(n_dropped_a, 0), max(n_dropped_b, 0)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def minimum_detectable_effect(
        self,
        benchmark: str,
        config_a: str,
        config_b: str,
        metric: str,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict[str, Any]:
        """The smallest difference this run could have detected.

        A non-significant result has two readings — the techniques are the
        same, or the experiment was too small to tell — and they call for
        opposite next steps. Nothing in a p-value distinguishes them. This does:
        if the observed difference is smaller than the MDE, the run could not
        have found an effect of that size even had it been real, and the honest
        conclusion is "inconclusive" rather than "no difference".

        Computed *post hoc*, from the variance actually observed:

            MDE = (z_alpha/2 + z_power) * sd(differences) / sqrt(n)

        This is the reason it is not offered before the run. An a-priori
        calculation needs a variance estimate nobody has yet, and picking one
        turns the sample-size decision into a guess dressed as arithmetic.

        Normal quantiles, not the t distribution: this is a scale reference
        rather than a test, and it is anti-conservative at very small n (about
        0.91 of the t-based MDE at n=12). Reported n makes that visible.

        Power is the chance that a run this size detects a true effect of the
        MDE. The report also carries the pair's own sign-flip p-value
        (uncorrected), because a difference below the MDE can still be
        significant: it was detected, and the MDE says only that a replication
        at this size would detect an effect that small less often than
        ``power``. The report used to call such a pair "would have been
        missed", next to a `results significance` table calling it significant.
        """
        import math

        ids, a, b, dropped_a, dropped_b = self._paired_scores(benchmark, config_a, config_b, metric)
        n = len(ids)
        if n < 2:
            raise ValueError(
                f"Only {n} sample(s) scored {metric!r} in both {config_a!r} and "
                f"{config_b!r} on {benchmark!r}; a spread cannot be estimated "
                "from fewer than two."
            )

        # candidate minus baseline, so an improvement reads positive.
        #
        # This was `a - b` -- baseline minus candidate -- under a report header
        # saying "candidate vs baseline", so a +0.10 improvement printed as
        # -0.1001. `underpowered` uses `abs()`, so the flag stayed correct and
        # the report looked internally consistent while its headline number was
        # backwards. `results interaction`, added in the same release, already
        # renders candidate-minus-baseline; the two commands disagreed on sign.
        diffs = [y - x for x, y in zip(a, b)]
        mean_diff = sum(diffs) / n
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        sd = math.sqrt(variance)

        mde = (_normal_quantile(1 - alpha / 2) + _normal_quantile(power)) * sd / math.sqrt(n)

        # A run with zero spread detects any non-zero difference, which is true
        # and useless to say as "MDE = 0". Flagged rather than dressed up.
        degenerate = sd == 0.0
        underpowered = (not degenerate) and abs(mean_diff) < mde

        p_value = self.significance_test(config_a, config_b, metric, benchmark, alpha=alpha)[
            "p_value"
        ]
        significant = p_value < alpha
        detects = f"a run this size detects a true difference of {mde:.4f} {power:.0%} of the time"
        if degenerate:
            interpretation = (
                "every paired difference was identical, so any non-zero effect "
                "would have been detected"
            )
        elif significant:
            interpretation = (
                f"the difference ({mean_diff:+.4f}) is significant on its own "
                f"(p={p_value:.4f}, uncorrected; `results significance` corrects "
                f"across every pair). {detects[0].upper()}{detects[1:]}"
                + (
                    ", and a smaller one like this less often, so a replication at "
                    "this size could miss it"
                    if underpowered
                    else ""
                )
            )
        elif underpowered:
            interpretation = (
                f"not significant (p={p_value:.4f}), and the observed difference "
                f"({mean_diff:+.4f}) is below {mde:.4f}: {detects}, and a smaller "
                "one less often, so a real effect this size could easily go "
                "undetected -- inconclusive, not evidence of no difference"
            )
        else:
            interpretation = (
                f"not significant (p={p_value:.4f}); {detects}, and a larger one more often"
            )

        return {
            "benchmark": benchmark,
            "config_a": config_a,
            "config_b": config_b,
            "metric": metric,
            "observed_difference": mean_diff,
            "sd_of_differences": sd,
            "n_paired": n,
            "n_dropped_a": dropped_a,
            "n_dropped_b": dropped_b,
            "alpha": alpha,
            "power": power,
            "minimum_detectable_effect": mde,
            "underpowered": underpowered,
            "zero_variance": degenerate,
            "n_needed_for_observed_effect": (
                _samples_for_effect(abs(mean_diff), sd, alpha, power) if mean_diff else None
            ),
            "p_value": p_value,
            "significant": significant,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Interaction between two techniques
    # ------------------------------------------------------------------

    def aligned_scores(
        self,
        benchmark: str,
        config_names: List[str],
        metric: str,
    ) -> Tuple[List[Any], Dict[str, List[float]]]:
        """Per-sample scores for N configs, aligned on one shared sample set.

        The intersection is taken **once, across all N**. Aligning pairwise and
        combining afterwards gives each pair a different sample set, and the
        double difference below subtracts quantities measured on different
        queries -- which produces a number, and a confident one.

        Returns ``(sample_ids, {config_name: scores})`` with every list the
        same length and position *i* the same query throughout.
        """
        indexes: Dict[str, Dict[Any, float]] = {}
        for name in config_names:
            tasks = self._by_bench_config.get((benchmark, name), [])
            if not tasks:
                raise ValueError(f"No tasks for ({benchmark}, {name})")
            index = _mean_by_sample(tasks, metric)
            if not index:
                raise ValueError(f"No sample scored {metric!r} for ({benchmark}, {name})")
            indexes[name] = index

        shared_set = set(indexes[config_names[0]])
        for name in config_names[1:]:
            shared_set &= set(indexes[name])
        shared = sorted(shared_set, key=str)

        return shared, {name: [indexes[name][sid] for sid in shared] for name in config_names}

    def interaction_test(
        self,
        benchmark: str,
        metric: str,
        *,
        neither: str,
        only_a: str,
        only_b: str,
        both: str,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Do two techniques compose, or do they overlap?

        The question every RAG paper raises and none answers: technique A is
        reported against a baseline, technique B is reported against a
        baseline, and whether A+B is worth A alone is left to the reader. It
        cannot be answered from two papers, because the interaction is not a
        function of the two main effects.

        Estimates the double difference::

            delta_A  = only_a - neither
            delta_B  = only_b - neither
            delta_AB = both   - neither
            interaction = delta_AB - (delta_A + delta_B)

        Positive means the techniques compound -- together they beat the sum of
        their parts. Negative means they overlap: both are fixing the same
        queries, and the second one buys less than its solo number promised.
        Around zero means they are additive and independent.

        All four cells are aligned on one shared sample set, and the test works
        on each query's own double difference, so the pairing that makes the
        estimate meaningful is kept. The p-value is a sign-flip test of those
        per-query interactions and the interval a t interval, for the reason
        given in :meth:`significance_test`: the bootstrap they replaced rejected
        8.3% of true nulls at n=20.

        ``n_bootstrap`` is the number of random sign patterns drawn when the
        sample is too large to enumerate; the name is kept for compatibility.
        """
        import numpy as np

        cells = [neither, only_a, only_b, both]
        if len(set(cells)) != 4:
            raise ValueError(
                f"The four cells must be four distinct configs, got {cells}. "
                "A 2x2 with a repeated cell reports an interaction of zero by "
                "construction."
            )

        ids, scores = self.aligned_scores(benchmark, cells, metric)
        n = len(ids)
        if n < 2:
            raise ValueError(
                f"Only {n} sample(s) scored {metric!r} in all four cells of the "
                f"2x2 on {benchmark!r}. An interaction cannot be estimated from "
                "fewer than two paired observations."
            )

        arr = {name: np.asarray(scores[name], dtype=float) for name in cells}
        d_a = arr[only_a] - arr[neither]
        d_b = arr[only_b] - arr[neither]
        d_ab = arr[both] - arr[neither]
        inter = d_ab - d_a - d_b

        observed = float(np.mean(inter))
        p_value, method = _sign_flip_p_value(inter, n_resamples=n_bootstrap, seed=seed)
        lo, hi = _t_interval(inter, alpha)

        main_a = float(np.mean(d_a))
        main_b = float(np.mean(d_b))
        combined = float(np.mean(d_ab))
        expected = main_a + main_b

        # How much of the additive prediction the combination actually
        # delivered. Undefined when the two techniques predict no change at
        # all, and reported as None rather than as 1.0 -- "they composed
        # perfectly" is not what "neither did anything" means.
        efficiency = (combined / expected) if abs(expected) > 1e-12 else None

        if p_value >= alpha:
            verdict = "additive (no detectable interaction)"
        elif observed > 0:
            verdict = "synergistic (together they beat the sum of their parts)"
        else:
            verdict = "redundant (they overlap; the second buys less than it does alone)"

        return {
            "benchmark": benchmark,
            "metric": metric,
            "cells": {
                "neither": neither,
                "only_a": only_a,
                "only_b": only_b,
                "both": both,
            },
            "means": {name: float(np.mean(arr[name])) for name in cells},
            "main_effect_a": main_a,
            "main_effect_b": main_b,
            "combined_effect": combined,
            "expected_if_additive": expected,
            "interaction": observed,
            "composition_efficiency": efficiency,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "p_value": p_value,
            "significant": bool(p_value < alpha),
            "effect_size": self._cohens_dz(inter),
            # The four-way intersection, not any pair's. Reporting a pairwise n
            # here would make the interval look better than the design supports.
            "n_paired": n,
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "ci_method": "t",
            "test": f"paired sign-flip ({method.replace('_', ' ')})",
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(self, metric: str = "ndcg@10") -> str:
        """A text leaderboard, ranked within each benchmark.

        Configs are ranked only against configs that scored the same queries
        on the same benchmark. This used to pool every task of a config across
        benchmarks and drop failed ones, so a config that skipped the hard
        benchmark outranked one that ran it, the printed ``± std (n=2)`` was
        the spread between benchmarks, and an arm that scored 30 of 100
        queries ranked above one that scored all 100.

        A config that scored a different query set, failed some runs, failed
        every run or never ran the benchmark is still listed, unranked or
        annotated, so its absence cannot pass for a result.
        """
        lines = [
            f"Leaderboard: {metric}" + (" (lower is better)" if lower_is_better(metric) else ""),
            "=" * 50,
            "Ranked within each benchmark, over configs that scored the same queries.",
        ]
        for bench in self.benchmarks:
            entries = []
            for config in self.configs:
                tasks = self._by_bench_config.get((bench, config), [])
                ok = [t for t in tasks if not t.error and metric in t.metrics]
                failed = sum(1 for t in tasks if t.error)
                queries = set(_mean_by_sample(ok, metric))
                entries.append((config, tasks, ok, failed, queries))

            # The query set every complete config scored: the union, so a
            # config missing any query is the one set apart.
            universe = set().union(*(e[4] for e in entries))
            width = max(len(e[0]) for e in entries)
            ranked, unranked = [], []
            for config, tasks, ok, failed, queries in entries:
                failures = f"; {failed} of {len(tasks)} runs failed" if failed and ok else ""
                if not tasks:
                    unranked.append(f"  -  {config:{width}s}  not run")
                elif not ok:
                    reason = "failed" if failed else f"no {metric} recorded"
                    unranked.append(f"  -  {config:{width}s}  {reason} ({len(tasks)} runs)")
                else:
                    agg = _aggregate([t.metrics[metric] for t in ok])
                    runs = f"{agg.n_runs} run" + ("s" if agg.n_runs != 1 else "")
                    spread = f" ± {agg.std:.4f}" if agg.n_runs > 1 else ""
                    n_queries = f", {len(queries)} queries" if queries else ""
                    cell = f"{agg.mean:.4f}{spread} ({runs}{n_queries}){failures}"
                    if queries and queries != universe:
                        unranked.append(
                            f"  -  {config:{width}s}  {cell}; scored {len(queries)} of "
                            f"{len(universe)} queries, not ranked against the others"
                        )
                    else:
                        ranked.append((agg.mean, config, cell))

            lines += ["", bench]
            # Ranking latency or noise sensitivity descending put the worst
            # config first.
            ranked.sort(key=lambda r: r[0], reverse=not lower_is_better(metric))
            for rank, (_, config, cell) in enumerate(ranked, 1):
                lines.append(f"  {rank}. {config:{width}s}  {cell}")
            lines += unranked
            if not ranked and len(entries) > 1:
                lines.append(
                    "  No config scored every query. `lasagna results significance` "
                    "compares two configs on the queries both scored."
                )
        notes = self._no_context_notes()
        if notes:
            lines += [""] + notes
        return "\n".join(lines)

    def _no_context_notes(self) -> List[str]:
        """One line per arm whose generator read no passages (closed_book, a
        ``no_rag`` arm), saying why it has no faithfulness to rank by."""
        arms = sorted(
            {
                r.config_name
                for r in self._results
                if not r.error and (r.system or {}).get("generator_context") == "none"
            }
        )
        return [
            f"{arm}: no context: faithfulness and context metrics not reported; "
            f"compare exact_match, token_f1, factual_correctness"
            for arm in arms
        ]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self, metric: str = "ndcg@10") -> str:
        """Generate a GitHub-friendly Markdown table."""
        agg = self.aggregate_by_benchmark(metric)
        if not agg:
            return "_No results_"

        configs = self.configs
        lines = [f"## {self.experiment_name or 'Benchmark Results'}: {metric}\n"]

        # Header
        header = "| Benchmark | " + " | ".join(configs) + " |"
        sep = "|---|" + "|".join(["---"] * len(configs)) + "|"
        lines.extend([header, sep])

        # Rows
        for bench in self.benchmarks:
            cells = []
            bench_data = agg.get(bench, {})
            best_mean = max((m.mean for m in bench_data.values()), default=0)
            for config in configs:
                m = bench_data.get(config)
                if m:
                    # A spread needs two runs: one read "0.9806 ± 0.0000".
                    val = f"{m.mean:.4f} ± {m.std:.4f}" if m.n_runs > 1 else f"{m.mean:.4f}"
                    if m.mean == best_mean and len(bench_data) > 1:
                        val = f"**{val}**"
                    cells.append(val)
                else:
                    cells.append("-")
            lines.append(f"| {bench} | " + " | ".join(cells) + " |")
        notes = self._no_context_notes()
        if notes:
            lines.append("")
            lines.extend(f"_{note}._" for note in notes)

        # Latency / cost summary (if any task has latency data)
        has_latency = any(r.avg_latency_ms > 0 for r in self._results if not r.error)
        if has_latency:
            lines.append("")
            lines.append("### Latency & Token Summary")
            lines.append("")
            lines.append(
                "| Config | Avg Latency | P95 Latency | System LLM Calls | "
                "System Tokens | Eval Tokens |"
            )
            lines.append("|--------|-------------|-------------|------------------|-----|-----|")
            # Aggregate per config
            config_stats: Dict[str, List] = {}
            for r in self._results:
                if r.error or r.avg_latency_ms <= 0:
                    continue
                config_stats.setdefault(r.config_name, []).append(r)
            for config in configs:
                tasks = config_stats.get(config, [])
                if not tasks:
                    continue
                avg_lat = sum(t.avg_latency_ms for t in tasks) / len(tasks)
                avg_p95 = sum(t.p95_latency_ms for t in tasks) / len(tasks)
                # Tokens, split by who spent them. `system` is the pipeline
                # under test; `eval` is the judge, and folding the two together
                # makes an expensive judge read as an expensive pipeline.
                sys_llm = sum(t.system_llm_calls for t in tasks)
                sys_tok = sum(t.system_input_tokens + t.system_output_tokens for t in tasks)
                eval_tok = sum(t.eval_input_tokens + t.eval_output_tokens for t in tasks)
                # "0" and "nobody was counting" must not print the same.
                measured = any(t.tokens_measured for t in tasks)
                sys_cell = f"{sys_tok:,}" if measured else "not measured"
                eval_cell = f"{eval_tok:,}" if measured else "not measured"
                lines.append(
                    f"| {config} | {avg_lat:.0f}ms | {avg_p95:.0f}ms | "
                    f"{sys_llm} | {sys_cell} | {eval_cell} |"
                )

        # Failed tasks
        if self.failed_tasks:
            lines.append(f"\n_⚠ {len(self.failed_tasks)} task(s) failed_")

        return "\n".join(lines)

    def to_latex(
        self,
        metric: str = "ndcg@10",
        bold_best: bool = True,
        include_std: bool = True,
        significance_set: Optional[set] = None,
    ) -> str:
        """Generate a LaTeX table for papers.

        Args:
            metric: Metric to display.
            bold_best: Bold the best result per benchmark.
            include_std: Include standard deviation.
            significance_set: Optional set of ``(benchmark, config)`` tuples
                where the config is significantly different from the best.
                When provided, a dagger marker is appended to those cells.
        """
        agg = self.aggregate_by_benchmark(metric)
        if not agg:
            return "% No results"

        configs = self.configs
        has_sig = significance_set is not None and len(significance_set) > 0
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{self.experiment_name or 'Benchmark Results'}: {_escape_latex(metric)}}}",
            "\\begin{tabular}{l" + "c" * len(configs) + "}",
            "\\toprule",
            "Benchmark & " + " & ".join(_escape_latex(c) for c in configs) + " \\\\",
            "\\midrule",
        ]

        for bench in self.benchmarks:
            bench_data = agg.get(bench, {})
            best_mean = max((m.mean for m in bench_data.values()), default=0)
            cells = []
            for config in configs:
                m = bench_data.get(config)
                if m:
                    val = f"{m.mean:.4f}"
                    if include_std and m.n_runs > 1:
                        val += f" \\pm {m.std:.4f}"
                    if bold_best and m.mean == best_mean and len(bench_data) > 1:
                        val = f"\\textbf{{{val}}}"
                    if include_std:
                        val = f"${val}$"
                    # Append dagger if significantly different from best
                    if has_sig and (bench, config) in significance_set:
                        val += "$^{\\dagger}$"
                    cells.append(val)
                else:
                    cells.append("--")  # LaTeX: an en dash, the empty-cell mark
            lines.append(f"{_escape_latex(bench)} & " + " & ".join(cells) + " \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
        if has_sig:
            lines.append("\\vspace{2pt}")
            lines.append(
                "\\footnotesize{$^{\\dagger}$ Significantly different from best "
                "($p < 0.05$, paired sign-flip test, Holm-corrected).}"
            )
        lines.append("\\end{table}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-query view
    # ------------------------------------------------------------------

    def per_query(self, benchmark: str = "") -> List[Dict[str, Any]]:
        """One row per (benchmark, config, sample), retrieval and generation
        side by side.

        The two eval paths write two different row shapes into the same
        ``per_sample_scores`` list -- retrieval rows carrying ``ndcg@10`` and
        ``recall@10``, generation rows carrying ``faithfulness`` and
        ``exact_match`` -- and nothing ever joined them, even though both carry
        ``_sample_id``. So the answer to "did this config lose at retrieval or
        at generation?" was in the file and could not be read out of it.

        That question is most of error analysis. A config that loses overall
        because it retrieves the right document and then generates badly needs
        the opposite fix from one that generates well over the wrong document,
        and an aggregate score cannot tell them apart.

        Aggregated in two steps, because a query's rows and a query's runs need
        opposite treatment. Within one run, an end-to-end sample contributes a
        retrieval row *and* a generation row; costs like latency and tokens are
        summed across them, because they are two stages of answering the same
        query. Across runs, everything is averaged, the same way the
        significance tests treat repeated runs.

        Doing it in one pass got both wrong: latency averaged the retrieval
        row's 40ms against the generation row's 900ms and reported 470ms for a
        query that took 940ms, and ``n_runs`` counted rows, so a single run over
        an e2e benchmark reported ``n_runs=2``.
        """
        #: Per-query costs, summed within a run rather than averaged.
        summed = {
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "llm_calls",
            "eval_input_tokens",
            "eval_output_tokens",
            "eval_llm_calls",
        }

        # (benchmark, config, sample) -> run_index -> field -> value
        per_run: Dict[Any, Dict[int, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        counts: Dict[Any, Dict[int, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        for result in self._results:
            if result.error or (benchmark and result.benchmark_name != benchmark):
                continue
            for i, sample in enumerate(result.per_sample_scores or []):
                sample_id = sample.get("_sample_id", i)
                key = (result.benchmark_name, result.config_name, sample_id)
                bucket = per_run[key][result.run_index]

                for field_name, value in sample.items():
                    if field_name == "_sample_id":
                        continue
                    if field_name in summed and isinstance(value, (int, float)):
                        bucket[field_name] = bucket.get(field_name, 0.0) + float(value)
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        # A metric can legitimately appear on both row shapes;
                        # average within the run rather than double-counting.
                        counts[key][result.run_index][field_name] += 1
                        seen = counts[key][result.run_index][field_name]
                        prior = bucket.get(field_name, 0.0)
                        bucket[field_name] = prior + (float(value) - prior) / seen
                    else:
                        # Trajectories, retrieved ids, source benchmark. Not
                        # measurements, so averaging is undefined; last wins.
                        bucket[field_name] = value

        rows: List[Dict[str, Any]] = []
        for (bench, config, sample_id), runs in per_run.items():
            row: Dict[str, Any] = {
                "benchmark": bench,
                "config": config,
                "sample_id": sample_id,
                "n_runs": len(runs),
            }
            numeric: Dict[str, List[float]] = defaultdict(list)
            for bucket in runs.values():
                for field_name, value in bucket.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric[field_name].append(float(value))
                    else:
                        row[field_name] = value
            for field_name, values in numeric.items():
                row[field_name] = sum(values) / len(values)
            rows.append(row)

        return sorted(rows, key=lambda r: (r["benchmark"], r["config"], str(r["sample_id"])))

    def per_query_dataframe(self, benchmark: str = "") -> Any:
        """:meth:`per_query` as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install lasagna-rag[eval]")

        return pd.DataFrame(self.per_query(benchmark))

    def to_dataframe(self) -> Any:
        """Convert to a pandas DataFrame with latency/cost columns."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install lasagna-rag[eval]")

        rows = []
        for r in self._results:
            row = {
                "benchmark": r.benchmark_name,
                "config": r.config_name,
                "run": r.run_index,
                "duration_s": r.duration_seconds,
                "n_samples": r.n_samples,
                "avg_latency_ms": r.avg_latency_ms,
                "p50_latency_ms": r.p50_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "system_llm_calls": r.system_llm_calls,
                "system_input_tokens": r.system_input_tokens,
                "system_output_tokens": r.system_output_tokens,
                "eval_llm_calls": r.eval_llm_calls,
                "eval_input_tokens": r.eval_input_tokens,
                "eval_output_tokens": r.eval_output_tokens,
                "total_llm_calls": r.total_llm_calls,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "tokens_measured": r.tokens_measured,
                "error": r.error,
            }
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Persist results to JSON.

        The run's provenance lives in ``experiment_config``: the metric
        definitions and package version it was computed under, the
        environment (``core.provenance.capture``), the judge, the settings the
        environment overrode (redacted) and the Hugging Face commit of each
        model it ran. The two versions are repeated at the top for a reader.
        They are the run's own: a file loaded without them is saved without
        them, never stamped with this build's.
        """
        meta = self.experiment_config or {}
        data = {
            "experiment_name": self.experiment_name,
            "metrics_version": meta.get("metrics_version"),
            "package_version": meta.get("package_version"),
            "experiment_config": self.experiment_config,
            "results": [r.to_dict() for r in self._results],
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {out}")

    @classmethod
    def load(cls, path: str) -> "BenchmarkResults":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        task_results = [TaskResult.from_dict(r) for r in data.get("results", [])]
        meta = dict(data.get("experiment_config") or {})
        for name in ("metrics_version", "package_version"):
            if meta.get(name) is None and data.get(name) is not None:
                meta[name] = data[name]
        return cls(
            task_results=task_results,
            experiment_name=data.get("experiment_name", ""),
            experiment_config=meta,
        )

    def compare(self, other: "BenchmarkResults", metric: str = "ndcg@10") -> str:
        """Compare two experiment results side by side.

        Reports whether the two are comparable at all before showing numbers.
        Previously this unioned the benchmark names and printed both tables
        whatever they measured — two runs on different embedding models, splits
        or sample counts looked exactly like a like-for-like comparison.

        A mismatch is shown as a field-level diff rather than a refusal: the
        first thing anyone does is compare their run against a published one
        that differs somewhere, and "incomparable" alone does not say which
        knob to turn.
        """
        from .bundle import compare_keys, comparability_key, held_fixed_from

        lines = [f"Comparison: {self.experiment_name} vs {other.experiment_name}\n"]
        lines.append(f"Metric: {metric}\n")

        # Held-fixed configuration, per side, from the first result of each,
        # with the stages the side's arms vary left out as the bundle does.
        def _held(results: "BenchmarkResults") -> Optional[Dict[str, Any]]:
            if not results._results:
                return None
            rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results._results]
            across = list(
                {row.get("config_name"): row.get("config_overrides") or {} for row in rows}.values()
            )
            return held_fixed_from(rows[0], results.experiment_config or {}, across)

        held_a, held_b = _held(self), _held(other)
        if held_a and held_b:
            key_a, key_b = comparability_key(held_a), comparability_key(held_b)
            if key_a == key_b:
                lines.append(f"Comparable: yes (key {key_a})\n")
            else:
                diffs = compare_keys(held_a, held_b)
                lines.append(
                    "Comparable: NO. These runs did not hold the same things fixed.\n"
                    "Differences that make the numbers below incommensurable:"
                )
                lines.extend(f"    {d}" for d in diffs)
                lines.append("")

        all_benchmarks = sorted(set(self.benchmarks) | set(other.benchmarks))
        for bench in all_benchmarks:
            lines.append(f"\n  {bench}:")
            for label, results in [("  A", self), ("  B", other)]:
                agg = results.aggregate_by_benchmark(metric).get(bench, {})
                for config, m in agg.items():
                    lines.append(f"    {label}/{config}: {m}")

        return "\n".join(lines)

    def to_bundle(self, submitter: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Assemble a self-contained, publishable record of this run."""
        from .bundle import build_bundle

        return build_bundle(self, submitter=submitter)


def _aggregate(values: List[float]) -> AggregatedMetrics:
    """Compute mean +/- std."""
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = variance**0.5
    else:
        std = 0.0
    return AggregatedMetrics(mean=mean, std=std, n_runs=n, values=values)


#: What each character LaTeX treats specially becomes in a table cell.
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    """LaTeX escaping for table cells and captions.

    Arm names are free text. Only ``_ & % # $`` were escaped, so a name with a
    brace, a backslash, ``~`` or ``^`` broke the table it was pasted into.
    """
    return "".join(_LATEX_SPECIAL.get(char, char) for char in str(text))


# ---------------------------------------------------------------------------
# The verdict on a significance table, shared by the CLI box and the HTML view
# ---------------------------------------------------------------------------


def format_p(p: float) -> str:
    """Four decimals, or scientific notation below 0.0001 rather than 0.0000.

    An exact sign-flip p over 25 queries can be 6e-8, and "0.0000" reads as a
    probability of zero.
    """
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def format_effect_size(effect: Optional[float]) -> str:
    """Render Cohen's ``d_z``, including the case where it has no finite value.

    An infinite ``d_z`` means every query moved by the same amount -- the most
    consistent effect there is. Printing that as a number is the problem this
    exists to avoid: ``0.00`` reads as "no effect", and the raw float reads as
    a measurement of twenty quadrillion standard deviations.
    """
    import math

    if effect is None:
        return "-"
    if math.isinf(effect):
        return "+∞" if effect > 0 else "-∞"
    if math.isnan(effect):
        return "-"
    return f"{effect:+.2f}"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


@dataclass(frozen=True)
class VerdictLine:
    """One line of a verdict, as ``(role, text)`` spans.

    ``role`` is ``"text"``, ``"winner"`` (the arm that won) or ``"command"``
    (a command to run), so each renderer emphasizes them its own way without
    parsing markup out of the text.
    """

    spans: Tuple[Tuple[str, str], ...] = ()

    @property
    def text(self) -> str:
        return "".join(text for _, text in self.spans)


@dataclass(frozen=True)
class Verdict:
    """What a Holm-corrected significance table supports, as plain data.

    ``style`` is ``"significant"`` when at least one pair survives the
    correction and ``"none"`` otherwise.
    """

    style: str
    title: str
    lines: Tuple[VerdictLine, ...]


#: How many surviving pairs the verdict names before pointing at the table.
_VERDICT_SHOWN = 5


def verdict_lines(rows: List[Dict[str, Any]], metric: str, benchmark: str, alpha: float) -> Verdict:
    """The answer a :meth:`BenchmarkResults.pairwise_significance` table supports.

    Names each pair that survives Holm correction, which side won (the lower
    side for a lower-is-better metric) and by how much. When nothing survives,
    says so and points at ``lasagna results power``, because "not significant"
    has two readings (no difference, or too few queries to see one) that call
    for opposite next steps. ``lasagna results significance`` prints this as a
    box and ``lasagna results view`` as a panel, so they cannot disagree.
    """
    winners = sorted(
        (r for r in rows if r.get("significant")),
        key=lambda r: r.get("p_adjusted", r["p_value"]),
    )
    family = _plural(len(rows), "pair")
    if not winners:
        return Verdict(
            style="none",
            title="no significant difference",
            lines=(
                VerdictLine(
                    (
                        (
                            "text",
                            f"No pair survives Holm correction at family-wise alpha={alpha} "
                            f"(0 of {family}).",
                        ),
                    )
                ),
                VerdictLine(
                    (
                        (
                            "text",
                            "Either the arms perform alike or this run has too few queries "
                            "to tell; ",
                        ),
                        ("command", "lasagna results power"),
                        ("text", " says which."),
                    )
                ),
            ),
        )

    lower = lower_is_better(metric)
    lines: List[VerdictLine] = []
    for row in winners[:_VERDICT_SHOWN]:
        a, b, diff = str(row["config_a"]), str(row["config_b"]), row["difference"]
        a_wins = diff < 0 if lower else diff > 0
        win, lose = (a, b) if a_wins else (b, a)
        effect = row.get("effect_size")
        # Names, metric and benchmark are spans of their own: free text that
        # each renderer escapes, kept apart from the fixed words around it.
        lines.append(VerdictLine((("winner", win), ("text", " beats "), ("text", lose))))
        lines.append(
            VerdictLine(
                (
                    ("text", "  "),
                    ("text", metric),
                    ("text", f" {'-' if lower else '+'}{abs(diff):.4f} on "),
                    ("text", benchmark),
                    (
                        "text",
                        f", d_z {format_effect_size(abs(effect) if effect is not None else None)}"
                        f", p (Holm) {format_p(row.get('p_adjusted', row['p_value']))}, "
                        f"{row.get('n_paired', '?')} queries",
                    ),
                )
            )
        )
    if len(winners) > _VERDICT_SHOWN:
        more = len(winners) - _VERDICT_SHOWN
        lines.append(VerdictLine((("text", f"  and {more} more in the table above"),)))
    lines.append(VerdictLine())
    survive = "survives" if len(winners) == 1 else "survive"
    lines.append(
        VerdictLine(
            (
                (
                    "text",
                    f"{len(winners)} of {family} {survive} correction at "
                    f"family-wise alpha={alpha}.",
                ),
            )
        )
    )
    return Verdict(
        style="significant", title="significant after Holm correction", lines=tuple(lines)
    )
