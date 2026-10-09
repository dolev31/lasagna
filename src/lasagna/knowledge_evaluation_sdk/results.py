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


@dataclass
class AggregatedMetrics:
    """Mean +/- std for a metric across multiple runs."""

    mean: float
    std: float
    n_runs: int
    values: List[float] = field(default_factory=list)

    def __str__(self) -> str:
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
    def _bootstrap_p_value(boot_arr, observed_diff) -> float:
        """Two-tailed bootstrap p-value, with a continuity correction.

        ``(count + 1) / (n + 1)`` rather than ``count / n``. Without it the
        estimate can be exactly 0.0, which is not a probability a finite
        resample can support -- it means "smaller than 1/n_bootstrap", and
        reporting it as zero invites "p < 0.001" claims the data cannot carry.

        The two implementations in this file disagreed on this: the ablation
        path corrected, this one did not, so the same comparison returned
        different p-values depending on which entry point was used.
        """
        import numpy as np

        exceed = int(np.sum(np.abs(boot_arr - observed_diff) >= abs(observed_diff)))
        return float((exceed + 1) / (len(boot_arr) + 1))

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
        """Paired bootstrap significance test between two configs on the SAME benchmark.

        Uses per-query scores from **every** run of each config, averaged per
        query before resampling. The ``benchmark`` parameter is REQUIRED —
        cross-benchmark significance is not valid (different query sets).

        Args:
            config_a: First pipeline config name.
            config_b: Second pipeline config name.
            metric: Metric name to compare.
            benchmark: Benchmark name (required for valid pairing).
            n_bootstrap: Number of bootstrap resamples.
            alpha: Significance level.
            seed: Random seed.

        Returns:
            Dict with ``p_value``, ``significant``, ``confidence_interval_95``,
            ``effect_size``, ``mean_a``, ``mean_b``, ``difference``, ``n_samples``.
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

        # Paired bootstrap
        rng = np.random.RandomState(seed)
        diffs = a - b
        observed_diff = float(np.mean(diffs))

        boot_diffs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            boot_diffs.append(float(np.mean(diffs[idx])))

        boot_diffs_arr = np.asarray(boot_diffs)
        # Two-tailed p-value: shift the bootstrap distribution onto the null
        # (zero difference), then count how often it exceeds the observed
        # effect in magnitude. Continuity-corrected -- see _bootstrap_p_value.
        p_value = self._bootstrap_p_value(boot_diffs_arr, observed_diff)
        ci_lower = float(np.percentile(boot_diffs_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_diffs_arr, 100 * (1 - alpha / 2)))

        effect_size = self._cohens_dz(diffs)

        return {
            "p_value": p_value,
            "significant": p_value < alpha,
            "confidence_interval_95": (ci_lower, ci_upper),
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
        """Domain-stratified paired bootstrap for macro-averaged metrics.

        Resamples within each domain, computes per-domain metric, then
        averages across domains. Prevents large domains (e.g., nfcorpus)
        from dominating the bootstrap distribution.

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

        # Domain-stratified bootstrap
        rng = np.random.RandomState(seed)
        observed_macro_a = np.mean([np.mean(domains_a[d]) for d in all_domains])
        observed_macro_b = np.mean([np.mean(domains_b[d]) for d in all_domains])
        observed_diff = float(observed_macro_a - observed_macro_b)

        boot_diffs = []
        for _ in range(n_bootstrap):
            macro_a, macro_b = [], []
            for d in all_domains:
                n_d = len(domains_a[d])
                idx = rng.randint(0, n_d, size=n_d)
                arr_a = np.asarray(domains_a[d])
                arr_b = np.asarray(domains_b[d])
                macro_a.append(float(np.mean(arr_a[idx])))
                macro_b.append(float(np.mean(arr_b[idx])))
            boot_diffs.append(np.mean(macro_a) - np.mean(macro_b))

        boot_arr = np.asarray(boot_diffs)
        p_value = self._bootstrap_p_value(boot_arr, observed_diff)
        ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

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
            "confidence_interval_95": (ci_lower, ci_upper),
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

        Args:
            metric: Metric to compare.
            benchmark: Benchmark (required).
            correction: ``"holm"`` (Holm-Bonferroni) or ``"none"``.
            alpha: Family-wise significance level.

        Returns:
            List of dicts with ``config_a``, ``config_b``, ``p_value``,
            ``significant``, ``corrected_alpha``.
        """
        configs = [c for c in self.configs if self._by_bench_config.get((benchmark, c))]
        pairs = [
            (configs[i], configs[j])
            for i in range(len(configs))
            for j in range(i + 1, len(configs))
        ]

        raw_results = []
        for ca, cb in pairs:
            try:
                sig = self.significance_test(ca, cb, metric, benchmark)
                raw_results.append(
                    {
                        "config_a": ca,
                        "config_b": cb,
                        "p_value": sig["p_value"],
                        "effect_size": sig["effect_size"],
                        "difference": sig["difference"],
                    }
                )
            except ValueError:
                continue

        if not raw_results:
            return []

        # Apply Holm-Bonferroni step-down correction:
        # Sort by p-value ascending. Test i has threshold alpha/(m-i).
        # Once a test fails to reject, all subsequent tests also fail.
        if correction == "holm":
            raw_results.sort(key=lambda x: x["p_value"])
            m = len(raw_results)
            rejected_so_far = True
            for i, r in enumerate(raw_results):
                corrected_alpha = alpha / (m - i)
                r["corrected_alpha"] = corrected_alpha
                if rejected_so_far and r["p_value"] <= corrected_alpha:
                    r["significant"] = True
                else:
                    rejected_so_far = False
                    r["significant"] = False
        else:
            for r in raw_results:
                r["corrected_alpha"] = alpha
                r["significant"] = r["p_value"] < alpha

        return raw_results

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

        Normal quantiles, not the t distribution: a bootstrap makes no
        parametric claim about the differences, so this is a scale reference
        rather than a test, and it is anti-conservative at very small n.
        Reported n makes that visible.
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
            "interpretation": (
                "every paired difference was identical, so any non-zero effect "
                "would have been detected"
                if degenerate
                else (
                    f"the observed difference ({mean_diff:+.4f}) is smaller than the "
                    f"smallest this run could detect ({mde:.4f}), so a real effect of "
                    f"this size would have been missed -- inconclusive, not null"
                    if underpowered
                    else f"the run could detect differences of {mde:.4f} or larger"
                )
            ),
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

        All four cells are aligned on one shared sample set, and the bootstrap
        resamples *samples* -- carrying all four cells of a sampled query
        together, so the pairing that makes the double difference meaningful
        survives the resampling.
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
        rng = np.random.default_rng(seed)
        # Resample query indices, not cells: every draw takes all four numbers
        # for a query, which is what keeps the design paired.
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot = inter[idx].mean(axis=1)

        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p_value = self._bootstrap_p_value(boot, observed)

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
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(self, metric: str = "ndcg@10") -> str:
        """Generate a simple text leaderboard ranked by mean metric."""
        agg = self.aggregate_by_pipeline(metric)
        ranked = sorted(agg.items(), key=lambda x: x[1].mean, reverse=True)

        lines = [f"Leaderboard — {metric}", "=" * 50]
        for rank, (config, metrics) in enumerate(ranked, 1):
            lines.append(f"  {rank}. {config:30s} {metrics}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self, metric: str = "ndcg@10") -> str:
        """Generate a GitHub-friendly Markdown table."""
        agg = self.aggregate_by_benchmark(metric)
        if not agg:
            return "_No results_"

        configs = self.configs
        lines = [f"## {self.experiment_name or 'Benchmark Results'} — {metric}\n"]

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
                    val = f"{m.mean:.4f} ± {m.std:.4f}"
                    if m.mean == best_mean and len(bench_data) > 1:
                        val = f"**{val}**"
                    cells.append(val)
                else:
                    cells.append("—")
            lines.append(f"| {bench} | " + " | ".join(cells) + " |")

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
            f"\\caption{{{self.experiment_name or 'Benchmark Results'} — {_escape_latex(metric)}}}",
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
                    if include_std:
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
                    cells.append("—")
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
                "\\footnotesize{$^{\\dagger}$ Significantly different from best ($p < 0.05$, paired bootstrap).}"
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
        """Persist results to JSON."""
        data = {
            "experiment_name": self.experiment_name,
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
        return cls(
            task_results=task_results,
            experiment_name=data.get("experiment_name", ""),
            experiment_config=data.get("experiment_config", {}),
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

        # Held-fixed configuration, per side, from the first result of each.
        def _held(results: "BenchmarkResults") -> Optional[Dict[str, Any]]:
            if not results._results:
                return None
            first = results._results[0]
            row = first.to_dict() if hasattr(first, "to_dict") else dict(first)
            return held_fixed_from(row, results.experiment_config or {})

        held_a, held_b = _held(self), _held(other)
        if held_a and held_b:
            key_a, key_b = comparability_key(held_a), comparability_key(held_b)
            if key_a == key_b:
                lines.append(f"Comparable: yes (key {key_a})\n")
            else:
                diffs = compare_keys(held_a, held_b)
                lines.append(
                    "Comparable: NO — these runs did not hold the same things fixed.\n"
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


def _escape_latex(text: str) -> str:
    """Minimal LaTeX escaping for table cells."""
    for char in ["_", "&", "%", "#", "$"]:
        text = text.replace(char, f"\\{char}")
    return text
