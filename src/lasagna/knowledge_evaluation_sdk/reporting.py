"""Publication-ready report generation.

Generates Markdown reports, LaTeX tables, and optional plots for
benchmark experiment results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .results import BenchmarkResults


def _format_effect_size(effect: Optional[float]) -> str:
    """Render Cohen's ``d_z`` for a report, including its non-finite case.

    ``d_z`` is infinite when every query moved by the same amount -- a real
    result, and the strongest one there is. A bare ``:.3f`` prints that as
    ``inf`` in the middle of a Markdown table.
    """
    import math

    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return "—"
    if math.isinf(effect):
        return "+∞" if effect > 0 else "−∞"
    return f"{effect:.3f}"


class ReportGenerator:
    """Generate publication-ready reports from benchmark results.

    Example::

        gen = ReportGenerator(results)
        gen.save_all("./output")
        # Creates: report.md, table.tex, figures/

        print(gen.markdown_report())
        print(gen.latex_table())
    """

    def __init__(
        self,
        results: BenchmarkResults,
        primary_metric: str = "ndcg@10",
    ) -> None:
        self._results = results
        self._primary_metric = primary_metric

    def markdown_report(self, metrics: Optional[List[str]] = None) -> str:
        """Generate a full GitHub-friendly Markdown report.

        Args:
            metrics: List of metrics to include. Defaults to all available.

        Returns:
            Complete Markdown report string.
        """
        r = self._results
        metric_names = metrics or r.get_metric_names() or [self._primary_metric]
        lines = []

        # Header
        lines.append(f"# {r.experiment_name or 'Benchmark Results'}")
        lines.append("")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Benchmarks**: {', '.join(r.benchmarks)}")
        lines.append(f"**Configs**: {', '.join(r.configs)}")
        lines.append(f"**Total tasks**: {r.total_tasks}")
        if r.failed_tasks:
            lines.append(f"**Failed**: {len(r.failed_tasks)}")
        lines.append("")

        # Summary table for each metric
        for metric in metric_names:
            lines.append(r.to_markdown(metric))
            lines.append("")

        # Leaderboard
        lines.append("## Leaderboard")
        lines.append("")
        lines.append("```")
        lines.append(r.leaderboard(self._primary_metric))
        lines.append("```")
        lines.append("")

        # Significance tests (if >1 config)
        if len(r.configs) > 1:
            lines.append("## Statistical Analysis")
            lines.append("")
            for bench in r.benchmarks:
                try:
                    pairwise = r.pairwise_significance(
                        self._primary_metric,
                        bench,
                    )
                    if pairwise:
                        lines.append(f"### {bench}")
                        lines.append("")
                        lines.append(
                            "| Config A | Config B | p-value | Significant | Effect Size |"
                        )
                        lines.append(
                            "|----------|----------|---------|-------------|-------------|"
                        )
                        for pw in pairwise:
                            sig_marker = "Yes" if pw["significant"] else "No"
                            lines.append(
                                f"| {pw['config_a']} | {pw['config_b']} "
                                f"| {pw['p_value']:.4f} | {sig_marker} "
                                f"| {_format_effect_size(pw['effect_size'])} |"
                            )
                        lines.append("")
                except (ValueError, ImportError):
                    pass

        # Tradeoff analysis (if latency data is available)
        has_latency = any(t.avg_latency_ms > 0 for t in r._results if not t.error)
        if has_latency and len(r.configs) > 1:
            lines.append("## Tradeoff Analysis")
            lines.append("")
            try:
                from .private_eval.tradeoff_analyzer import TradeoffAnalyzer

                analyzer = TradeoffAnalyzer()
                # Build config_results aggregated across all tasks per config.
                # Track sums and counts for a correct running mean.
                config_data: Dict[str, Dict[str, Any]] = {}
                _metric_counts: Dict[str, Dict[str, int]] = {}
                for task in r._results:
                    if task.error:
                        continue
                    if task.config_name not in config_data:
                        config_data[task.config_name] = {
                            "metrics": dict(task.metrics),
                            "per_sample_scores": list(task.per_sample_scores),
                        }
                        _metric_counts[task.config_name] = {k: 1 for k in task.metrics}
                    else:
                        entry = config_data[task.config_name]
                        counts = _metric_counts[task.config_name]
                        entry["per_sample_scores"].extend(task.per_sample_scores)
                        for k, v in task.metrics.items():
                            if k in entry["metrics"]:
                                counts[k] += 1
                                # Running sum; divide at the end
                                entry["metrics"][k] += v
                            else:
                                entry["metrics"][k] = v
                                counts[k] = 1
                # Divide sums by counts to get true means
                for cfg, counts in _metric_counts.items():
                    for k, n in counts.items():
                        if n > 1:
                            config_data[cfg]["metrics"][k] /= n
                efficiencies = analyzer.efficiency_table(
                    config_data,
                    quality_metric=self._primary_metric,
                )
                recommendations = analyzer.tier_recommendations(efficiencies)
                lines.append(analyzer.to_markdown(efficiencies, recommendations))
                lines.append("")
            except (ImportError, Exception):
                pass

        # Failed tasks
        if r.failed_tasks:
            lines.append("## Failed Tasks")
            lines.append("")
            for task in r.failed_tasks:
                lines.append(f"- **{task.task_key}**: {task.error}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by Lasagna Evaluation SDK*")

        return "\n".join(lines)

    def latex_table(
        self,
        metric: Optional[str] = None,
        bold_best: bool = True,
        include_std: bool = True,
        significance_markers: bool = True,
    ) -> str:
        """Generate a publication-ready LaTeX table.

        Args:
            metric: Metric to display. Defaults to primary metric.
            bold_best: Bold the best result per benchmark.
            include_std: Include standard deviation.
            significance_markers: Add dagger (\\dagger) for p<0.05
                vs the best config on each benchmark.

        Returns:
            LaTeX table string.
        """
        m = metric or self._primary_metric

        if not significance_markers:
            return self._results.to_latex(m, bold_best=bold_best, include_std=include_std)

        # Build a set of (benchmark, config) pairs that are significantly
        # different from the best config on that benchmark.
        sig_pairs: set = set()  # {(benchmark, config)} where config is sig. diff from best
        r = self._results
        agg = r.aggregate_by_benchmark(m)

        for bench in r.benchmarks:
            bench_data = agg.get(bench, {})
            if len(bench_data) < 2:
                continue
            # Corrected, and against the best config only.
            #
            # This ran N-1 uncorrected `significance_test` calls -- one per
            # config, against whichever arm happened to score highest -- and put
            # a dagger next to each that cleared p<0.05. That is the textbook
            # way to manufacture a false positive, in a LaTeX table headed for a
            # paper, from a package whose stated purpose is preventing exactly
            # that. `pairwise_significance` already implements Holm; use it.
            #
            # Selecting the best arm and then testing against it is itself a
            # source of optimism, so the daggers mean "differs from the best
            # arm, family-wise corrected across the whole grid" -- not "is the
            # winner". `to_latex` renders them; the caption should say so.
            best_config = max(bench_data, key=lambda c: bench_data[c].mean)
            try:
                for row in r.pairwise_significance(m, bench, correction="holm"):
                    if not row.get("significant"):
                        continue
                    pair = {row["config_a"], row["config_b"]}
                    if best_config not in pair:
                        continue
                    other = (pair - {best_config}).pop()
                    sig_pairs.add((bench, other))
            except (ValueError, ImportError, KeyError):
                pass

        return r.to_latex(
            m,
            bold_best=bold_best,
            include_std=include_std,
            significance_set=sig_pairs,
        )

    def radar_plot(
        self,
        configs: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """Generate a radar (spider) plot comparing configs across metrics.

        Args:
            configs: Configs to compare. Defaults to all.
            metrics: Metrics to plot. Defaults to all.
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib + numpy required: pip install lasagna-rag[eval]")

        configs = configs or self._results.configs
        metrics = metrics or self._results.get_metric_names()

        if not metrics or not configs:
            logger.warning("No data for radar plot")
            return None

        # Compute mean score per config per metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for config in configs:
            values = []
            for metric in metrics:
                agg = self._results.aggregate_by_pipeline(metric)
                if config in agg:
                    values.append(agg[config].mean)
                else:
                    values.append(0.0)
            values += values[:1]
            ax.plot(angles, values, "o-", label=config, linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(self._results.experiment_name or "Benchmark Comparison")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Radar plot saved to {save_path}")

        return fig

    def heatmap(
        self,
        metric: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """Generate a heatmap of benchmark x config scores.

        Args:
            metric: Metric to display. Defaults to primary metric.
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib + numpy required: pip install lasagna-rag[eval]")

        m = metric or self._primary_metric
        agg = self._results.aggregate_by_benchmark(m)

        benchmarks = self._results.benchmarks
        configs = self._results.configs

        data = np.zeros((len(benchmarks), len(configs)))
        for i, bench in enumerate(benchmarks):
            for j, config in enumerate(configs):
                if bench in agg and config in agg[bench]:
                    data[i, j] = agg[bench][config].mean

        fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), max(4, len(benchmarks) * 0.6)))

        try:
            import seaborn as sns

            sns.heatmap(
                data,
                annot=True,
                fmt=".4f",
                xticklabels=configs,
                yticklabels=benchmarks,
                cmap="YlOrRd",
                ax=ax,
            )
        except ImportError:
            im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(configs)))
            ax.set_yticks(range(len(benchmarks)))
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_yticklabels(benchmarks)
            # Annotate
            for i in range(len(benchmarks)):
                for j in range(len(configs)):
                    ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im)

        ax.set_title(f"{self._results.experiment_name or 'Results'} — {m}")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Heatmap saved to {save_path}")

        return fig

    def save_all(self, output_dir: str) -> None:
        """Save all outputs: report, LaTeX, and plots.

        Creates:
            - ``report.md``
            - ``table.tex``
            - ``figures/radar.png``
            - ``figures/heatmap.png``
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Markdown report
        report = self.markdown_report()
        (out / "report.md").write_text(report)
        logger.info(f"Report saved to {out / 'report.md'}")

        # LaTeX table
        latex = self.latex_table()
        (out / "table.tex").write_text(latex)
        logger.info(f"LaTeX table saved to {out / 'table.tex'}")

        # Plots (optional — skip if matplotlib not available)
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            fig_dir = out / "figures"
            fig_dir.mkdir(exist_ok=True)

            try:
                self.radar_plot(save_path=str(fig_dir / "radar.png"))
            except Exception as exc:
                logger.debug(f"Radar plot skipped: {exc}")

            try:
                self.heatmap(save_path=str(fig_dir / "heatmap.png"))
            except Exception as exc:
                logger.debug(f"Heatmap skipped: {exc}")

        except ImportError:
            logger.debug("matplotlib not available — plots skipped")
