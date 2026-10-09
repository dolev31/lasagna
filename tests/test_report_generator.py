"""`ReportGenerator` is exported, documented, and had never been run.

`tests/test_public_api_surface.py` lists the name, which checks that the export
exists and nothing else. No test ever constructed one or called a method, so
the LaTeX tables and Markdown reports this package offers for writing up
results were unverified — including against changes to the very API they
consume.

That is not hypothetical. `pairwise_significance`'s ``effect_size`` can now be
``±inf`` (see `test_statistics_consistency.py` for why), and
`markdown_report` formatted it with a bare ``:.3f``, which prints ``inf`` in
the middle of a results table.

These tests run offline. The plotting methods need matplotlib and skip
without it; the text methods are the ones a paper depends on.
"""

from __future__ import annotations


import pytest

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults


def _results(per_query):
    task_results = []
    for config, scores in per_query.items():
        task_results.append(
            {
                "task_key": f"bench|{config}|cfg|corpus|live|run_0",
                "run_index": 0,
                "benchmark_name": "bench",
                "config_name": config,
                "metrics": {"ndcg@10": sum(scores) / len(scores)},
                "per_sample_scores": [
                    {"_sample_id": f"q{i}", "_source_benchmark": "beir/nfcorpus", "ndcg@10": s}
                    for i, s in enumerate(scores)
                ],
                "error": None,
            }
        )
    from lasagna.knowledge_evaluation_sdk.results import TaskResult

    return BenchmarkResults(task_results=[TaskResult.from_dict(r) for r in task_results])


@pytest.fixture
def report(tmp_path):
    from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator

    n = 30
    results = _results(
        {
            "baseline": [0.40 + 0.01 * (i % 7) for i in range(n)],
            "candidate": [0.55 + 0.01 * (i % 5) for i in range(n)],
        }
    )
    return ReportGenerator(results)


class TestItProducesAReport:
    def test_markdown_report_is_produced(self, report):
        text = report.markdown_report()
        assert isinstance(text, str)
        assert text.strip()

    def test_it_names_the_configs(self, report):
        text = report.markdown_report()
        assert "baseline" in text
        assert "candidate" in text

    def test_it_includes_the_significance_section(self, report):
        """The section that made this class worth keeping — and the one that
        depends on `pairwise_significance`, whose return shape changed."""
        assert "Statistical Analysis" in report.markdown_report()

    def test_latex_table_is_produced(self, report):
        text = report.latex_table()
        assert "\\begin{tabular}" in text or "tabular" in text

    def test_save_all_writes_files(self, report, tmp_path):
        report.save_all(str(tmp_path))
        written = list(tmp_path.rglob("*"))
        assert written, "save_all wrote nothing"


class TestTheInfiniteEffectSizeDoesNotLeakIntoAReport:
    """A bare ``:.3f`` on an infinite d_z prints ``inf`` into a results table.

    Caught because this file exists at all: the change that made ``effect_size``
    non-finite was three commits earlier and nothing here ran.
    """

    @pytest.fixture
    def constant_difference_report(self):
        from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator

        n = 30
        return ReportGenerator(
            _results(
                {
                    "a": [0.50 + 0.001 * i for i in range(n)],
                    "b": [0.80 + 0.001 * i for i in range(n)],
                }
            )
        )

    def test_the_report_does_not_contain_a_raw_inf(self, constant_difference_report):
        text = constant_difference_report.markdown_report()
        assert "inf" not in text.lower().replace("information", ""), (
            "a non-finite effect size reached the report as a raw float"
        )

    def test_it_renders_the_symbol_instead(self, constant_difference_report):
        assert "∞" in constant_difference_report.markdown_report()

    def test_the_formatter_handles_every_case(self):
        import math

        from lasagna.knowledge_evaluation_sdk.reporting import _format_effect_size

        assert _format_effect_size(math.inf) == "+∞"
        assert _format_effect_size(-math.inf) == "−∞"
        assert _format_effect_size(None) == "—"
        assert _format_effect_size(float("nan")) == "—"
        assert _format_effect_size(1.5) == "1.500"


class TestItSurvivesDegenerateInput:
    """A report generator that raises on a one-config run is worse than one
    that says there was nothing to compare."""

    def test_a_single_config_does_not_raise(self):
        from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator

        text = ReportGenerator(_results({"only": [0.5] * 10})).markdown_report()
        assert isinstance(text, str)

    def test_a_run_with_no_per_sample_scores_does_not_raise(self):
        from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator

        from lasagna.knowledge_evaluation_sdk.results import TaskResult

        results = BenchmarkResults(
            task_results=[
                TaskResult.from_dict(r)
                for r in [
                    {
                        "task_key": f"bench|{c}|cfg|corpus|live|run_0",
                        "run_index": 0,
                        "benchmark_name": "bench",
                        "config_name": c,
                        "metrics": {"ndcg@10": 0.5},
                        "per_sample_scores": [],
                        "error": None,
                    }
                    for c in ("x", "y")
                ]
            ]
        )
        assert isinstance(ReportGenerator(results).markdown_report(), str)


def test_an_untestable_pair_is_listed_in_the_report():
    """`pairwise_significance` now returns pairs it could not test, with p None."""
    from lasagna.knowledge_evaluation_sdk.reporting import ReportGenerator
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    results = _results({"a": [0.4 + 0.01 * (i % 7) for i in range(20)], "b": [0.6] * 20})
    results = BenchmarkResults(
        list(results._results)
        + [
            TaskResult(
                task_key="c",
                benchmark_name="bench",
                config_name="c",
                run_index=0,
                per_sample_scores=[{"_sample_id": "elsewhere", "ndcg@10": 0.5}],
            )
        ]
    )
    text = ReportGenerator(results).markdown_report()
    assert "untestable" in text
