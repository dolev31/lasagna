"""`lasagna results significance` — the test a paper reports, finally on the CLI.

`BenchmarkResults.pairwise_significance` has existed since the beginning, is
documented in `docs/metrics.md`, and had two unit tests. It was reachable only
from Python: `grep significance src/lasagna/cli.py` returned nothing, and the
only in-tree caller was `ReportGenerator`, which nothing calls either.

So the CLI offered `compare` (one pair), `power` (one pair) and `per-query` (one
pair) but not the all-pairs test with a multiple-comparison correction. Running
`compare` across a grid one pair at a time is exactly how a false positive gets
manufactured: at alpha=0.05 six configs make fifteen comparisons and roughly a
54% chance one clears the bar by luck. The package that exists to stop people
publishing noise made the correct tool the inconvenient one.

Driven through typer's real runner, so argument parsing and output formatting
are exercised rather than just the function body.
"""

from __future__ import annotations

import json

import pytest

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402

runner = typer_testing.CliRunner()


def _results_file(tmp_path, per_query):
    """A results.json with per-query scores for each config.

    Significance needs paired per-query numbers; aggregate means cannot produce
    a p-value, and the command must say so rather than inventing one.
    """
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
                    {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)
                ],
                "error": None,
            }
        )
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": task_results}))
    return str(path)


@pytest.fixture
def three_configs(tmp_path):
    """One config clearly better, two indistinguishable."""
    n = 40
    return _results_file(
        tmp_path,
        {
            "baseline": [0.50 + 0.001 * i for i in range(n)],
            "same_as_baseline": [0.50 + 0.001 * i + (0.002 if i % 2 else -0.002) for i in range(n)],
            "much_better": [0.80 + 0.001 * i for i in range(n)],
        },
    )


class TestItRuns:
    def test_the_command_exists(self):
        result = runner.invoke(app, ["results", "significance", "--help"])
        assert result.exit_code == 0

    def test_it_reports_pairs(self, three_configs):
        result = runner.invoke(app, ["results", "significance", three_configs, "--all"])
        assert result.exit_code == 0, result.output
        assert "3 pairs" in result.output

    def test_it_finds_the_real_difference(self, three_configs):
        result = runner.invoke(app, ["results", "significance", three_configs])
        assert "much_better" in result.output

    def test_it_does_not_report_the_null_pair_as_significant(self, three_configs):
        """`baseline` vs `same_as_baseline` differ only by alternating noise.

        An earlier version of this test looked for any significant line
        mentioning `same_as_baseline` and failed -- correctly, because
        `much_better` vs `same_as_baseline` genuinely is significant. The pair
        has to be named, not just one member of it.
        """
        result = runner.invoke(app, ["results", "significance", three_configs, "--all"])
        for line in result.output.splitlines():
            if "baseline " in line and "same_as_baseline" in line and "much_better" not in line:
                assert "significant" not in line
                return
        pytest.fail("the baseline/same_as_baseline pair was not reported at all")


class TestTheDegenerateEffectSize:
    """Every query moving by exactly the same amount is the strongest, most
    consistent effect there is. `_cohens_dz` returned 0.0 for it — the same
    value a genuinely null result gives — so the two opposite conclusions were
    indistinguishable in the output.

    `minimum_detectable_effect` already flagged this case as ``degenerate``;
    the effect size did not.
    """

    @pytest.fixture
    def constant_difference(self, tmp_path):
        n = 30
        return _results_file(
            tmp_path,
            {
                "a": [0.50 + 0.001 * i for i in range(n)],
                "b": [0.80 + 0.001 * i for i in range(n)],
            },
        )

    def test_a_perfectly_consistent_effect_is_not_reported_as_zero(self, constant_difference):
        result = runner.invoke(app, ["results", "significance", constant_difference, "--all"])
        assert result.exit_code == 0, result.output
        assert "+0.00" not in result.output and "-0.00" not in result.output

    @pytest.fixture
    def constant_difference_other_way(self, tmp_path):
        """The same data with the configs swapped, so d_z is +inf.

        Needed because the fixture above happens to yield -inf, and an earlier
        version of this class tested only that: disabling the `+inf` branch in
        the CLI changed nothing and every test still passed.
        """
        n = 30
        return _results_file(
            tmp_path,
            {
                "z_later_alphabetically": [0.50 + 0.001 * i for i in range(n)],
                "a_earlier": [0.80 + 0.001 * i for i in range(n)],
            },
        )

    @pytest.mark.parametrize(
        "fixture_name", ["constant_difference", "constant_difference_other_way"]
    )
    def test_it_is_rendered_as_infinite(self, fixture_name, request):
        path = request.getfixturevalue(fixture_name)
        result = runner.invoke(app, ["results", "significance", path, "--all"])
        assert result.exit_code == 0, result.output
        assert "∞" in result.output
        assert "inf" not in result.output.lower(), (
            "raw float formatting leaked through instead of the ∞ branch"
        )

    def test_both_signs_are_rendered(self):
        """Directly, without depending on which way a fixture happens to sort."""
        import math

        from lasagna import cli

        assert cli._format_effect_size(math.inf) == "+∞"
        assert cli._format_effect_size(-math.inf) == "-∞"
        assert cli._format_effect_size(0.0) == "+0.00"
        assert cli._format_effect_size(None) == "—"

    def test_the_underlying_value_is_infinite_not_zero(self):
        """Asserted on the library, not the rendering, so a consumer reading
        `effect_size` from Python sees the same thing."""
        import math

        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

        assert BenchmarkResults._cohens_dz([0.3] * 10) == math.inf
        assert BenchmarkResults._cohens_dz([-0.3] * 10) == -math.inf

    def test_a_genuinely_null_result_is_still_zero(self):
        """The case 0.0 was always right for. If this and the one above both
        returned 0.0, neither could be told from the other."""
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

        assert BenchmarkResults._cohens_dz([0.0] * 10) == 0.0


class TestTheCorrectionIsTheWholePoint:
    def test_holm_is_the_default(self, three_configs):
        result = runner.invoke(app, ["results", "significance", three_configs])
        assert "Holm-Bonferroni" in result.output

    def test_uncorrected_is_available_and_labelled_as_such(self, three_configs):
        result = runner.invoke(
            app, ["results", "significance", three_configs, "--correction", "none"]
        )
        assert result.exit_code == 0, result.output
        assert "uncorrected" in result.output

    def test_uncorrected_says_how_many_false_positives_to_expect(self, three_configs):
        """A number the reader can check the surviving pairs against."""
        result = runner.invoke(
            app, ["results", "significance", three_configs, "--correction", "none"]
        )
        assert "by chance" in result.output

    def test_correction_is_never_less_strict_than_none(self, three_configs):
        """The property that makes the default safe. Asserted rather than
        assumed, because a correction applied in the wrong direction would look
        identical in a summary line."""

        def survivors(correction):
            out = runner.invoke(
                app,
                ["results", "significance", three_configs, "--correction", correction, "--all"],
            ).output
            return out.count("significant")

        assert survivors("holm") <= survivors("none")

    def test_an_unknown_correction_is_refused(self, three_configs):
        result = runner.invoke(
            app, ["results", "significance", three_configs, "--correction", "bonferroni"]
        )
        assert result.exit_code != 0


class TestItRefusesRatherThanGuessing:
    def test_multiple_benchmarks_require_choosing_one(self, tmp_path):
        """Pooling a correction across benchmarks would correct for comparisons
        nobody made."""
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_key": f"{bench}|{config}|cfg|corpus|live|run_0",
                            "run_index": 0,
                            "benchmark_name": bench,
                            "config_name": config,
                            "metrics": {"ndcg@10": 0.5},
                            "per_sample_scores": [
                                {"_sample_id": f"q{i}", "ndcg@10": 0.5} for i in range(5)
                            ],
                            "error": None,
                        }
                        for bench in ("bench_a", "bench_b")
                        for config in ("x", "y")
                    ]
                }
            )
        )
        result = runner.invoke(app, ["results", "significance", str(path)])
        assert result.exit_code != 0
        assert "--benchmark" in result.output

    def test_naming_the_benchmark_resolves_it(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_key": f"{bench}|{config}|cfg|corpus|live|run_0",
                            "run_index": 0,
                            "benchmark_name": bench,
                            "config_name": config,
                            "metrics": {"ndcg@10": 0.5},
                            "per_sample_scores": [
                                {"_sample_id": f"q{i}", "ndcg@10": 0.5 + 0.01 * i}
                                for i in range(10)
                            ],
                            "error": None,
                        }
                        for bench in ("bench_a", "bench_b")
                        for config in ("x", "y")
                    ]
                }
            )
        )
        result = runner.invoke(
            app, ["results", "significance", str(path), "--benchmark", "bench_a"]
        )
        assert result.exit_code == 0, result.output
        assert "bench_a" in result.output

    def test_a_single_config_cannot_be_compared_to_itself(self, tmp_path):
        path = _results_file(tmp_path, {"only_one": [0.5] * 10})
        result = runner.invoke(app, ["results", "significance", path])
        assert result.exit_code != 0
        assert "two configs" in result.output
