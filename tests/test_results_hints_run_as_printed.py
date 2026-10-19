"""The command a results verdict names runs when copied.

`lasagna results significance` ended "lasagna results power says which.", and
`lasagna results power <run>` fails with "Missing argument 'config_a'": the hint
named a command that cannot run as written. The verdict now prints the command
with the file and the closest pair filled in, and every hint the results
commands print is run here to prove it parses.
"""

from __future__ import annotations

import json
import shlex

import pytest

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402

runner = typer_testing.CliRunner()


def _run_file(tmp_path, per_bench, name="results.json"):
    """A results.json: ``per_bench`` maps benchmark -> config -> per-query scores."""
    rows = []
    for bench, configs in per_bench.items():
        for config, scores in configs.items():
            rows.append(
                {
                    "task_key": f"{bench}|{config}|cfg|corpus|live|run_0",
                    "run_index": 0,
                    "benchmark_name": bench,
                    "config_name": config,
                    "metrics": {"ndcg@10": sum(scores) / len(scores)},
                    "per_sample_scores": [
                        {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)
                    ],
                    "error": None,
                }
            )
    path = tmp_path / name
    path.write_text(json.dumps({"results": rows}))
    return path


def _tied(n=40):
    return {
        "minilm": [0.50 + 0.001 * i for i in range(n)],
        "tinybert": [0.50 + 0.001 * i + (0.002 if i % 2 else -0.002) for i in range(n)],
    }


def _hint(output: str) -> list:
    """The `lasagna ...` command a verdict printed, as argv (without `lasagna`)."""
    lines = [ln.strip(" │") for ln in output.splitlines() if ln.strip(" │").startswith("lasagna ")]
    assert len(lines) == 1, output
    return shlex.split(lines[0])[1:]


def test_the_verdict_names_a_power_command_with_the_closest_pair(tmp_path):
    path = _run_file(tmp_path, {"smoke": _tied()})
    out = runner.invoke(app, ["results", "significance", str(path)]).output
    argv = _hint(out)
    assert argv[:2] == ["results", "power"]
    assert argv[2:] == [str(path), "minilm", "tinybert"]
    ran = runner.invoke(app, argv)
    assert ran.exit_code == 0, ran.output
    assert "Missing argument" not in ran.output


def test_the_power_hint_picks_the_benchmark_and_metric_it_tested(tmp_path):
    path = _run_file(tmp_path, {"smoke": _tied(), "other": _tied()})
    out = runner.invoke(app, ["results", "significance", str(path), "--benchmark", "smoke"]).output
    argv = _hint(out)
    assert argv[-2:] == ["--benchmark", "smoke"]
    ran = runner.invoke(app, argv)
    assert ran.exit_code == 0, ran.output


def test_a_path_with_a_space_is_quoted_in_the_hint(tmp_path):
    folder = tmp_path / "my runs"
    folder.mkdir()
    path = _run_file(folder, {"smoke": _tied()})
    out = runner.invoke(app, ["results", "significance", str(path)]).output
    argv = _hint(out)
    assert argv[2] == str(path)
    assert runner.invoke(app, argv).exit_code == 0


def test_the_html_page_names_the_same_runnable_command(tmp_path):
    from lasagna.knowledge_evaluation_sdk.viewer import render_html

    path = _run_file(tmp_path, {"smoke": _tied()})
    page = render_html(path)
    assert "results power" in page
    assert "Missing" not in page
    import html
    import re

    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    # The page knows the file by name only, so the command is relative to the run's folder.
    assert "lasagna results power results.json minilm tinybert" in " ".join(text.split())


def test_a_leaderboard_that_ranks_nobody_names_the_significance_command_whole(tmp_path):
    from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

    path = _run_file(tmp_path, {"smoke": {"a": [0.5, 0.6], "b": [0.5, 0.6]}})
    # Each arm scored a different pair of the three queries: nobody scored all.
    data = json.loads(path.read_text())
    for row, ids in zip(data["results"], (["q0", "q1"], ["q1", "q2"])):
        for sample, qid in zip(row["per_sample_scores"], ids):
            sample["_sample_id"] = qid
    path.write_text(json.dumps(data))
    text = BenchmarkResults.load(str(path)).leaderboard()
    assert "No config scored every query" in text
    assert "`lasagna results significance <results.json>`" in text
