"""`lasagna results leaderboard` reads one run's results.json, not only bundles.

Given a results.json it scanned the file as a folder of bundles, found none and
printed "0 row(s) from 0 bundle(s)" with exit code 0, so the only way to
reprint a run's leaderboard was the deprecated `benchmark leaderboard`, which
the Colab notebook used. That alias, given a folder of bundles, failed with
"Unknown format <typer.models.OptionInfo ...>". And `-o` said "Write markdown
here" when it names a folder and, with `--format latex`, gets `.tex` pages.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna.cli import app
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

runner = CliRunner()

N = 20


def _task(config: str, score: float) -> dict:
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {"ndcg@10": score},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "ndcg@10": score, "_retrieved_ids": [f"{config}-{i}"]}
            for i in range(N)
        ],
        "error": None,
        "config_overrides": {"reranker": config},
        "profile": "speed",
    }


@pytest.fixture
def results_json(tmp_path):
    path = tmp_path / "results" / "results.json"
    path.parent.mkdir()
    config = {
        "experiment_name": "one run",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "metrics_version": METRICS_VERSION,
        "environment": {"lasagna_version": "0.1.0", "python": "3.12.0", "platform": "test"},
    }
    tasks = [_task("control", 0.50), _task("mine", 0.60)]
    path.write_text(
        json.dumps({"experiment_name": "one run", "experiment_config": config, "results": tasks})
    )
    return path


def test_a_results_json_prints_the_runs_own_leaderboard(results_json):
    result = runner.invoke(app, ["results", "leaderboard", str(results_json), "-m", "ndcg@10"])
    assert result.exit_code == 0, result.output
    assert "1. mine" in result.output and "2. control" in result.output
    assert "0 row(s)" not in result.output
    assert "deprecated" not in result.output


def test_a_runs_folder_reads_as_its_results_json(results_json):
    result = runner.invoke(app, ["results", "leaderboard", str(results_json.parent)])
    assert result.exit_code == 0, result.output
    assert "1. mine" in result.output and "0 row(s)" not in result.output


def test_a_results_json_renders_as_latex_and_as_pages(results_json, tmp_path):
    latex = runner.invoke(app, ["results", "leaderboard", str(results_json), "--format", "latex"])
    assert latex.exit_code == 0, latex.output
    assert latex.output.count("\\begin{tabular}") == 1
    assert latex.output.index("mine") < latex.output.index("control")

    out = tmp_path / "pages"
    written = runner.invoke(
        app, ["results", "leaderboard", str(results_json), "--format", "latex", "-o", str(out)]
    )
    assert written.exit_code == 0, written.output
    assert "2 rows from 1 bundle" in " ".join(written.output.split())
    assert (out / "bench.tex").read_text().count("\\begin{tabular}") == 1
    assert json.loads((out / "index.json").read_text())["n_rows"] == 2

    markdown = runner.invoke(app, ["results", "leaderboard", str(results_json), "-o", str(out)])
    assert markdown.exit_code == 0, markdown.output
    page = (out / "bench.md").read_text()
    assert page.index("mine") < page.index("control")
    # A run's own file is no published bundle: the page says it was not
    # validated and links no path on this machine
    # (tests/test_researcher_messages_say_what_ran.py).
    assert "results.json)" not in page and "not validated" in page


def test_a_file_that_is_neither_says_so(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}))
    for args in ([], ["--format", "latex"]):
        result = runner.invoke(app, ["results", "leaderboard", str(other), *args])
        assert result.exit_code == 1, args
        assert "neither a bundle nor a results.json" in " ".join(result.output.split())

    missing = runner.invoke(app, ["results", "leaderboard", str(tmp_path / "nope.json")])
    assert missing.exit_code == 1
    assert "No such file or folder" in missing.output


def test_the_deprecated_spelling_reads_a_folder_and_a_file(results_json, tmp_path):
    folder = tmp_path / "bundles"
    folder.mkdir()
    bundle = runner.invoke(
        app, ["results", "bundle", str(results_json), "-o", str(folder / "run.json")]
    )
    assert bundle.exit_code == 0, bundle.output

    from_folder = runner.invoke(app, ["benchmark", "leaderboard", str(folder)])
    assert from_folder.exit_code == 0, from_folder.output
    assert "OptionInfo" not in from_folder.output
    assert "2 rows from 1 bundle" in " ".join(from_folder.output.split())

    from_file = runner.invoke(app, ["benchmark", "leaderboard", str(results_json)])
    assert from_file.exit_code == 0, from_file.output
    assert "1. mine" in from_file.output
    assert from_file.output.count("deprecated") == 1


def test_the_output_help_names_a_folder_and_both_formats():
    result = runner.invoke(app, ["results", "leaderboard", "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "Write markdown here" not in help_text
    assert "Folder to write one page per benchmark into (.md, or .tex with --format latex)" in (
        help_text
    )
    assert "a run's results.json (or its folder)" in help_text
