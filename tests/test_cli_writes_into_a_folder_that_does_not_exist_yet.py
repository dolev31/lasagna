"""``-o newdir/file`` creates ``newdir`` instead of ending in a traceback.

``lasagna results bundle results.json -o newdir/run.json`` (and ``results
schema``, ``results per-query``, ``export``) wrote with ``Path.write_text``
straight away, so a folder that did not exist yet ended in a
``FileNotFoundError`` traceback after the bundle had been built. The folder
is created; one that cannot be is a one-line error.
"""

from __future__ import annotations

import json

import pytest

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION  # noqa: E402

runner = typer_testing.CliRunner()


def _results_file(tmp_path):
    row = {
        "task_key": "dense",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": "dense",
        "metrics": {"ndcg@10": 0.5},
        "per_sample_scores": [{"_sample_id": "q1", "ndcg@10": 0.5}],
        "error": None,
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"metrics_version": METRICS_VERSION, "results": [row]}))
    return str(path)


@pytest.mark.parametrize(
    "command, name",
    [
        (["results", "bundle", "{results}"], "run.json"),
        (["results", "per-query", "{results}"], "rows.jsonl"),
        (["results", "schema"], "schema.json"),
    ],
)
def test_the_output_folder_is_created(tmp_path, command, name):
    out = tmp_path / "newdir" / "deeper" / name
    args = [a.format(results=_results_file(tmp_path)) for a in command] + ["-o", str(out)]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_folder_that_cannot_be_made_is_one_line(tmp_path):
    blocker = tmp_path / "taken"
    blocker.write_text("a file, not a folder")

    result = runner.invoke(
        app, ["results", "bundle", _results_file(tmp_path), "-o", str(blocker / "run.json")]
    )

    assert result.exit_code == 1
    assert "Cannot create the folder" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
