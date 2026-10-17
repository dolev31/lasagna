"""Every results command takes the folder a run prints after "Output:".

``lasagna results significance <that folder> --all`` printed a 30-line
IsADirectoryError traceback, and so did interaction, power, bundle, compare
and per-query; only view and leaderboard read the folder. One resolver now
turns a folder into its results.json for all of them, with a plain error when
the folder has none.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION


def _row(config: str, scores) -> dict:
    return {
        "task_key": f"smoke|{config}|run_0",
        "benchmark_name": "smoke",
        "config_name": config,
        "run_index": 0,
        "metrics": {"ndcg@10": sum(scores) / len(scores)},
        "per_sample_scores": [{"sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)],
        "n_samples": len(scores),
        "split": "test",
        "sampling_seed": 42,
        "corpus_provenance": "native",
        "resolved_mode": "live",
        "config_overrides": {"reranker": config} if config != "base" else {},
        "system": {"embedder_model": "all-MiniLM-L6-v2", "chunker": "fixed"},
    }


@pytest.fixture
def run_folder(tmp_path):
    folder = tmp_path / "my_run_20261017_104132"
    folder.mkdir()
    (folder / "results.json").write_text(
        json.dumps(
            {
                "experiment_name": "my run",
                "metrics_version": METRICS_VERSION,
                "package_version": "0.1.0",
                "experiment_config": {
                    "metrics_version": METRICS_VERSION,
                    "environment": {
                        "lasagna_version": "0.1.0",
                        "python": "3.12.9",
                        "platform": "darwin",
                    },
                },
                "results": [
                    _row("base", [0.2, 0.4, 0.6, 0.5, 0.3, 0.7]),
                    _row("cross_encoder", [0.3, 0.5, 0.7, 0.6, 0.4, 0.9]),
                ],
            }
        )
    )
    return folder


def _invoke(*argv):
    result = CliRunner().invoke(cli.app, ["results", *map(str, argv)], terminal_width=200)
    assert not isinstance(result.exception, IsADirectoryError), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    assert "Traceback" not in result.output
    return result


@pytest.mark.parametrize(
    "argv",
    [
        ["significance", "{run}", "--all"],
        ["power", "{run}", "base", "cross_encoder"],
        ["bundle", "{run}", "-o", "{tmp}/run.json"],
        ["compare", "{run}", "{run}"],
        ["per-query", "{run}"],
        ["view", "{run}", "-o", "{tmp}/view.html"],
        ["leaderboard", "{run}"],
    ],
)
def test_a_results_command_reads_the_run_folder(run_folder, tmp_path, argv):
    result = _invoke(*(a.format(run=run_folder, tmp=tmp_path) for a in argv))
    assert result.exit_code == 0, result.output


def test_interaction_reads_the_folder_and_says_what_it_needs(run_folder):
    result = _invoke("interaction", run_folder)
    # Two arms are no factorial design: refused in a sentence, not a traceback.
    assert result.exit_code == 1
    assert "Configs in this run: base, cross_encoder" in " ".join(result.output.split())


def test_a_folder_without_results_is_named(tmp_path):
    result = _invoke("significance", tmp_path, "--all")
    words = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "is a folder with no results.json in it" in words


def test_validate_says_it_checks_a_bundle(run_folder):
    result = _invoke("validate", run_folder)
    words = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "results validate checks a bundle" in words
    assert "lasagna results bundle" in words
