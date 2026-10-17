"""Every `lasagna results` command refuses a metric the run never recorded.

`results view -m ndgc@10` said "No metric 'ndgc@10' in these results. They
record: ...". The other commands took the typo at face value:
`results significance -m ndgc@10` (or `-m ndcg@5`, a metric no task recorded)
printed "No pair survives Holm correction" and exited 0, a statistical verdict
on numbers that do not exist; `results leaderboard` listed every arm as
"no ndgc@10 recorded" and exited 0, and `results compare` printed an empty
table and exited 0. Each now names the closest recorded metric, lists them all,
and exits 1.
"""

from __future__ import annotations

import json

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402

runner = typer_testing.CliRunner()

N = 20


def _task(config: str, base: float) -> dict:
    scores = [base + 0.001 * i for i in range(N)]
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {"ndcg@10": sum(scores) / N, "recall@10": 0.5},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "ndcg@10": s, "recall@10": 0.5} for i, s in enumerate(scores)
        ],
        "error": None,
        "config_overrides": {"reranker": config},
    }


@pytest.fixture
def results_json(tmp_path):
    """Four arms of a 2x2, so `interaction` finds its cells without flags."""
    config = {
        "experiment_name": "typo",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "environment": {"lasagna_version": "0.1.0"},
        "metrics_version": METRICS_VERSION,
    }
    tasks = [_task("neither", 0.5), _task("a", 0.6), _task("b", 0.55), _task("a+b", 0.65)]
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"experiment_name": "typo", "experiment_config": config, "results": tasks})
    )
    return str(path)


def _refused(args, metric: str = "ndgc@10"):
    result = runner.invoke(app, args)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, result.output
    assert f"No metric '{metric}'" in words, words
    assert "They record: ndcg@10, recall@10." in words, words
    return words


@pytest.mark.parametrize(
    "command",
    [
        ["results", "significance", "{r}"],
        ["results", "leaderboard", "{r}"],
        ["results", "leaderboard", "{r}", "--format", "latex"],
        ["results", "compare", "{r}", "{r}"],
        ["results", "power", "{r}", "neither", "a"],
        ["results", "interaction", "{r}"],
        ["results", "view", "{r}", "-o", "{out}"],
        ["benchmark", "compare", "{r}", "{r}"],
        ["benchmark", "leaderboard", "{r}"],
    ],
    ids=lambda c: " ".join(c[:2] + [a for a in c if a.startswith("--")]),
)
def test_a_typo_names_the_closest_recorded_metric(command, results_json, tmp_path):
    args = [a.format(r=results_json, out=tmp_path / "v.html") for a in command]
    words = _refused(args + ["-m", "ndgc@10"])
    assert "did you mean 'ndcg@10'?" in words


def test_a_metric_no_task_recorded_gets_no_verdict(results_json):
    words = _refused(["results", "significance", results_json, "-m", "ndcg@5"], "ndcg@5")
    assert "Holm" not in words


def test_compare_names_the_file_that_lacks_the_metric(results_json, tmp_path):
    other = tmp_path / "other.json"
    data = json.loads(open(results_json).read())
    for task in data["results"]:
        task["metrics"] = {"recall@10": 0.5}
        for sample in task["per_sample_scores"]:
            sample.pop("ndcg@10")
    other.write_text(json.dumps(data))
    result = runner.invoke(app, ["results", "compare", results_json, str(other)])
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "No metric 'ndcg@10' in other.json" in words
    assert "They record: recall@10." in words


def test_a_folder_of_bundles_is_checked_against_what_the_bundles_record(results_json, tmp_path):
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    made = runner.invoke(app, ["results", "bundle", results_json, "-o", str(bundles / "run.json")])
    assert made.exit_code == 0, made.output
    _refused(["results", "leaderboard", str(bundles), "-m", "ndgc@10"])


def test_a_recorded_metric_still_works(results_json):
    result = runner.invoke(app, ["results", "significance", results_json, "-m", "recall@10"])
    assert result.exit_code == 0, result.output
