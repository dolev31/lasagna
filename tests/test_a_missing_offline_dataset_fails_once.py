"""Offline, a benchmark whose data is not cached fails once, before any task runs.

`examples/notebooks/compare_your_reranker.py` with an empty SciFact cache
printed the dataset error once per arm and again in the failed-task list (six
times), then a green "Experiment Complete ... Failed: 3" box and "Next: lasagna
results significance ...", for results that did not exist.
"""

from __future__ import annotations

import os
import subprocess
import sys

from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import datasets
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult
from tests._paths import PACKAGE_DIR

SWEEP = """
name: does coverage help
benchmarks:
  - name: beir/scifact
    n_samples: 100
tracks: [retrieval]
output_dir: results
n_runs: 1
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, mmr]
"""


def test_a_dataset_that_is_not_cached_is_named_with_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(datasets, "dataset_cached", lambda repo: False)
    problem = datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"], "smoke": []})
    assert isinstance(problem, datasets.DatasetNotCached)
    assert str(problem).startswith("beir/scifact is not in the Hugging Face cache")
    assert "run once online" in str(problem) and "unset HF_HUB_OFFLINE" in str(problem)
    monkeypatch.setattr(datasets, "dataset_cached", lambda repo: True)
    assert datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]}) is None


def test_online_nothing_is_checked(monkeypatch):
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(datasets, "dataset_cached", lambda repo: False)
    assert datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]}) is None


def test_a_datasets_cache_folder_counts_as_cached(monkeypatch, tmp_path):
    import datasets.config as datasets_config

    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))
    assert not datasets.dataset_cached("mteb/scifact")
    (tmp_path / "mteb___scifact").mkdir()
    assert datasets.dataset_cached("mteb/scifact")


def test_the_sweep_stops_with_one_error_before_any_task(tmp_path):
    (tmp_path / "sweep.yaml").write_text(SWEEP)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_", "HF_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        HF_HOME=str(tmp_path / "hf"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONPATH=str(PACKAGE_DIR.parent),
        COLUMNS="300",
    )
    run = subprocess.run(
        [sys.executable, "-m", "lasagna", "benchmark", "experiment", "sweep.yaml"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = run.stdout + run.stderr
    assert run.returncode == 1, out[-3000:]
    assert out.count("not in the Hugging Face cache") == 1, out[-3000:]
    assert "beir/scifact is not in the Hugging Face cache and offline mode is on" in out
    assert "dataset_info.json" not in out and "Couldn't reach" not in out
    assert "Experiment Complete" not in out and "Next:" not in out


def test_when_every_task_fails_the_summary_says_so(monkeypatch, tmp_path):
    from lasagna.knowledge_evaluation_sdk import experiment

    failed = [
        TaskResult(
            task_key=f"smoke|{arm}|run_0",
            benchmark_name="smoke",
            config_name=arm,
            run_index=0,
            error="boom",
        )
        for arm in ("a", "b")
    ]

    async def run(self, progress_callback=None):
        return BenchmarkResults(task_results=failed, experiment_name="x", experiment_config={})

    monkeypatch.setattr(experiment.Experiment, "generate_tasks", lambda self: [])
    monkeypatch.setattr(experiment.Experiment, "run", run)
    (tmp_path / "sweep.yaml").write_text(
        "name: x\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\noutput_dir: out\n"
        "pipeline: {base_profile: speed}\n"
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.app, ["benchmark", "experiment", "sweep.yaml"], terminal_width=200
    )
    assert result.exit_code == 1, result.output
    assert "Experiment failed: every task failed" in result.output
    assert "Experiment Complete" not in result.output
    assert "Next:" not in result.output
    assert "failed smoke|a|run_0: boom" in " ".join(result.output.split())
