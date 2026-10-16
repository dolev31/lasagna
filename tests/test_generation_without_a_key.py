"""Generation without an LLM key is refused up front, and retrieval numbers survive it.

``benchmark validate-data`` told a user with no key to run ``--tracks
retrieval,generation``. That run ingested, retrieved, then failed every
answer with a loguru traceback ("20 of 20 answers could not be generated"),
and the task's retrieval metrics were thrown away with the generation ones.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from tests._paths import PACKAGE_DIR


def _no_llm(monkeypatch):
    import lasagna.providers as providers

    def _missing():
        raise providers.LLMProviderNotConfigured("no key")

    monkeypatch.setattr(providers, "get_llm_provider", _missing)


def _some_llm(monkeypatch):
    import lasagna.providers as providers

    monkeypatch.setattr(providers, "get_llm_provider", lambda: SimpleNamespace())


@pytest.fixture
def data(tmp_path):
    rows = [
        {
            "question": f"Which animal is fact {i} about?",
            "answer": "tardigrade",
            "contexts": [f"Fact {i}: tardigrades survive."],
        }
        for i in range(3)
    ]
    path = tmp_path / "mine.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_validate_data_suggests_retrieval_only_without_a_key(monkeypatch, data):
    _no_llm(monkeypatch)
    out = CliRunner().invoke(cli.app, ["benchmark", "validate-data", data]).output
    assert "--tracks retrieval\n" in out.replace(" \n", "\n")
    assert "retrieval,generation" not in out
    assert "needs an LLM API key" in out


def test_validate_data_suggests_both_tracks_with_a_key(monkeypatch, data):
    _some_llm(monkeypatch)
    out = CliRunner().invoke(cli.app, ["benchmark", "validate-data", data]).output
    assert "--tracks retrieval,generation" in out


def test_benchmark_run_refuses_generation_before_anything_loads(monkeypatch, data):
    _no_llm(monkeypatch)

    def _never(config):
        raise AssertionError("the experiment was built; the key check came too late")

    monkeypatch.setattr(exp, "Experiment", _never)
    result = CliRunner().invoke(
        cli.app, ["benchmark", "run", data, "--tracks", "retrieval,generation"]
    )
    assert result.exit_code == 1
    assert "The generation track needs an LLM" in result.output
    assert "--tracks retrieval" in result.output


def test_a_generation_only_task_fails_before_the_corpus_is_ingested(monkeypatch):
    _no_llm(monkeypatch)

    async def _never(*a, **k):
        raise AssertionError("the corpus was ingested for a task that cannot run")

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _never)
    task = BenchmarkTask(
        benchmark_name="b",
        config_name="c",
        run_index=0,
        mode="live",
        config_overrides={},
        tracks=("generation",),
    )
    sample = BenchmarkSample(query_id="q", question="q?", ground_truth="a", context="a")
    result = asyncio.run(exp._eval_live(task, [sample]))
    assert "The generation track needs an LLM" in result.error


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_retrieval_metrics_survive_when_generation_cannot_run(tmp_path, data):
    """Both tracks, no key: retrieval is scored and reported, generation is named."""
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"name: both\nbenchmarks: [{{path: {data}}}]\npipeline: {{base_profile: speed}}\n"
        f"n_runs: 1\noutput_dir: {tmp_path / 'out'}\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith(("_API_KEY", "_APIKEY", "_BASE_URL")) and k not in ("HF_MODEL_ID",)
    }
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")
    run = subprocess.run(
        [sys.executable, "-m", "lasagna", "benchmark", "experiment", str(sweep)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Traceback" not in run.stderr
    assert "generation not scored" in run.stdout
    row = json.loads((tmp_path / "out" / "results.json").read_text())["results"][0]
    assert row["error"] is None
    assert row["metrics"]["ndcg@10"] > 0
    assert "exact_match" not in row["metrics"]
    assert "needs an LLM" in row["system"]["generation_failed"]
