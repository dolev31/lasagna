"""A benchmark runs while `lasagna serve` holds the local store.

`lasagna benchmark run smoke` with `lasagna serve` up printed "Output: ...",
then "Could not create the arm's collection: The Milvus Lite store ... is in
use", and left a results folder of failed tasks. A benchmark needs nothing from
the user's store, so when the default store is held the benchmark commands now
use one of their own under `<data dir>/bench/dbs` and say so; a store chosen
explicitly is not moved, and the run stops before it makes anything, with the
choices. `Experiment.run` from Python refuses the same way before its folder.
"""

from __future__ import annotations


import json
import os
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult

fcntl = pytest.importorskip("fcntl")  # the lock Milvus Lite takes on POSIX

runner = CliRunner()


def _hold(store) -> int:
    """Take the Milvus Lite store's lock the way its server does; returns the fd."""
    store.mkdir(parents=True, exist_ok=True)
    fd = os.open(store / "LOCK", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A data directory of its own, with no store chosen explicitly."""
    from lasagna import providers
    from lasagna.core.config import reset_rag_config

    for name in ("LASAGNA_DB_DIR", "RAG_DB_DIR"):
        monkeypatch.setenv(name, "unset")  # recorded, so teardown restores it
        monkeypatch.delenv(name)
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("RAG_VECTOR_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("MILVUS_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_rag_config()
    yield tmp_path / "data"
    providers.reset_providers()
    reset_rag_config()


@pytest.fixture
def sweep(monkeypatch, tmp_path):
    """`benchmark experiment` on a sweep whose run records the store it was given."""
    from lasagna.knowledge_evaluation_sdk import experiment
    from lasagna.providers import get_config_provider

    seen = {}

    async def run(self, progress_callback=None):
        seen["db_dir"] = str(get_config_provider().db_dir)
        tasks = [
            TaskResult(
                task_key=f"smoke|{arm}|run_0",
                benchmark_name="smoke",
                config_name=arm,
                run_index=0,
                metrics={"ndcg@10": 0.5},
                per_sample_scores=[{"_sample_id": "q1", "ndcg@10": 0.5}],
            )
            for arm in ("a", "b")
        ]
        return BenchmarkResults(tasks, "x", {"metrics_version": METRICS_VERSION})

    monkeypatch.setattr(experiment.Experiment, "generate_tasks", lambda self: [])
    monkeypatch.setattr(experiment.Experiment, "run", run)
    (tmp_path / "sweep.yaml").write_text(
        "name: x\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\noutput_dir: out\n"
        "pipeline: {base_profile: speed}\n"
    )
    return seen


def test_a_held_default_store_moves_the_benchmark_to_its_own(data_dir, sweep):
    fd = _hold(data_dir / "dbs" / "rag_milvus.db")
    try:
        result = runner.invoke(cli.app, ["benchmark", "experiment", "sweep.yaml"])
    finally:
        os.close(fd)
    words = " ".join(result.output.split())
    assert result.exit_code == 0, words
    assert "Another process is using the local store, so this benchmark uses its own" in words
    assert sweep["db_dir"] == str(data_dir / "bench" / "dbs")


def test_a_free_store_is_used_as_before(data_dir, sweep):
    result = runner.invoke(cli.app, ["benchmark", "experiment", "sweep.yaml"])
    assert result.exit_code == 0, result.output
    assert "uses its own" not in result.output
    assert sweep["db_dir"] == str(data_dir / "dbs")


def test_a_store_chosen_explicitly_is_not_moved(data_dir, sweep, monkeypatch, tmp_path):
    chosen = tmp_path / "mine"
    monkeypatch.setenv("RAG_DB_DIR", str(chosen))
    fd = _hold(chosen / "rag_milvus.db")
    try:
        result = runner.invoke(cli.app, ["benchmark", "experiment", "sweep.yaml"])
    finally:
        os.close(fd)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "so this benchmark cannot open it" in words
    assert "RAG_DB_DIR=<another directory>" in words
    assert "Output:" not in words and "db_dir" not in sweep
    assert not (tmp_path / "out").exists()


def test_an_experiment_from_python_refuses_before_making_its_folder(monkeypatch, tmp_path):
    import asyncio

    from lasagna.core import local_server
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _parse_config

    store = tmp_path / "rag_milvus.db"
    monkeypatch.setattr(local_server, "local_store_path", lambda: str(store))
    out = tmp_path / "out"
    exp = Experiment(
        config=_parse_config(
            {
                "name": "x",
                "benchmarks": [{"name": "smoke"}],
                "tracks": ["retrieval"],
                "output_dir": str(out),
                "pipeline": {"base_profile": "speed"},
            }
        )
    )
    fd = _hold(store)
    try:
        with pytest.raises(RuntimeError, match="so this experiment cannot open it"):
            asyncio.run(exp.run())
    finally:
        os.close(fd)
    assert not out.exists()


def test_the_benchmark_commands_choose_the_store_before_anything_reads_the_configuration(
    tmp_path,
):
    """The switch resets the configuration; a module holding the old one would keep it."""
    probe = textwrap.dedent(
        """
        import json, sys
        from typer.testing import CliRunner
        from lasagna import cli

        seen = []

        def probe():
            import lasagna.core.config as config
            seen.append({
                "created": config._rag_config is not None,
                "bound": sorted(
                    n for n, m in sys.modules.items()
                    if n.startswith("lasagna") and n != "lasagna.core.config"
                    and m is not None and "rag_config" in vars(m)
                ),
            })
            raise SystemExit(0)

        cli._benchmark_store = probe
        for argv in (["benchmark", "run", "smoke"], ["benchmark", "experiment", sys.argv[1]]):
            CliRunner().invoke(cli.app, argv)
        print(json.dumps(seen))
        """
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("name: x\nbenchmarks: [{name: smoke}]\n")
    env = {**os.environ, "HOME": str(tmp_path), "LASAGNA_DATA_DIR": str(tmp_path / "data")}
    out = subprocess.run(
        [sys.executable, "-c", probe, str(sweep)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen == [{"created": False, "bound": []}] * 2
