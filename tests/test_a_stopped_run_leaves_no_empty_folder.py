"""A run that stops leaves nothing empty behind, and says what it kept.

`lasagna benchmark run beir/nfcorpus` offline with the dataset not cached was
refused before any task, and left `results/benchmark_beir_nfcorpus_<time>/`
holding only an empty `.checkpoints` folder, one per attempt. A run in which
every task failed left an empty `.checkpoints` beside its results.json. Now an
empty checkpoint folder is removed, and so is the run's own timestamped folder
when it holds nothing else. A run that stopped with finished tasks
checkpointed keeps them and says that `--resume` continues from them.
"""

from __future__ import annotations

import asyncio

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.runner import TaskResult


def _sweep(**extra):
    return {
        "name": "stops",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "tracks": ["retrieval"],
        "n_runs": 2,
        "pipeline": {"base_profile": "speed"},
        **extra,
    }


def _ok(task):
    return TaskResult(
        task_key=task.task_key,
        benchmark_name=task.benchmark_name,
        config_name=task.config_name,
        run_index=task.run_index,
        metrics={"ndcg@10": 0.5},
    )


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "results"
    monkeypatch.setattr(exp, "_results_root", lambda: str(root))
    return root


def test_a_run_refused_before_any_task_leaves_no_folder(results_root, monkeypatch):
    monkeypatch.setattr(
        exp, "_offline_dataset_problem", lambda names: RuntimeError("smoke is not cached")
    )

    async def never(task, samples):
        raise AssertionError("no task runs")

    with pytest.raises(RuntimeError, match="not cached"):
        asyncio.run(exp.Experiment.from_dict(_sweep(), eval_fn=never).run())
    assert not results_root.exists() or not any(results_root.iterdir())


def test_a_run_whose_every_task_failed_keeps_only_its_results(tmp_path):
    out = tmp_path / "out"

    async def fails(task, samples):
        raise RuntimeError("model not downloaded")

    with pytest.warns(RuntimeWarning, match="failed"):
        asyncio.run(exp.Experiment.from_dict(_sweep(output_dir=str(out)), eval_fn=fails).run())
    assert (out / "results.json").is_file()
    assert not (out / ".checkpoints").exists()


def test_a_run_that_stops_after_a_finished_task_keeps_it_for_resume(tmp_path, monkeypatch):
    out = tmp_path / "out"
    warned = []
    monkeypatch.setattr(exp.logger, "warning", lambda message: warned.append(message))
    calls = []

    async def second_one_crashes(task, samples):
        calls.append(task.task_key)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return _ok(task)

    sweep = _sweep(output_dir=str(out), max_concurrency=1)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(exp.Experiment.from_dict(sweep, eval_fn=second_one_crashes).run())
    assert len(list((out / ".checkpoints").glob("*.json"))) == 1
    assert not (out / "results.json").exists()
    (message,) = [m for m in warned if "--resume" in m]
    assert message == (
        f"The run stopped before writing results. {out} holds 1 finished task(s) as "
        f"checkpoints: run the same command with --resume to continue from them."
    )


def test_a_folder_named_by_output_dir_is_never_removed(tmp_path, monkeypatch):
    out = tmp_path / "mine"
    monkeypatch.setattr(exp, "_offline_dataset_problem", lambda names: RuntimeError("refused"))

    async def never(task, samples):
        raise AssertionError("no task runs")

    with pytest.raises(RuntimeError):
        asyncio.run(exp.Experiment.from_dict(_sweep(output_dir=str(out)), eval_fn=never).run())
    assert out.is_dir() and not (out / ".checkpoints").exists()
