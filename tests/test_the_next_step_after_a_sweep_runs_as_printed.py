"""The "Next:" line a sweep ends with is a command that works on that run.

Every sweep ended with "Next: lasagna results significance <file> --all". On a
run of three benchmarks that command stops at "This run covers 3 benchmarks
... Pass --benchmark"; after a factorial it runs the wrong test (a 2x2 asks
whether the techniques compose: `results interaction`); and a run where two
arms of three failed printed it beside a "Failed: 2" box, with one arm left to
compare. The line is now built from the run, and a partial failure says how
many tasks failed.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult

runner = CliRunner()


def _task(benchmark: str, arm: str, lift: float = 0.0, error=None) -> TaskResult:
    scores = [0.3 + lift + 0.02 * (i % 5) for i in range(12)]
    return TaskResult(
        task_key=f"{benchmark}|{arm}|run_0",
        benchmark_name=benchmark,
        config_name=arm,
        run_index=0,
        metrics={} if error else {"ndcg@10": sum(scores) / len(scores)},
        per_sample_scores=[]
        if error
        else [{"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)],
        error=error,
    )


def _sweep(tmp_path, monkeypatch, tasks, pipeline: str = "{base_profile: speed}"):
    """`benchmark experiment` on a sweep whose run returns ``tasks``."""
    from lasagna.knowledge_evaluation_sdk import experiment

    async def run(self, progress_callback=None):
        results = BenchmarkResults(
            task_results=tasks,
            experiment_name="x",
            experiment_config={"metrics_version": METRICS_VERSION},
        )
        results.save(str(tmp_path / "out" / "results.json"))
        return results

    monkeypatch.setattr(experiment.Experiment, "generate_tasks", lambda self: [])
    monkeypatch.setattr(experiment.Experiment, "run", run)
    (tmp_path / "sweep.yaml").write_text(
        "name: x\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\noutput_dir: out\n"
        f"pipeline: {pipeline}\n"
    )
    monkeypatch.chdir(tmp_path)
    return runner.invoke(cli.app, ["benchmark", "experiment", "sweep.yaml"], terminal_width=300)


def _next_commands(output: str) -> list:
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Next:"))
    commands = [lines[start][len("Next:") :].strip()]
    for line in lines[start + 1 :]:
        if not line.startswith(" " * 8) or not line.strip().startswith("lasagna "):
            break
        commands.append(line.strip())
    return commands


def _run_printed(command: str):
    argv = command.split()
    assert argv[0] == "lasagna"
    return runner.invoke(cli.app, argv[1:], terminal_width=300)


def test_a_run_of_several_benchmarks_gets_one_working_line_each(tmp_path, monkeypatch):
    benchmarks = ("smoke", "beir/scifact", "tickets@7d85fdce34c4")
    tasks = [_task(b, arm, lift) for b in benchmarks for arm, lift in (("a", 0), ("b", 0.1))]
    result = _sweep(tmp_path, monkeypatch, tasks)
    assert result.exit_code == 0, result.output
    commands = _next_commands(result.output)
    assert len(commands) == 3
    for benchmark, command in zip(sorted(benchmarks), commands):
        assert command.startswith("lasagna results significance ")
        assert command.endswith(f"--benchmark {benchmark}")
        ran = _run_printed(command)
        assert ran.exit_code == 0, ran.output
        assert "Pass --benchmark" not in ran.output


def test_a_factorial_points_at_the_interaction_test(tmp_path, monkeypatch):
    cells = {"neither": 0.0, "reranking": 0.1, "mmr": 0.02, "reranking+mmr": 0.11}
    tasks = [_task("smoke", arm, lift) for arm, lift in cells.items()]
    result = _sweep(
        tmp_path,
        monkeypatch,
        tasks,
        "{base_profile: speed, factorial: {factor_a: enable_reranking, factor_b: enable_mmr}}",
    )
    assert result.exit_code == 0, result.output
    (command,) = _next_commands(result.output)
    assert command.startswith("lasagna results interaction ")
    ran = _run_printed(command)
    assert ran.exit_code == 0, ran.output
    assert "interaction" in ran.output


def test_a_partial_failure_says_how_many_failed_and_offers_only_what_works(tmp_path, monkeypatch):
    tasks = [
        _task("smoke", "ok"),
        _task("smoke", "neg_chunk", error="boom"),
        _task("smoke", "alpha7", error="boom"),
    ]
    result = _sweep(tmp_path, monkeypatch, tasks)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "2 of 3 tasks failed" in words
    (command,) = _next_commands(result.output)
    # One arm finished: nothing to test, so the step is the run's page.
    assert command.startswith("lasagna results view ")
    assert "significance" not in command


@pytest.mark.parametrize("n_benchmarks", [1, 2])
def test_one_benchmark_needs_no_flag(tmp_path, monkeypatch, n_benchmarks):
    names = ["smoke", "nq"][:n_benchmarks]
    tasks = [_task(b, arm, lift) for b in names for arm, lift in (("a", 0), ("b", 0.1))]
    result = _sweep(tmp_path, monkeypatch, tasks)
    commands = _next_commands(result.output)
    assert ("--benchmark" in commands[0]) == (n_benchmarks > 1)
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert len(saved["results"]) == 2 * n_benchmarks
