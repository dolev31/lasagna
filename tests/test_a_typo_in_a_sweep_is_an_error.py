"""A key a sweep file does not read is an error, with the likely fix.

`rerankrs: [cross_encoder, "my_ideas:keyword_overlap"]` gave a one-arm run of
the stock pipeline with exit 0 and no warning: a number that reads exactly
like a score for the component and is not one. Misplaced and misspelled keys
at every level, and arm settings nothing reads, are refused before ingestion.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _parse_config

BASE = {"name": "t", "benchmarks": [{"name": "smoke"}], "tracks": ["retrieval"]}


def _parse(**extra):
    return _parse_config({**BASE, **extra})


def test_a_misspelled_pipeline_key_names_the_intended_one():
    with pytest.raises(ValueError, match=r"'rerankrs' \(did you mean 'rerankers'\?\)"):
        _parse(pipeline={"base_profile": "speed", "rerankrs": ["cross_encoder"]})


def test_a_pipeline_key_at_the_top_level_says_where_it_goes():
    with pytest.raises(ValueError, match=r"put it under `pipeline:`"):
        _parse(config_overrides=[{"name": "a"}])


def test_a_misspelled_top_level_key_is_refused():
    with pytest.raises(ValueError, match=r"'n_run' \(did you mean 'n_runs'\?\)"):
        _parse(n_run=3)


def test_benchmark_and_factorial_entries_are_checked():
    with pytest.raises(ValueError, match="n_sample"):
        _parse_config({**BASE, "benchmarks": [{"name": "smoke", "n_sample": 5}]})
    with pytest.raises(ValueError, match="factor_c"):
        _parse(pipeline={"factorial": {"factor_a": "enable_hyde", "factor_c": "x"}})


def test_an_arm_setting_nothing_reads_is_refused_before_ingestion():
    cfg = _parse(
        pipeline={
            "base_profile": "speed",
            "config_overrides": [{"name": "typo", "enable_rerankng": True}],
        }
    )
    with pytest.raises(
        ValueError, match=r"'enable_rerankng' \(did you mean 'enable_reranking'\?\)"
    ):
        Experiment(cfg).generate_tasks()


def test_a_correct_sweep_still_parses():
    cfg = _parse(
        pipeline={
            "base_profile": "speed",
            "config_overrides": [
                {"name": "a", "enable_reranking": False},
                {"name": "b", "enable_reranking": True, "metric_k": 20},
            ],
        }
    )
    assert len(Experiment(cfg).generate_tasks()) == 2


def test_the_cli_prints_the_refusal_not_a_traceback(tmp_path):
    from typer.testing import CliRunner

    from lasagna import cli

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "name: t\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\n"
        "pipeline: {base_profile: speed, rerankrs: [cross_encoder]}\n"
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep)])
    out = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "did you mean 'rerankers'" in out and "Traceback" not in out


def test_python_says_when_every_task_failed(tmp_path, monkeypatch):
    import asyncio

    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _parse(pipeline={"base_profile": "speed"})

    async def failing(task, samples):
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error="Couldn't reach the Hub",
        )

    with pytest.warns(RuntimeWarning, match="All 1 task.*Couldn't reach the Hub"):
        asyncio.run(Experiment(cfg, eval_fn=failing).run())
