"""`--resume` resumes the experiment's last run, or says there is none.

Without `output_dir:` every run writes to a new timestamped folder, so
`--resume` looked in a fresh, empty one and silently re-ran every task. And
the folder was named from the free-text experiment name, so a printed path
like `help?_20261009_214730/results.json` failed in zsh as a glob.
"""

from __future__ import annotations

import json
from pathlib import Path

from lasagna.knowledge_evaluation_sdk.experiment import ExperimentConfig, _slug


def _run_dir(home: Path, name: str, checkpoints: int) -> Path:
    d = home / ".lasagna" / "research" / "results" / name
    (d / ".checkpoints").mkdir(parents=True)
    for i in range(checkpoints):
        (d / ".checkpoints" / f"t{i}.json").write_text(json.dumps({}))
    return d


def test_the_folder_name_is_shell_safe():
    assert _slug("does longer_first help?") == "does_longer_first_help"
    assert _slug("  ") == "experiment"
    cfg = ExperimentConfig(name="a b/c?")
    assert Path(cfg.effective_output_dir()).name.startswith("a_b_c_")


def test_resume_picks_the_newest_run_with_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _run_dir(tmp_path, "my_sweep_20261001_120000", 3)
    newest = _run_dir(tmp_path, "my_sweep_20261002_120000", 2)
    _run_dir(tmp_path, "my_sweep_20261003_120000", 0)  # started, nothing finished
    _run_dir(tmp_path, "my_sweep_extra_20261004_120000", 5)  # another experiment
    assert ExperimentConfig(name="my sweep").latest_output_dir() == str(newest)


def test_nothing_to_resume_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ExperimentConfig(name="never ran").latest_output_dir() is None


def test_the_cli_refuses_a_resume_with_nothing_to_resume(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from lasagna import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "name: never ran\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\n"
        "pipeline: {base_profile: speed}\n"
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep), "--resume"])
    assert result.exit_code != 0
    assert "Nothing to resume" in " ".join(result.output.split())


def test_identical_repeated_runs_are_noticed():
    from types import SimpleNamespace

    from lasagna.knowledge_evaluation_sdk.experiment import runs_were_identical

    def r(run, score, latency):
        return SimpleNamespace(
            benchmark_name="b",
            config_name="a",
            run_index=run,
            error=None,
            per_sample_scores=[{"_sample_id": "q", "ndcg@10": score, "latency_ms": latency}],
        )

    assert runs_were_identical([r(0, 0.5, 10), r(1, 0.5, 12)])  # latency is not a result
    assert not runs_were_identical([r(0, 0.5, 10), r(1, 0.6, 10)])
    assert not runs_were_identical([r(0, 0.5, 10)])  # one run: nothing to compare
