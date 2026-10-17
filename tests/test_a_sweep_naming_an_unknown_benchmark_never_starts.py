"""A sweep naming a benchmark that does not exist stops before it starts, once.

`benchmarks: [{name: smok}]` printed "Starting experiment", then every task
failed with a quoted KeyError carrying all 33 benchmark names; `--dry-run`
printed that list three times and exited 0. A local `path` that does not exist
also got as far as "Starting experiment". Both are now refused before anything
starts, with one message naming the closest benchmark, in the real run and in
the dry run alike.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lasagna.cli import app
from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _parse_config

TYPO = """
name: typo
benchmarks: [{name: smok}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, mmr]
n_runs: 1
output_dir: out
"""

MISSING_FILE = """
name: missing file
benchmarks: [{path: ./missing.jsonl}]
tracks: [retrieval]
pipeline:
  base_profile: speed
n_runs: 1
output_dir: out
"""


def _run(tmp_path, monkeypatch, sweep: str, *flags: str):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sweep.yaml").write_text(sweep)
    result = CliRunner().invoke(app, ["benchmark", "experiment", "sweep.yaml", *flags])
    return result, " ".join(result.output.split())


@pytest.mark.parametrize("flags", [(), ("--dry-run", "--no-probe"), ("--dry-run",)])
def test_an_unknown_name_is_one_error_with_the_closest_name(tmp_path, monkeypatch, flags):
    result, words = _run(tmp_path, monkeypatch, TYPO, *flags)
    assert result.exit_code == 1, result.output
    assert "Unknown benchmark 'smok' (did you mean 'smoke'?)" in words
    assert "lasagna benchmark list" in words
    assert "Starting experiment" not in words
    assert "Dry run of" not in words
    # One sentence, not the 33 names once per task.
    assert words.count("beir/scifact") == 0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flags", [(), ("--dry-run", "--no-probe")])
def test_a_missing_local_file_is_refused_before_starting(tmp_path, monkeypatch, flags):
    result, words = _run(tmp_path, monkeypatch, MISSING_FILE, *flags)
    assert result.exit_code == 1, result.output
    assert "missing.jsonl: no such file or folder" in words
    assert "Starting experiment" not in words


def test_the_library_refuses_every_unknown_name_at_once():
    cfg = _parse_config(
        {
            "name": "t",
            "benchmarks": [{"name": "smok"}, {"name": "smoke"}, {"name": "beir/scifct"}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed"},
        }
    )
    with pytest.raises(ValueError) as caught:
        Experiment(cfg).generate_tasks()
    message = str(caught.value)
    assert "'smok' (did you mean 'smoke'?)" in message
    assert "'beir/scifct' (did you mean 'beir/scifact'?)" in message
    assert "Available" not in message
