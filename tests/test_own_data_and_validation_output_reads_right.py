"""What `benchmark run` on your own file, `validate-data` and `component validate` print.

`benchmark run ./my.jsonl` warned that the user's own file was "an easier task
than the published benchmark", named its results folder
`benchmark_._my.jsonl_<time>`, and printed "1 run(s)" and "embedded in 0.0
min". `validate-data` on a file nothing could be measured on printed "valid
jsonl data" and then exited 1. A factorial whose arm was inert printed no
"Stage isolation:" verdict at all. `component validate cross_encoder`, a name
any sweep can use, was refused as not being module:function.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from tests._paths import PACKAGE_DIR

ROWS = (
    '{"id": "q1", "question": "Who introduced the solidus?", "answer": "Constantine", '
    '"contexts": ["The solidus was a gold coin introduced by Constantine in 312 AD."]}\n'
    '{"id": "q2", "question": "What replaced the solidus?", "answer": "The hyperpyron", '
    '"contexts": ["The hyperpyron replaced the debased solidus in 1092."]}\n'
)


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_a_run_on_your_own_file_reads_as_yours(tmp_path):
    (tmp_path / "my.jsonl").write_text(ROWS)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="300",
    )
    run = subprocess.run(
        [sys.executable, "-m", "lasagna", "benchmark", "run", "./my.jsonl", "--profile", "speed"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = run.stdout + run.stderr
    assert run.returncode == 0, out[-3000:]
    assert "published benchmark" not in out
    assert "1 run," in out and "run(s)" not in out
    assert "0.0 min" not in out
    folders = [p.name for p in (tmp_path / "home" / ".lasagna" / "research" / "results").iterdir()]
    assert folders and all(f.startswith("benchmark_my_") for f in folders), folders


def test_small_latencies_are_never_zero():
    assert cli._milliseconds(0.004) == "under 0.01 ms"
    assert cli._milliseconds(0.42) == "0.42 ms"
    assert cli._milliseconds(123.456) == "123.5 ms"


def test_data_nothing_can_be_measured_on_is_one_error_not_valid(tmp_path):
    path = tmp_path / "questions_only.jsonl"
    path.write_text('{"question": "q"}\n')
    result = CliRunner().invoke(
        cli.app, ["benchmark", "validate-data", str(path)], terminal_width=300
    )
    assert result.exit_code == 1
    assert "valid" not in result.output
    assert "Nothing can be measured on questions_only.jsonl" in result.output
    assert "1 problem; fix it and run this again." in result.output


def test_an_inert_arm_still_gets_the_isolation_verdict(monkeypatch, capsys):
    from lasagna.knowledge_evaluation_sdk import stages

    report = {"undeclared": [], "unverified": [], "inert": ["hybrid: returned the control's lists"]}
    monkeypatch.setattr(stages, "isolation_report", lambda rows: report)
    monkeypatch.setattr(stages, "compared_pairs", lambda rows: 3)
    cli._print_isolation(SimpleNamespace(_results=[{}, {}]))
    out = " ".join(capsys.readouterr().out.split())
    assert "Stage isolation: the configuration that reached each stage differs only" in out
    assert "(3 pairs compared), but 1 arm changed nothing there:" in out
    assert "Inert arm: hybrid" in out


def test_a_registered_component_validates_by_its_name():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["component", "validate", "fixed"], terminal_width=300)
    assert result.exit_code == 0, result.output
    assert "Contract satisfied" in result.output
    typo = runner.invoke(cli.app, ["component", "validate", "fixd"], terminal_width=300)
    assert typo.exit_code == 1
    assert "No component is registered as 'fixd' (did you mean 'fixed'?)" in typo.output
