"""A run whose tasks failed must say which and exit non-zero.

Judge outages, unusable judgements and failed generations used to be recorded
as 0.0 with error=None, so the summary said "Failed: 0" and the report showed
0.0000 +/- 0.0000. They now fail the task; this checks the command reports it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from lasagna.cli import _exit_if_tasks_failed


def test_failed_tasks_exit_non_zero_and_are_named(capsys):
    bad = SimpleNamespace(
        task_key="beir/scifact|arm|0", error="judge produced no usable scores for faithfulness"
    )
    with pytest.raises(typer.Exit) as exc:
        _exit_if_tasks_failed(SimpleNamespace(failed_tasks=[bad]))
    assert exc.value.exit_code == 1


def test_a_clean_run_exits_normally():
    _exit_if_tasks_failed(SimpleNamespace(failed_tasks=[]))
