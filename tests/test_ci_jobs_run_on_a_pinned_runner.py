"""Every CI job names the runner image it was tested on.

Every job said `runs-on: ubuntu-latest`, and GitHub moves that label to the
next Ubuntu release on its own schedule (Ubuntu 26 from 2026-10-19). The first
run after the move installs other system libraries and Python builds than the
ones the workflows were written against, with no commit to point at. A pinned
image changes only when this file does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

#: The image the workflows were last run on.
PINNED = "ubuntu-24.04"


def _jobs(path: Path):
    return (yaml.safe_load(path.read_text()) or {}).get("jobs") or {}


def test_there_are_workflows_to_check():
    assert {p.name for p in WORKFLOWS} >= {"tests.yml", "release.yml"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_runs_on_the_pinned_image(path):
    jobs = _jobs(path)
    assert jobs, f"{path.name} defines no jobs"
    wrong = {name: job.get("runs-on") for name, job in jobs.items() if job.get("runs-on") != PINNED}
    assert not wrong, f"{path.name}: jobs not on {PINNED}: {wrong}"
