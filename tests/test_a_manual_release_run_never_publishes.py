"""A manual run of release.yml builds, checks and smoke-tests, and never uploads.

The dry run (Actions > Release > Run workflow) is how the release path is tested
before a real release, so the publish job must run only for a published
GitHub Release.
"""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


def test_only_a_published_release_runs_the_publish_job():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    assert jobs["publish"]["if"] == "github.event_name == 'release'"
    assert all(job.get("if") is None for name, job in jobs.items() if name != "publish")
