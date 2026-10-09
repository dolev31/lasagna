"""A rendered gate report must never contain machine-local paths.

`results.json` was already safe because `RunResult.to_dict()` redacts. The
markdown was not: `render()` read the raw run object, so every `reproduce:`
command carried the developer's home directory, their local layout, and the
names of unrelated projects sitting next to this one.

The markdown is the artefact people actually share — it is what gets pasted
into a GitHub issue when a check fails. Redaction belongs in the renderer so
there is no way to produce an unredacted report at all, which is stronger than
remembering to scrub at each call site.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from e2e.contract import Check, PhaseResult, RunResult, Status  # noqa: E402
from e2e.report import render  # noqa: E402


HOME = os.path.expanduser("~")


def _run_with(repro: str = "", detail: str = "") -> RunResult:
    check = Check(
        name="some_check",
        status=Status.FAILED,
        detail=detail,
        repro=repro,
    )
    return RunResult(
        started_at="2026-01-01T00:00:00Z",
        environment={"python": "3.12.0"},
        phases=[PhaseResult(key="p1", title="Install", checks=[check])],
    )


class TestRenderRedacts:
    def test_home_directory_never_reaches_the_markdown(self):
        """The exact regression: a repro command built from an absolute path."""
        report = render(_run_with(repro=f"cd {HOME}/PycharmProjects/lasagna && pytest"))
        assert HOME not in report
        assert "<home>" in report or "<repo>" in report

    def test_it_redacts_detail_too_not_only_repro(self):
        """Tracebacks land in `detail`, and they are full of absolute paths."""
        report = render(_run_with(detail=f"FileNotFoundError: {HOME}/.cache/thing"))
        assert HOME not in report

    def test_the_repo_path_becomes_a_tag(self):
        report = render(_run_with(repro=f"python {REPO}/e2e/run.py"))
        assert str(REPO) not in report

    def test_the_interpreter_path_is_redacted(self):
        """The gate often runs under some unrelated project's virtualenv."""
        report = render(_run_with(repro=f"{sys.executable} -m pytest"))
        assert sys.executable not in report

    def test_a_clean_report_is_unchanged_in_substance(self):
        """Redaction must not eat ordinary content."""
        report = render(_run_with(detail="3 tests failed"))
        assert "3 tests failed" in report
        assert "some_check" in report


class TestCommittedReportsStayClean:
    def test_no_report_on_disk_leaks(self):
        """Catches a report written before the renderer was fixed.

        Kept separate from the render test so the failure message distinguishes
        "the renderer is broken" from "delete this stale artefact".
        """
        reports = REPO / "e2e" / "reports"
        if not reports.is_dir():
            pytest.skip("no reports in this checkout")
        leaking = [
            str(f.relative_to(REPO))
            for f in reports.rglob("*")
            if f.is_file() and HOME in f.read_text(errors="replace")
        ]
        assert not leaking, (
            f"reports contain local paths: {leaking}. If these predate the "
            f"renderer fix, delete them; they cannot be retro-redacted."
        )


class TestTheCliDoesNotLeakPathsOnStderr:
    """`doctor --issue` redacts its output; the logs beside it did not.

    The command exists to produce something safe to paste into a public issue,
    and `core/diagnostics.py` redacts every string on the way out. But nothing
    configured logging, so loguru's default DEBUG sink printed the settings path
    and the resolved db_dir to stderr one line earlier. Anyone copying their
    terminal copies both.
    """

    def _run(self, *args, env_extra=None):
        import os
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        env.pop("LASAGNA_LOG_LEVEL", None)
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, "-m", "lasagna", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    def test_stderr_carries_no_home_path(self):
        result = self._run("doctor", "--issue")
        assert HOME not in result.stderr, (
            f"CLI logs leak the home directory beside redacted output:\n{result.stderr[:400]}"
        )

    def test_stdout_stays_redacted(self):
        """Guards against 'fixing' the leak by disabling redaction."""
        result = self._run("doctor", "--issue")
        assert HOME not in result.stdout
        assert "lasagna" in result.stdout.lower(), "the report should still have content"

    def test_verbose_restores_the_logs(self):
        """Quieting the default must not remove the ability to debug."""
        result = self._run("-v", "doctor")
        assert "DEBUG" in result.stderr, "-v should show DEBUG lines"

    def test_an_invalid_log_level_does_not_crash(self):
        """A typo in an env var must not break every command."""
        result = self._run("doctor", env_extra={"LASAGNA_LOG_LEVEL": "NOPE"})
        assert "Traceback" not in result.stderr
        assert result.returncode in (0, 1), f"unexpected exit {result.returncode}"
