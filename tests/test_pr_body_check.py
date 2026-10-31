"""The pull request check fails a description or diff below the bar, and only those.

`scripts/check_pr_body.py` is what `.github/workflows/pr-quality.yml` runs on
every pull request. A check that passes everything is a template nobody fills
in; one that fails a good pull request teaches contributors to ignore it. So
both directions are tested: the unfilled template fails, a filled one passes,
and each diff rule fails its own bad line and passes the fixed one.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_documented_claims import _MARKETING
from tests.test_nothing_lasagna_prints_has_an_em_dash import _DOUBLE_HYPHEN, DASH

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_pr_body", REPO_ROOT / "scripts" / "check_pr_body.py"
)
check_pr_body = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_pr_body)

TEMPLATE = (REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-quality.yml"

FILLED = """\
## Why

`lasagna component validate` rejects an agentic_pipeline because the stage is
missing from its table, so nobody can check one before opening a pull request.

## What changed

The validator knows the agentic_pipeline stage. `--stage` lists it in its help.

## How it was tested

`tests/test_component_validate.py::test_an_agentic_pipeline_validates` fails
before the change and passes after it. Ran `scripts/ci-local.sh`. In the stack
built from this branch, `docker compose exec api lasagna component validate
self_ask` passes (image `lasagna-api`, `docker compose ps` shows it healthy).

## Evidence

No number can move: the validator only reads a component, it never runs a benchmark.

## Agent use

Written with: Claude Code
Read in full by: @octocat

## Checklist

- [x] This PR makes one logical change.
- [x] `scripts/ci-local.sh` passes, or at least `ruff check .`, `ruff format --check .` and the test suite.
- [x] Every new `noqa`, `type: ignore` or `pragma: no cover` names its code and gives a reason.
- [x] `CHANGELOG.md` has a line under Unreleased, or nothing a user sees changed.
- [x] I ran this change in the Docker Compose stack built from this branch, and "How it was tested" shows the commands.
- [x] I read every line of the diff, and I can explain each one.
"""


def _replace(body: str, section: str, text: str) -> str:
    start = body.index(f"## {section}\n")
    end = body.find("\n## ", start + 1)
    end = len(body) if end < 0 else end
    return body[:start] + f"## {section}\n\n{text}\n" + body[end:]


def _diff(path: str, lines: list[str], *, new: bool = False) -> str:
    head = f"diff --git a/{path} b/{path}\n"
    if new:
        head += "new file mode 100644\n--- /dev/null\n"
    else:
        head += f"--- a/{path}\n"
    head += f"+++ b/{path}\n@@ -1,0 +1,{len(lines)} @@\n"
    return head + "".join(f"+{line}\n" for line in lines)


def _with_tests_and_changelog(diff: str) -> str:
    return (
        diff
        + _diff("tests/test_x.py", ["def test_x():", "    assert True"])
        + _diff("CHANGELOG.md", ["- A line."])
    )


class TestTheDescription:
    def test_the_unfilled_template_fails(self):
        found = check_pr_body.problems(TEMPLATE)
        text = "\n".join(found)
        for section in ("Why", "What changed", "How it was tested", "Evidence"):
            assert f'"## {section}" is empty' in text, section
        assert "Written with:" in text and "@handle" in text
        assert "is not ticked" in text

    def test_a_filled_description_passes(self):
        assert check_pr_body.problems(FILLED) == []

    def test_an_empty_description_fails(self):
        assert check_pr_body.problems("")
        assert check_pr_body.problems(None)

    def test_the_template_and_the_check_name_the_same_sections(self):
        headings = re.findall(r"^## (.+)$", TEMPLATE, re.M)
        assert tuple(headings) == check_pr_body.SECTIONS

    def test_every_required_box_is_in_the_template_and_unticked(self):
        boxes = re.findall(r"^- \[( )\] (.+)$", TEMPLATE, re.M)
        assert len(boxes) == len(check_pr_body.REQUIRED_BOXES)
        for fragment in check_pr_body.REQUIRED_BOXES:
            assert any(fragment in label for _, label in boxes), fragment

    @pytest.mark.parametrize("placeholder", ["TBD", "n/a", "todo", "see diff", "..."])
    def test_a_placeholder_is_not_an_answer(self, placeholder):
        body = _replace(FILLED, "What changed", placeholder)
        assert any(f'says only "{placeholder}"' in p for p in check_pr_body.problems(body))

    def test_a_why_of_a_few_words_fails(self):
        body = _replace(FILLED, "Why", "Fix the bug.")
        assert any("a few words" in p for p in check_pr_body.problems(body))

    def test_no_number_can_move_needs_a_reason(self):
        body = _replace(FILLED, "Evidence", "No number can move:")
        assert any("but not why" in p for p in check_pr_body.problems(body))

    def test_evidence_can_be_a_result(self):
        body = _replace(FILLED, "Evidence", "p(Holm) 1.0e-04, +0.0913 ndcg@10 on beir/scifact.")
        assert check_pr_body.problems(body) == []

    def test_the_reader_is_a_person_by_handle(self):
        for reader in ("", "me", "the agent"):
            body = FILLED.replace("Read in full by: @octocat", f"Read in full by: {reader}")
            assert any("@handle" in p for p in check_pr_body.problems(body)), reader

    def test_written_with_must_be_filled_and_none_is_an_answer(self):
        blank = FILLED.replace("Written with: Claude Code", "Written with:")
        assert any("Written with:" in p for p in check_pr_body.problems(blank))
        human = FILLED.replace("Written with: Claude Code", "Written with: none")
        assert check_pr_body.problems(human) == []

    def test_an_unticked_or_removed_box_fails(self):
        unticked = FILLED.replace("- [x] I read every line", "- [ ] I read every line")
        assert any("is not ticked" in p for p in check_pr_body.problems(unticked))
        removed = "\n".join(line for line in FILLED.splitlines() if "I read every line" not in line)
        assert any("was removed" in p for p in check_pr_body.problems(removed))

    def test_a_change_not_run_in_the_compose_stack_fails(self):
        local_only = _replace(
            FILLED,
            "How it was tested",
            "`tests/test_component_validate.py::test_an_agentic_pipeline_validates` "
            "fails before the change and passes after it.",
        )
        assert any("docker compose" in p for p in check_pr_body.problems(local_only))
        for spelling in ("docker compose up -d --build", "Docker Compose run", "docker-compose ps"):
            ran = _replace(FILLED, "How it was tested", f"Ran `{spelling}` and the test.")
            assert not any("docker compose" in p for p in check_pr_body.problems(ran)), spelling

    def test_the_compose_box_is_required(self):
        unticked = FILLED.replace("- [x] I ran this change", "- [ ] I ran this change")
        assert any("Docker Compose stack" in p for p in check_pr_body.problems(unticked))

    def test_windows_line_endings_are_read(self):
        assert check_pr_body.problems(FILLED.replace("\n", "\r\n")) == []

    def test_the_template_reads_as_the_rest_of_the_docs(self):
        """`_shipped_markdown` skips dotted directories, so this file is checked here."""
        assert DASH not in TEMPLATE
        assert not [line for line in TEMPLATE.splitlines() if _DOUBLE_HYPHEN.search(line)]
        assert not _MARKETING.search(TEMPLATE)
        assert check_pr_body.GUIDE in TEMPLATE


class TestTheDiff:
    @pytest.mark.parametrize(
        ("bad", "good"),
        [
            ("x = f()  # noqa", "x = f()  # noqa: BLE001 - reported, not raised"),
            (
                "x = f()  # noqa: BLE001",
                "x = f()  # noqa: E402,F401 - imported for its side effect",
            ),
            ("x: int = f()  # type: ignore", "x: int = f()  # type: ignore[assignment]"),
            ("if gpu:  # pragma: no cover", "if gpu:  # pragma: no cover - no GPU in CI"),
            ("# TODO handle the empty corpus", "# TODO(#12) handle the empty corpus"),
            ("# FIXME: slow", "# FIXME(#7): slow"),
            ("    pytest.skip()", '    pytest.skip("needs the bge weights")'),
            ("@pytest.mark.skip", '@pytest.mark.skip(reason="flaky upstream, see #9")'),
            (
                "try: f()\nexcept OSError: pass",
                "try: f()\nexcept OSError: pass  # a missing cache is fine",
            ),
            (
                "except KeyError:\n    pass",
                "except KeyError:  # an unknown key keeps the default\n    pass",
            ),
        ],
    )
    def test_each_rule_fails_its_bad_line_and_passes_the_fixed_one(self, bad, good):
        def run(text):
            diff = _with_tests_and_changelog(_diff("src/lasagna/x.py", text.split("\n")))
            return check_pr_body.diff_problems(diff)

        assert run(bad), bad
        assert run(good) == [], good

    def test_an_except_that_handles_the_error_passes(self):
        lines = ["try:", "    f()", "except OSError:", "    log.warning('no cache')"]
        diff = _with_tests_and_changelog(_diff("src/lasagna/x.py", lines))
        assert check_pr_body.diff_problems(diff) == []

    def test_only_added_lines_are_read(self):
        diff = (
            "diff --git a/src/lasagna/x.py b/src/lasagna/x.py\n"
            "--- a/src/lasagna/x.py\n+++ b/src/lasagna/x.py\n"
            "@@ -1,3 +1,3 @@\n"
            " x = f()  # noqa\n"
            "-y = 1\n"
            "+y = 2\n"
            " except OSError:\n"
        )
        assert check_pr_body.diff_problems(_with_tests_and_changelog(diff)) == []

    def test_the_line_number_is_the_new_files(self):
        diff = (
            "diff --git a/src/lasagna/x.py b/src/lasagna/x.py\n"
            "--- a/src/lasagna/x.py\n+++ b/src/lasagna/x.py\n"
            "@@ -10,2 +40,3 @@\n"
            " a = 1\n"
            "+b = g()  # noqa\n"
            " c = 3\n"
        )
        found = check_pr_body.diff_problems(_with_tests_and_changelog(diff))
        assert found and found[0].startswith("src/lasagna/x.py:41:")

    def test_a_new_module_opens_with_a_docstring(self):
        bare = _diff("src/lasagna/new.py", ["from __future__ import annotations"], new=True)
        found = check_pr_body.diff_problems(_with_tests_and_changelog(bare))
        assert any("docstring" in p for p in found)
        documented = _diff(
            "src/lasagna/new.py",
            ["# Licensed under Apache 2.0.", '"""What this module is for."""'],
            new=True,
        )
        assert check_pr_body.diff_problems(_with_tests_and_changelog(documented)) == []

    def test_an_edited_module_is_not_asked_for_a_docstring(self):
        diff = _with_tests_and_changelog(_diff("src/lasagna/old.py", ["x = 1"]))
        assert check_pr_body.diff_problems(diff) == []

    def test_code_needs_a_changelog_line_and_a_test(self):
        code = _diff("src/lasagna/x.py", ["x = 1"])
        found = "\n".join(check_pr_body.diff_problems(code))
        assert "CHANGELOG.md did not" in found and "nothing under tests/ did" in found

    def test_the_labels_waive_the_changelog_and_the_test(self):
        code = _diff("src/lasagna/x.py", ["x = 1"])
        assert check_pr_body.diff_problems(code, {"no-changelog", "no-new-test"}) == []
        only_changelog = check_pr_body.diff_problems(code, {"no-changelog"})
        assert len(only_changelog) == 1 and "tests/" in only_changelog[0]

    def test_docs_and_tests_alone_need_neither(self):
        diff = _diff("docs/guide.md", ["A sentence."]) + _diff("tests/test_x.py", ["x = 1"])
        assert check_pr_body.diff_problems(diff) == []

    def test_the_rules_apply_to_tests_and_scripts_too(self):
        diff = _diff("tests/test_x.py", ["x = f()  # noqa"]) + _diff("scripts/y.py", ["# TODO"])
        assert len(check_pr_body.diff_problems(diff)) == 2

    def test_the_checker_and_this_file_are_exempt(self):
        """They spell out the patterns they reject."""
        diff = _diff("scripts/check_pr_body.py", ["# TODO"]) + _diff(
            "tests/test_pr_body_check.py", ["x = f()  # noqa"]
        )
        assert check_pr_body.diff_problems(diff) == []


class TestTheCommandAndTheWorkflow:
    def test_the_command_reads_a_file_and_says_what_is_missing(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(TEMPLATE, encoding="utf-8")
        script = str(REPO_ROOT / "scripts" / "check_pr_body.py")
        failed = subprocess.run(
            [sys.executable, script, "--file", str(body)], capture_output=True, text=True
        )
        assert failed.returncode == 1
        assert "::error::" in failed.stdout and check_pr_body.GUIDE in failed.stdout
        body.write_text(FILLED, encoding="utf-8")
        passed = subprocess.run(
            [sys.executable, script, "--file", str(body)], capture_output=True, text=True
        )
        assert passed.returncode == 0, passed.stdout

    def test_the_workflow_runs_the_check_on_every_change_to_the_pull_request(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML reads the key `on` as True.
        triggers = workflow[True]
        assert set(triggers) == {"pull_request"}, "never pull_request_target"
        assert {"opened", "edited", "synchronize", "labeled", "unlabeled"} <= set(
            triggers["pull_request"]["types"]
        )
        assert workflow["permissions"] == {"contents": "read"}
        (job,) = workflow["jobs"].values()
        run = next(step["run"] for step in job["steps"] if "run" in step)
        assert "scripts/check_pr_body.py --diff" in run and "${{" not in run

    def test_the_waiver_labels_are_named_in_contributing(self):
        contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "## Quality bar" in contributing
        for label in ("no-changelog", "no-new-test"):
            assert f"`{label}`" in contributing, label
