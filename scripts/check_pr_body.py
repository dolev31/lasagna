"""Refuse a pull request whose description or diff is below the bar.

    python scripts/check_pr_body.py --file body.md --diff origin/main

Run by `.github/workflows/pr-quality.yml` on every pull request that is not a
draft, and by hand before you open one. The bar is written out in
CONTRIBUTING.md under "Quality bar"; this script checks the part a machine can
check, so a reviewer's time goes to the part it cannot.

Checks the description (from `--file`, `-` for stdin, or `$PR_BODY`):

- every section of `.github/PULL_REQUEST_TEMPLATE.md` is there and filled in,
  not left as the template's comments or a placeholder such as "TBD";
- "Agent use" says what wrote the change and names, by GitHub handle, the
  person who read every line of it;
- "Evidence" is a result, or a line that starts "No number can move:" and
  gives the reason;
- "How it was tested" names a `docker compose` command, since every change is
  run in the production environment, the Compose stack built from its branch;
- every box in the checklist is ticked.

With `--diff BASE`, checks the lines `git diff BASE...HEAD` adds:

- a `noqa` names its code and gives a reason (`# noqa: BLE001 - why`);
- a `type: ignore` names its code (`# type: ignore[arg-type]`);
- a `pragma: no cover` gives a reason;
- a TODO or FIXME links its issue (`TODO(#12)`);
- a skipped test says why it is skipped;
- an `except` whose body is only `pass` carries a comment that says why the
  error is safe to ignore, or handles it;
- a new module under `src/lasagna/` opens with a docstring;
- a change to code under `src/lasagna/` comes with a CHANGELOG.md line and a
  change under `tests/`, unless a maintainer added the `no-changelog` or
  `no-new-test` label.

Only added lines are checked, so code that was already there never fails a
pull request that did not touch it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
GUIDE = "https://github.com/dolev31/lasagna/blob/main/CONTRIBUTING.md#quality-bar"

#: The template's headings, in order. The test suite checks the two agree.
SECTIONS = ("Why", "What changed", "How it was tested", "Evidence", "Agent use", "Checklist")

#: A fragment of each checklist line that must be there, ticked.
REQUIRED_BOXES = (
    "one logical change",
    "scripts/ci-local.sh",
    "names its code and gives a reason",
    "CHANGELOG.md",
    "Docker Compose stack",
    "I read every line of the diff",
)

#: A Compose command, the proof a change ran in the production environment.
_COMPOSE = re.compile(r"docker[ -]compose\b", re.IGNORECASE)

#: What a section holds when nobody filled it in.
PLACEHOLDERS = {"todo", "tbd", "wip", "n/a", "na", "none", "-", "...", "see diff", "see title"}

#: A "Why" shorter than this is a title, not a reason.
MIN_WHY_WORDS = 5

#: The checker and its tests spell out the patterns they reject.
EXEMPT = ("scripts/check_pr_body.py", "tests/test_pr_body_check.py")

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)
_BOX = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+)$", re.M)
_HANDLE = re.compile(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")

_NOQA = re.compile(r"#\s*noqa\b(?P<rest>.*)", re.I)
_NOQA_OK = re.compile(r"^:\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*\s+-\s+\S")
_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?!\[)")
_PRAGMA = re.compile(r"#\s*pragma:\s*no\s+cover(?!\s+-\s+\S)")
_TODO = re.compile(r"\b(?:TODO|FIXME)\b(?!\(#\d+\))")
_EMPTY_SKIP = re.compile(r"pytest\.skip\(\s*\)|pytest\.mark\.skip(?:\(\s*\))?\s*(?:#.*)?$")
_EXCEPT = re.compile(r"^\s*except\b[^#]*:\s*(?P<rest>[^#]*)")
_PASS = re.compile(r"^\s*pass\s*$")
_DOCSTRING_START = re.compile(r"^[rRbBuU]?(\"\"\"|''')")


def _sections(body: str) -> dict[str, str]:
    """Each `## ` heading's text, with the template's comments taken out."""
    text = _COMMENT.sub("", body.replace("\r\n", "\n"))
    heads = list(_HEADING.finditer(text))
    out = {}
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[head.group(1)] = text[head.end() : end].strip()
    return out


def _field(section: str, name: str) -> str | None:
    """What follows `name:` on its line, or None when the line is missing."""
    match = re.search(rf"^\s*{re.escape(name)}:[ \t]*(.*)$", section, re.M)
    return match.group(1).strip() if match else None


def _is_placeholder(text: str) -> bool:
    text = text.strip().lower()
    return text in PLACEHOLDERS or text.strip(".!") in PLACEHOLDERS | {""}


def problems(body: str) -> list[str]:
    """What the description is missing, one sentence each."""
    found = []
    sections = _sections(body or "")
    for name in SECTIONS:
        if name not in sections:
            found.append(f'the description has no "## {name}" section')
        elif not sections[name]:
            found.append(f'"## {name}" is empty')
        elif _is_placeholder(sections[name]):
            found.append(f'"## {name}" says only "{sections[name]}"')

    why = sections.get("Why", "")
    if why and not _is_placeholder(why) and len(why.split()) < MIN_WHY_WORDS:
        found.append('"## Why" is a few words; say what problem this solves')

    evidence = sections.get("Evidence", "")
    if evidence.lower().startswith("no number can move"):
        reason = evidence.split(":", 1)[1].strip() if ":" in evidence else ""
        if len(reason.split()) < 3:
            found.append('"## Evidence" says no number can move but not why')

    tested = sections.get("How it was tested", "")
    if tested and not _is_placeholder(tested) and not _COMPOSE.search(tested):
        found.append(
            '"## How it was tested" shows no `docker compose` command; run the change in '
            "the Compose stack built from this branch (CONTRIBUTING.md#tested-in-the-production-environment)"
        )

    agent = sections.get("Agent use", "")
    if agent:
        written = _field(agent, "Written with")
        reader = _field(agent, "Read in full by")
        if not written:
            found.append(
                '"## Agent use" needs "Written with:" filled in (the agent you used, or "none")'
            )
        if reader is None or not _HANDLE.search(reader):
            found.append(
                '"## Agent use" needs "Read in full by:" to name a person by GitHub @handle'
            )

    checklist = sections.get("Checklist", "")
    boxes = _BOX.findall(checklist)
    for fragment in REQUIRED_BOXES:
        match = [ticked for ticked, label in boxes if fragment in label]
        if not match:
            found.append(f'the checklist line with "{fragment}" was removed')
        elif not any(t in "xX" for t in match):
            found.append(f'the checklist line with "{fragment}" is not ticked')
    return found


def _files(diff_text: str) -> dict[str, dict]:
    """Each file in a unified diff: whether it is new, and its added lines."""
    files: dict[str, dict] = {}
    current = None
    in_header = False
    line_no = 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", raw)
            current = match.group(2) if match else None
            in_header = True
            if current:
                files[current] = {"new": False, "added": []}
        elif current is None:
            continue
        elif raw.startswith("@@"):
            in_header = False
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", raw)
            line_no = int(match.group(1)) if match else 0
        elif in_header:
            # "new file mode", "index", "--- a/...", "+++ b/...": not content.
            if raw.startswith("new file mode") or raw == "--- /dev/null":
                files[current]["new"] = True
        elif raw.startswith("+"):
            files[current]["added"].append((line_no, raw[1:]))
            line_no += 1
        elif raw.startswith(" "):
            line_no += 1
    return files


def _line_problems(path: str, added: list[tuple[int, str]]) -> list[str]:
    found = []
    by_line = dict(added)
    for n, text in added:
        where = f"{path}:{n}"
        noqa = _NOQA.search(text)
        if noqa and not _NOQA_OK.match(noqa.group("rest")):
            found.append(f"{where}: a noqa names its code and says why (# noqa: BLE001 - why)")
        if _TYPE_IGNORE.search(text):
            found.append(f"{where}: a type: ignore names its code (# type: ignore[arg-type])")
        if _PRAGMA.search(text):
            found.append(f"{where}: a pragma: no cover says why (# pragma: no cover - why)")
        if _TODO.search(text):
            found.append(f"{where}: a TODO or FIXME links its issue, as TODO(#12)")
        if _EMPTY_SKIP.search(text):
            found.append(f'{where}: a skipped test says why (reason="...")')
        handler = _EXCEPT.match(text)
        if handler and "#" not in text:
            rest = handler.group("rest").strip()
            if rest == "pass" or (not rest and _PASS.match(by_line.get(n + 1, "x"))):
                found.append(
                    f"{where}: an except whose body is only pass hides the error; "
                    "handle it, or say in a comment why it is safe to ignore"
                )
    return found


def _missing_docstring(added: list[tuple[int, str]]) -> bool:
    for _, text in added:
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return not _DOCSTRING_START.match(stripped)
    return False


def diff_problems(diff_text: str, labels: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """What the lines a pull request adds break, one sentence each."""
    found = []
    files = _files(diff_text)
    for path, info in files.items():
        if path in EXEMPT or not path.endswith(".py"):
            continue
        found += _line_problems(path, info["added"])
        if info["new"] and path.startswith("src/lasagna/") and _missing_docstring(info["added"]):
            found.append(f"{path}: a new module opens with a docstring that says what it is for")

    code = [p for p in files if p.startswith("src/lasagna/") and p.endswith(".py")]
    if code and "CHANGELOG.md" not in files and "no-changelog" not in labels:
        found.append(
            "the code under src/lasagna/ changed but CHANGELOG.md did not; add a line "
            "under Unreleased (a maintainer adds no-changelog when nothing a user sees changed)"
        )
    if code and not any(p.startswith("tests/") for p in files) and "no-new-test" not in labels:
        found.append(
            "the code under src/lasagna/ changed but nothing under tests/ did; add the "
            "test that fails without the change (a maintainer adds no-new-test for a pure refactor)"
        )
    return found


def _git_diff(base: str) -> str:
    return subprocess.run(
        ["git", "diff", "--no-color", "--no-ext-diff", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="the description, or - for stdin (default: $PR_BODY)")
    parser.add_argument("--diff", metavar="BASE", help="also check `git diff BASE...HEAD`")
    parser.add_argument("--labels", default="", help="the pull request's labels, comma-separated")
    args = parser.parse_args(argv)

    if args.file == "-":
        body = sys.stdin.read()
    elif args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    else:
        body = os.environ.get("PR_BODY", "")

    found = problems(body)
    if args.diff:
        labels = {label.strip() for label in args.labels.split(",") if label.strip()}
        found += diff_problems(_git_diff(args.diff), labels)

    for p in found:
        print(f"::error::{p}")
    if found:
        print(f"The bar, and how to meet it: {GUIDE}")
    else:
        print("the description and the diff meet the bar")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
