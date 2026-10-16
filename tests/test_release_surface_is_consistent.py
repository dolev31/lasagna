"""Small inconsistencies a reader of the first release would trip on.

`-p` meant `--port` on `serve` and `--profile` everywhere else. The Kubernetes
manifests still used the old project name, `rag-knowledge`. Shipped code
carried comments that read like tickets ("P1-3 FIX:", "CRITICAL FIX:"), which
say what was once wrong, not what the code does. CITATION.cff named
2026-08-18 as the release date of a release that has no date yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from lasagna import cli

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lasagna"


def _commands(group, prefix=""):
    for name, command in getattr(group, "commands", {}).items():
        yield f"{prefix}{name}", command
        yield from _commands(command, f"{prefix}{name} ")


def test_dash_p_means_profile_wherever_it_exists():
    root = typer.main.get_command(cli.app)
    wrong = [
        f"{name}: {param.opts}"
        for name, command in _commands(root)
        for param in getattr(command, "params", [])
        if "-p" in getattr(param, "opts", []) and "--profile" not in param.opts
    ]
    assert not wrong, wrong


def test_the_kubernetes_manifests_use_the_projects_name():
    stale = [
        str(p) for p in (SRC / "deploy" / "k8s").rglob("*.yaml") if "rag-knowledge" in p.read_text()
    ]
    assert not stale, stale


def test_no_shipped_comment_reads_like_a_ticket():
    ticket = re.compile(r"\b(P[0-9]-[0-9]+ FIX|CRITICAL FIX|(HIGH|MEDIUM|LOW) PRIORITY FIX)\b")
    found = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if ticket.search(line)
    ]
    assert not found, found


def test_the_citation_is_dated_with_the_changelog():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_release", ROOT / "scripts" / "check_release.py"
    )
    check_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_release)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
    date = check_release._changelog_date(changelog, version)
    released = check_release._cff_field((ROOT / "CITATION.cff").read_text(), "date-released")
    assert released == date, f"CITATION.cff says {released}, CHANGELOG.md says {date}"
