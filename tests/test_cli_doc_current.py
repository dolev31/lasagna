"""docs/reference/cli.md is written from the Typer app, and this fails when it drifts.

``scripts/gen_cli_doc.py`` reads every command, option, default and help string
from ``lasagna.cli``. Add or change one without regenerating the page and this
names the script to run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_page_matches_the_command_line(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_cli_doc.py"), "--check"],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_page_has_no_em_dash():
    assert "—" not in (REPO / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")


def _options_named(command, wanted):
    """The options named in ``wanted``, found anywhere in the command tree."""
    found = {}

    def walk(cmd):
        for param in cmd.params:
            for opt in getattr(param, "opts", ()):
                if opt in wanted and opt not in found:
                    found[opt] = param
        for sub in getattr(cmd, "commands", {}).values():
            walk(sub)

    walk(command)
    return found


def test_option_type_labels_do_not_depend_on_the_typer_version(monkeypatch):
    """The page says "text", whichever Typer is installed (#35).

    Typer 0.27 names its string type ``str`` and Typer 0.26 names the same class
    ``text``, so a label taken from the type's name rendered the page two ways.
    Read the labels from the real app, not from a hand-built parameter, so the
    test follows the Typer that is actually installed.
    """
    import typer.main
    from lasagna import cli as cli_module

    # scripts/ is not a package, so the path entry is the only way in; pytest
    # takes it back out when the test ends.
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import gen_cli_doc  # noqa: PLC0415 - scripts/ is a flat directory, not a package

    options = _options_named(
        typer.main.get_command(cli_module.app),
        {"--output", "--session", "--diff", "--top-k"},
    )
    assert {"--output", "--diff", "--top-k"} <= set(options), sorted(options)

    expected = {
        "--output": "text",
        "--session": "text",
        "--diff": "<text text>",
        # `lasagna search --top-k` is an integer option: the fix must not turn
        # every type into the string one.
        "--top-k": "int",
    }
    for opt, label in expected.items():
        assert gen_cli_doc._type(options[opt]) == label, opt
