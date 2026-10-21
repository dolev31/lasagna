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
