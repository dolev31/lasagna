"""docs/reference/settings.md lists every setting, and is regenerated with them.

The settings were documented by hand in a table that had fallen behind: 43
settings the configuration reads appeared nowhere. The page is now written by
``scripts/gen_settings_doc.py`` from the defaults, and this fails when a setting
is added, removed or re-described without regenerating it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "reference" / "settings.md"


def test_the_page_matches_the_settings(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_settings_doc.py"), "--check"],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_every_setting_the_configuration_reads_is_on_the_page():
    from lasagna.core import config as config_module
    from lasagna.core.config import RAGConfig

    listed = set(re.findall(r"^\| `([a-z0-9_]+)` \|", PAGE.read_text(), re.M))
    settings = (set(RAGConfig()._config) | set(config_module._PROPERTY_ONLY_SETTINGS)) - {
        "upload_dir",
        "db_path",
        "citation_db_path",
        "lifecycle",
    }
    assert not settings - listed, sorted(settings - listed)
    assert not listed - settings, sorted(listed - settings)


def test_every_row_has_a_description():
    rows = re.findall(r"^\| `[a-z0-9_]+` \|.*\|\s*(.*?)\s*\|$", PAGE.read_text(), re.M)
    assert rows and all(rows), "a setting has no description; add a comment beside its default"
