"""The deploy README says what the Quick start needs, and the deploy files use plain punctuation.

The Quick start ran `docker compose`, `make`, `openssl` and `sed -i.bak` with no
"Requires" line, and the first pointer to `lasagna serve` for a reader without
Docker was a third of the way down. The compose file's comments also used " -- "
as a dash, which the house style replaces with a comma, a colon or parentheses.
"""

from __future__ import annotations

import re

from tests._paths import PACKAGE_DIR

DEPLOY = PACKAGE_DIR / "deploy"


def _quick_start() -> str:
    text = (DEPLOY / "README.md").read_text(encoding="utf-8")
    start = text.index("## Quick start")
    return text[start : text.index("\n## ", start + 1)]


def test_the_quick_start_opens_with_what_it_requires():
    section = _quick_start()
    before_commands = section.split("```", 1)[0]
    for tool in ("Docker", "Compose v2", "make", "openssl"):
        assert tool in before_commands, before_commands


def test_the_quick_start_points_to_lasagna_serve_for_readers_without_docker():
    before_commands = _quick_start().split("```", 1)[0]
    assert re.search(r"No Docker\?.*lasagna serve", before_commands, re.S), before_commands
    assert "](../../../README.md#rest-api)" in before_commands


def test_the_deploy_files_use_no_spaced_double_hyphen_as_a_dash():
    found = []
    for name in ("README.md", "docker-compose.yml"):
        for number, line in enumerate((DEPLOY / name).read_text(encoding="utf-8").splitlines(), 1):
            if " -- " in line:
                found.append(f"{name}:{number}: {line.strip()}")
    assert not found, "\n".join(found)
