"""The dev container sets up what CONTRIBUTING.md asks for, Docker included.

A Codespace opened from `.devcontainer/devcontainer.json` is the free way to
meet the production-environment rule when a machine has no Docker daemon, so
it must have Docker, and its install must be the one CONTRIBUTING.md
documents: a container that installs something else tests something else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEVCONTAINER = REPO / ".devcontainer" / "devcontainer.json"


def _config() -> dict:
    """devcontainer.json is JSON with comments; drop the comment lines."""
    text = DEVCONTAINER.read_text(encoding="utf-8")
    return json.loads(re.sub(r"(?m)^\s*//.*$", "", text))


def test_it_runs_docker_for_the_compose_stack():
    features = _config()["features"]
    assert any(
        name.startswith("ghcr.io/devcontainers/features/docker-in-docker") for name in features
    )


def test_it_installs_what_contributing_documents():
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    documented = re.search(r'uv pip install -e "(\.\[[a-z,-]+\])" (--group \w+)', contributing)
    assert documented, "CONTRIBUTING.md no longer shows the uv install command"
    extras, group = documented.groups()
    command = _config()["postCreateCommand"]
    assert f"uv pip install --system -e '{extras}' {group}" in command


def test_it_uses_the_python_the_release_is_pinned_on():
    """constraints/<version>.txt pins the tested versions on Python 3.12."""
    python = _config()["features"]["ghcr.io/devcontainers/features/python:1"]
    assert python["version"] == "3.12"
