"""The dev container sets up what CONTRIBUTING.md asks for, Docker included.

A Codespace opened from `.devcontainer/devcontainer.json` is the free way to
meet the production-environment rule when a machine has no Docker daemon, so
it must have Docker, and its install must be the one CONTRIBUTING.md
documents: a container that installs something else tests something else. It
installs under the release's constraints file, with CPU-only torch first: from
PyPI on Linux x86_64, torch brings several GB of CUDA libraries a Codespace has
no GPU for, on the disk the Compose stack also needs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEVCONTAINER = REPO / ".devcontainer"


def _config() -> dict:
    """devcontainer.json is JSON with comments.

    Only whole-line `//` comments are stripped: a trailing comment or a
    `/* */` block would make this fail with a JSON error, so keep them out.
    """
    text = (DEVCONTAINER / "devcontainer.json").read_text(encoding="utf-8")
    return json.loads(re.sub(r"(?m)^\s*//.*$", "", text))


def _setup() -> str:
    command = _config()["postCreateCommand"]
    assert command == "sh .devcontainer/post-create.sh", command
    return (DEVCONTAINER / "post-create.sh").read_text(encoding="utf-8")


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
    install = re.search(r'uv pip install --system -e "(\.\[[a-z,-]+\])" (--group \w+) -c', _setup())
    assert install, "post-create.sh no longer runs the constrained editable install"
    assert install.groups() == (extras, group)


def test_it_installs_cpu_torch_at_the_pinned_version_first():
    setup = _setup()
    cpu = setup.index("--index-url https://download.pytorch.org/whl/cpu")
    assert cpu < setup.index('-e ".[')
    assert "grep -E '^torch==' \"$constraints\"" in setup


def test_it_installs_the_pre_commit_hooks():
    assert _setup().rstrip().endswith("pre-commit install")


def test_it_uses_the_python_the_constraints_are_tested_on():
    python = _config()["features"]["ghcr.io/devcontainers/features/python:1"]["version"]
    version = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)[1]
    header = (REPO / "constraints" / f"{version}.txt").read_text(encoding="utf-8").splitlines()[0]
    assert f"Python {python}" in header, header
