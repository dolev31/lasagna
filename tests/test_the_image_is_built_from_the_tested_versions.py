"""The image installs the versions the release was tested with, and no service runs :latest.

The Dockerfile ignored ``constraints/<version>.txt``: each build took whatever
torch, transformers and sentence-transformers were newest that day, so two
builds of one release could compute different numbers. The jaeger service ran
``jaegertracing/all-in-one:latest``, which changes under the stack without a
commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests._paths import DEPLOY_DIR

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = (DEPLOY_DIR / "Dockerfile").read_text()


def _version() -> str:
    return re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)[1]


def _run_commands() -> list:
    """Each RUN instruction, continuation lines joined."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    return [line for line in joined.splitlines() if line.startswith("RUN ")]


def test_the_constraints_named_are_this_releases_and_exist():
    named = re.findall(r"^ARG CONSTRAINTS=(\S+)$", DOCKERFILE, re.M)
    assert named, "the Dockerfile names no constraints file"
    assert set(named) == {f"constraints/{_version()}.txt"}
    assert (REPO / named[0]).is_file()


def test_every_install_of_the_models_or_the_package_is_constrained():
    installs = [
        cmd
        for cmd in _run_commands()
        if "pip install" in cmd and ("sentence-transformers" in cmd or '".[' in cmd)
    ]
    assert len(installs) == 2, installs
    for cmd in installs:
        assert re.search(r"pip install --no-cache-dir -c \S*constraints\.txt", cmd), cmd


def test_no_install_overrides_a_tested_version():
    # setuptools<78 was pinned for milvus-lite's pkg_resources; the tested
    # milvus-lite needs no pkg_resources, and the pin contradicted setuptools==84.
    assert "setuptools<" not in DOCKERFILE


def test_every_service_image_has_a_version_tag():
    files = [DEPLOY_DIR / "docker-compose.yml", *sorted((DEPLOY_DIR / "backends").glob("*.yml"))]
    for path in files:
        services = (yaml.safe_load(path.read_text()) or {}).get("services") or {}
        for name, service in services.items():
            image = (service or {}).get("image")
            if not image:
                continue
            tag = image.rsplit(":", 1)[1] if ":" in image.split("/")[-1] else ""
            assert tag and tag != "latest", f"{path.name}: {name} runs {image}"
