"""The image installs the extras ``IMAGE_EXTRAS`` names, beyond the production set.

Every pull request is tested in the production environment, the Compose stack
built from its branch (CONTRIBUTING.md, "Tested in the production environment").
The image installed only the serving extras, so ``lasagna benchmark`` and a
sweep, which need the ``eval`` extra, could not run inside the stack, and a
change to the evaluation harness could not be tested there. ``IMAGE_EXTRAS``
adds extras at build time; it is empty in production.

The install step is executed here with a stand-in ``pip``, so the test checks
what the shell does with each value, not only the text.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "src" / "lasagna" / "deploy"
DOCKERFILE = (DEPLOY_DIR / "Dockerfile").read_text()


def _install(tmp_path: Path, backend: str = "milvus", extras: str = "") -> tuple:
    """(exit code, the extras spec pip was given, stderr+stdout) for one build."""
    match = re.search(r'RUN (case "\$VECTOR_BACKEND" in .*?"gunicorn>=22\.0\.0")', DOCKERFILE, re.S)
    assert match, "the Dockerfile has no install step keyed on VECTOR_BACKEND"
    script = match.group(1)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "pip-calls"
    pip = bin_dir / "pip"
    pip.write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > {calls}\n')
    pip.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", script],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "VECTOR_BACKEND": backend,
            "IMAGE_EXTRAS": extras,
        },
        capture_output=True,
        text=True,
    )
    spec = ""
    if calls.exists():
        spec = next((a for a in calls.read_text().split("\n") if a.startswith(".[")), "")
    return result.returncode, spec, result.stdout + result.stderr


def test_production_installs_only_the_production_set(tmp_path):
    code, spec, out = _install(tmp_path)
    assert code == 0, out
    assert spec == ".[production,pgvector,opensearch,flag-embedding]"


@pytest.mark.parametrize(
    ("backend", "extras", "spec"),
    [
        ("milvus", "eval", ".[production,pgvector,opensearch,flag-embedding,eval]"),
        (
            "milvus",
            "eval,providers",
            ".[production,pgvector,opensearch,flag-embedding,eval,providers]",
        ),
        ("qdrant", "eval", ".[production,pgvector,opensearch,flag-embedding,vector-backends,eval]"),
    ],
)
def test_image_extras_are_added_to_the_install(tmp_path, backend, extras, spec):
    code, got, out = _install(tmp_path, backend, extras)
    assert code == 0, out
    assert got == spec


@pytest.mark.parametrize("extras", ["eval]", "eval; rm -rf /", "eval providers", "EVAL", "$(id)"])
def test_anything_but_extra_names_fails_the_build(tmp_path, extras):
    code, spec, out = _install(tmp_path, "milvus", extras)
    assert code != 0
    assert spec == "", "pip ran with a value the build should have refused"
    assert "give extras names separated by commas" in out


def test_compose_passes_image_extras_from_env_and_defaults_to_empty():
    compose = yaml.safe_load((DEPLOY_DIR / "docker-compose.yml").read_text())
    args = compose["x-build"]["args"]
    assert args["IMAGE_EXTRAS"] == "${IMAGE_EXTRAS:-}"
    assert 'ARG IMAGE_EXTRAS=""' in DOCKERFILE
    assert "# IMAGE_EXTRAS=" in (DEPLOY_DIR / ".env.example").read_text()


def test_the_image_records_its_extras():
    assert 'LABEL org.lasagna.image-extras="$IMAGE_EXTRAS"' in DOCKERFILE
