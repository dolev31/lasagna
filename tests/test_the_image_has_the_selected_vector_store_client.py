"""The Docker image carries the client library of the vector store it is built for.

The Dockerfile installed ``.[production,pgvector,opensearch,flag-embedding]``:
no ``qdrant-client``, ``weaviate-client`` or ``pinecone``. So the stack built
with ``backends/qdrant.yml`` (the CI ``live-compose`` job's choice), and the
weaviate and pinecone fragments, started an api that could not load its
backend and never passed its readiness check.

The Dockerfile's ``VECTOR_BACKEND`` build argument adds the
``vector-backends`` extra, those three fragments set it, the build fails if
the client does not import, and ``make validate`` refuses an image built for
another backend (``up`` reuses an image that exists).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "src" / "lasagna" / "deploy"
NEEDS_CLIENT = {"qdrant": "qdrant_client", "weaviate": "weaviate", "pinecone": "pinecone"}
SERVICES = ("api", "celery-worker", "celery-beat", "mcp")
SHIPPED = sorted(p.stem for p in (DEPLOY_DIR / "backends").glob("*.yml"))


def _install_choice(backend: str) -> tuple:
    """(extra, modules checked) the Dockerfile's install step picks for ``backend``."""
    text = (DEPLOY_DIR / "Dockerfile").read_text()
    match = re.search(r'RUN (case "\$VECTOR_BACKEND" in .*?esac)', text, re.S)
    assert match, "the Dockerfile has no install step keyed on VECTOR_BACKEND"
    script = match.group(1).replace("\\\n", "\n")
    result = subprocess.run(
        ["sh", "-c", script + ' && printf "%s|%s" "$extra" "$check"'],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "VECTOR_BACKEND": backend},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    extra, check = result.stdout.split("|")
    return extra, check.split()


@pytest.mark.parametrize("backend", SHIPPED)
def test_every_shipped_backend_names_a_known_install(backend):
    extra, check = _install_choice(backend)
    if backend in NEEDS_CLIENT:
        assert extra == ",vector-backends"
        assert NEEDS_CLIENT[backend] in check
    else:
        assert extra == ""


def test_an_unknown_backend_fails_the_build():
    text = (DEPLOY_DIR / "Dockerfile").read_text()
    assert "Unknown VECTOR_BACKEND=" in text


@pytest.mark.parametrize("backend", sorted(NEEDS_CLIENT))
def test_the_fragment_builds_every_service_with_its_client(backend):
    fragment = yaml.safe_load((DEPLOY_DIR / "backends" / f"{backend}.yml").read_text())
    for name in SERVICES:
        args = fragment["services"][name].get("build", {}).get("args", {})
        assert args.get("VECTOR_BACKEND") == backend, f"{backend}/{name}: {args}"


def test_the_image_records_what_it_was_built_for():
    text = (DEPLOY_DIR / "Dockerfile").read_text()
    assert 'LABEL org.lasagna.vector-backend="$VECTOR_BACKEND"' in text


# -- make validate and an image that already exists ---------------------------

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def _validate(tmp_path, backend, built_for):
    """`make validate` with a stand-in `docker` reporting an image built for ``built_for``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        '  "compose -f") echo deploy-api; echo redis:7-alpine ;;\n'
        '  "image inspect") [ -n "$BUILT_FOR" ] || exit 1; echo "$BUILT_FOR" ;;\n'
        "esac\n"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return subprocess.run(
        [
            "make",
            "-s",
            "-C",
            str(DEPLOY_DIR),
            "validate",
            f"VECTOR_BACKEND={backend}",
            "RAG_JWT_SECRET=x",
            "POSTGRES_PASSWORD=y",
            "COMPOSE_PROFILES=",
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "BUILT_FOR": built_for,
        },
    )


@needs_make
@pytest.mark.parametrize("built_for", ["milvus", "<no value>"])
def test_validate_refuses_an_image_without_the_client(tmp_path, built_for):
    result = _validate(tmp_path, "qdrant", built_for)
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "deploy-api has no qdrant client" in out
    assert "make build VECTOR_BACKEND=qdrant" in out


@needs_make
@pytest.mark.parametrize(
    "backend, built_for",
    [
        ("qdrant", "qdrant"),
        ("weaviate", "all"),
        ("weaviate", "qdrant"),  # the extra installs all three clients
        ("qdrant", ""),
        ("pgvector", "milvus"),
    ],
)
def test_validate_accepts_a_matching_or_missing_image(tmp_path, backend, built_for):
    result = _validate(tmp_path, backend, built_for)
    assert result.returncode == 0, result.stdout + result.stderr
