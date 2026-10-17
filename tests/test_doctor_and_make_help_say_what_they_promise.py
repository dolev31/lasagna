"""`doctor --issue` prints no absolute path, and the deploy Makefile describes itself truly.

- `lasagna doctor` promises "no absolute paths", and redacted only the home
  directory and the temp directory: a project outside `$HOME` printed its
  settings file's full path in the block meant for a public issue.
- After the deploy quick start's `cp .env.example .env`, `make` listed every
  target as "Makefile" (grep named the file once .env was included), and
  `validate-k8s` was never listed (the pattern allowed no digits).
- The Makefile's header said `VECTOR_BACKEND=pgvector make up` overrides the
  backend; while `.env` sets it, only `make up VECTOR_BACKEND=pgvector` does,
  which is what the deploy README says.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR

DEPLOY = PACKAGE_DIR / "deploy"

#: An absolute path with a directory part, outside a URL.
_ABSOLUTE = re.compile(r"(?<![\w:/>.~<-])/(?:[\w.@-]+/)+[\w.@-]*")


@pytest.fixture
def outside_home(tmp_path):
    """A project and a data directory, neither under HOME nor the temp directory."""
    project = tmp_path / "srv" / "project"
    project.mkdir(parents=True)
    settings = project / "settings.lasagna.toml"
    settings.write_text('[rag]\nvector_backend = "milvus"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="200",
        RAG_SETTINGS_FILE=str(settings),
        LASAGNA_DATA_DIR=str(tmp_path / "srv" / "lasagna-data"),
        # The temp directory is redacted on its own; keep it out of the paths.
        TMPDIR=str(tmp_path / "elsewhere"),
    )
    (tmp_path / "elsewhere").mkdir()
    return project, env


@pytest.mark.parametrize("flag", ["--issue", "--json", None])
def test_doctor_prints_no_absolute_path(outside_home, flag):
    project, env = outside_home
    args = [sys.executable, "-m", "lasagna", "doctor", *([flag] if flag else [])]
    done = subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, timeout=300)
    out = done.stdout
    assert "settings" in out, out + done.stderr[-2000:]
    assert str(project.parent) not in out, out
    leaked = _ABSOLUTE.findall(out)
    assert not leaked, leaked
    assert "./settings.lasagna.toml" in out


def _make(tmp_path, *args: str, **env: str) -> str:
    deploy = tmp_path / "deploy"
    if not deploy.exists():
        shutil.copytree(DEPLOY, deploy, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(deploy / ".env.example", deploy / ".env")
    done = subprocess.run(
        ["make", *args],
        cwd=deploy,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return re.sub(r"\x1b\[[0-9;]*m", "", done.stdout + done.stderr)


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_help_lists_every_target_by_name_with_dot_env(tmp_path):
    listed = {line.split()[0] for line in _make(tmp_path, "help").splitlines() if line.strip()}
    declared = set(
        re.findall(r"^([a-zA-Z0-9_-]+):.*## ", (DEPLOY / "Makefile").read_text(), flags=re.M)
    )
    assert "validate-k8s" in declared
    assert listed == declared, listed ^ declared


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_header_names_the_override_that_works(tmp_path):
    header = (DEPLOY / "Makefile").read_text().split(".DEFAULT_GOAL")[0]
    assert "VECTOR_BACKEND=pgvector make up" not in header
    assert "make up VECTOR_BACKEND=pgvector" in header
    assert "backends/pgvector.yml" in _make(tmp_path, "-n", "down", "VECTOR_BACKEND=pgvector")
    # As an environment variable it loses to .env, as the header and README say.
    assert "backends/pgvector.yml" not in _make(tmp_path, "-n", "down", VECTOR_BACKEND="pgvector")
    readme = " ".join((DEPLOY / "README.md").read_text().split())
    assert "(`VECTOR_BACKEND=qdrant make up`) has no effect while `.env` sets it" in readme
