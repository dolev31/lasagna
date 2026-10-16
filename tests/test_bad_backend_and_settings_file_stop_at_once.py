"""A misspelled vector backend or a missing settings file stops the command at once.

`RAG_VECTOR_BACKEND=milvuss lasagna benchmark run smoke` ran, failed its task,
and advised "point RAG_DB_DIR at a new directory". `RAG_SETTINGS_FILE=/nope.toml`
only logged a warning (hidden at the CLI's level) and ran on the packaged
settings, a configuration other than the one named.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR


@pytest.fixture
def lasagna(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="300",
    )

    def run(*argv, **extra):
        return subprocess.run(
            [sys.executable, "-m", "lasagna", *argv],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=300,
        )

    return run


@pytest.mark.parametrize(
    "argv",
    [
        ("benchmark", "run", "smoke", "--profile", "speed"),
        ("search", "anything"),
        ("upload", "README.md"),
    ],
)
def test_a_misspelled_backend_is_named_before_anything_runs(lasagna, tmp_path, argv):
    (tmp_path / "README.md").write_text("# notes\n")
    run = lasagna(*argv, RAG_VECTOR_BACKEND="milvuss")
    out = run.stdout + run.stderr
    assert run.returncode == 1, out[-2000:]
    assert "unknown backend 'milvuss' (did you mean 'milvus'?)" in out
    assert "RAG_DB_DIR" not in out and "Running benchmark" not in out


def test_a_settings_file_that_does_not_exist_is_an_error(lasagna, tmp_path):
    missing = tmp_path / "nope.toml"
    run = lasagna("status", RAG_SETTINGS_FILE=str(missing))
    out = run.stdout + run.stderr
    assert run.returncode == 2, out[-2000:]
    assert f"RAG_SETTINGS_FILE is {missing}, and no file is there" in out
    assert "Configuration" not in out  # nothing ran on the packaged settings
