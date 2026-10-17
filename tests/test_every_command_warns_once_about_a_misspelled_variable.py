"""Every command warns once about an environment variable that names no setting.

`RAG_CHUNK_SISE=500 lasagna status` printed nothing, while doctor, configure
and search warned: only a command that loaded the configuration checked the
environment. The CLI now checks it for every command, once, on stderr, without
loading the configuration or creating its folders.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

WARNING = "RAG_CHUNK_SISE is not a Lasagna setting and was ignored. Did you mean RAG_CHUNK_SIZE?"


def _run(tmp_path, *args, **env):
    environment = {**os.environ, "HOME": str(tmp_path), **env}
    environment.pop("LASAGNA_DATA_DIR", None)
    return subprocess.run(
        [sys.executable, "-m", "lasagna", *args],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
        timeout=120,
    )


@pytest.mark.parametrize(
    "args",
    [["status"], ["component", "list"], ["benchmark", "list"], ["results", "schema"]],
    ids=["status", "component-list", "benchmark-list", "results-schema"],
)
def test_the_command_warns_once_on_stderr(tmp_path, args):
    done = _run(tmp_path, *args, RAG_CHUNK_SISE="500")
    assert done.stderr.count(WARNING) == 1, done.stderr
    assert WARNING not in done.stdout


def test_a_real_setting_is_not_warned_about(tmp_path):
    done = _run(tmp_path, "component", "list", RAG_CHUNK_SIZE="500")
    assert done.returncode == 0, done.stderr
    assert "is not a Lasagna setting" not in done.stderr


def test_the_check_loads_no_configuration_and_makes_no_folder(tmp_path, monkeypatch):
    from lasagna.core import config

    monkeypatch.setenv("RAG_CHUNK_SISE", "500")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_warned_unknown_env", set())
    warned = []
    monkeypatch.setattr(config.logger, "warning", lambda message: warned.append(message))
    before = config._rag_config
    config.warn_unknown_environment()
    assert config._rag_config is before
    assert list(tmp_path.iterdir()) == []
    assert [m for m in warned if "RAG_CHUNK_SISE" in m] == [
        f"{WARNING} The settings are listed in docs/reference/settings.md."
    ]
