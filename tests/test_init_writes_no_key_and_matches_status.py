"""`lasagna init` says what `status` will show, and never writes an API key.

It wrote OPENAI_API_KEY to ./.env (mode 0644), then said "Lasagna reads no
other key from .env", and `doctor` reported no key. Its "Cloud API" answer
put "Embeddings: text-embedding-3-small" in the summary while `status` showed
all-MiniLM-L6-v2 (nothing read that answer: the profile chooses the model),
and the provider read "OpenAI (GPT-4o)", a model Lasagna no longer calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from lasagna.providers.llm_provider import SimpleLLMProvider
from tests._paths import PACKAGE_DIR

DRIVE = """
import sys
import questionary
from typer.testing import CliRunner

answers = iter(sys.argv[1:])

class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value

questionary.select = lambda *a, **k: _Answer(next(answers))
questionary.password = lambda *a, **k: _Answer("sk-typed-into-the-wizard")

from lasagna import cli

result = CliRunner().invoke(cli.app, ["init"], env={"COLUMNS": "200"})
print(result.output)
sys.exit(result.exit_code)
"""


@pytest.fixture
def run(tmp_path):
    (tmp_path / "drive.py").write_text(DRIVE)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("RAG_", "LASAGNA_")) and not k.endswith("_API_KEY")
    }
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="200",
    )

    def _run(*argv, **extra):
        done = subprocess.run(
            [sys.executable, *argv],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=300,
        )
        return done

    return _run


def _status_embedding(run, **env) -> str:
    status = run("-m", "lasagna", "status", **env)
    assert status.returncode == 0, status.stdout + status.stderr[-2000:]
    line = next(line for line in status.stdout.splitlines() if "Embedding:" in line)
    return line.split("Embedding:")[1].strip(" │")


def test_init_writes_no_key_and_tells_which_to_export(run, tmp_path):
    init = run("drive.py", "milvus", "speed")
    assert init.returncode == 0, init.stdout + init.stderr[-2000:]
    env_file = (tmp_path / ".env").read_text()
    assert "API_KEY" not in env_file and "sk-typed" not in env_file
    assert env_file.startswith("RAG_SETTINGS_FILE=")
    assert "export OPENAI_API_KEY=<your key>" in init.stdout
    assert "GPT-4o" not in init.stdout
    assert "Embeddings: all-MiniLM-L6-v2" in init.stdout
    assert _status_embedding(run) == "all-MiniLM-L6-v2"


def test_with_the_key_exported_init_and_doctor_both_see_it(run):
    init = run("drive.py", "milvus", "standard", OPENAI_API_KEY="sk-from-the-shell")
    assert init.returncode == 0, init.stdout + init.stderr[-2000:]
    model = SimpleLLMProvider.SUPPORTED_PLATFORMS["openai"]
    assert f"openai, {model} (OPENAI_API_KEY is set in this shell)" in init.stdout
    assert "Embeddings: BAAI/bge-m3" in init.stdout
    assert _status_embedding(run) == "BAAI/bge-m3"
    doctor = run("-m", "lasagna", "doctor", "--json", OPENAI_API_KEY="sk-from-the-shell")
    checks = {c["name"]: c["status"] for c in json.loads(doctor.stdout)["checks"]}
    assert checks["llm_credentials"] == "ok"
