"""`lasagna init` asks only what it uses, and says where the LLM really comes from.

It asked "LLM provider?" and threw the answer away: it wrote only
`vector_backend` to the settings file, yet the summary said "LLM: OpenAI,
gpt-6-luna" while nothing would call OpenAI, since Lasagna picks the provider
from the API key exported in the shell. The question is gone; the summary
names the provider the exported key selects, or says that none is set and
which variable to export.
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
import json
import sys
import questionary
from typer.testing import CliRunner

answers = {"Vector database?": "milvus", "Quality profile?": sys.argv[1]}
asked = []

class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value

def select(message, *args, **kwargs):
    asked.append(message)
    return _Answer(answers.get(message))

questionary.select = select

from lasagna import cli

result = CliRunner().invoke(cli.app, ["init"], env={"COLUMNS": "200"})
print(result.output)
print("ASKED=" + json.dumps(asked))
sys.exit(result.exit_code)
"""


@pytest.fixture
def init(tmp_path):
    (tmp_path / "drive.py").write_text(DRIVE)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("RAG_", "LASAGNA_"))
        and not k.endswith(("_API_KEY", "_APIKEY"))
        and k not in ("OLLAMA_BASE_URL", "VLLM_BASE_URL", "LLAMACPP_MODEL_PATH", "HF_MODEL_ID")
    }
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), COLUMNS="200")

    def _run(profile: str, **extra):
        done = subprocess.run(
            [sys.executable, "drive.py", profile],
            cwd=tmp_path,
            env={**env, "HF_HUB_OFFLINE": "1", **extra},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr[-2000:]
        out, _, asked = done.stdout.rpartition("ASKED=")
        return " ".join(out.split()), json.loads(asked)

    return _run


def test_only_the_backend_and_the_profile_are_asked(init):
    words, asked = init("speed")
    assert asked == ["Vector database?", "Quality profile?"]
    assert "Profile: speed" in words


def test_with_a_key_exported_the_summary_names_the_provider_it_selects(init):
    words, _ = init("standard", ANTHROPIC_API_KEY="sk-ant-from-the-shell")
    model = SimpleLLMProvider.SUPPORTED_PLATFORMS["anthropic"]
    assert f"LLM: anthropic, {model} (ANTHROPIC_API_KEY is set in this shell)" in words
    assert "OpenAI" not in words


def test_with_no_key_the_summary_says_none_is_set_and_what_to_export(init):
    words, _ = init("speed")
    assert "LLM: none" in words
    assert "export OPENAI_API_KEY=<your key>" in words
    assert "the provider follows the key you export" in words
