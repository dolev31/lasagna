"""What a contributor reads about the code is what the code does.

Found by the sixth persona audit:

* docs/configuration.md, the SDK configuration page and the README called
  `embedding_model` process-wide, while two clients in one process on
  different sessions keep their own models (the field is the process default
  and also the session's setting, which wins).
* The Makefile said Milvus standalone "requires x86_64", and the stress suite
  "no ARM64 Docker image", while backends/milvus.yml said x86_64 and ARM64
  (Docker Hub publishes milvusdb/milvus:v2.4.24 for linux/amd64 and
  linux/arm64).
* CONTRIBUTING said a component "must not read state it did not declare" and
  that a test enforces it; the test checks only mutation.
* "intelligent query routing" in the settings reference (generated from a
  docstring) and "Two honest warnings" in the guide.
* `results --help` cut the leaderboard summary mid-sentence.
* A missing settings file's error printed its path twice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

from lasagna import cli

REPO = Path(__file__).resolve().parents[1]


def _words(path: str) -> str:
    return " ".join((REPO / path).read_text(encoding="utf-8").split())


def test_the_embedding_model_is_documented_as_the_sessions():
    from lasagna.sdk._config_bridge import INIT_TIME_FIELDS, RUNTIME_FIELD_MAP

    # The mechanism the docs describe: a process default and a session setting.
    assert "embedding_model" in INIT_TIME_FIELDS and "embedding_model" in RUNTIME_FIELD_MAP
    for page in ("README.md", "docs/configuration.md", "src/lasagna/sdk/docs/configuration.md"):
        text = re.sub(r"\s>\s", " ", _words(page))
        assert "embedding model is process-wide" not in text, page
        assert "two clients in one process on different sessions can use different models" in (
            text.lower()
        ), page


def test_every_page_agrees_the_milvus_image_runs_on_both_architectures():
    assert "x86_64 and ARM64" in _words("src/lasagna/deploy/backends/milvus.yml")
    for page in ("src/lasagna/deploy/Makefile", "tests/stress/conftest.py"):
        text = _words(page)
        assert "requires x86_64" not in text and "no ARM64 Docker image" not in text, page
    assert "Its image runs on x86_64 and ARM64" in _words("src/lasagna/deploy/Makefile")


def test_contributing_says_what_the_isolation_checks_enforce():
    text = _words("CONTRIBUTING.md")
    assert "must not read state it did not declare" not in text
    assert "is not checked by anything" in text
    assert "mutates its input" in text


@pytest.mark.parametrize(
    "page, word",
    [
        ("docs/reference/settings.md", "intelligent"),
        ("docs/guide.md", "honest warnings"),
        ("src/lasagna/sdk/docs/configuration.md", "intelligent"),
    ],
)
def test_plain_wording(page, word):
    assert word not in _words(page).lower()


def test_every_help_summary_is_a_whole_sentence():
    """Typer shows a docstring's first line as the summary in its parent's --help."""
    root = typer.main.get_command(cli.app)
    cut = []

    def walk(cmd, path):
        for name, sub in getattr(cmd, "commands", {}).items():
            if getattr(sub, "commands", None):
                walk(sub, f"{path} {name}")
            elif not sub.hidden:
                first = (sub.callback.__doc__ or "").strip().splitlines()[0].strip()
                if not first.endswith((".", "?")):
                    cut.append(f"{path} {name}: {first!r}")

    walk(root, "lasagna")
    assert not cut, cut
    help_text = " ".join(CliRunner().invoke(cli.app, ["results", "--help"]).output.split())
    assert "grouped by │" not in help_text


def test_a_missing_settings_file_names_its_path_once(monkeypatch, tmp_path):
    from lasagna.core.config import ConfigError, RAGConfig

    missing = tmp_path / "nope.toml"
    monkeypatch.setenv("LASAGNA_SETTINGS_FILE", str(missing))
    with pytest.raises(ConfigError) as raised:
        RAGConfig()
    message = str(raised.value)
    assert message.count(str(missing)) == 1, message

    monkeypatch.setenv("LASAGNA_SETTINGS_FILE", "nope.toml")
    with pytest.raises(ConfigError) as raised:
        RAGConfig()
    assert "a relative path is read from the package folder" in str(raised.value)
