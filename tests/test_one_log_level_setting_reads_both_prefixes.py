"""The log level is one setting, read from `LASAGNA_LOG_LEVEL` or `RAG_LOG_LEVEL`.

The docs say every setting reads both prefixes, but there were two names for
two things: `LASAGNA_LOG_LEVEL` set the CLI's and MCP's console level and was
not the `log_level` setting, and `RAG_LOG_LEVEL` set `log_level` (the REST
server's level) and not the CLI's. `log_level` now reads both, `LASAGNA_`
first, and every entry point takes its level from it. A value it does not
take still never stops a command, as the CLI promised for its own name.
"""

from __future__ import annotations

import sys

import pytest

from lasagna import cli
from lasagna.core.config import RAGConfig


@pytest.fixture(autouse=True)
def _no_ambient_level(monkeypatch):
    for name in ("LASAGNA_LOG_LEVEL", "RAG_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["LASAGNA_LOG_LEVEL", "RAG_LOG_LEVEL"])
def test_either_name_sets_the_setting(monkeypatch, name):
    monkeypatch.setenv(name, "error")
    assert RAGConfig().log_level == "ERROR"


def test_the_lasagna_name_wins(monkeypatch):
    monkeypatch.setenv("LASAGNA_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    assert RAGConfig().log_level == "ERROR"


@pytest.mark.parametrize("name", ["LASAGNA_LOG_LEVEL", "RAG_LOG_LEVEL"])
def test_a_typo_is_ignored_not_fatal(monkeypatch, name):
    monkeypatch.setenv(name, "verbose")
    assert RAGConfig().log_level == "INFO"


@pytest.mark.parametrize("name", ["LASAGNA_LOG_LEVEL", "RAG_LOG_LEVEL"])
def test_the_cli_takes_its_level_from_either_name(monkeypatch, name):
    from loguru import logger

    monkeypatch.setenv(name, "ERROR")
    try:
        assert cli._configure_cli_logging() == "ERROR"
        monkeypatch.setenv(name, "verbose")
        assert cli._configure_cli_logging() == "WARNING"
    finally:
        logger.remove()
        logger.add(sys.stderr)


def test_the_rest_server_takes_its_level_from_either_name(monkeypatch):
    from lasagna.api.rest import limits

    seen = {}
    monkeypatch.setattr(limits, "log_format", lambda: "text")
    monkeypatch.setattr(limits.logger, "remove", lambda *a: None)
    monkeypatch.setattr(limits.logger, "add", lambda *a, **k: seen.update(k))
    monkeypatch.setenv("LASAGNA_LOG_LEVEL", "error")
    limits.apply_log_format()
    assert seen["level"] == "ERROR"


def test_the_settings_reference_lists_both_names():
    from tests._paths import PACKAGE_DIR

    reference = (PACKAGE_DIR.parents[1] / "docs" / "reference" / "settings.md").read_text()
    row = next(line for line in reference.splitlines() if line.startswith("| `log_level` |"))
    assert "`LASAGNA_LOG_LEVEL`, `RAG_LOG_LEVEL`" in row
    assert "is the exception" not in reference
