"""A bad environment setting stops the process with its name, value and the fix.

An invalid value was logged and ignored (``RAG_CHUNK_SIZE=big`` ran with 800),
an out-of-range one was dropped, a misspelled boolean read as false
(``RAG_ENABLE_RERANKING=flase``), ``on`` read as false, and a misspelled name
(``RAG_CHUNK_SIZ``) did nothing at all, so a run used the defaults while its
operator believed otherwise. Settings that only the settings file defines
(``use_agentic_rag``) could not be set from the environment, though the
documentation said they could.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from loguru import logger

from lasagna.core import config as config_module
from lasagna.core.config import RAGConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def warnings(monkeypatch):
    monkeypatch.setattr(config_module, "_warned_unknown_env", set(), raising=False)
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    yield seen
    logger.remove(sink)


def _error(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    error = getattr(config_module, "ConfigError", ValueError)
    with pytest.raises(error) as raised:
        RAGConfig()
    return str(raised.value)


def test_an_unparseable_number_names_the_variable_the_value_and_the_type(monkeypatch):
    message = _error(monkeypatch, RAG_CHUNK_SIZE="big")
    assert "RAG_CHUNK_SIZE='big'" in message
    assert "whole number" in message
    assert "unset the variable" in message


def test_a_value_out_of_range_is_refused_with_the_range(monkeypatch):
    message = _error(monkeypatch, RAG_HYBRID_ALPHA="1.5")
    assert "RAG_HYBRID_ALPHA='1.5'" in message and "0.0 to 1.0" in message


def test_a_misspelled_boolean_is_refused_rather_than_read_as_false(monkeypatch):
    message = _error(monkeypatch, RAG_ENABLE_RERANKING="flase")
    assert "RAG_ENABLE_RERANKING" in message and "true or false" in message


def test_a_value_outside_a_fixed_set_suggests_the_closest(monkeypatch):
    message = _error(monkeypatch, RAG_DATABASE_BACKEND="postgres")
    assert "sqlite, postgresql" in message and "did you mean postgresql" in message


def test_every_bad_variable_is_reported_at_once(monkeypatch):
    message = _error(monkeypatch, RAG_CHUNK_SIZE="big", RAG_MMR_LAMBDA="2")
    assert "RAG_CHUNK_SIZE" in message and "RAG_MMR_LAMBDA" in message


def test_the_error_is_a_value_error():
    assert issubclass(config_module.ConfigError, ValueError)


def test_on_and_off_are_booleans(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_RERANKING", "on")
    monkeypatch.setenv("RAG_ENABLE_MMR", "off")
    config = RAGConfig()
    assert config.enable_reranking is True
    assert config.enable_mmr is False


def test_an_empty_value_means_unset(monkeypatch):
    # What `${VAR:-}` leaves in a compose file.
    monkeypatch.setenv("RAG_CHUNK_SIZE", "")
    monkeypatch.setenv("RAG_ENABLE_RERANKING", "")
    config = RAGConfig()
    assert config.chunk_size == RAGConfig().chunk_size
    assert "chunk_size" not in config._env_overrides


def test_a_setting_whose_default_is_none_is_parsed_to_its_type(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_TRUNCATE_DIM", "256")
    assert RAGConfig().embedding_truncate_dim == 256
    monkeypatch.setenv("RAG_EMBEDDING_TRUNCATE_DIM", "none")
    assert RAGConfig().embedding_truncate_dim is None


def test_a_setting_only_the_settings_file_defines_can_be_set_from_the_environment(
    monkeypatch,
):
    assert RAGConfig().use_agentic_rag is True
    monkeypatch.setenv("RAG_USE_AGENTIC_RAG", "false")
    assert RAGConfig().use_agentic_rag is False


def test_an_unknown_name_warns_once_with_the_closest_setting(monkeypatch, warnings):
    monkeypatch.setenv("RAG_CHUNK_SIZ", "500")
    RAGConfig()
    RAGConfig()
    said = [w for w in warnings if "RAG_CHUNK_SIZ " in w or "RAG_CHUNK_SIZ is" in w]
    assert len(said) == 1, warnings
    assert "Did you mean RAG_CHUNK_SIZE?" in said[0]


def test_a_variable_the_package_reads_elsewhere_does_not_warn(monkeypatch, warnings):
    monkeypatch.setenv("RAG_JWT_SECRET", "x")
    monkeypatch.setenv("RAG_GC_TTL_DAYS", "3")
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini")
    RAGConfig()
    assert not [w for w in warnings if "not a Lasagna setting" in w], warnings


#: How the package reads an environment variable.
_READS = re.compile(
    r"""(?:environ\.get|getenv|environ\.setdefault|environ\[|_env\(|env_get\()\s*\(?\s*["'](RAG_[A-Z0-9_]+)["']"""
)


def _variables_read() -> set:
    names = set()
    for path in (REPO / "src" / "lasagna").rglob("*.py"):
        names |= set(_READS.findall(path.read_text()))
    deploy = REPO / "src" / "lasagna" / "deploy"
    for path in deploy.rglob("*"):
        if path.is_file() and path.suffix in ("", ".yml", ".yaml", ".example", ".env"):
            names |= set(re.findall(r"^\s*(RAG_[A-Z0-9_]+)\s*[:=]", path.read_text(), re.M))
    return names


def test_no_variable_the_package_reads_is_reported_as_unknown():
    """The warning must never fire on a variable something in the package uses."""
    settings = set(RAGConfig()._config) | set(config_module._PROPERTY_ONLY_SETTINGS)
    read = _variables_read()
    assert "RAG_JWT_SECRET" in read and "RAG_DB_DIR" in read  # the scan sees reads
    unknown = sorted(
        name
        for name in read
        if name[4:].lower() not in settings
        and name not in config_module.NON_SETTING_ENV
        and not name.startswith(config_module.NON_SETTING_ENV_PREFIXES)
    )
    assert not unknown, (
        f"{unknown} are read by the package but are neither settings nor listed in "
        f"lasagna.core.config.NON_SETTING_ENV, so setting them would warn as unknown"
    )


def test_one_set_of_true_words_serves_the_server_and_the_settings():
    from lasagna.api.rest import auth_mode

    assert auth_mode._TRUTHY is config_module.TRUTHY
    assert {"1", "true", "yes", "on"} == set(config_module.TRUTHY)
