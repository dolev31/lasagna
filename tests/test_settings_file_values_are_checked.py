"""A bad value in a settings file stops the process, naming the file, key and value.

Environment values were checked (type, range, choices) and refused with the
variable named, but values from the settings file (``LASAGNA_SETTINGS_FILE``,
or the packaged ``settings.lasagna.toml``) were merged unchecked:
``chunk_size = 5`` or ``database_backend = "postgres"`` ran as written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import lasagna
from lasagna.core.config import ConfigError, RAGConfig


def _settings(tmp_path, monkeypatch, body: str) -> Path:
    path = tmp_path / "my-settings.toml"
    path.write_text("[rag]\n" + body)
    monkeypatch.setenv("LASAGNA_SETTINGS_FILE", str(path))
    return path


def test_bad_values_are_refused_with_the_file_the_key_and_the_value(tmp_path, monkeypatch):
    path = _settings(
        tmp_path,
        monkeypatch,
        'chunk_size = 5\ndatabase_backend = "postgres"\nenable_mmr = "yes please"\n'
        "hybrid_alpha = 0.4\n",
    )

    with pytest.raises(ConfigError) as raised:
        RAGConfig()

    message = str(raised.value)
    assert str(path) in message
    assert "[rag] chunk_size = 5: chunk_size takes 50 to 100000" in message
    assert "database_backend = 'postgres'" in message and "did you mean postgresql" in message
    assert "enable_mmr = 'yes please'" in message and "true or false" in message
    assert "hybrid_alpha" not in message
    assert "remove the line" in message


def test_a_whole_number_is_a_number_and_a_valid_file_loads(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, "hybrid_alpha = 1\nchunk_size = 600\nenable_mmr = false\n")

    config = RAGConfig()

    assert config.chunk_size == 600


@pytest.mark.parametrize("name", ["settings.lasagna.toml", "settings.lasagna_production.toml"])
def test_the_shipped_settings_files_pass(monkeypatch, name):
    monkeypatch.setenv("LASAGNA_SETTINGS_FILE", str(Path(lasagna.__file__).parent / name))
    RAGConfig()
