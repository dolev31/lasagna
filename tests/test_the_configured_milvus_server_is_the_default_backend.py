"""``get_backend("milvus")`` with no arguments uses the configured Milvus server.

The default Milvus configuration read the session configuration through
``lasagna.session_config``, a module that no longer exists; the ImportError was
caught as "not ready yet", so ``RAG_VECTOR_BACKEND_CONFIG`` (or the settings
file's ``vector_backend_config``) was ignored and Milvus Lite was opened.
"""

from __future__ import annotations

import pytest

from lasagna.core import config as config_module
from lasagna.core.config import RAGConfig
from lasagna.vector_backends.registry import _get_default_milvus_config


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch):
    """A process configuration read after each test set its variables."""

    def reload():
        monkeypatch.setattr(config_module, "rag_config", RAGConfig())

    return reload


def test_a_uri_in_the_backend_config_is_used(monkeypatch, fresh_config):
    monkeypatch.delenv("MILVUS_URI", raising=False)
    monkeypatch.setenv(
        "RAG_VECTOR_BACKEND_CONFIG", '{"uri": "http://milvus.example:19530", "token": "t"}'
    )
    fresh_config()

    assert _get_default_milvus_config() == {"uri": "http://milvus.example:19530", "token": "t"}


def test_milvus_uri_is_used_when_the_config_names_no_server(monkeypatch, fresh_config):
    monkeypatch.setenv("MILVUS_URI", "http://other.example:19530")
    monkeypatch.delenv("RAG_VECTOR_BACKEND_CONFIG", raising=False)
    fresh_config()

    assert _get_default_milvus_config()["uri"] == "http://other.example:19530"


def test_the_config_wins_over_milvus_uri_as_it_does_at_startup(monkeypatch, fresh_config):
    monkeypatch.setenv("MILVUS_URI", "http://other.example:19530")
    monkeypatch.setenv("RAG_VECTOR_BACKEND_CONFIG", '{"uri": "http://milvus.example:19530"}')
    fresh_config()

    assert _get_default_milvus_config()["uri"] == "http://milvus.example:19530"


def test_a_file_and_a_server_together_are_refused(monkeypatch, fresh_config, tmp_path):
    monkeypatch.setenv("MILVUS_URI", "http://other.example:19530")
    monkeypatch.setenv("RAG_VECTOR_BACKEND_CONFIG", f'{{"db_path": "{tmp_path}/lite.db"}}')
    fresh_config()

    with pytest.raises(config_module.ConfigError, match="two stores"):
        _get_default_milvus_config()


def test_without_a_server_milvus_lite_is_opened(monkeypatch, fresh_config):
    monkeypatch.delenv("MILVUS_URI", raising=False)
    monkeypatch.delenv("RAG_VECTOR_BACKEND_CONFIG", raising=False)
    fresh_config()

    assert _get_default_milvus_config()["db_path"].endswith("rag_milvus.db")
