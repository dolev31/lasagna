"""A Milvus server named in the configuration is the store the backend opens.

``backends/milvus.yml`` sets ``RAG_VECTOR_BACKEND_CONFIG={"uri": "http://milvus:19530"}``.
The startup backend always passed a ``db_path`` as well (the default
``<db dir>/rag_milvus.db``), and ``MilvusBackend.connect`` took ``db_path``
over ``uri``, so the stack configured for a Milvus server opened a Milvus Lite
file inside each container: four containers, four private stores, and the
server never saw a vector. ``MILVUS_URI`` was ignored the same way.

The rule now: a server address (``uri`` in the backend config, else
``MILVUS_URI``) wins over the default local file; a ``db_path`` set together
with one is an error naming both.
"""

from __future__ import annotations

import pytest
from loguru import logger

from lasagna.api import _collection_manager as cm
from lasagna.core.config import ConfigError, RAGConfig
from lasagna.vector_backends import milvus_backend

SERVER = "http://milvus:19530"


@pytest.fixture
def opened(monkeypatch):
    """What MilvusClient was asked to open, without opening anything."""
    calls = []

    class _Client:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(milvus_backend, "MilvusClient", _Client)
    monkeypatch.setattr(cm, "get_backend", lambda name, **kw: milvus_backend.MilvusBackend(**kw))
    monkeypatch.delenv("MILVUS_URI", raising=False)
    monkeypatch.delenv("MILVUS_TOKEN", raising=False)
    return calls


def _startup_backend(monkeypatch, backend_config: str):
    """The backend the server builds at startup from these environment settings."""
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "milvus")
    monkeypatch.setenv("RAG_VECTOR_BACKEND_CONFIG", backend_config)
    monkeypatch.setattr(cm, "rag_config", RAGConfig())
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        backend = cm.CollectionManager.create_vector_backend()
        backend.connect()
    finally:
        logger.remove(sink)
    return backend, "".join(lines)


def test_the_compose_milvus_fragment_connects_to_the_server(monkeypatch, opened):
    backend, log = _startup_backend(monkeypatch, '{"uri": "%s"}' % SERVER)

    assert not backend.is_lite
    assert opened[-1][0] == ()
    assert opened[-1][1]["uri"] == SERVER
    assert f"Milvus server at {SERVER} (from vector_backend_config)" in log
    assert "rag_milvus.db" not in log


def test_milvus_uri_in_the_environment_connects_to_the_server(monkeypatch, opened):
    monkeypatch.setenv("MILVUS_URI", SERVER)
    monkeypatch.setenv("MILVUS_TOKEN", "root:Milvus")
    backend, log = _startup_backend(monkeypatch, "{}")

    assert not backend.is_lite
    assert opened[-1][1]["uri"] == SERVER
    assert opened[-1][1]["token"] == "root:Milvus"
    assert "(from MILVUS_URI)" in log


def test_without_a_server_the_default_local_file_is_opened(monkeypatch, opened, tmp_path):
    monkeypatch.setenv("LASAGNA_DB_DIR", str(tmp_path))
    backend, log = _startup_backend(monkeypatch, "{}")

    assert backend.is_lite
    assert opened[-1][0][0].endswith("rag_milvus.db")
    assert "Milvus Lite at " in log


def test_a_configured_db_path_alone_is_used(monkeypatch, opened, tmp_path):
    path = str(tmp_path / "mine.db")
    backend, log = _startup_backend(monkeypatch, '{"db_path": "%s"}' % path)

    assert backend.is_lite
    assert opened[-1][0][0] == path
    assert f"Milvus Lite at {path}" in log


@pytest.mark.parametrize("server_in_env", [False, True])
def test_db_path_with_a_server_is_refused_naming_both(monkeypatch, opened, server_in_env):
    if server_in_env:
        monkeypatch.setenv("MILVUS_URI", SERVER)
        config = '{"db_path": "/data/milvus.db"}'
    else:
        config = '{"db_path": "/data/milvus.db", "uri": "%s"}' % SERVER

    with pytest.raises(ConfigError) as err:
        _startup_backend(monkeypatch, config)

    message = str(err.value)
    assert "/data/milvus.db" in message and SERVER in message
    assert ("MILVUS_URI" if server_in_env else "vector_backend_config") in message
    assert opened == []


def test_the_log_line_hides_a_password_in_the_uri():
    kwargs, where = cm.milvus_connection_args(
        {"uri": "https://user:hunter2@milvus.example:19530"}, lambda: "unused", environ={}
    )
    assert kwargs["uri"] == "https://user:hunter2@milvus.example:19530"
    assert "hunter2" not in where


def test_milvus_backend_refuses_both_a_file_and_a_server():
    with pytest.raises(ValueError, match="db_path.*uri"):
        milvus_backend.MilvusBackend(db_path="/data/milvus.db", uri=SERVER)
