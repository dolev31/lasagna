"""A store another process holds is reported once, by the reply, not by an ERROR log.

`lasagna serve` started while another process held the Milvus Lite store
logged "ERROR | RAG backend init failed (will retry on first request): The
Milvus Lite store at /abs/path ... is in use" on top of what each request
answers (503 with the same sentence), and every MCP tool call logged an ERROR
line with a traceback beside the reply that already said what to do. The
backend's own log line was moved to DEBUG earlier; these were the ones left.
Any other failure still logs at ERROR.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger


class DataDirLockedError(ConnectionError):
    """Stands in for the vector store's class: it is recognized by name."""


@pytest.fixture
def logs():
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="DEBUG")
    yield records
    logger.remove(sink)


def _errors(records):
    return [r["message"] for r in records if r["level"].no >= logger.level("WARNING").no]


def _held():
    try:
        raise DataDirLockedError("The Milvus Lite store at /x/rag_milvus.db is in use")
    except ConnectionError as exc:
        raise RuntimeError("Failed to initialize RAG Backend") from exc


@pytest.mark.parametrize("held", [True, False])
def test_serve_starting_on_a_held_store_logs_no_error(monkeypatch, logs, held):
    from fastapi.testclient import TestClient

    import lasagna.api.rag_backend as rag_backend
    from lasagna.api.rest.app import create_app

    def broken():
        if held:
            _held()
        raise ValueError("embedding size 384 does not match 1024")

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(rag_backend, "get_rag_backend", broken)
    monkeypatch.setattr(rag_backend, "close_backend", lambda: None)
    with TestClient(create_app()):
        pass

    errors = _errors(logs)
    if held:
        assert not [m for m in errors if "in use" in m or "init failed" in m], errors
        debug = [r["message"] for r in logs if r["level"].name == "DEBUG"]
        assert any("RAG backend not initialized" in m for m in debug), debug
    else:
        assert any("384 does not match 1024" in m for m in errors), errors


@pytest.mark.parametrize("held", [True, False])
def test_an_mcp_tool_on_a_held_store_logs_no_error(monkeypatch, logs, held):
    from lasagna.mcp import server

    def broken():
        if held:
            _held()
        raise ValueError("embedding size 384 does not match 1024")

    monkeypatch.setattr(server, "_get_backend", broken)
    monkeypatch.setattr(server, "_get_config", lambda: SimpleNamespace())
    reply = server.get_system_status()

    errors = _errors(logs)
    if held:
        assert "Failed to initialize RAG Backend" in reply
        assert errors == [], errors
    else:
        assert "384 does not match 1024" in reply
        failure = [r for r in logs if r["level"].name == "ERROR"]
        assert failure and failure[0]["exception"] is not None
