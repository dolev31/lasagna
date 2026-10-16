"""While ``lasagna serve`` runs, the CLI goes through it and MCP has an endpoint in it.

Milvus Lite admits one process per store, and ``serve`` held it: ``status``,
``search``, ``upload`` and ``lasagna-mcp`` loaded their models for about 7 s
and then failed with "Another process is using the local store". ``serve``
records its address in ``<data dir>/serve.json``; a command that finds the
store held goes through that server's REST API, or stops at once with the fix
when no server it can reach holds it. ``serve --mcp`` serves the MCP tools at
``/mcp/sse`` from the same process.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import local_server


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_recorded_server_is_found_while_it_runs(data_dir):
    (data_dir).mkdir(parents=True)
    record = data_dir / "serve.json"
    record.write_text(json.dumps({"pid": os.getppid(), "url": "http://127.0.0.1:8765"}))
    assert local_server.running_server() == "http://127.0.0.1:8765"
    record.write_text(json.dumps({"pid": _dead_pid(), "url": "http://127.0.0.1:8765"}))
    assert local_server.running_server() is None


def test_serve_withdraws_only_its_own_record(data_dir):
    local_server.announce("http://127.0.0.1:8765")
    assert json.loads((data_dir / "serve.json").read_text())["pid"] == os.getpid()
    local_server.withdraw()
    assert not (data_dir / "serve.json").exists()


def test_a_held_lock_is_seen(tmp_path):
    store = tmp_path / "rag_milvus.db"
    store.mkdir()
    lock = store / "LOCK"
    lock.write_text("")
    assert not local_server.lite_store_locked(str(store))
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert local_server.lite_store_locked(str(store))
    finally:
        os.close(fd)


def test_status_goes_through_a_running_server(monkeypatch):
    seen = {}

    class _Remote:
        async def list_documents(self):
            return []

        async def close(self):
            pass

    async def _config(url, session_id):
        seen["config"] = (url, session_id)
        return {"embedding_model": "all-MiniLM-L6-v2"}

    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: "http://127.0.0.1:8765")
    from lasagna.sdk import KnowledgeClient

    monkeypatch.setattr(
        KnowledgeClient,
        "from_url",
        classmethod(lambda cls, url, **kw: seen.update(url=url) or _Remote()),
    )
    monkeypatch.setattr(cli, "_served_config", _config)
    r = CliRunner().invoke(cli.app, ["status", "-s", "my-corpus"], terminal_width=200)
    assert r.exit_code == 0, r.output
    assert seen["url"] == "http://127.0.0.1:8765"
    assert seen["config"] == ("http://127.0.0.1:8765", "my-corpus")
    assert "all-MiniLM-L6-v2" in r.output


@pytest.mark.parametrize("command", [["status"], ["upload", "README.md"], ["search", "q"]])
def test_a_store_held_by_something_else_stops_at_once(monkeypatch, command):
    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: None)
    from lasagna.core import model_downloads

    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: None)

    def _never(*a, **k):
        raise AssertionError("a client was built for a store another process holds")

    monkeypatch.setattr(cli, "_get_client", _never)
    r = CliRunner().invoke(cli.app, command, terminal_width=200)
    out = " ".join(r.output.split())
    assert r.exit_code == 1
    assert "Another process is using the local store" in out and "lasagna serve --mcp" in out


def test_serve_refuses_a_held_store_before_starting(monkeypatch):
    import uvicorn

    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: "http://127.0.0.1:8000")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("the server started"))
    r = CliRunner().invoke(cli.app, ["serve", "--port", "8765"], terminal_width=200)
    assert r.exit_code == 1
    assert "already running at http://127.0.0.1:8000" in " ".join(r.output.split())


def test_serve_mcp_mounts_the_mcp_endpoint(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_MCP", "true")
    from lasagna.api.rest.app import create_app

    app = create_app()
    assert any(getattr(route, "path", None) == "/mcp" for route in app.routes)


def test_a_second_response_after_the_event_stream_is_dropped():
    import asyncio

    from lasagna.api.rest.app import _one_response

    async def sse_then_empty(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"event: x\n\n"})
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent = []

    async def send(message):
        sent.append(message["type"])

    asyncio.run(_one_response(sse_then_empty)({"type": "http"}, None, send))
    assert sent == ["http.response.start", "http.response.body"]


def test_remote_upload_can_wait_for_processing():
    from lasagna.sdk.remote_backend import RemoteRAGBackend

    assert RemoteRAGBackend.get_document_by_id


def test_an_mcp_tool_says_where_to_connect_when_serve_holds_the_store(monkeypatch):
    import asyncio

    from lasagna.mcp import server

    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: "http://127.0.0.1:8765")
    monkeypatch.setattr(
        server, "get_rag_backend", lambda: pytest.fail("models loaded for a held store")
    )
    from loguru import logger

    errors = []
    sink = logger.add(lambda m: errors.append(m), level="ERROR")
    try:
        out = asyncio.run(server.search("q", session_id="my-corpus"))
    finally:
        logger.remove(sink)
    assert "http://127.0.0.1:8765/mcp/sse" in out and "lasagna serve --mcp" in out
    assert not errors, "the reply says it; an ERROR line with a traceback repeated it"
