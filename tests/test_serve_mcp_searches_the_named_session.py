"""`lasagna serve --mcp -s <session>` serves an MCP search that needs no session argument.

The README's `serve --mcp` block could not search: every MCP tool call over
/mcp/sse answered "Either session_id or session_token is required", and the
banner said requests without a session use 'default'. `serve` takes
`--session/-s`, the session MCP tool calls and REST requests use when they
name none.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.api.rest.session_token import extract_session_id
from tests._paths import PACKAGE_DIR


def test_a_rest_request_without_a_session_uses_the_servers(monkeypatch):
    monkeypatch.delenv("RAG_SESSION_ID", raising=False)
    assert extract_session_id() == "default"
    monkeypatch.setenv("RAG_SESSION_ID", "my-corpus")
    assert extract_session_id() == "my-corpus"
    assert extract_session_id(session_id_header="other") == "other"


def test_an_mcp_call_without_a_session_says_how_to_name_one(monkeypatch):
    from lasagna.mcp import server

    monkeypatch.delenv("RAG_SESSION_ID", raising=False)
    with pytest.raises(ValueError, match=r"lasagna serve --mcp -s <session>"):
        server._resolve_session("")
    monkeypatch.setenv("RAG_SESSION_ID", "my-corpus")
    assert server._resolve_session("")[0] == "my-corpus"


def test_serve_dash_s_names_the_session_in_the_banner(monkeypatch):
    import uvicorn

    from lasagna.core import local_server

    for var in ("RAG_SESSION_ID", "RAG_EMBED_MCP", "RAG_CONFIGURATOR"):
        monkeypatch.setenv(var, "")  # restored after the test; serve sets them
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: started.update(kw))
    monkeypatch.setattr(local_server, "local_store_held", lambda: False)
    monkeypatch.setattr(local_server, "announce", lambda base: None)
    monkeypatch.setattr(local_server, "withdraw", lambda: None)
    monkeypatch.setattr(cli, "_port_problem", lambda host, port: None)
    result = CliRunner().invoke(
        cli.app, ["serve", "--mcp", "-s", "my-corpus"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert started, "the server did not start"
    assert os.environ["RAG_SESSION_ID"] == "my-corpus"
    assert "Without X-Session-ID, requests use session 'my-corpus'" in result.output


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _mcp_search(url: str, query: str):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool("search", {"query": query})


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_the_readme_block_answers_an_sse_search(tmp_path):
    pytest.importorskip("mcp.client.sse")
    (tmp_path / "a.md").write_text("# Ablation\n\nThe ablation showed a 0.04 drop in NDCG@10.\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")

    def lasagna(*argv):
        done = subprocess.run(
            [sys.executable, "-m", "lasagna", *argv], cwd=tmp_path, env=env, capture_output=True
        )
        assert done.returncode == 0, done.stdout + done.stderr[-2000:]

    lasagna("configure", "--profile", "speed", "-s", "my-corpus")
    lasagna("upload", "a.md", "-s", "my-corpus")
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "lasagna", "serve", "--mcp", "-s", "my-corpus", "--port", str(port)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        result = asyncio.run(
            asyncio.wait_for(
                _mcp_search(f"http://127.0.0.1:{port}/mcp/sse", "what did the ablation show?"),
                timeout=120,
            )
        )
    finally:
        server.terminate()
        server.wait(timeout=30)
    text = " ".join(getattr(c, "text", "") for c in result.content)
    assert not result.isError, text
    assert "0.04 drop in NDCG@10" in text
