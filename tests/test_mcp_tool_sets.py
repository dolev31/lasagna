"""The MCP server hands an agent no deletion tools unless asked, and speaks stdio.

Every tool was always registered, so an agent connected to a knowledge base
could call ``clear_session``, ``bulk_delete_session`` or ``rebuild_collection``
the first time it decided to "clean up". The default transport was SSE, so
``lasagna-mcp`` with no arguments opened a network port, while every MCP client
that launches a server (Claude Code, Claude Desktop, Cursor) speaks stdio.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402

DESTRUCTIVE = {
    "clear_session",
    "bulk_delete_session",
    "rebuild_collection",
    "delete_document",
    "delete_rag_profile",
    "reset_rag_config",
    "set_lifecycle_policy",
}
ALIASES = {"search_knowledge", "search_with_context"}
TOOLS_ENV = "LASAGNA_MCP_TOOLS"


def _listed(**kwargs) -> set:
    srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, **kwargs)
    return {tool.name for tool in asyncio.run(srv.list_tools())}


def _call(tool: str, arguments: dict, **kwargs):
    async def go():
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, **kwargs)
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            listed = {t.name for t in (await client.list_tools()).tools}
            return listed, await client.call_tool(tool, arguments)

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def _no_ambient_tool_set(monkeypatch):
    monkeypatch.delenv(TOOLS_ENV, raising=False)


class TestTheDefaultSetIsSafe:
    def test_no_destructive_tool_is_listed_by_default(self):
        listed = _listed()
        assert listed, "nothing listed: the parse is wrong, not the server"
        assert not listed & DESTRUCTIVE, sorted(listed & DESTRUCTIVE)

    def test_a_destructive_tool_cannot_be_called_by_default(self):
        _, result = _call("clear_session", {"session_id": "s1", "confirm": True})
        assert result.isError is True
        assert "Unknown tool" in result.content[0].text

    def test_the_deprecated_search_aliases_are_callable_but_not_listed(self):
        """Existing clients keep working; new agents see one search tool."""
        listed, result = _call("search_knowledge", {"query": "  ", "session_id": "s1"})
        assert not listed & ALIASES
        assert "search" in listed
        # The alias ran: the refusal is the search tool's own, not "Unknown tool".
        assert result.isError is True
        assert "the query is empty" in result.content[0].text

    def test_everything_else_is_still_served(self):
        assert _listed() == _listed(tools="all") - DESTRUCTIVE - ALIASES


class TestAllIsOptIn:
    def test_all_lists_every_tool(self):
        listed = _listed(tools="all")
        assert DESTRUCTIVE | ALIASES <= listed
        assert listed == {fn.__name__ for fn in mcp_server._TOOLS}

    def test_the_environment_selects_the_set(self, monkeypatch):
        monkeypatch.setenv(TOOLS_ENV, "all")
        assert DESTRUCTIVE <= _listed()

    def test_an_unknown_set_names_the_accepted_values(self):
        with pytest.raises(ValueError, match="--tools safe .* --tools all"):
            mcp_server.resolve_tool_set("everything")


def _run(*args, tmp_path, **env_extra) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != TOOLS_ENV}
    env.update(HOME=str(tmp_path), **env_extra)
    return subprocess.run(
        [sys.executable, "-m", "lasagna.mcp", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


class TestTheCommandLine:
    def test_the_default_transport_is_stdio(self, tmp_path):
        """With stdin at EOF a stdio server exits cleanly; an SSE one would bind.

        INFO, because a stdio server logs warnings only by default."""
        result = _run("--log-level", "INFO", tmp_path=tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
        assert "Transport: stdio" in result.stderr
        assert "safe set" in result.stderr

    def test_a_port_without_sse_is_refused_with_the_fix(self, tmp_path):
        """`lasagna-mcp --port 8005` used to mean SSE; it must not wait on stdin."""
        result = _run("--port", "8005", tmp_path=tmp_path)
        assert result.returncode == 2
        assert "Add --transport sse" in result.stderr

    def test_an_unknown_set_in_the_environment_is_refused(self, tmp_path):
        result = _run(tmp_path=tmp_path, LASAGNA_MCP_TOOLS="everything")
        assert result.returncode == 2
        assert "--tools all" in result.stderr


def test_the_path_flags_win_over_the_environment(monkeypatch, tmp_path):
    """LASAGNA_DB_DIR outranks RAG_DB_DIR, and the flag used to set only the latter."""
    from lasagna.core import paths

    for name in ("LASAGNA_DB_DIR", "RAG_DB_DIR", "LASAGNA_UPLOAD_DIR", "RAG_UPLOAD_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "ambient"))
    assert mcp_server._apply_path_flags(str(tmp_path / "db"), str(tmp_path / "up")) is True
    assert paths.db_dir() == tmp_path / "db"
    assert paths.upload_dir() == tmp_path / "up"
