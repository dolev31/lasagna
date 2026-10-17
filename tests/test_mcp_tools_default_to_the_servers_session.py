"""No MCP tool requires `session_id`: without one, a tool acts on the server's session.

The README says a tool acts on the session `RAG_SESSION_ID` names (or `lasagna
serve --mcp -s`) unless its `session_id` argument names another. Eleven tools
declared `session_id` as required anyway (`get_document_status`,
`get_document_info`, `set_active_profile`, `patch_rag_config`, ...), so a client
that followed the README got a raw pydantic error with a URL, and
`list_profiles` and `get_profile_stats` defaulted to the session "default"
rather than the server's, so a profile saved through one tool was missing from
the other's listing. Every tool now resolves its session the same way: the
argument, else the server's session, else a plain error saying how to set one.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402


def _tools():
    srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
    return asyncio.run(srv.list_tools())


def _call(tool: str, arguments: dict):
    async def go():
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            return await client.call_tool(tool, arguments)

    result = asyncio.run(go())
    return result, result.content[0].text if result.content else ""


def test_no_tool_requires_a_session_id():
    tools = _tools()
    assert len(tools) > 20
    requiring = sorted(
        t.name for t in tools if "session_id" in (t.inputSchema.get("required") or [])
    )
    assert requiring == []


def test_every_session_parameter_says_what_it_defaults_to():
    for tool in _tools():
        prop = tool.inputSchema.get("properties", {}).get("session_id")
        if prop is None or "Unused" in prop.get("description", ""):
            continue
        assert "RAG_SESSION_ID" in prop.get("description", ""), tool.name


def test_a_tool_without_a_session_id_acts_on_the_servers_session(monkeypatch):
    monkeypatch.setenv("RAG_SESSION_ID", "mcp-default-corpus")
    result, text = _call("set_active_profile", {"profile_name": "speed"})
    assert not result.isError, text
    assert text == "Profile 'speed' activated for session 'mcp-default-corpus'."

    result, text = _call("get_active_profile", {})
    assert json.loads(text)["active_profile_name"] == "speed"


def test_profile_tools_list_what_the_servers_session_saved(monkeypatch):
    monkeypatch.setenv("RAG_SESSION_ID", "mcp-profile-owner")
    result, text = _call(
        "save_rag_profile",
        {
            "profile_name": "mine_mcp",
            "config": json.dumps({"rerank_top_k": 7}),
            "set_active": False,
        },
    )
    assert not result.isError, text
    try:
        _, listing = _call("list_profiles", {"include_builtin": False})
        assert "mine_mcp" in {p["name"] for p in json.loads(listing)}
    finally:
        _call("delete_rag_profile", {"profile_name": "mine_mcp"})


def test_with_no_session_anywhere_the_error_says_how_to_set_one(monkeypatch):
    monkeypatch.delenv("RAG_SESSION_ID", raising=False)
    result, text = _call("get_document_status", {"document_id": "doc_x"})
    assert result.isError
    assert "No session named" in text and "RAG_SESSION_ID" in text
    assert "pydantic" not in text and "Field required" not in text
