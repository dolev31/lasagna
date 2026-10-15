"""An MCP tool failure must reach the client as a tool error, not as content.

Tools returned failures as text or {"error": ...} with isError=False, so an
agent read "EMBEDDING DIMENSION MISMATCH" or "No LLMProvider configured" as
retrieved knowledge, and the shipped server never told a client a call failed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402


def _call(tool: str, arguments: dict):
    async def go():
        # "all": rebuild_collection is not in the default safe set.
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(go())


def test_a_refused_call_is_an_mcp_error():
    result = _call("rebuild_collection", {"session_id": "s", "confirm": False})
    assert result.isError is True
    assert "confirm must be true" in result.content[0].text


def test_the_python_function_still_returns_the_text():
    reply = mcp_server.rebuild_collection(session_id="s", confirm=False)
    assert "confirm must be true" in json.loads(reply)["error"]


@pytest.mark.parametrize(
    "reply, failed",
    [
        ("Search failed: no LLM", True),
        ("Failed to set profile: unknown", True),
        (json.dumps({"error": "bad", "hint": "do x"}), True),
        (json.dumps({"status": "unhealthy", "error": "db down"}), False),  # a report
        (json.dumps({"results": []}), False),
        ("<knowledge_base>[1] text</knowledge_base>", False),
    ],
)
def test_the_error_convention(reply, failed):
    assert (mcp_server._tool_failure(reply) is not None) is failed
