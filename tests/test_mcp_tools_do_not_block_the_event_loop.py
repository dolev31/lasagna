"""A long MCP tool call leaves the server free to answer everyone else.

FastMCP runs a synchronous tool on the event loop. Measured on the SSE
server: during one client's 46 s ``upload_document`` a bare ``GET /health``
took 43 s and a second client's ``list_documents`` waited for the upload. In
the embedded deployment (MCP mounted in the REST app, the documented Milvus
Lite setup) a ``reindex_document`` stalled the REST API's own liveness probe.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402


def test_every_registered_tool_runs_off_the_event_loop():
    srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999)
    tools = srv._tool_manager._tools
    assert tools
    blocking = sorted(name for name, tool in tools.items() if not tool.is_async)
    assert blocking == [], f"run on the event loop: {blocking}"


def test_the_loop_keeps_answering_while_a_sync_tool_runs():
    def slow_tool(session_id: str = "") -> str:
        time.sleep(1.0)
        return "done"

    srv = FastMCP("probe")
    srv.add_tool(mcp_server._flag_failures(slow_tool))

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            call = asyncio.create_task(client.call_tool("slow_tool", {}))
            started = time.perf_counter()
            await asyncio.sleep(0.05)  # a heartbeat the loop must serve on time
            heartbeat = time.perf_counter() - started
            result = await call
            return heartbeat, result

    heartbeat, result = asyncio.run(go())
    assert result.content[0].text == "done"
    assert heartbeat < 0.5, f"the loop was blocked for {heartbeat:.2f}s"


def test_a_sync_tool_keeps_its_signature_and_its_failure_convention():
    def tool(document_id: str, top_k: int = 5) -> str:
        return "Failed to do it: no"

    srv = FastMCP("probe")
    srv.add_tool(mcp_server._flag_failures(tool))
    schema = srv._tool_manager._tools["tool"].parameters
    assert set(schema["properties"]) == {"document_id", "top_k"}
    assert schema["required"] == ["document_id"]

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            return await client.call_tool("tool", {"document_id": "d"})

    result = asyncio.run(go())
    assert result.isError is True
