"""An MCP tool refuses an argument it does not declare, naming the one meant.

FastMCP dropped undeclared arguments without a word: `search {"query":
"ablation", "session": "nope"}` searched the server's own session, and
`{"topk": 1}` returned the default three results. Every tool now answers such a
call with a tool error that names the closest declared argument and lists the
ones it takes, before the tool runs. The tools and their arguments are read
from `tools/list`, so a tool added later is covered without editing this file.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402


def _misspelled(name: str, declared) -> str:
    """``name`` with its last character dropped, or doubled when that is declared."""
    for candidate in (name[:-1], name + name[-1]):
        if candidate not in declared:
            return candidate
    raise AssertionError(name)  # pragma: no cover


def _calls():
    """(tool, arguments, declared names, misspelled name) for every tool, and the replies."""

    async def go():
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            tools = (await client.list_tools()).tools
            out = []
            for tool in tools:
                declared = list((tool.inputSchema or {}).get("properties") or {})
                stray = await client.call_tool(tool.name, {"not_an_argument_of_any_tool": 1})
                wrong = None
                if declared:
                    typo = _misspelled(declared[0], declared)
                    wrong = (typo, declared[0], await client.call_tool(tool.name, {typo: "x"}))
                out.append((tool.name, declared, stray, wrong))
            return out

    return asyncio.run(go())


@pytest.fixture(scope="module")
def replies():
    return _calls()


def test_every_tool_is_covered(replies):
    assert len(replies) == len(mcp_server._TOOLS)


def test_every_tool_refuses_an_argument_it_does_not_declare(replies):
    for name, declared, stray, _ in replies:
        text = stray.content[0].text
        assert stray.isError is True, (name, text)
        assert f"{name}: unknown argument 'not_an_argument_of_any_tool'" in text, text
        takes = ", ".join(declared) or "no arguments"
        assert text.endswith(f"{name} takes: {takes}."), text
        assert "pydantic" not in text and "Traceback" not in text


def test_a_misspelled_argument_names_the_declared_one(replies):
    for name, _, _, wrong in replies:
        if wrong is None:
            continue
        typo, meant, reply = wrong
        text = reply.content[0].text
        assert reply.isError is True, (name, text)
        assert f"unknown argument {typo!r} (did you mean {meant!r}?)" in text, text


def _search(arguments):
    async def go():
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999)
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            return await client.call_tool("search", arguments)

    return asyncio.run(go())


@pytest.mark.parametrize(
    "arguments, typo, meant",
    [
        ({"query": "ablation", "session": "nope"}, "session", "session_id"),
        ({"query": "ablation", "topk": 1}, "topk", "top_k"),
    ],
)
def test_the_two_calls_the_audit_made(arguments, typo, meant):
    reply = _search(arguments)
    assert reply.isError is True
    assert reply.content[0].text.startswith(
        f"search: unknown argument {typo!r} (did you mean {meant!r}?). search takes: query,"
    )
