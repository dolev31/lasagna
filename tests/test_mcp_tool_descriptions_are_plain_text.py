"""MCP tool descriptions are short plain text, and each parameter documents itself.

Clients showed each tool's docstring as written: reST ``:param name:`` fields
and double backticks, about 26 KB across the tools with ``--tools all``,
while the input schema, where a client looks for what an argument means,
carried no descriptions at all.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from lasagna.mcp import server  # noqa: E402


@pytest.fixture(scope="module")
def tools():
    srv = server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
    return {t.name: t for t in asyncio.run(srv.list_tools())}


def test_no_description_carries_rest_markup(tools):
    for name, tool in tools.items():
        text = tool.description or ""
        assert ":param" not in text and ":returns:" not in text, name
        assert "``" not in text, name


def test_the_descriptions_are_short(tools):
    assert sum(len(t.description or "") for t in tools.values()) < 10_000


def test_parameters_are_described_in_the_schema(tools):
    properties = tools["search"].inputSchema["properties"]
    assert properties["query"]["description"] == "Natural-language search query."
    assert "only documents belonging to this session" in properties["session_id"]["description"]
    undocumented = [
        (name, param)
        for name, tool in tools.items()
        for param, spec in tool.inputSchema.get("properties", {}).items()
        if f":param {param}:" in (getattr(server, name).__doc__ or "")
        and not spec.get("description")
    ]
    assert not undocumented, undocumented


def test_the_split_keeps_lists_and_the_return_value():
    description, params = server._split_docstring(
        """Do a thing.

        It does:

        - one step
        - another step

        :param x: The input,
            on two lines.
        :returns: The ``result``.
        """
    )
    assert description == (
        "Do a thing.\n\nIt does:\n\n- one step\n- another step\n\nReturns: The `result`."
    )
    assert params == {"x": "The input, on two lines."}
