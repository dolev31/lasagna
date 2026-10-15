"""The MCP tool surface is a public API, so a change to it must be deliberate.

Tool names, descriptions and input schemas are what an agent and a client
config are written against. They changed with no review signal: a renamed
parameter or a reworded description shipped as an incidental edit. GOVERNANCE.md
puts the tool schemas under the deprecation policy; this snapshot is how a
change to them shows up in a diff.

To accept a deliberate change::

    LASAGNA_UPDATE_SNAPSHOTS=1 pytest tests/test_mcp_tool_schema_snapshot.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from lasagna.mcp import server as mcp_server  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "mcp_tools.json"


def _surface() -> dict:
    def listed(tool_set):
        srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools=tool_set)
        return asyncio.run(srv.list_tools())

    safe = {tool.name for tool in listed("safe")}
    return {
        tool.name: {
            "listed_in": ["all", "safe"] if tool.name in safe else ["all"],
            # Python 3.13 strips docstring indentation at compile time; 3.10
            # to 3.12 do not. Compare the text a client renders.
            "description": inspect.cleandoc(tool.description or ""),
            "input_schema": tool.inputSchema,
        }
        for tool in sorted(listed("all"), key=lambda t: t.name)
    }


def test_the_tool_surface_matches_the_snapshot():
    current = _surface()
    if os.environ.get("LASAGNA_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    recorded = json.loads(SNAPSHOT.read_text())
    assert sorted(current) == sorted(recorded), "tools were added or removed"
    changed = sorted(name for name in current if current[name] != recorded[name])
    assert not changed, (
        f"the schema or description of {changed} changed. If that is deliberate, "
        f"rerun with LASAGNA_UPDATE_SNAPSHOTS=1 and note it in CHANGELOG.md."
    )
