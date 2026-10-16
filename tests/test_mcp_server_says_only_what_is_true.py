"""The MCP server reports its own version and tool count, logs little, and invents nothing.

Over stdio one search wrote 144 lines to the client's stderr: DEBUG level, ANSI
color, the logger named ``rag_knowledge_mcp``, and FastMCP's "Processing
request" lines. ``serverInfo.version`` was the MCP SDK's ("1.30.0"). The startup
log said "Tools: 25" while ``tools/list`` returned 23. ``--help`` read
``usage: __main__.py`` and never named ``RAG_SESSION_ID``. ``set_active_profile``
cited a "deep_search" profile that does not exist, and every search result
carried the example "The cost was $1,370 [1]", which an agent could take as a
fact from the documents. ``set_lifecycle_policy``, which changes what garbage
collection deletes in every session, was in the default set.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

import lasagna  # noqa: E402
from lasagna.mcp import server as mcp_server  # noqa: E402
from tests._paths import PACKAGE_DIR  # noqa: E402

README = Path(__file__).resolve().parents[1] / "README.md"


def _session(tools):
    async def go():
        srv = mcp_server.create_mcp_server(tools=tools)
        async with create_connected_server_and_client_session(srv._mcp_server) as client:
            init = await client.initialize()
            listed = await client.list_tools()
        return srv, init, listed

    return asyncio.run(go())


def test_server_info_reports_lasagnas_version():
    _srv, init, _listed = _session("safe")
    assert init.serverInfo.version == lasagna.__version__


@pytest.mark.parametrize("tools", ["safe", "all"])
def test_the_logged_count_is_what_a_client_is_shown(tools):
    srv, _init, listed = _session(tools)
    assert srv.listed_tool_count() == len(listed.tools)


def test_the_readme_counts_are_the_real_ones():
    _s, _i, safe = _session("safe")
    _s, _i, every = _session("all")
    text = " ".join(README.read_text().split())
    assert f"serves the `safe` set, {len(safe.tools)} tools" in text
    assert f"serves all {len(every.tools)}" in text


def test_lifecycle_policy_is_outside_the_safe_set_and_session_config_inside():
    _s, _i, safe = _session("safe")
    names = {t.name for t in safe.tools}
    assert "set_lifecycle_policy" not in names
    assert "patch_rag_config" in names


def test_no_tool_names_a_profile_that_does_not_exist():
    _s, _i, every = _session("all")
    text = " ".join(t.description or "" for t in every.tools)
    assert "deep_search" not in text


def test_search_context_carries_no_invented_example():
    from lasagna.agentic.citation_manager import Citation, CitationManager

    citation = Citation(
        citation_id="x",
        index=1,
        chunk_id="c",
        document_id="d",
        document_name="notes.md",
        content="Tardigrades survive in space.",
        content_preview="Tardigrades",
        score=0.6,
        normalized_score=1.0,
        chunk_index=0,
    )
    context = CitationManager().format_context_with_citations([citation])
    assert "$1,370" not in context and "Example" not in context
    assert "[1]" in context and "Tardigrades survive in space." in context


def _run_mcp(*args, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(PACKAGE_DIR.parent))
    return subprocess.run(
        [sys.executable, "-m", "lasagna.mcp", *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_help_names_the_command_and_the_session_variable(tmp_path):
    out = _run_mcp("--help", tmp_path=tmp_path).stdout
    assert out.startswith("usage: lasagna-mcp")
    assert "RAG_SESSION_ID" in out


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


CLIENT = """
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "lasagna.mcp"],
                                   env=dict(os.environ))
    with open("server_stderr.txt", "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                await s.list_tools()
                res = await s.call_tool("search", {"query": "what survives in space?"})
                print(res.content[0].text[:600])

asyncio.run(main())
"""

SETUP = """
import asyncio
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    open("notes.md", "w").write("# Notes\\n\\nTardigrades survive in space.\\n")
    client = KnowledgeClient(session_id="my-corpus", config=config_for_speed())
    doc = await client.upload("notes.md")
    await client.documents.wait_for_processing(doc.document_id)
    await client.close()

asyncio.run(main())
"""


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_a_stdio_search_leaves_the_clients_log_empty(tmp_path):
    (tmp_path / "setup.py").write_text(SETUP)
    (tmp_path / "client.py").write_text(CLIENT)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        RAG_SESSION_ID="my-corpus",
    )
    for script in ("setup.py", "client.py"):
        run = subprocess.run(
            [sys.executable, script],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert run.returncode == 0, run.stderr[-3000:]
    assert "Tardigrades" in run.stdout
    log = (tmp_path / "server_stderr.txt").read_text()
    assert not re.search(r"DEBUG|INFO|\x1b\[", log), log[:2000]
