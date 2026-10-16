"""A failed MCP search logs no query text, and a session without an LLM still searches.

A failed search wrote a 134-line loguru traceback to the client's log with
every frame's local variables, the query text among them six times: the MCP
server's sink lacked ``diagnose=False``. The tool error ended "or call
configure_providers()", a Python call an agent cannot make. And on the
default profile every MCP query went to the agentic loop, so without an LLM
key every search failed, while the README said only complex queries need one.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._paths import PACKAGE_DIR


def _logger_add_calls():
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                yield path.relative_to(PACKAGE_DIR), node


def test_every_log_sink_leaves_local_variables_out_of_tracebacks():
    calls = list(_logger_add_calls())
    assert len(calls) >= 5
    missing = [
        f"{path}:{node.lineno}"
        for path, node in calls
        if not any(
            kw.arg == "diagnose" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in node.keywords
        )
    ]
    assert not missing, f"logger.add without diagnose=False: {missing}"


FAILING_SEARCH = """
import asyncio
from loguru import logger
from lasagna.mcp import server

logger.enable("lasagna")  # as the server's main() does
server._owns_logging = True
server._configure_rag_logging("WARNING")

def prepare(tool, query, session_id, *args):
    held = query  # a local variable holding the query, as the real frames have
    raise RuntimeError("the vector store is unreachable")

server._prepare_search = prepare
print(asyncio.run(server.search("secret-query-text", session_id="s")))
"""


def test_a_failed_search_logs_the_error_without_the_query(tmp_path):
    (tmp_path / "fail.py").write_text(FAILING_SEARCH)
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)}
    run = subprocess.run(
        [sys.executable, "fail.py"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Search failed: the vector store is unreachable" in run.stdout
    assert "the vector store is unreachable" in run.stderr
    assert "secret-query-text" not in run.stderr, run.stderr[-3000:]


def test_the_missing_llm_error_tells_the_operator_what_to_set(monkeypatch):
    from lasagna import providers

    for name, _platform in providers._API_KEY_TO_PLATFORM:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers, "_llm_provider", None)
    monkeypatch.setattr(providers, "_auto_configure_from_env", lambda: None)
    with pytest.raises(providers.LLMProviderNotConfigured) as info:
        providers.get_llm_provider()
    message = str(info.value)
    assert "configure_providers" not in message
    assert "where Lasagna runs" in message and "OPENAI_API_KEY" in message


class _Backend:
    def search_with_citations(self, **kw):
        chunk = SimpleNamespace(chunk_id="c1", content="the ablation lowered NDCG", chunk_index=0)
        return [SimpleNamespace(chunk=chunk, score=0.9)], [], "<knowledge_base>[1]</knowledge_base>"


def test_without_an_llm_an_agentic_session_runs_its_standard_pipeline(monkeypatch):
    from lasagna import providers
    from lasagna.mcp import server

    config = SimpleNamespace(use_agentic_rag=True, rerank_top_k=5)
    monkeypatch.setattr(server, "_prepare_search", lambda *a: (_Backend(), config, None))
    monkeypatch.setattr(server, "_store_citations", lambda *a, **k: None)
    monkeypatch.setattr(server, "_log_citation_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(server, "_validate_page_mapping", lambda *a, **k: None)

    def no_llm():
        raise providers.LLMProviderNotConfigured("No LLMProvider configured.")

    monkeypatch.setattr(providers, "get_llm_provider", no_llm)

    def agentic(**kw):
        raise AssertionError("the agentic loop ran without an LLM")

    monkeypatch.setattr("lasagna.agentic.agentic_rag.AgenticRAG.create", agentic)
    out = asyncio.run(server.search("what did the ablation show?", session_id="s"))
    assert out.startswith("<knowledge_base>")
    assert '"degraded_stages": ["agentic"]' in out


def test_the_readme_says_what_happens_without_a_key():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "without an API key those queries run the profile's standard pipeline" in (
        " ".join(readme.replace(">", " ").split())
    )
