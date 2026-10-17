"""The release gate's cold-start download includes every model the default uses.

The gate's one query, "What is the melting point of gallium?", is a short
factual question, which the default configuration's router sends to its fast
tier, and the fast tier does not rerank. So the reranker was never fetched, and
``model_download`` (4.6 GB) measured BGE-M3 alone, twice over, while the docs
it is checked against described BGE-M3 and the mxbai reranker. The journey now
asks a second question that is reranked.
"""

from __future__ import annotations

import json
import sys
import types

from e2e.phases import p2_cold_start as p2
from lasagna.core.config import rag_config
from lasagna.retrieval.enhancement import classify_query
from lasagna.retrieval.routing import RoutingTier, get_query_router


def _tier(query: str) -> RoutingTier:
    return get_query_router().route(query, None, classify_query(query), rag_config).tier


def _script_constant(name: str) -> str:
    line = next(x for x in p2._FIRST_SEARCH.splitlines() if x.startswith(f"{name} = "))
    return line.split(" = ", 1)[1].strip().strip('"')


def test_the_first_query_skips_reranking_so_a_second_one_is_needed():
    assert rag_config.enable_reranking and rag_config.enable_query_routing
    assert _tier(_script_constant("QUERY")) is RoutingTier.FAST


def test_the_second_query_is_one_the_default_configuration_reranks():
    assert _tier(_script_constant("RERANKED_QUERY")) is RoutingTier.STANDARD


class _Client:
    """Stands in for KnowledgeClient: records the queries, finds the answer."""

    queries: list = []

    def __init__(self, session_id=None):
        pass

    async def upload(self, path):
        return types.SimpleNamespace(document_id="doc1", chunk_count=2, status="ready")

    async def list_documents(self):
        return [types.SimpleNamespace(document_id="doc1", chunk_count=2, status="ready")]

    async def search_documents(self, query, top_k=5):
        _Client.queries.append(query)
        chunk = types.SimpleNamespace(content="Its melting point is 29.76 degrees Celsius")
        return [types.SimpleNamespace(chunk=chunk, score=0.9)]

    async def close(self):
        pass


def test_the_journey_reranks_after_timing_the_first_search(monkeypatch, tmp_path, capsys):
    doc = tmp_path / "gallium.txt"
    doc.write_text(p2.SAMPLE_DOC)
    monkeypatch.setenv("E2E_DOC", str(doc))
    monkeypatch.setenv("E2E_SESSION", p2.SESSION_ID)
    monkeypatch.setenv("E2E_CHUNK_POLL_S", "0")
    monkeypatch.setitem(sys.modules, "lasagna.sdk", types.SimpleNamespace(KnowledgeClient=_Client))
    _Client.queries = []

    exec(compile(p2._FIRST_SEARCH, "first_search.py", "exec"), {"__name__": "__main__"})

    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("E2E_JSON:"))
    out = json.loads(line[len("E2E_JSON:") :])
    assert _Client.queries == [_script_constant("QUERY"), _script_constant("RERANKED_QUERY")]
    assert out["ok"] and out["answer_present"] and out["reranked_n_results"] == 1
    assert out["first_search_elapsed"] <= out["elapsed"]
    expected = [rag_config.embedding_model]
    if rag_config.enable_reranking:
        expected.append(rag_config.reranker_model)
    assert out["models"] == expected


def test_a_model_counts_as_fetched_only_with_a_snapshot(tmp_path):
    (tmp_path / "models--BAAI--bge-m3" / "snapshots").mkdir(parents=True)
    (tmp_path / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").mkdir(
        parents=True
    )
    (tmp_path / "models--mixedbread-ai--mxbai-rerank-large-v1").mkdir()
    assert p2._in_cache(tmp_path, "BAAI/bge-m3")
    assert p2._in_cache(tmp_path, "all-MiniLM-L6-v2")
    assert not p2._in_cache(tmp_path, "mixedbread-ai/mxbai-rerank-large-v1")
