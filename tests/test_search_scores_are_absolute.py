"""Search reports an absolute score, and every surface returns the same results.

Scores were min-max normalized per query, so the top hit always read 1.00,
even for a query the documents cannot answer. REST and MCP then dropped
results below ``similarity_threshold`` (0.3) on that relative score while the
CLI and SDK did not: one query in one session returned 5 results from the CLI,
3 from REST at top_k 5 and 2 from MCP at top_k 3. The evaluation's generation
track still filters with the session threshold on the normalized score, and
must keep doing so exactly.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from tests._paths import PACKAGE_DIR

#: Dense-search scores of three chunks for one query: none of them is a match.
SCORES = [0.62, 0.55, 0.51]


class _Retriever(BaseRetriever):
    _forwards_session_config = True

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(content=f"c{i}", score=s, document_id=f"d{i}", chunk_id=f"c{i}")
            for i, s in enumerate(SCORES)
        ]


class _Backend:
    backend_name = "fake"

    def search(self, **kw):
        raise AssertionError("the store's own search ran instead of the retriever")


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

    _Backend.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._maybe_reload_embedding_manager = lambda config: None
    o.backend = backend
    return o


#: The speed profile's pipeline: dense search, no reranking, a 0.3 threshold.
SPEED = dict(
    enable_hybrid_search=False,
    enable_reranking=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=False,
    enable_hyde=False,
    enable_mmr=False,
    enable_context_reordering=False,
    enable_hierarchical=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
    similarity_threshold=0.3,
)


def _search(orchestrator, **kw):
    return orchestrator.search(
        "q", "s", top_k=5, _retriever_override=_Retriever(), query_config_overrides=SPEED, **kw
    )


def test_the_top_hit_reports_its_own_score_not_one(orchestrator):
    out = _search(orchestrator, min_score=None)
    assert [round(r.score, 4) for r in out] == SCORES


def test_with_no_threshold_nothing_is_dropped(orchestrator):
    assert len(_search(orchestrator, min_score=None)) == 3


def test_a_threshold_the_caller_names_applies_to_the_reported_score(orchestrator):
    out = _search(orchestrator, min_score=0.54)
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]


def test_the_session_threshold_keeps_its_old_cut_for_internal_callers(orchestrator):
    # The evaluation's generation track searches with the sentinel: the 0.3
    # threshold on the per-query min-max score drops the lowest chunk (0.0),
    # as it always has. The scores it reports are still the absolute ones.
    out = _search(orchestrator)
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
    assert [round(r.score, 4) for r in out] == SCORES[:2]


def test_serving_surfaces_default_to_no_threshold():
    from lasagna.api.rag_backend import RAGBackend
    from lasagna.api.rest.models import SearchRequest
    from lasagna.sdk.client import KnowledgeClient

    assert SearchRequest(query="q").min_score is None
    assert (
        inspect.signature(KnowledgeClient.search_documents).parameters["min_score"].default is None
    )
    assert (
        inspect.signature(RAGBackend.search_with_citations).parameters["min_score"].default is None
    )


def test_a_rest_result_carries_its_rank():
    from lasagna.api.rest.routes.search import _result_to_response

    r = SimpleNamespace(
        chunk=SimpleNamespace(
            chunk_id="c", content="x", document_id="d", chunk_index=0, page_number=None, metadata={}
        ),
        score=0.61,
        document=SimpleNamespace(filename="a.md"),
    )
    out = _result_to_response(r, 2)
    assert (out.rank, out.score) == (2, 0.61)


def test_a_citation_reports_the_search_score():
    from lasagna.agentic.citation_manager import Citation

    fields = dict(
        citation_id="x",
        index=1,
        chunk_id="c",
        document_id="d",
        document_name="a.md",
        content="text",
        content_preview="text",
        score=0.61,
        normalized_score=1.0,
        chunk_index=0,
    )
    assert Citation(**fields).to_dict()["score"] == 0.61


SURFACES = r"""
import asyncio, json, os, sys
from lasagna.sdk import KnowledgeClient, config_for_speed

Q = "how do I bake sourdough bread at home"
DOC = "\n\n".join(
    f"Section {i}. Lasagna measures retrieval components on benchmark {i} with paired tests."
    for i in range(8)
)

async def main():
    open("notes.md", "w").write(DOC)
    client = KnowledgeClient(session_id="corpus", config=config_for_speed())
    doc = await client.upload("notes.md")
    await client.documents.wait_for_processing(doc.document_id)
    sdk = await client.search_documents(Q, top_k=3)
    await client.close()

    from fastapi.testclient import TestClient
    from lasagna.api.rest.app import create_app

    with TestClient(create_app()) as c:
        rest = c.post("/api/v1/search", headers={"X-Session-ID": "corpus"},
                      json={"query": Q, "top_k": 3}).json()["results"]
    from lasagna.mcp import server
    out = await server.search(Q, top_k=3)
    mcp = json.loads(out.split("<search_metadata>")[1].split("</search_metadata>")[0])
    print(json.dumps({
        "sdk": [round(r.score, 4) for r in sdk.results],
        "rest": [round(r["score"], 4) for r in rest],
        "mcp": [round(c["score"], 4) for c in mcp["citations"]],
    }))

asyncio.run(main())
"""


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_sdk_rest_and_mcp_return_the_same_results(tmp_path):
    (tmp_path / "surfaces.py").write_text(SURFACES)
    # None of this suite's settings: its shared data directory is held open.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        RAG_SESSION_ID="corpus",
    )
    run = subprocess.run(
        [sys.executable, "surfaces.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    seen = json.loads(run.stdout.strip().splitlines()[-1])
    assert len(seen["sdk"]) == 3, seen
    assert seen["sdk"] == seen["rest"] == seen["mcp"], seen
    assert max(seen["sdk"]) < 0.99, "an unanswerable query's top hit read as a perfect match"
