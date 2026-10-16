"""Two searches at once, on sessions with different embedding models, each run MMR with its own.

Every search shares one reranker. The orchestrator made a session's embedder
current for MMR by assigning it to that reranker (``bind_embedding_manager``)
and reranked a few steps later. When a second search on another model assigned
its own embedder in between, the first search's MMR re-embedded its passages
with the other session's model and compared them with those vectors: the
diversity step ranked one session's results with another session's embeddings
(or failed on two vector sizes). The embedder now travels with the request.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.retrieval.search.ranker import MMRReranker, MMRRerankerAdapter

SESSIONS = {"s-large": "BAAI/bge-large-en-v1.5", "s-m3": "BAAI/bge-m3"}
#: What a search reaches with MMR on: dense retrieval, then the reranker.
SETTINGS = dict(
    enable_hybrid_search=False,
    enable_reranking=True,
    enable_mmr=True,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=False,
    enable_hyde=False,
    enable_context_reordering=False,
    enable_hierarchical=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
)


class _Embedder:
    """One session's model: records the passages MMR asked it to embed."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.embedded = []

    def encode_documents_batch(self, texts, show_progress=False):
        self.embedded.extend(texts)
        return [np.array([1.0, float(i)]) for i, _ in enumerate(texts)]

    def similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def encode_query(self, query):
        return np.ones(2)

    def get_cache_stats(self):
        return {}


class _Retriever(BaseRetriever):
    """Returns passages that name the session; both searches meet here."""

    _forwards_session_config = True

    def __init__(self, barrier):
        self.barrier = barrier

    def retrieve(self, query, collection, top_k=5, **kwargs):
        # Both searches have made their embedder current before either ranks:
        # the interleaving under which a shared attribute is read back after
        # the other search changed it.
        self.barrier.wait()
        session = query.split()[-1]
        return [
            ScoredDocument(
                content=f"{session} passage {i}",
                score=0.9 - i / 10,
                document_id=f"{session}-d{i}",
                chunk_id=f"{session}-c{i}",
            )
            for i in range(3)
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
    configs = {
        sid: rag_config.with_query_overrides({"embedding_model": model, **SETTINGS})
        for sid, model in SESSIONS.items()
    }
    embedders = {model: _Embedder(model) for model in SESSIONS.values()}

    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = embedders["BAAI/bge-m3"]
    o.hybrid_searcher = None
    # One MMR reranker, built with no embedder, shared by every search.
    o.reranker = MMRRerankerAdapter()
    o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: configs[sid]
    o._session_embedder = lambda config: embedders[config.embedding_model]
    o.backend = backend
    return SimpleNamespace(orch=o, embedders=embedders)


def test_concurrent_searches_run_mmr_with_their_own_sessions_model(orchestrator):
    retriever = _Retriever(threading.Barrier(len(SESSIONS), timeout=10))
    results, errors = {}, []

    def search(session):
        try:
            results[session] = orchestrator.orch.search(
                f"query from {session}", session, top_k=3, _retriever_override=retriever
            )
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=search, args=(s,)) for s in SESSIONS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors, errors
    for session, model in SESSIONS.items():
        embedded = orchestrator.embedders[model].embedded
        assert embedded and {text.split()[0] for text in embedded} == {session}, (model, embedded)
        assert len(results[session]) == 3


def test_a_search_leaves_the_shared_reranker_unchanged(orchestrator):
    orchestrator.orch.search(
        "query from s-large",
        "s-large",
        top_k=3,
        _retriever_override=_Retriever(threading.Barrier(1)),
    )

    assert orchestrator.embedders["BAAI/bge-large-en-v1.5"].embedded
    inner = orchestrator.orch.reranker._inner
    assert isinstance(inner, MMRReranker)
    assert inner.embedding_manager is None
