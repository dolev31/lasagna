"""``search_params`` cannot change the metric a collection is searched with.

``metric_type`` is a server setting and ``search_params`` a session one, and
both Milvus search calls spread ``search_params`` after the metric, so a
``metric_type`` inside it won: an L2 search of a cosine index, its distances
read as similarities.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.core.config import get_rag_config
from lasagna.retrieval.search import hybrid
from lasagna.vector_backends.base import MetricType, SearchParams
from lasagna.vector_backends.milvus_backend import MilvusBackend


def test_the_milvus_search_keeps_the_configured_metric():
    seen = {}

    class _Client:
        def describe_collection(self, name, **kw):
            return {"fields": [{"name": "session_id"}]}

        def get_load_state(self, name, **kw):
            return {"state": "Loaded"}

        def search(self, **kw):
            seen.update(kw)
            return [[]]

    backend = MilvusBackend(db_path="never-opened.db")
    backend._client = _Client()
    backend.search(
        collection_name="c",
        query_embedding=[0.1, 0.2],
        limit=3,
        search_params=SearchParams(
            metric_type=MetricType.COSINE, params={"ef": 64, "metric_type": "L2"}
        ),
    )
    assert seen["search_params"] == {"ef": 64, "metric_type": "COSINE"}


def test_the_native_hybrid_search_keeps_the_configured_metric(monkeypatch):
    requests = []

    def request(**kw):
        requests.append(kw)
        return kw

    monkeypatch.setattr(hybrid, "AnnSearchRequest", request, raising=False)
    monkeypatch.setattr(hybrid, "RRFRanker", lambda *a, **k: None, raising=False)
    client = SimpleNamespace(hybrid_search=lambda **kw: [[]])
    searcher = hybrid.MilvusNativeHybridSearcher(client, "c", alpha=0.5)
    searcher._native_bm25_supported = True
    config = get_rag_config().with_query_overrides(
        {"metric_type": "COSINE", "search_params": {"ef": 64, "metric_type": "L2"}}
    )
    searcher.hybrid_search("q", [0.1, 0.2], "s1", top_k=3, config=config)
    dense = next(r for r in requests if r["anns_field"] == "embedding")
    assert dense["param"] == {"ef": 64, "metric_type": "COSINE"}
