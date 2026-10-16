"""Hybrid search on a Milvus server runs the fusion path, without reporting degraded.

A Milvus server claims native hybrid search, but native hybrid reads a
``sparse_embedding`` field that the collections Lasagna creates do not have.
Every hybrid search tried it, failed, logged a warning and reported
``degraded_stages: ["hybrid"]``, and because the backend counted as native the
BM25 index was never filled, so the "fallback" was dense-only. Native hybrid
is claimed per collection now, only where the field exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.core.config import get_rag_config
from lasagna.retrieval.search.hybrid import HybridSearcher
from lasagna.utils import degraded
from lasagna.vector_backends.milvus_backend import MilvusBackend

COLLECTION = "rag_documents_v2__baai_bge_m3__1024"
CHUNKS = [
    ("d0_chunk_0", "d0", "tide tables for the harbor"),
    ("d1_chunk_0", "d1", "the bridge opens at noon"),
    ("d2_chunk_0", "d2", "lighthouse keeper logbook"),
]


def _server(fields):
    backend = MilvusBackend(uri="http://milvus.example:19530")
    backend.native_calls = []
    backend.describe_fields = lambda name: [{"name": f, "params": {}} for f in fields]
    backend.describe_collection = lambda name: {"fields": [{"name": f} for f in fields]}

    def hybrid_search(**kw):
        backend.native_calls.append(kw)
        if "sparse_embedding" not in fields:  # what a Milvus server answers
            raise RuntimeError("fieldName(sparse_embedding) not found")
        return []

    backend.hybrid_search = hybrid_search
    backend.iter_all = lambda **kw: [
        {"chunk_id": c, "document_id": d, "content": t, "metadata": {}} for c, d, t in CHUNKS
    ]
    backend.query = lambda **kw: [{"chunk_id": c} for c in kw["filter_expr"].value]
    backend.session_filter = lambda collection, session: None
    return backend


def _search(backend):
    searcher = HybridSearcher()
    searcher.set_vector_backend(backend, COLLECTION)
    searcher.hydrate_session_index("s1", backend, COLLECTION, None)
    config = get_rag_config().with_query_overrides(
        {"enable_hybrid_search": True, "hybrid_alpha": 0.5, "sparse_method": "bm25"}
    )
    with degraded.collecting() as stages:
        hits, metrics = searcher.search_hybrid(
            "lighthouse",
            [],
            "s1",
            top_k=5,
            query_embedding=[0.1] * 4,
            config=config,
            return_metrics=True,
        )
    return hits, metrics, stages


@pytest.mark.parametrize(
    "fields", [["chunk_id", "content", "embedding", "metadata"]], ids=["lasagna schema"]
)
def test_a_collection_without_the_sparse_field_uses_bm25_fusion_quietly(fields):
    backend = _server(fields)

    hits, metrics, stages = _search(backend)

    assert backend.native_calls == []
    assert stages == []
    assert metrics["hybrid_mode"] == "rank_bm25_fallback"
    assert [h.document_id for h in hits] == ["d2"]


def test_a_collection_with_the_sparse_field_uses_native_hybrid():
    backend = _server(["chunk_id", "content", "embedding", "sparse_embedding"])

    _, metrics, stages = _search(backend)

    assert len(backend.native_calls) == 1
    assert metrics["hybrid_mode"] == "native_milvus"
    assert stages == []


def test_milvus_lite_never_claims_native_hybrid(tmp_path):
    backend = MilvusBackend(db_path=str(tmp_path / "lite.db"))
    assert backend.collection_supports_native_hybrid(COLLECTION) is False
    assert SimpleNamespace(c=backend.capabilities).c.supports_hybrid_search is False
