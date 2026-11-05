"""Hybrid search asks about the collection it searches, through a connection pool too (#50, #52).

The Compose stack sets ``RAG_CONNECTION_POOL_SIZE=10``, so its Milvus backend
is a `PooledVectorBackend`. The pool did not pass on
``collection_supports_native_hybrid``, and the hybrid searcher took a backend
without it for one that runs native hybrid search on every collection. The
collections Lasagna creates have no ``sparse_embedding`` field, so in the
stack every hybrid search tried the native call and failed. The same answer
skipped the benchmark's BM25 prebuild, so on a corpus ingested with hybrid
off, a hybrid arm then found its BM25 index empty and ran dense-only, scoring
exactly the control.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from lasagna.retrieval.search.hybrid import HybridSearcher
from lasagna.utils import degraded
from lasagna.vector_backends.base import VectorStoreAdapter
from lasagna.vector_backends.milvus_backend import MilvusBackend
from lasagna.vector_backends.pool import PooledVectorBackend

COLLECTION = "rag_bench_emb_de51dc75"
LASAGNA_FIELDS = ["chunk_id", "content", "embedding", "metadata"]
CHUNKS = [
    ("d0_chunk_0", "d0", "tide tables for the harbor"),
    ("d1_chunk_0", "d1", "the bridge opens at noon"),
    ("d2_chunk_0", "d2", "lighthouse keeper logbook"),
]


def _server(fields_by_collection):
    """A Milvus server backend whose collections have the given fields."""
    backend = MilvusBackend(uri="http://milvus.example:19530")
    backend.connect = lambda: None
    backend.native_calls = []
    backend.describe_fields = lambda name: [
        {"name": f, "params": {}} for f in fields_by_collection.get(name, [])
    ]

    def hybrid_search(**kw):
        backend.native_calls.append(kw)
        if "sparse_embedding" not in fields_by_collection.get(kw["collection_name"], []):
            raise RuntimeError("fieldName(sparse_embedding) not found")
        return []

    backend.hybrid_search = hybrid_search
    # The pool passes arguments on positionally.
    backend.iter_all = lambda *a, **kw: [
        {"chunk_id": c, "document_id": d, "content": t, "metadata": {}} for c, d, t in CHUNKS
    ]
    backend.session_filter = lambda *a, **kw: None
    return backend


def _pooled(server):
    pool = PooledVectorBackend(factory=lambda: server, pool_size=1)
    pool.connect()
    return pool


def test_a_pool_answers_for_the_collection_as_its_server_does():
    server = _server(
        {COLLECTION: LASAGNA_FIELDS, "with_sparse": LASAGNA_FIELDS + ["sparse_embedding"]}
    )
    pool = _pooled(server)
    assert pool.capabilities.supports_hybrid_search is True
    assert pool.collection_supports_native_hybrid(COLLECTION) is False
    assert pool.collection_supports_native_hybrid("with_sparse") is True


def test_through_a_pool_a_collection_without_the_field_uses_bm25_fusion_quietly():
    from lasagna.core.config import get_rag_config

    server = _server({COLLECTION: LASAGNA_FIELDS})
    pool = _pooled(server)
    searcher = HybridSearcher()
    searcher.set_vector_backend(pool, COLLECTION)
    searcher.hydrate_session_index("s1", pool, COLLECTION, None)
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
    assert server.native_calls == []
    assert stages == []
    assert metrics["hybrid_mode"] == "rank_bm25_fallback"
    assert [h.document_id for h in hits] == ["d2"]


def test_the_session_collection_is_asked_about_not_the_default():
    """Each embedder has its own collection; the default may have the field and it may not."""
    server = _server({"default": LASAGNA_FIELDS + ["sparse_embedding"], COLLECTION: LASAGNA_FIELDS})
    searcher = HybridSearcher()
    searcher.set_vector_backend(server, "default")
    assert searcher._backend_supports_native_hybrid("s1") is True
    # Before the first search registers the session (the benchmark's prebuild):
    assert searcher._backend_supports_native_hybrid("s1", COLLECTION) is False
    searcher.set_session_backend("s1", server, COLLECTION)
    assert searcher._backend_supports_native_hybrid("s1") is False


def test_no_backend_answer_is_not_a_yes():
    searcher = HybridSearcher()
    caps = SimpleNamespace(supports_hybrid_search=True, supports_native_bm25=True)
    searcher.set_vector_backend(SimpleNamespace(capabilities=caps, backend_name="x"), COLLECTION)
    assert searcher._backend_supports_native_hybrid("s1") is False
    searcher._collection_name = ""
    assert searcher._backend_supports_native_hybrid("s1") is False


def test_the_legacy_native_searcher_serves_only_its_own_collection():
    searcher = HybridSearcher()
    searcher._native_searcher = SimpleNamespace(
        collection_name="default", is_native_bm25_supported=lambda: True
    )
    searcher._collection_name = "default"
    assert searcher._legacy_native_serves("s1") is True
    assert searcher._legacy_native_serves("s1", COLLECTION) is False


def test_a_milvus_server_answers_a_hybrid_search_without_the_field_with_dense_results():
    """A direct caller is told, not handed the server's refusal or a silent dense list."""
    backend = MilvusBackend(uri="http://milvus.example:19530")
    backend.describe_fields = lambda name: [{"name": f} for f in LASAGNA_FIELDS]
    dense = [SimpleNamespace(id="c0")]
    backend.search = lambda **kw: dense
    with degraded.collecting() as stages:
        got = backend.hybrid_search(COLLECTION, query_text="q", query_embedding=[0.1] * 4)
        why = degraded.reasons()
    assert got is dense
    assert stages == ["hybrid"]
    assert (
        why["hybrid"]
        == f"collection {COLLECTION} has no sparse_embedding field; dense results only"
    )


#: Methods a backend keeps to itself: raw client passthroughs its own code
#: calls, never through the adapter interface (checked: no caller outside the
#: backend's module).
_BACKEND_INTERNAL = {
    "client",
    "insert_raw",
    "search_raw",
    "query_raw",
    "delete_raw",
    "filter",
    "translate",
}


def _backend_classes():
    import importlib

    found = []
    for name in ("milvus", "qdrant", "pgvector", "opensearch", "weaviate", "pinecone"):
        try:
            module = importlib.import_module(f"lasagna.vector_backends.{name}_backend")
        except ImportError:  # an optional client library not installed
            continue
        found += [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, VectorStoreAdapter) and cls.__module__ == module.__name__
        ]
    return found


@pytest.mark.parametrize("backend_class", _backend_classes(), ids=lambda c: c.__name__)
def test_the_pool_defines_everything_its_backends_define(backend_class):
    """The class of #50: the pool inherits the base default of what a backend overrides.

    It happened to ``read_fresh``, ``wait_until_searchable``, ``is_lite`` and
    ``collection_supports_native_hybrid``, each time making a pooled server
    behave as the default says rather than as the server does.
    """
    own = set()
    for klass in backend_class.__mro__:
        if klass is VectorStoreAdapter:
            break
        own |= {n for n in vars(klass) if not n.startswith("_")}
    missing = sorted(n for n in own - _BACKEND_INTERNAL if n not in vars(PooledVectorBackend))
    assert missing == [], f"PooledVectorBackend does not pass on {missing}"


def test_the_benchmark_prebuild_fills_bm25_for_a_hybrid_arm_on_a_pooled_server(monkeypatch):
    """#52: a corpus ingested with hybrid off, searched by an arm that turns it on."""
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    pool = _pooled(_server({COLLECTION: LASAGNA_FIELDS}))
    searcher = HybridSearcher()
    searcher.set_vector_backend(pool, "rag_documents")
    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="m", dimension=4),
        hybrid_searcher=searcher,
        _vector_backend=pool,
    )

    class _Pipeline:
        _component_names: dict = {}
        _instances: dict = {}

        def __init__(self):
            self._config = {"enable_hybrid_search": True, "collection_name": COLLECTION}

        def query(self, question, top_k, session_id, config):
            return [SimpleNamespace(chunk=SimpleNamespace(document_id="d2", content="x"))]

    async def _ingested(*a, **k):
        return {
            "session_id": "bench_corpus_s",
            "corpus_to_valid": {"d2": "d2"},
            "valid_to_corpus": {"d2": "d2"},
            "corpus_fingerprint": "fp",
        }

    async def _released(*a, **k):
        return None

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingested)
    monkeypatch.setattr(exp, "_release_corpus", _released)
    monkeypatch.setattr(exp, "_apply_overrides_to_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    monkeypatch.setattr(
        pipeline_mod.KnowledgePipeline, "from_profile", classmethod(lambda cls, p: _Pipeline())
    )
    task = BenchmarkTask(
        benchmark_name="smoke",
        config_name="hybrid",
        run_index=0,
        mode="live",
        config_overrides={"enable_hybrid_search": True},
        tracks=("retrieval",),
    )
    sample = BenchmarkSample(query_id="q", question="lighthouse?", expected_docs={"d2": 1})
    result = asyncio.run(exp._eval_live(task, [sample]))
    assert result.error is None
    index = searcher.get_session_index("bench_corpus_s")
    assert sorted(d["chunk_id"] for d in index.documents) == [c for c, _, _ in CHUNKS]
