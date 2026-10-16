"""An archived document leaves every worker's search results, not only the archiving one's.

Each API worker keeps its own search cache and checks the session's content
generation on every hit. Uploads and deletes bump it; archiving, restoring and
the garbage collector's TTL and error phases did not. So after one worker
archived a document, another worker kept serving it from its cache, as a hit
that skips the database, until the entry expired; and a restored document
stayed missing there the same way.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.api._document_pipeline import DocumentPipeline
from lasagna.core import database
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.database import RAGDatabase
from lasagna.infrastructure.cache.search_cache import SearchCache
from lasagna.lifecycle.gc import GarbageCollector
from lasagna.lifecycle.manager import DocumentLifecycleManager
from lasagna.lifecycle.policy import LifecyclePolicy
from lasagna.vector_backends.base import VectorStoreAdapter

SESSION = "s-shared"
DOCS = ("d1", "d2")
SETTINGS = dict(
    enable_search_cache=True,
    enable_hybrid_search=False,
    enable_reranking=False,
    enable_mmr=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_hyde=False,
    enable_context_reordering=False,
    enable_hierarchical=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
    similarity_threshold=0.0,
)


class _Retriever(BaseRetriever):
    """The vector store: archiving leaves the vectors in place."""

    _forwards_session_config = True

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(
                content=f"passage of {d}", score=0.9 - i / 10, document_id=d, chunk_id=f"{d}-c0"
            )
            for i, d in enumerate(DOCS)
        ]


class _Backend:
    backend_name = "fake"
    session_filter = VectorStoreAdapter.session_filter


class _Embedder:
    model_name = "stub"

    def encode_query(self, query):
        return [1.0, 0.0]

    def get_cache_stats(self):
        return {}


@pytest.fixture
def db(tmp_path, monkeypatch):
    store = RAGDatabase(str(tmp_path / "metadata.db"))
    monkeypatch.setattr(database, "_rag_database", store)
    for doc_id in DOCS:
        store.save_document(
            {
                "document_id": doc_id,
                "session_id": SESSION,
                "filename": f"{doc_id}.txt",
                "file_size": 10,
                "file_type": "txt",
                "status": "ready",
                "chunks_count": 1,
            },
            file_path=f"/tmp/{doc_id}.txt",
        )
    return store


class _Worker:
    """One API process: its own orchestrator and its own search cache."""

    def __init__(self, monkeypatch):
        backend = _Backend()
        self.cache = SearchCache()
        config = rag_config.with_query_overrides(SETTINGS)
        o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
        o.embedding_manager = _Embedder()
        o.hybrid_searcher = None
        o.reranker = o.query_enhancer = o._auto_merger = None
        o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
        o._document_pipeline = DocumentPipeline.__new__(DocumentPipeline)
        o._default_backend = o.backend = backend
        o._get_session_config = lambda sid: config
        o._session_embedder = lambda config: o.embedding_manager
        self.orchestrator = o
        self.keyword_index_at = []
        self.pipeline = SimpleNamespace(
            hybrid_searcher=SimpleNamespace(
                note_local_change=lambda sid, gen: self.keyword_index_at.append(gen)
            ),
            _invalidate_search_cache=self.cache.invalidate_session,
            _remove_from_session_focus=lambda doc_id, sid: None,
            delete_document=lambda doc_id, sid: True,
        )
        self._monkeypatch = monkeypatch

    def search(self):
        self._monkeypatch.setattr(orch_mod, "get_search_cache", lambda config=None: self.cache)
        results = self.orchestrator.search(
            "tides", SESSION, top_k=2, _retriever_override=_Retriever()
        )
        return sorted(r.document.document_id for r in results)

    def lifecycle(self, db, **policy):
        return DocumentLifecycleManager(
            db=db,
            pipeline=self.pipeline,
            backend=None,
            policy=LifecyclePolicy(**policy),
            collection_name="chunks",
        )


@pytest.fixture
def workers(db, monkeypatch):
    return _Worker(monkeypatch), _Worker(monkeypatch)


def test_a_document_archived_by_another_worker_is_not_served_from_this_workers_cache(db, workers):
    reader, archiver = workers
    assert reader.search() == ["d1", "d2"]
    assert reader.search() == ["d1", "d2"]
    assert reader.cache.stats()["hits"] == 1, "the second search must be a cache hit"

    archiver.lifecycle(db).soft_delete("d1", SESSION)

    assert reader.search() == ["d2"]
    # The chunks stay in the store, so the archiving process keeps its keyword index.
    assert archiver.keyword_index_at == [db.get_content_generation(SESSION)]


def test_a_document_restored_by_another_worker_comes_back_in_this_workers_results(db, workers):
    reader, archiver = workers
    archiver.lifecycle(db).soft_delete("d1", SESSION)
    assert reader.search() == ["d2"]
    assert reader.search() == ["d2"]  # cached without d1

    archiver.lifecycle(db).restore("d1", SESSION)

    assert reader.search() == ["d1", "d2"]


def test_a_document_the_gc_expires_leaves_every_workers_cache(db, workers):
    reader, collector = workers
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with db.transaction() as conn:
        conn.execute("UPDATE rag_documents SET expires_at = ? WHERE document_id = 'd1'", (past,))
    assert reader.search() == ["d1", "d2"]
    assert reader.search() == ["d1", "d2"]

    gc = GarbageCollector(
        db=db, pipeline=collector.pipeline, policy=LifecyclePolicy(document_ttl_days=1)
    )
    assert gc._phase2_expire_by_ttl().docs_cleaned == 1

    assert reader.search() == ["d2"]


def test_archiving_an_archived_document_changes_nothing(db):
    db.soft_delete_document("d1")
    before = db.get_content_generation(SESSION)
    assert db.soft_delete_document("d1") is False
    assert db.get_content_generation(SESSION) == before
