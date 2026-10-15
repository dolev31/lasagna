"""Each embedding model has its own vector collection, and a collection knows its model.

Every model wrote to one collection, ``rag_documents_v2``, told apart only by
its dimension. Configuring the speed profile for one session and uploading
rebuilt that collection at 384 dimensions (with "Dropping vector collection for
rebuild" on a first upload), and the next search in the default session failed
with "EMBEDDING DIMENSION MISMATCH", "would be DELETED", and a remedy that was a
Python method. bge-large and bge-m3, both 1024, shared vectors silently.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

import lasagna.api._collection_manager as cm
from lasagna.core import config as core_config
from lasagna.core.config import get_rag_config
from lasagna.core.database import RAGDatabase
from lasagna.infrastructure.embedding.manager import _config_signature


class _MemoryStore:
    """A vector store holding collection sizes and row counts, recording drops."""

    backend_name = "milvus"
    is_lite = True

    def __init__(self):
        self.collections = {}
        self.dropped = []

    def add(self, name, dim, rows):
        self.collections[name] = {"dim": dim, "rows": rows}

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, name, schema, index_config=None):
        dim = next(f.dimension for f in schema.fields if f.dtype == "FLOAT_VECTOR")
        self.add(name, dim, 0)

    def drop_collection(self, name):
        self.collections.pop(name)
        self.dropped.append(name)

    def get_collection_stats(self, name):
        return {"row_count": self.collections[name]["rows"]}

    def describe_fields(self, name):
        return [{"name": "embedding", "params": {"dim": self.collections[name]["dim"]}}]

    def describe_collection(self, name):
        return SimpleNamespace(fields=self.describe_fields(name))

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda **kw: None)

    def create_index(self, *args, **kwargs):
        pass

    def list_indexes(self, name):
        return ["embedding"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A manager over an in-memory vector store, stamping into a fresh database."""
    stamps = RAGDatabase(str(tmp_path / "rag.db"))
    monkeypatch.setattr(cm, "_stamp_store", lambda: stamps, raising=False)
    getattr(core_config, "forget_adopted_collections", lambda: None)()

    # The embedder is never loaded: a manager whose signature matches the config.
    def _manager(config):
        return SimpleNamespace(
            signature=_config_signature(config),
            model_name=config.embedding_model,
            dimension=getattr(config, "effective_embedding_dim", config.embedding_dim),
        )

    import lasagna.infrastructure.embedding as embedding
    import lasagna.retrieval.search as search

    monkeypatch.setattr(cm, "get_embedding_manager", _manager)
    monkeypatch.setattr(embedding, "reset_embedding_manager", lambda: None)
    monkeypatch.setattr(search, "reset_reranker", lambda: None)

    vectors = _MemoryStore()
    manager = cm.CollectionManager(vectors, "milvus", {}, _manager(get_rag_config()))
    yield SimpleNamespace(vectors=vectors, manager=manager, stamps=stamps)
    getattr(core_config, "forget_adopted_collections", lambda: None)()


def _config(**settings):
    return get_rag_config().with_query_overrides(settings)


BGE_M3 = {"embedding_model": "BAAI/bge-m3", "embedding_dim": 1024}
BGE_LARGE = {"embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dim": 1024}
MINILM = {"embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384}


def test_two_models_of_the_same_size_get_different_collections():
    large, m3 = _config(**BGE_LARGE).collection_name, _config(**BGE_M3).collection_name
    assert large != m3
    assert m3 == "rag_documents_v2__baai_bge_m3__1024"


def test_the_shipped_default_name_means_one_collection_per_model():
    # Every built-in profile carries collection_name = "rag_documents_v2".
    named = _config(collection_name="rag_documents_v2", **MINILM)
    assert named.collection_name == "rag_documents_v2__all_minilm_l6_v2__384"


def test_an_explicit_name_is_used_as_given():
    # The evaluation harness names a collection per arm; those must not move.
    assert _config(collection_name="rag_bench_emb_c4c84ef1", **MINILM).collection_name == (
        "rag_bench_emb_c4c84ef1"
    )


def test_a_stamped_collection_opened_with_another_model_is_refused_naming_both(store):
    store.manager.maybe_reload_embedding_manager(_config(collection_name="my_docs", **BGE_M3))
    store.vectors.collections["my_docs"]["rows"] = 40

    with pytest.raises(getattr(cm, "CollectionIdentityError", ValueError)) as raised:
        store.manager.maybe_reload_embedding_manager(
            _config(collection_name="my_docs", **BGE_LARGE)
        )
    message = str(raised.value)
    assert "BAAI/bge-m3" in message and "BAAI/bge-large-en-v1.5" in message
    assert "lasagna configure --profile standard -s default" in message
    assert "DELETED" not in message and "force_rebuild" not in message
    assert store.vectors.dropped == []


def test_a_size_mismatch_names_the_fix_and_never_a_python_method(store):
    store.vectors.add("my_docs", 384, rows=12)
    with pytest.raises(getattr(cm, "CollectionIdentityError", ValueError)) as raised:
        store.manager.maybe_reload_embedding_manager(_config(collection_name="my_docs", **BGE_M3))
    message = str(raised.value)
    assert "384" in message and "1024" in message
    assert "DELETED" not in message
    assert "backend." not in message and "()" not in message
    assert store.vectors.dropped == []


def test_a_new_profile_starts_its_own_collection_without_dropping_anything(store):
    """configure speed for one session, upload, then search the default session."""
    default = _config(**BGE_M3)
    store.manager.maybe_reload_embedding_manager(default)  # backend start: the default's
    seen = []
    sink = logger.add(lambda m: seen.append(m.record), level="INFO")
    try:
        speed = _config(**MINILM)
        store.manager.maybe_reload_embedding_manager(speed)  # first upload under speed
        store.vectors.collections[speed.collection_name]["rows"] = 30
        store.vectors.collections[default.collection_name]["rows"] = 5
        store.manager.maybe_reload_embedding_manager(default)  # the default session searches
    finally:
        logger.remove(sink)

    assert store.vectors.dropped == []
    assert store.vectors.collections[speed.collection_name]["dim"] == 384
    assert store.vectors.collections[default.collection_name]["dim"] == 1024
    assert not [r for r in seen if r["level"].no >= 30], [r["message"] for r in seen]
    assert not [r for r in seen if "Dropping vector collection" in r["message"]]


def test_rebuilding_an_empty_collection_is_information_not_a_warning(store):
    store.vectors.add("my_docs", 384, rows=0)
    seen = []
    sink = logger.add(lambda m: seen.append(m.record), level="INFO")
    try:
        store.manager.maybe_reload_embedding_manager(_config(collection_name="my_docs", **BGE_M3))
    finally:
        logger.remove(sink)
    assert store.vectors.collections["my_docs"]["dim"] == 1024
    assert not [r for r in seen if r["level"].no >= 30], [r["message"] for r in seen]


def test_a_legacy_collection_with_documents_is_adopted_under_its_old_name(store):
    store.vectors.add("rag_documents_v2", 384, rows=50)
    speed = _config(**MINILM)

    store.manager.maybe_reload_embedding_manager(speed)

    assert speed.collection_name == "rag_documents_v2"
    stamp = store.stamps.get_collection_meta("rag_documents_v2")
    assert (stamp["embedding_model"], stamp["embedding_dim"]) == ("all-MiniLM-L6-v2", 384)
    assert store.vectors.collections["rag_documents_v2"]["rows"] == 50


def test_a_legacy_collection_at_another_size_is_left_alone_and_listed(store):
    store.vectors.add("rag_documents_v2", 384, rows=50)
    default = _config(**BGE_M3)

    store.manager.maybe_reload_embedding_manager(default)  # no dimension error

    assert default.collection_name == "rag_documents_v2__baai_bge_m3__1024"
    assert store.vectors.collections["rag_documents_v2"] == {"dim": 384, "rows": 50}
    assert cm.legacy_collections(store.vectors) == [
        {"name": "rag_documents_v2", "rows": 50, "embedding_dim": 384}
    ]


def test_an_adopted_legacy_collection_is_not_listed_as_unused(store):
    store.vectors.add("rag_documents_v2", 384, rows=50)
    store.manager.maybe_reload_embedding_manager(_config(**MINILM))
    assert cm.legacy_collections(store.vectors) == []


def test_documents_without_a_stamp_are_stamped_by_the_model_that_opens_them(store):
    store.vectors.add("my_docs", 1024, rows=7)
    store.manager.maybe_reload_embedding_manager(_config(collection_name="my_docs", **BGE_M3))
    assert store.stamps.get_collection_meta("my_docs")["embedding_model"] == "BAAI/bge-m3"


def test_a_profile_switch_checks_the_collection_the_new_profile_uses(monkeypatch):
    """The guard read the shared collection, so a speed session's documents
    blocked every other session from switching to bge-m3."""
    import lasagna.api._profile_guard as guard

    looked_at = []

    def stored(collection, vector_backend=None):
        looked_at.append(collection)
        return 384 if collection == "rag_documents_v2__all_minilm_l6_v2__384" else None

    monkeypatch.setattr(guard, "stored_embedding_dim", stored)
    guard.refuse_embedding_change(
        before={"collection_name": "rag_documents_v2", **MINILM},
        after={"collection_name": "rag_documents_v2", **BGE_M3},
        change="Switching to profile 'standard'",
        kept="speed",
    )
    assert looked_at == ["rag_documents_v2__baai_bge_m3__1024"]


class TestTheStampTable:
    """The table both metadata stores keep (SQLite here; Postgres in the parity suite)."""

    def test_the_first_stamp_wins(self, tmp_path):
        db = RAGDatabase(str(tmp_path / "rag.db"))
        db.stamp_collection("c", "BAAI/bge-m3", 1024, lasagna_version="0.1.0")
        kept = db.stamp_collection("c", "other", 384)
        assert (kept["embedding_model"], kept["embedding_dim"]) == ("BAAI/bge-m3", 1024)
        assert kept["lasagna_version"] == "0.1.0" and kept["created_at"]

    def test_a_stamp_is_per_backend_and_can_be_forgotten(self, tmp_path):
        db = RAGDatabase(str(tmp_path / "rag.db"))
        db.stamp_collection("c", "m", 384, vector_backend="qdrant")
        assert db.get_collection_meta("c") is None
        assert db.get_collection_meta("c", vector_backend="qdrant")["embedding_dim"] == 384
        assert db.forget_collection_meta("c", vector_backend="qdrant")
        assert db.list_collection_meta() == []


def test_an_orphan_scan_ignores_documents_kept_in_another_collection():
    """A document in a speed session's collection is not missing from the
    default one; repair soft-deleted it."""
    from lasagna.lifecycle.orphan_detector import OrphanDetector

    rows = {
        "a": {"document_id": "a", "collection_name": "rag_documents_v2__all_minilm_l6_v2__384"},
        "b": {"document_id": "b", "collection_name": "rag_documents_v2__baai_bge_m3__1024"},
        "c": {"document_id": "c", "collection_name": None},
    }
    db = SimpleNamespace(
        list_all_active_document_ids=lambda session_id, only_ready=True: list(rows),
        get_documents_batch=lambda ids: [rows[i] for i in ids],
    )
    vectors = SimpleNamespace(list_document_ids=lambda name, session_id=None, limit=0: [])
    report = OrphanDetector(db=db, backend=vectors).detect("rag_documents_v2__baai_bge_m3__1024")
    assert sorted(o.document_id for o in report.orphans) == ["b", "c"]


def test_configuring_speed_for_one_session_leaves_the_default_session_searchable(
    store, monkeypatch
):
    """The reported sequence, through the real session configuration."""
    import uuid

    import lasagna.api._profile_guard as guard
    import lasagna.providers as providers
    from lasagna.core.session_config import get_knowledge_config_manager
    from lasagna.providers.session_provider import PersistentSessionProvider

    def stored(collection, vector_backend=None):
        entry = store.vectors.collections.get(collection)
        return entry["dim"] if entry and entry["rows"] else None

    monkeypatch.setattr(guard, "stored_embedding_dim", stored)
    monkeypatch.setattr(providers, "_session_provider", PersistentSessionProvider())
    speed_session, other_session = f"a-{uuid.uuid4().hex[:8]}", f"b-{uuid.uuid4().hex[:8]}"

    store.manager.initialize_collection()  # backend start
    get_knowledge_config_manager().set_active_profile("speed", session_id=speed_session)
    speed = get_rag_config().with_session(speed_session)
    assert speed.embedding_dim == 384
    store.manager.maybe_reload_embedding_manager(speed)  # its first upload
    store.vectors.collections[speed.collection_name]["rows"] = 20

    other = get_rag_config().with_session(other_session)
    store.manager.maybe_reload_embedding_manager(other)  # a default-profile search
    assert speed.collection_name != other.collection_name
    assert store.vectors.dropped == []


def test_an_upload_records_the_collection_its_vectors_go_to(monkeypatch):
    """Delete looked vectors up in the process-wide collection, so a speed
    session's deleted document stayed searchable."""
    import lasagna.core.database as database
    from lasagna.api._document_pipeline import DocumentPipeline
    from lasagna.core.schema import Document

    saved = {}

    class _DB:
        def save_document(self, document, file_path, collection_name=None):
            saved["collection_name"] = collection_name

    monkeypatch.setattr(database, "get_rag_database", lambda: _DB())
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._collection_manager = SimpleNamespace(_default_backend_name="milvus")
    pipeline._storage = SimpleNamespace(backend_name="local")
    speed = _config(**MINILM)
    monkeypatch.setattr(pipeline, "_get_session_config", lambda session_id: speed)
    doc = Document(
        document_id="doc_1", session_id="s1", filename="a.txt", file_type=".txt", file_size=1
    )
    pipeline._save_document_metadata(doc, "/tmp/a.txt")
    assert saved["collection_name"] == "rag_documents_v2__all_minilm_l6_v2__384"


def test_listing_a_documents_chunks_reads_its_sessions_collection(monkeypatch):
    from lasagna.api._document_pipeline import DocumentPipeline

    asked = []

    class _Vectors:
        def session_filter(self, collection, session_id):
            asked.append(collection)
            from lasagna.vector_backends.filters import FilterExpression

            return FilterExpression.eq("session_id", session_id)

        def query(self, collection_name, filter_expr, output_fields, limit):
            asked.append(collection_name)
            return [{"content": "x", "metadata": {"chunk_index": 0}}]

    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._collection_manager = SimpleNamespace(resolve_backend=lambda sid: _Vectors())
    speed = _config(**MINILM)
    assert speed.collection_name != get_rag_config().collection_name
    monkeypatch.setattr(pipeline, "_get_session_config", lambda session_id: speed)
    chunks = pipeline.get_document_chunks("doc_1", "s1")
    assert chunks and set(asked) == {speed.collection_name}


def test_hybrid_search_is_pointed_at_the_sessions_own_collection(monkeypatch):
    """On a backend with native hybrid search, a session in another model's
    collection was searched in the default collection."""
    import numpy as np

    from lasagna.api import _search_orchestrator as orch_mod

    registered = []
    vectors = SimpleNamespace(
        backend_name="milvus",
        session_filter=lambda c, s: None,
        search=lambda **kw: [],
    )
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = SimpleNamespace(
        _collection_name="rag_documents_v2__baai_bge_m3__1024",
        set_session_backend=lambda sid, backend, collection: registered.append(collection),
    )
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: vectors)
    o._default_backend = vectors
    speed = _config(**MINILM)
    o._get_session_config = lambda sid: speed
    o._maybe_reload_embedding_manager = lambda config: None
    arm = dict(
        enable_hybrid_search=False,
        enable_reranking=False,
        enable_query_routing=False,
        enable_query_classification=False,
        enable_search_cache=False,
        enable_hyde=False,
        **MINILM,
    )
    o.search("q", "s1", top_k=5, query_config_overrides=arm)
    assert registered == [speed.collection_name]
