"""Two sessions on different embedding models, served at once, keep their vectors apart.

The upload and search paths made a session's embedder current by assigning it
to an attribute every request shares (``DocumentPipeline.embedding_manager``,
``SearchOrchestrator.embedding_manager``), then read that attribute back a few
lines later. Two concurrent uploads on bge-large and bge-m3 (both 1024
dimensions, so no insert fails) could each read the other's model: one
collection received the other model's vectors, silently. Each switch also went
through ``reset_embedding_manager()``, which emptied the cache of loaded
models, so sessions taking turns loaded a model on every request.

Only the model class is faked here; the worker task, the pipeline, the
collection manager and the embedder cache are the real ones.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import lasagna.api._collection_manager as cm
import lasagna.api._document_pipeline as dp
import lasagna.api._search_orchestrator as so
import lasagna.infrastructure.embedding.manager as em
from lasagna.api._document_pipeline import DocumentPipeline
from lasagna.core import config as core_config
from lasagna.core.config import get_rag_config
from lasagna.core.database import RAGDatabase
from lasagna.core.schema import Document, DocumentStatus
from lasagna.infrastructure.tasks import document_tasks

LARGE = {"embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dim": 1024}
M3 = {"embedding_model": "BAAI/bge-m3", "embedding_dim": 1024}
# The value every vector of a model carries, so a stored vector names its model.
CODE = {"BAAI/bge-large-en-v1.5": 1.0, "BAAI/bge-m3": 2.0}
QUIET = {
    "enable_hybrid_search": False,
    "enable_contextual_retrieval": False,
    "enable_graph_retrieval": False,
    "enable_late_chunking": False,
    "enable_raptor": False,
    "enable_hierarchical": False,
    "enable_query_classification": False,
    "enable_query_routing": False,
    "enable_rag_fusion": False,
    "enable_query_expansion": False,
    "enable_search_cache": False,
    "enable_hyde": False,
    "enable_crag": False,
}


class _Model:
    """Stands in for EmbeddingManager: no weights, vectors that name the model."""

    created: list = []

    def __init__(self, config=None):
        config = config or core_config.rag_config
        self._signature = em._config_signature(config)
        self.model_name = config.embedding_model
        self.dimension = self.embedding_dim = config.embedding_dim
        _Model.created.append(self.model_name)

    @property
    def signature(self):
        return self._signature

    def _vector(self):
        return np.full(self.dimension, CODE.get(self.model_name, 9.0), dtype=np.float32)

    def encode_documents_batch(self, texts, show_progress=False):
        return [self._vector() for _ in texts]

    def encode_query(self, query):
        return self._vector()

    def get_cache_stats(self):
        return {}


class _Store:
    """An in-memory vector store that records what each collection receives."""

    backend_name = "milvus"
    is_lite = True

    def __init__(self):
        self.collections = {}
        self.inserted = {}

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, name, schema, index_config=None):
        dim = next(f.dimension for f in schema.fields if f.dtype == "FLOAT_VECTOR")
        self.collections[name] = {"dim": dim, "rows": 0}

    def drop_collection(self, name):
        self.collections.pop(name)

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

    def insert(self, collection_name, data, partition_key=None):
        self.collections[collection_name]["rows"] += len(data)
        self.inserted.setdefault(collection_name, set()).update(row["embedding"][0] for row in data)


@pytest.fixture
def world(tmp_path, monkeypatch):
    stamps = RAGDatabase(str(tmp_path / "rag.db"))
    monkeypatch.setattr(cm, "_stamp_store", lambda: stamps, raising=False)
    getattr(core_config, "forget_adopted_collections", lambda: None)()
    monkeypatch.setattr(em, "EmbeddingManager", _Model)
    monkeypatch.setattr(em, "_embedding_manager", None)
    monkeypatch.setattr(em, "_shared_managers", type(em._shared_managers)())
    monkeypatch.setattr(_Model, "created", [])
    quota = MagicMock()
    for module in (dp, so):
        monkeypatch.setattr(module, "get_quota_manager", lambda: quota)
    monkeypatch.setattr("lasagna.core.quota.get_quota_manager", lambda: quota)

    configs = {
        "s-large": get_rag_config().with_query_overrides({**LARGE, **QUIET}),
        "s-m3": get_rag_config().with_query_overrides({**M3, **QUIET}),
    }
    store = _Store()
    manager = cm.CollectionManager(store, "milvus", {}, em.get_embedding_manager(get_rag_config()))
    manager._get_session_config = lambda sid: configs[sid]

    # Both sessions reach the point where their collection is ready before
    # either goes on: the interleaving under which a shared attribute is
    # read back after the other session changed it.
    barrier = threading.Barrier(2, timeout=10)
    waited = threading.local()
    ensure = manager.ensure_backend_collection

    def ensure_then_wait(*args, **kwargs):
        ensure(*args, **kwargs)
        if getattr(world_state, "race", False) and not getattr(waited, "done", False):
            waited.done = True
            barrier.wait()

    world_state = SimpleNamespace(race=False)
    manager.ensure_backend_collection = ensure_then_wait
    yield SimpleNamespace(
        configs=configs, store=store, manager=manager, state=world_state, quota=quota
    )
    getattr(core_config, "forget_adopted_collections", lambda: None)()


def _pipeline(world):
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline.embedding_manager = world.manager.embedding_manager
    pipeline._collection_manager = world.manager
    pipeline._storage = SimpleNamespace(ensure_local=lambda p: p, get_local_path=lambda p: p)
    pipeline._rebuild_lock = threading.Lock()
    pipeline._rebuilding = False
    pipeline.hybrid_searcher = MagicMock()
    pipeline._should_use_docling = lambda path, config: False
    pipeline._extract_text_with_ocr = lambda path, file_type, config: f"text of {path}"
    pipeline._update_document_status = lambda *a, **k: None
    return pipeline


def _worker_backend(world, pipeline):
    docs = {}
    for session in world.configs:
        docs[f"doc-{session}"] = Document(
            document_id=f"doc-{session}",
            session_id=session,
            filename="a.txt",
            file_size=10,
            file_type="txt",
            status=DocumentStatus.PROCESSING,
            chunks_count=1,
        )
    return SimpleNamespace(
        _document_pipeline=pipeline,
        _get_document_metadata=lambda doc_id: docs[doc_id],
        get_session_config=lambda sid: world.configs[sid],
    )


def _run_together(*calls):
    errors = []

    def run(call):
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(c,)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors


def _chunks(text, document_id, session_id, filename, embedding_manager, config, **kw):
    return [(text, {"document_id": document_id, "session_id": session_id})]


def test_concurrent_uploads_write_each_models_vectors_to_its_own_collection(world, monkeypatch):
    monkeypatch.setattr(dp, "chunk_document", _chunks)
    backend = _worker_backend(world, _pipeline(world))
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: backend)
    world.state.race = True

    _run_together(
        *(
            (lambda s=s: document_tasks._process_document_impl(f"doc-{s}", s, f"/up/{s}.txt"))
            for s in world.configs
        )
    )

    large = world.configs["s-large"].collection_name
    m3 = world.configs["s-m3"].collection_name
    assert large != m3
    assert world.store.inserted == {large: {1.0}, m3: {2.0}}


class _Stop(Exception):
    """Ends a search right after the query is embedded."""


def _orchestrator(world):
    orch = so.SearchOrchestrator.__new__(so.SearchOrchestrator)
    orch.embedding_manager = world.manager.embedding_manager
    orch.hybrid_searcher = MagicMock()
    orch.reranker = SimpleNamespace(name="none")
    orch.query_enhancer = None
    orch.citation_manager = None
    orch._collection_manager = world.manager
    orch._document_pipeline = None
    orch._auto_merger = None
    orch._default_backend = world.store
    orch._get_session_config = lambda sid: world.configs[sid]

    def stop_after_encoding(session_id=None):
        raise _Stop

    world.manager.resolve_backend = stop_after_encoding
    return orch


def test_concurrent_searches_embed_each_query_with_its_sessions_model(world, monkeypatch):
    encoded = {}

    def encode_query(self, query):
        encoded[query] = self.model_name
        return self._vector()

    monkeypatch.setattr(_Model, "encode_query", encode_query)
    orch = _orchestrator(world)
    world.state.race = True

    def search(session):
        with pytest.raises(_Stop):
            orch.search(f"query from {session}", session, routing_tier="standard")

    _run_together(*((lambda s=s: search(s)) for s in world.configs))

    assert encoded == {
        "query from s-large": "BAAI/bge-large-en-v1.5",
        "query from s-m3": "BAAI/bge-m3",
    }


def test_sessions_taking_turns_load_each_model_once(world, monkeypatch):
    monkeypatch.setattr(dp, "chunk_document", _chunks)
    backend = _worker_backend(world, _pipeline(world))
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: backend)
    loaded_at_start = len(_Model.created)

    for _ in range(3):
        for session in world.configs:
            document_tasks._process_document_impl(f"doc-{session}", session, "/up/a.txt")

    loaded = _Model.created[loaded_at_start:]
    # bge-m3 is the default model, already loaded; bge-large loads once.
    assert loaded == ["BAAI/bge-large-en-v1.5"], loaded
