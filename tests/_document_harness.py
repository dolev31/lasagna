"""A document pipeline with real metadata, files and quota, and stand-in models and vectors.

What a document's life touches is real: the SQLite documents table, the local
file store, the quota manager and `DocumentPipeline` itself. The embedder, the
chunker and the vector store are small fakes, so a test runs in milliseconds
with no model and no network. The vector store either replaces a row that has
the same id (``upsert=True``: Qdrant, pgvector) or keeps both (Milvus
``insert``). ``LASAGNA_MILVUS_DB_PATH`` swaps in Milvus Lite itself.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from lasagna.core.config import get_rag_config
from lasagna.core.model_downloads import ModelNotDownloaded
from lasagna.vector_backends.filters import FilterExpression, FilterOperator

SESSION = "harness-session"

#: Two embedding models, so a reindex can move a document between collections.
MODEL_A = {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 8}
MODEL_B = {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 8}

#: Settings every harness config carries: no Docling, no caches, no LLM stages.
_PLAIN = {
    "docling_enabled": False,
    "enable_search_cache": False,
    "enable_hybrid_search": False,
    "enable_hierarchical": False,
    "enable_contextual_retrieval": False,
    "enable_graph_retrieval": False,
    "enable_raptor": False,
    "enable_late_chunking": False,
}


def matches(expr: Optional[FilterExpression], row: Dict[str, Any]) -> bool:
    """Whether ``row`` satisfies ``expr``, for the operators the pipeline uses."""
    if expr is None or expr.is_empty():
        return True
    op = expr.operator
    if op == FilterOperator.AND:
        return all(matches(c, row) for c in expr.children or [])
    if op == FilterOperator.OR:
        return any(matches(c, row) for c in expr.children or [])
    if op == FilterOperator.NOT:
        return not matches((expr.children or [None])[0], row)
    value = row.get(expr.field)
    if op == FilterOperator.EQ:
        return value == expr.value
    if op == FilterOperator.NE:
        return value != expr.value
    if op == FilterOperator.IN:
        return value in expr.value
    if op == FilterOperator.NOT_IN:
        return value not in expr.value
    raise NotImplementedError(op)


class MemoryVectors:
    """A vector store in a dict: ``upsert`` replaces a row with the same chunk id."""

    def __init__(self, *, upsert: bool = False, backend_name: str = "milvus"):
        self.upsert = upsert
        self.backend_name = backend_name
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self.fail_insert: Optional[Exception] = None

    def is_connected(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def insert(self, collection_name, data, partition_key=None):
        if self.fail_insert is not None:
            raise self.fail_insert
        rows = self.collections.setdefault(collection_name, [])
        for item in data if isinstance(data, list) else [data]:
            if self.upsert:
                rows[:] = [r for r in rows if r["chunk_id"] != item["chunk_id"]]
            rows.append(dict(item))
        return [d["chunk_id"] for d in data]

    def delete(self, collection_name, filter_expr, partition_key=None):
        rows = self.collections.get(collection_name, [])
        kept = [r for r in rows if not matches(filter_expr, r)]
        self.collections[collection_name] = kept
        return len(rows) - len(kept)

    def query(
        self,
        collection_name,
        filter_expr=None,
        output_fields=None,
        limit=100,
        offset=0,
        partition_key=None,
    ):
        rows = [r for r in self.collections.get(collection_name, []) if matches(filter_expr, r)]
        return [dict(r) for r in rows[offset : offset + limit]]

    def iter_all(self, collection_name, filter_expr=None, output_fields=None, **kw):
        yield from self.query(collection_name, filter_expr, limit=10**9)

    def session_filter(self, collection_name, session_id):
        return FilterExpression.eq("session_id", session_id)

    def flush(self, collection_name):
        pass


class LiteVectors:
    """Milvus Lite behind the same few calls, creating collections as they are used."""

    def __init__(self, path: str):
        from lasagna.vector_backends.milvus_backend import MilvusBackend

        self.backend = MilvusBackend(db_path=path)
        self.backend.connect()
        self.backend_name = "milvus"
        self.upsert = False
        self.fail_insert: Optional[Exception] = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def ensure(self, name: str, dim: int) -> None:
        if self.backend.has_collection(name):
            return
        from lasagna.core.schema import create_abstract_rag_schema
        from lasagna.vector_backends.base import IndexConfig, IndexType, MetricType

        index = IndexConfig(index_type=IndexType.AUTOINDEX, metric_type=MetricType.COSINE)
        self.backend.create_collection(name, create_abstract_rag_schema(dim), index)

    def insert(self, collection_name, data, partition_key=None):
        if self.fail_insert is not None:
            raise self.fail_insert
        self.ensure(collection_name, len(data[0]["embedding"]))
        ids = self.backend.insert(collection_name=collection_name, data=data)
        self.backend.flush(collection_name)
        return ids

    def delete(self, collection_name, filter_expr, partition_key=None):
        if not self.backend.has_collection(collection_name):
            return 0
        n = self.backend.delete(collection_name=collection_name, filter_expr=filter_expr)
        self.backend.flush(collection_name)
        return n

    def query(self, collection_name, filter_expr=None, output_fields=None, limit=100, **kw):
        if not self.backend.has_collection(collection_name):
            return []
        return self.backend.query(
            collection_name=collection_name,
            filter_expr=filter_expr,
            output_fields=output_fields or ["chunk_id", "document_id", "content", "session_id"],
            limit=limit,
        )

    @property
    def collections(self) -> Dict[str, List[Dict[str, Any]]]:
        names = self.backend._client.list_collections()
        return {n: self.query(n, FilterExpression.ne("chunk_id", ""), limit=10_000) for n in names}


class FakeEmbedder:
    """Deterministic vectors from the text, at the configured size."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(f"{self.model_name}|{text}".encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=self.dimension)
        return v / np.linalg.norm(v)

    def encode_documents_batch(self, texts, show_progress=False):
        return [self._vector(t) for t in texts]

    def encode_document(self, text):
        return self._vector(text)

    def encode_query(self, text):
        return self._vector(text)


class FakeCollections:
    """The collection manager's side of an upload: embedders and the store."""

    _default_backend_name = "milvus"

    def __init__(self, vectors):
        self.vectors = vectors
        self.not_downloaded: set = set()

    def embedder_for(self, config):
        if config.embedding_model in self.not_downloaded:
            raise ModelNotDownloaded(config.embedding_model, offline=True)
        return FakeEmbedder(config.embedding_model, config.embedding_dim)

    def resolve_backend(self, session_id=None):
        return self.vectors

    def get_vector_backend(self, session_id=None):
        return self.vectors

    def collection_is_empty(self, name, backend=None):
        return not self.vectors.query(name, None, limit=1)

    def recreate_collection_for_config(self, config, backend=None):
        self.vectors.collections[config.collection_name] = []

    def ensure_collection_for_config(self, config):
        pass

    def get_collection_embedding_dim(self, name):
        return 8


def split_words(text: str, size: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


class Harness:
    """A pipeline, a RAGBackend over it, and ways to change the session's settings."""

    def __init__(self, tmp_path, monkeypatch, vectors=None, *, quota=None):
        import lasagna.api._document_pipeline as pipeline_mod
        import lasagna.core.database as database
        import lasagna.core.quota as quota_mod
        from lasagna.api._document_pipeline import DocumentPipeline
        from lasagna.api.rag_backend import RAGBackend
        from lasagna.core.database import RAGDatabase
        from lasagna.processing.document_processor import DocumentProcessor
        from lasagna.storage.local_backend import LocalStorageBackend

        if vectors is None:
            lite = os.environ.get("LASAGNA_MILVUS_DB_PATH")
            vectors = LiteVectors(str(tmp_path / "harness_milvus.db")) if lite else MemoryVectors()
        self.vectors = vectors
        self.db = RAGDatabase(str(tmp_path / "rag.db"))
        self.uploads = tmp_path / "uploads"
        self.quota = quota or quota_mod.QuotaManager(enabled=False)
        self.settings: Dict[str, Any] = dict(MODEL_A)
        self.chunk_words = 4
        self.fail_chunking: Optional[Exception] = None

        monkeypatch.setattr(database, "get_rag_database", lambda: self.db)
        monkeypatch.setattr(quota_mod, "get_quota_manager", lambda: self.quota)
        monkeypatch.setattr(pipeline_mod, "get_quota_manager", lambda: self.quota)
        monkeypatch.setattr(pipeline_mod, "dispatch_processing", lambda *a, **k: False)
        monkeypatch.setattr(pipeline_mod, "get_backend", lambda name=None: self.vectors)
        monkeypatch.setattr(pipeline_mod, "chunk_document", self._chunk)

        self.collections = FakeCollections(vectors)
        searcher = SimpleNamespace(
            index_chunks=lambda *a, **k: None,
            remove_document=lambda *a, **k: None,
            note_local_change=lambda *a, **k: None,
            clear_session=lambda *a, **k: None,
        )
        processor = SimpleNamespace(
            detect_file_type=DocumentProcessor.detect_file_type,
            extract_text=lambda path, file_type: open(path, encoding="utf-8").read(),
        )
        pipeline = DocumentPipeline(
            document_processor=processor,
            embedding_manager=None,
            chunker=None,
            hybrid_searcher=searcher,
            collection_manager=self.collections,
            storage=LocalStorageBackend(str(self.uploads)),
        )
        pipeline._get_session_config = self.config
        self.pipeline = pipeline

        backend = object.__new__(RAGBackend)
        backend._initialized = True
        backend._document_pipeline = pipeline
        backend._collection_manager = self.collections
        backend.hybrid_searcher = searcher
        backend.get_session_config = self.config
        backend._get_session_config = self.config
        self.backend = backend

    # -- settings ---------------------------------------------------------

    def config(self, session_id: Optional[str] = None):
        return get_rag_config().with_query_overrides({**_PLAIN, **self.settings})

    def _chunk(self, text, document_id, session_id, filename, embedding_manager, config, **kw):
        if self.fail_chunking is not None:
            raise self.fail_chunking
        chunks = [
            (
                piece,
                {
                    "document_id": document_id,
                    "session_id": session_id,
                    "filename": filename,
                    "chunk_index": i,
                },
            )
            for i, piece in enumerate(split_words(text, self.chunk_words))
        ]
        return (chunks, None) if kw.get("return_parents") else chunks

    # -- documents ----------------------------------------------------------

    def write(self, tmp_path, name: str, text: str) -> str:
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    def upload(self, path: str, **kwargs):
        return self.pipeline.upload_document(path, SESSION, os.path.basename(path), **kwargs)

    def row(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_document(document_id)

    def chunks(self, document_id: str, collection: Optional[str] = None) -> List[Dict]:
        """The stored chunks of ``document_id`` in one collection, or in all of them."""
        names = [collection] if collection else list(self.vectors.collections)
        return [
            r
            for name in names
            for r in self.vectors.query(
                name, FilterExpression.eq("document_id", document_id), limit=10_000
            )
        ]

    def searchable(self, document_id: str) -> bool:
        """Ready, and its collection holds exactly the chunks its row counts."""
        row = self.row(document_id)
        if not row or row["status"] != "ready":
            return False
        stored = self.chunks(document_id, row.get("collection_name"))
        file_ok = bool(row.get("file_path")) and os.path.exists(row["file_path"])
        return file_ok and row["chunks_count"] > 0 and len(stored) == row["chunks_count"]

    def texts(self, document_id: str) -> List[str]:
        row = self.row(document_id)
        return sorted(c["content"] for c in self.chunks(document_id, row["collection_name"]))
