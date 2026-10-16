"""On a Milvus server, rows written moments ago count before a drop is decided.

A Milvus server's collection statistics count the rows of flushed segments
only: rows still in growing segments (written seconds ago, by this process or
another) read as ``row_count: 0``. `collection_is_empty` and the profile guard
counted that way with nothing flushed (``MilvusBackend.flush`` did nothing),
so a collection just written to read as empty, and the identity check rebuilt
it at a new size. They now flush first. Milvus Lite counts unflushed rows, so
there the flush stays a no-op (checked against a real store when
``LASAGNA_MILVUS_DB_PATH`` is set).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import lasagna.api._profile_guard as guard
from lasagna.api import _collection_manager as cm
from lasagna.vector_backends.milvus_backend import MilvusBackend

FIELDS = [
    {"name": "chunk_id", "type": "VARCHAR", "params": {"max_length": 128}, "is_primary": True},
    {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 384}},
]


class _Server:
    """A Milvus server holding 12 rows that are not flushed yet."""

    def __init__(self):
        self.growing, self.flushed, self.dropped = 12, 0, []

    def has_collection(self, name, **kw):
        return True

    def describe_collection(self, name, **kw):
        return {"collection_name": name, "description": "", "fields": FIELDS}

    def get_collection_stats(self, name, **kw):
        return {"row_count": self.flushed}

    def flush(self, name, **kw):
        self.flushed, self.growing = self.flushed + self.growing, 0

    def drop_collection(self, name, **kw):
        self.dropped.append(name)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(cm, "_stamp_store", lambda: None)
    backend = MilvusBackend(uri="http://milvus:19530")
    backend._client = _Server()
    return backend


def test_unflushed_rows_are_not_an_empty_collection(server):
    assert cm.collection_is_empty(server, "rag_documents") is False


def test_the_identity_check_keeps_a_collection_whose_rows_are_unflushed(server):
    manager = cm.CollectionManager(server, "milvus", {}, embedding_manager=None)
    config = SimpleNamespace(
        collection_name="rag_documents", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768
    )
    with pytest.raises(cm.CollectionIdentityError, match="384"):
        manager.check_collection_identity(config)
    assert server._client.dropped == []


def test_the_profile_guard_sees_unflushed_rows(server):
    assert guard.stored_embedding_dim("rag_documents", vector_backend=server) == 384


@pytest.mark.skipif(
    not os.environ.get("LASAGNA_MILVUS_DB_PATH"),
    reason="Milvus Lite is opt-in: set LASAGNA_MILVUS_DB_PATH to run",
)
def test_milvus_lite_counts_rows_it_has_not_flushed(tmp_path):
    from pymilvus import MilvusClient

    path = str(tmp_path / "lite.db")
    client = MilvusClient(path)
    try:
        client.create_collection("fresh", dimension=4)
        client.insert("fresh", [{"id": i, "vector": [0.1] * 4} for i in range(5)])
        backend = MilvusBackend(db_path=path)
        backend._client = client
        backend.flush("fresh")  # a no-op on Lite
        assert backend.get_collection_stats("fresh") == {"row_count": 5}
        assert cm.collection_is_empty(backend, "fresh") is False
    finally:
        client.close()
