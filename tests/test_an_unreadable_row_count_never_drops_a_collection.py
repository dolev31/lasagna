"""A Milvus collection whose row count cannot be read is never dropped as empty.

``MilvusBackend.get_collection_stats`` answered ``row_count: 0`` on any error
(a timeout, a server restart, a missing permission). `collection_is_empty`
treats "cannot tell" as "not empty" precisely because every session's
documents share the collection, but it never saw the error: the identity check
read the populated collection as empty and recreated it at the new size,
deleting every session's vectors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.api import _collection_manager as cm
from lasagna.vector_backends.milvus_backend import MilvusBackend


def _fields(dim):
    return [
        {"name": "chunk_id", "type": "VARCHAR", "params": {"max_length": 128}, "is_primary": True},
        {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": dim}},
        {"name": "metadata", "type": "JSON", "params": {}},
    ]


class _Client:
    """A Milvus server holding a 384-dimension collection whose statistics call fails."""

    def __init__(self):
        self.dropped = []

    def has_collection(self, name, **kw):
        return True

    def describe_collection(self, name, **kw):
        return {"collection_name": name, "description": "", "fields": _fields(384)}

    def get_collection_stats(self, name, **kw):
        raise TimeoutError("DEADLINE_EXCEEDED: GetCollectionStatistics")

    def flush(self, name, **kw):
        pass

    def drop_collection(self, name, **kw):
        self.dropped.append(name)


@pytest.fixture
def milvus(monkeypatch):
    monkeypatch.setattr(cm, "_stamp_store", lambda: None)
    backend = MilvusBackend(db_path="never-opened.db")
    backend._client = _Client()
    return backend


def test_failed_statistics_are_an_error_not_zero_rows(milvus):
    with pytest.raises(TimeoutError):
        milvus.get_collection_stats("rag_documents")
    assert cm.collection_is_empty(milvus, "rag_documents") is False


def test_the_identity_check_does_not_recreate_a_collection_it_cannot_count(milvus):
    manager = cm.CollectionManager(milvus, "milvus", {}, embedding_manager=None)
    config = SimpleNamespace(
        collection_name="rag_documents", embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768
    )
    with pytest.raises(cm.CollectionIdentityError, match="384"):
        manager.check_collection_identity(config)
    assert milvus._client.dropped == []
