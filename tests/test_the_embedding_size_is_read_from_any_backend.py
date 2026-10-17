"""The embedding size of a collection is read whatever the backend calls its vector field.

Found on the compose stack with Qdrant: once a model's collection existed,
applying a profile that uses that model answered 409, "could not read the
embedding size of 'rag_documents_v2__all_minilm_l6_v2__384'", although Qdrant
reported ``size: 384``. ``get_collection_embedding_dim`` looked only for a
field named ``embedding``, which is Milvus's name; Qdrant describes its vector
field as ``vector`` and Pinecone as ``values``, each with a ``dimension``.
"""

from __future__ import annotations

import pytest

from lasagna.api._collection_manager import CollectionManager


class _Backend:
    def __init__(self, fields):
        self.fields = fields

    def has_collection(self, collection_name):
        return True

    def describe_fields(self, collection_name):
        return self.fields


def _dim(fields):
    manager = CollectionManager.__new__(CollectionManager)
    return manager.get_collection_embedding_dim("c", backend=_Backend(fields))


@pytest.mark.parametrize(
    "fields",
    [
        # Qdrant
        [
            {"name": "id", "type": "VARCHAR"},
            {"name": "vector", "type": "FLOAT_VECTOR", "dimension": 384},
            {"name": "payload", "type": "JSON"},
        ],
        # Pinecone
        [
            {"name": "id", "type": "VARCHAR"},
            {"name": "values", "type": "FLOAT_VECTOR", "dimension": 384},
        ],
        # Milvus, as before
        [{"name": "chunk_id"}, {"name": "embedding", "params": {"dim": 384}}],
        # Milvus with its BM25 sparse field listed first
        [
            {"name": "sparse", "type": "DataType.SPARSE_FLOAT_VECTOR", "params": {}},
            {"name": "embedding", "type": "DataType.FLOAT_VECTOR", "params": {"dim": 384}},
        ],
    ],
    ids=["qdrant", "pinecone", "milvus", "milvus-with-sparse"],
)
def test_the_vector_field_gives_the_size(fields):
    assert _dim(fields) == 384


def test_a_backend_that_reports_no_size_gives_none():
    assert _dim([{"name": "content", "type": "text"}]) is None
