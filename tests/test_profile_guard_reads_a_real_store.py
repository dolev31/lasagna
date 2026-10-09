"""The profile guard's size check works against a real Milvus Lite store.

``lasagna configure`` refuses a profile whose embedding size differs from the
stored documents', using ``stored_dimension_mismatch``. Its only other test
replaces that function with a stub, so the detection itself had never run under
test. This runs it against a real collection.

Opt-in like the conformance suite (it starts Milvus Lite), so a plain
``pytest`` stays hermetic:

    LASAGNA_MILVUS_DB_PATH=./milvus_conformance.db pytest tests/test_profile_guard_reads_a_real_store.py
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LASAGNA_MILVUS_DB_PATH"),
    reason="Milvus Lite is opt-in: set LASAGNA_MILVUS_DB_PATH to run",
)


@pytest.fixture
def store(tmp_path):
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    backend = MilvusBackend(db_path=str(tmp_path / "guard.db"))
    backend.connect()
    yield backend
    backend.disconnect()


def _collection(backend, name: str, dim: int, rows: int) -> None:
    from lasagna.vector_backends.base import (
        CollectionSchema,
        FieldSchema,
        IndexConfig,
        IndexType,
        MetricType,
    )

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=dim),
            FieldSchema(name="content", dtype="VARCHAR", max_length=200),
        ]
    )
    backend.create_collection(
        name, schema, IndexConfig(index_type=IndexType.AUTOINDEX, metric_type=MetricType.COSINE)
    )
    if rows:
        backend.insert(
            name,
            [
                {"chunk_id": f"c{i}", "embedding": [1.0] + [0.0] * (dim - 1), "content": "x"}
                for i in range(rows)
            ],
        )


@pytest.fixture
def session():
    from lasagna.core.config import rag_config

    session_id = f"guard_{uuid.uuid4().hex[:8]}"
    return session_id, rag_config.with_session(session_id)


def test_documents_stored_at_another_size_are_detected(store, session):
    from lasagna.api._profile_guard import stored_dimension_mismatch

    session_id, config = session
    stored = 384 if int(config.embedding_dim) != 384 else 768
    _collection(store, config.collection_name, stored, rows=2)

    assert stored_dimension_mismatch(session_id, store) == (stored, int(config.embedding_dim))


def test_documents_stored_at_the_same_size_pass(store, session):
    from lasagna.api._profile_guard import stored_dimension_mismatch

    session_id, config = session
    _collection(store, config.collection_name, int(config.embedding_dim), rows=2)

    assert stored_dimension_mismatch(session_id, store) is None


def test_an_empty_collection_has_nothing_to_protect(store, session):
    from lasagna.api._profile_guard import stored_dimension_mismatch

    session_id, config = session
    _collection(store, config.collection_name, 384, rows=0)

    assert stored_dimension_mismatch(session_id, store) is None
