"""The profile guard reads the stored embedding size from a real Milvus Lite store.

Every config write that changes the embedding size is checked against the
stored documents (``refuse_embedding_change``), and the check rests on
``stored_embedding_dim``. Its other tests replace that function with a stub, so
reading a real collection had never run under test. This does.

Opt-in like the conformance suite (it starts Milvus Lite), so a plain
``pytest`` stays hermetic:

    LASAGNA_MILVUS_DB_PATH=./milvus_conformance.db pytest tests/test_profile_guard_reads_a_real_store.py
"""

from __future__ import annotations

import os

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


def test_the_stored_size_is_read_from_the_collection(store):
    from lasagna.api._profile_guard import stored_embedding_dim

    _collection(store, "guard_sized", 384, rows=2)
    assert stored_embedding_dim("guard_sized", store) == 384


def test_an_empty_collection_has_nothing_to_protect(store):
    from lasagna.api._profile_guard import stored_embedding_dim

    _collection(store, "guard_empty", 384, rows=0)
    assert stored_embedding_dim("guard_empty", store) is None


def test_a_missing_collection_has_nothing_to_protect(store):
    from lasagna.api._profile_guard import stored_embedding_dim

    assert stored_embedding_dim("guard_absent", store) is None


def test_a_change_to_another_size_is_refused_against_real_documents(store):
    from lasagna.api._profile_guard import EmbeddingSizeConflict, refuse_embedding_change

    _collection(store, "guard_refuse", 384, rows=2)
    before = {"embedding_dim": 384, "collection_name": "guard_refuse"}
    after = {"embedding_dim": 1024, "collection_name": "guard_refuse"}
    with pytest.raises(EmbeddingSizeConflict, match="384"):
        refuse_embedding_change(
            before=before, after=after, change="switching", kept="speed", vector_backend=store
        )


def test_a_change_to_the_stored_size_is_allowed(store):
    from lasagna.api._profile_guard import refuse_embedding_change

    _collection(store, "guard_allow", 768, rows=2)
    before = {"embedding_dim": 384, "collection_name": "guard_allow"}
    after = {"embedding_dim": 768, "collection_name": "guard_allow"}
    refuse_embedding_change(
        before=before, after=after, change="switching", kept="speed", vector_backend=store
    )
