"""The corpus pre-flight asks which expected documents exist, in bounded queries.

It used to list every document id in the collection by streaming all rows
through Milvus Lite's query iterator, which could block forever on a re-run or
a resume. It now asks `document_id in [...]` in batches, and splits a batch
whose reply fills the row cap so a many-chunk document is never missed.
"""

from __future__ import annotations

import os

import pytest

from lasagna.knowledge_evaluation_sdk import experiment
from lasagna.knowledge_evaluation_sdk.experiment import _present_document_ids


class _FakeBackend:
    """Each document has `chunks_per_doc` rows; replies are capped like Milvus."""

    def __init__(self, stored, chunks_per_doc, cap):
        self.stored, self.chunks_per_doc, self.cap, self.calls = set(stored), chunks_per_doc, cap, 0

    def query(self, collection_name, filter_expr, output_fields, limit):
        self.calls += 1
        wanted = set(
            filter_expr.to_milvus().split("[")[1].split("]")[0].replace('"', "").split(", ")
        )
        rows = [
            {"document_id": d}
            for d in sorted(wanted & self.stored)
            for _ in range(self.chunks_per_doc)
        ]
        return rows[: min(limit, self.cap)]


def test_present_ids_are_exact_even_when_replies_hit_the_cap(monkeypatch):
    monkeypatch.setattr(experiment, "_PRESENCE_QUERY_ROWS", 50)
    expected = {f"d{i:04d}" for i in range(1000)}
    stored = {f"d{i:04d}" for i in range(0, 1000, 2)}
    backend = _FakeBackend(stored, chunks_per_doc=7, cap=50)
    assert _present_document_ids(backend, "c", "s", expected) == stored


@pytest.mark.skipif(not os.environ.get("LASAGNA_MILVUS_DB_PATH"), reason="Milvus Lite is opt-in")
def test_against_milvus_lite(tmp_path):
    from lasagna.vector_backends.base import (
        CollectionSchema,
        FieldSchema,
        IndexConfig,
        IndexType,
        MetricType,
    )
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    backend = MilvusBackend(db_path=str(tmp_path / "p.db"))
    backend.connect()
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=64),
            FieldSchema(name="document_id", dtype="VARCHAR", max_length=64),
            FieldSchema(name="session_id", dtype="VARCHAR", max_length=64),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=4),
        ]
    )
    backend.create_collection(
        "p", schema, IndexConfig(index_type=IndexType.AUTOINDEX, metric_type=MetricType.COSINE)
    )
    rows = [
        {"chunk_id": f"{d}_c{k}", "document_id": d, "session_id": "s1", "embedding": [1.0, 0, 0, 0]}
        for d in ("a", "b")
        for k in range(5)
    ] + [
        {"chunk_id": "x_c0", "document_id": "x", "session_id": "other", "embedding": [1.0, 0, 0, 0]}
    ]
    backend.insert("p", rows)
    assert _present_document_ids(backend, "p", "s1", {"a", "b", "c", "x"}) == {"a", "b"}
    backend.disconnect()
