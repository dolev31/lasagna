"""A Milvus Lite database written by one process must be searchable from the next.

Milvus Lite 3.x reopens a database file with its collections in the
``released`` state, so a backend that never calls ``load_collection`` works in
the process that created the data and fails in every later one: the Python
quickstart's second run, ``lasagna search`` after ``lasagna upload``, and any
server restart. The conformance suite could not see it, because it never
reopens a database, so this test does exactly that, in two real processes.

Opt-in like the conformance suite, so a plain ``pytest`` stays hermetic:

    LASAGNA_MILVUS_DB_PATH=./milvus_conformance.db pytest tests/test_milvus_reopen_in_new_process.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LASAGNA_MILVUS_DB_PATH"),
    reason="Milvus Lite is opt-in: set LASAGNA_MILVUS_DB_PATH to run",
)

_PHASE = textwrap.dedent(
    """
    import sys

    from lasagna.vector_backends.base import (
        CollectionSchema, FieldSchema, IndexConfig, IndexType, MetricType,
    )
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    DIM, NAME = 8, "reopen_regression"
    phase, db = sys.argv[1], sys.argv[2]
    query = [1.0] + [0.0] * (DIM - 1)
    backend = MilvusBackend(db_path=db)
    backend.connect()
    if phase == "create":
        schema = CollectionSchema(fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=DIM),
            FieldSchema(name="content", dtype="VARCHAR", max_length=200),
        ])
        backend.create_collection(
            NAME, schema,
            IndexConfig(index_type=IndexType.AUTOINDEX, metric_type=MetricType.COSINE),
        )
        backend.insert(NAME, [{"chunk_id": "c0", "embedding": query, "content": "hello"}])
    print(len(backend.search(NAME, query, limit=1)))
    print(len(backend.query(NAME, output_fields=["chunk_id"], limit=10)))
    backend.disconnect()
    """
)


def _run(phase: str, db: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PHASE, phase, db],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"{phase} phase failed:\n{proc.stderr[-2000:]}"
    return proc.stdout.split()[-2:]


def test_collection_is_searchable_from_a_new_process(tmp_path):
    db = str(tmp_path / "reopen.db")
    assert _run("create", db) == ["1", "1"]
    # A fresh interpreter opening the same file: search and query must both work.
    assert _run("reopen", db) == ["1", "1"]
