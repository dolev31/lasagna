"""A delete on a Milvus server loads the collection first.

A Milvus server refuses a filtered delete on a collection that is not loaded
(``collection not loaded``). Queries and searches loaded the collection first;
deletes did not. In the default Compose stack the Celery worker clears what an
earlier attempt at a document left before it indexes, and that delete ran before
anything had loaded the collection, so every upload ended in ``error`` after its
retries ran out. The CI stack runs Qdrant, which has no such state, so nothing
caught it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")

from lasagna.vector_backends.base import FilterExpression  # noqa: E402 - imported after importorskip, which skips without pymilvus
from lasagna.vector_backends.milvus_backend import MilvusBackend  # noqa: E402 - imported after importorskip, which skips without pymilvus


class _Server:
    """What a Milvus server does: a delete needs the collection loaded."""

    def __init__(self):
        self.loaded = set()
        self.calls = []

    def get_load_state(self, name, timeout=None):
        return {"state": "<LoadState: Loaded>" if name in self.loaded else "<LoadState: NotLoad>"}

    def load_collection(self, name, timeout=None):
        self.calls.append(("load", name))
        self.loaded.add(name)

    def delete(self, collection_name, filter, **kwargs):
        if collection_name not in self.loaded:
            raise RuntimeError("collection not loaded[collection=470212715189633095]")
        self.calls.append(("delete", collection_name))
        return {"delete_count": 2}


@pytest.fixture
def backend(monkeypatch):
    made = MilvusBackend(uri="http://milvus:19530")
    server = _Server()
    monkeypatch.setattr(made, "_ensure_connected", lambda: server)
    made._test_server = server
    return made


def test_delete_loads_the_collection_before_it_deletes(backend):
    removed = backend.delete("chunks", FilterExpression.eq("document_id", "doc_1"))
    assert removed == 2
    assert backend._test_server.calls == [("load", "chunks"), ("delete", "chunks")]


def test_the_raw_delete_does_the_same(backend):
    backend.delete_raw("chunks", 'document_id == "doc_1"')
    assert backend._test_server.calls == [("load", "chunks"), ("delete", "chunks")]


def test_a_loaded_collection_is_not_loaded_again(backend):
    backend._test_server.loaded.add("chunks")
    backend.delete("chunks", FilterExpression.eq("document_id", "doc_1"))
    assert backend._test_server.calls == [("delete", "chunks")]


def test_a_second_delete_does_not_ask_again(backend):
    backend.delete("chunks", FilterExpression.eq("document_id", "doc_1"))
    backend.delete("chunks", FilterExpression.eq("document_id", "doc_2"))
    assert backend._test_server.calls == [
        ("load", "chunks"),
        ("delete", "chunks"),
        ("delete", "chunks"),
    ]


def test_a_recreated_collection_is_loaded_again(backend, monkeypatch):
    """Dropped and created again under the same name, it starts unloaded."""
    backend.delete("chunks", FilterExpression.eq("document_id", "doc_1"))
    backend._test_server.loaded.discard("chunks")  # the server dropped and recreated it
    monkeypatch.setattr(backend, "_to_milvus_schema", lambda schema: None)
    monkeypatch.setattr(backend._test_server, "create_collection", lambda **kw: None, raising=False)
    try:
        backend.create_collection("chunks", schema=None)
    except Exception:
        pass  # only the cache reset at the top of create_collection matters here
    backend.delete("chunks", FilterExpression.eq("document_id", "doc_1"))
    assert backend._test_server.calls[-2:] == [("load", "chunks"), ("delete", "chunks")]
