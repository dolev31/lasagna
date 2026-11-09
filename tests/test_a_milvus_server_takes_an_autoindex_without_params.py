"""An AUTOINDEX on a Milvus server is sent with its metric and nothing else.

A Milvus server refuses params with AUTOINDEX ("only metric type can be passed
when use AutoIndex"). IndexConfig's default params are HNSW's, so an AUTOINDEX
config carried them and a server refused it. Milvus Lite swaps in FLAT and never
saw it, and CI ran the conformance suite on Lite only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")

from lasagna.vector_backends.milvus_backend import MilvusBackend  # noqa: E402 - imported after importorskip, which skips without pymilvus


def test_an_autoindex_sends_only_its_metric(monkeypatch):
    from lasagna.vector_backends.base import IndexConfig, IndexType, MetricType

    added = {}

    class _Params:
        def add_index(self, **kwargs):
            added.update(kwargs)

    class _Client:
        def list_indexes(self, name, **kwargs):
            return []

        def prepare_index_params(self):
            return _Params()

        def create_index(self, name, params, **kwargs):
            pass

    backend = MilvusBackend(uri="http://milvus:19530")
    monkeypatch.setattr(backend, "_ensure_connected", lambda: _Client())
    backend.create_index(
        "chunks",
        "embedding",
        IndexConfig(index_type=IndexType.AUTOINDEX, metric_type=MetricType.COSINE),
    )
    assert added["index_type"] == "AUTOINDEX" and added["params"] == {}
    assert added["metric_type"] == "COSINE"


def test_another_index_keeps_its_params(monkeypatch):
    from lasagna.vector_backends.base import IndexConfig, IndexType, MetricType

    added = {}

    class _Params:
        def add_index(self, **kwargs):
            added.update(kwargs)

    class _Client:
        def list_indexes(self, name, **kwargs):
            return []

        def prepare_index_params(self):
            return _Params()

        def create_index(self, name, params, **kwargs):
            pass

    backend = MilvusBackend(uri="http://milvus:19530")
    monkeypatch.setattr(backend, "_ensure_connected", lambda: _Client())
    backend.create_index(
        "chunks",
        "embedding",
        IndexConfig(index_type=IndexType.HNSW, metric_type=MetricType.COSINE, params={"M": 16}),
    )
    assert added["index_type"] == "HNSW" and added["params"] == {"M": 16}


def test_the_collection_manager_sends_no_params_with_autoindex():
    """The other path to an index, through prepare_index_params, follows the same rule."""
    from types import SimpleNamespace

    from lasagna.api._collection_manager import CollectionManager

    added = {}

    class _Params:
        def add_index(self, **kwargs):
            added.update(kwargs)

    backend = SimpleNamespace(
        is_lite=False,
        backend_name="milvus",
        prepare_index_params=lambda: _Params(),
        create_index=lambda *a, **k: None,
    )
    config = SimpleNamespace(
        index_type="AUTOINDEX", metric_type="COSINE", index_params={"M": 16, "efConstruction": 256}
    )
    CollectionManager.create_index_on_backend(SimpleNamespace(), backend, "chunks", config)
    assert added["index_type"] == "AUTOINDEX" and added["params"] == {}
