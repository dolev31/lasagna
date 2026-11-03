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
