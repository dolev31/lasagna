"""Milvus Lite collections get a FLAT index.

Milvus Lite 3.x builds HNSW/AUTOINDEX with faiss inside the calling process
when a collection loads. On macOS, once torch has started its OpenMP runtime,
that build aborts ("OMP: Error #15") or deadlocks, so search from a new
process crashed once a collection held a few hundred chunks. FLAT has no
build step and is exact.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.api._collection_manager import _dense_index_type


def test_lite_gets_flat_whatever_the_setting():
    lite = SimpleNamespace(is_lite=True)
    for setting in ("AUTOINDEX", "HNSW", "IVF_FLAT"):
        assert _dense_index_type(lite, SimpleNamespace(index_type=setting)) == "FLAT"


def test_a_server_keeps_the_configured_index():
    server = SimpleNamespace(is_lite=False)
    assert _dense_index_type(server, SimpleNamespace(index_type="HNSW")) == "HNSW"
    assert _dense_index_type(server, SimpleNamespace(index_type="AUTOINDEX")) == "AUTOINDEX"
