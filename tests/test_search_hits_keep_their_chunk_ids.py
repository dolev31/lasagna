"""Search results must carry each chunk's real id, and MMR the current embedder.

pymilvus 3.x returns a hit's primary key under the field's own name
(``chunk_id``) and no ``id`` key; reading only ``id`` gave every chunk the id
"0", so citations and any map keyed by chunk id collapsed onto one chunk.
Separately, the reranker kept the embedding manager it was built with, so
after an embedder switch MMR compared vectors of two different dimensions.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.retrieval.search.ranker import bind_embedding_manager
from lasagna.vector_backends.milvus_backend import _hit_primary_key


def test_primary_key_is_read_under_either_name():
    assert _hit_primary_key({"id": "c1", "distance": 0.1}) == "c1"  # pymilvus 2.x
    assert _hit_primary_key({"chunk_id": "c2", "distance": 0.1}) == "c2"  # pymilvus 3.x
    assert _hit_primary_key({"entity": {"chunk_id": "c3"}}) == "c3"


def test_mmr_uses_the_embedder_of_the_current_search():
    old = SimpleNamespace(encode_documents_batch=lambda *a, **k: None, name="old-1024")
    new = SimpleNamespace(encode_documents_batch=lambda *a, **k: None, name="new-384")
    reranker = SimpleNamespace(mmr=SimpleNamespace(embedding_manager=old))
    bind_embedding_manager(reranker, new)
    assert reranker.mmr.embedding_manager is new

    wrapped = SimpleNamespace(_inner=SimpleNamespace(mmr=SimpleNamespace(embedding_manager=old)))
    bind_embedding_manager(wrapped, new)
    assert wrapped._inner.mmr.embedding_manager is new

    # A custom embedder without the manager API leaves MMR alone.
    bind_embedding_manager(reranker, object())
    assert reranker.mmr.embedding_manager is new
