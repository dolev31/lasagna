"""Search results must carry each chunk's real id, and MMR the current embedder.

pymilvus 3.x returns a hit's primary key under the field's own name
(``chunk_id``) and no ``id`` key; reading only ``id`` gave every chunk the id
"0", so citations and any map keyed by chunk id collapsed onto one chunk.
Separately, the reranker kept the embedding manager it was built with, so
after an embedder switch MMR compared vectors of two different dimensions.
The search's embedder is now passed for its rerank call only: assigned to the
shared reranker, another session's search could replace it in between.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lasagna.retrieval.search.ranker import MMRReranker, RerankedResult, search_embedder
from lasagna.vector_backends.milvus_backend import _hit_primary_key


def test_primary_key_is_read_under_either_name():
    assert _hit_primary_key({"id": "c1", "distance": 0.1}) == "c1"  # pymilvus 2.x
    assert _hit_primary_key({"chunk_id": "c2", "distance": 0.1}) == "c2"  # pymilvus 3.x
    assert _hit_primary_key({"entity": {"chunk_id": "c3"}}) == "c3"


def test_mmr_uses_the_embedder_of_the_current_search():
    class _Embedder:
        def __init__(self, name):
            self.name, self.calls = name, 0

        def encode_documents_batch(self, texts, show_progress=False):
            self.calls += 1
            return [np.array([1.0, float(i)]) for i, _ in enumerate(texts)]

        def similarity(self, a, b):
            return float(np.dot(a, b))

    old, new = _Embedder("old-1024"), _Embedder("new-384")
    passages = [
        RerankedResult(
            chunk_id=f"c{i}",
            document_id="d",
            content=f"p{i}",
            metadata={},
            original_score=1.0 - i / 10,
            rerank_score=1.0 - i / 10,
            normalized_score=1.0 - i / 10,
        )
        for i in range(2)
    ]
    reranker = SimpleNamespace(mmr=MMRReranker(embedding_manager=old))
    with search_embedder(new):
        reranker.mmr.rerank_mmr("q", passages, top_k=2, force=True)
    assert (old.calls, new.calls) == (0, 1)
    # The search's embedder is used for that call only; the reranker keeps its own.
    assert reranker.mmr.embedding_manager is old

    # A custom embedder without the manager API leaves MMR with its own.
    with search_embedder(object()):
        reranker.mmr.rerank_mmr("q", passages, top_k=2, force=True)
    assert (old.calls, new.calls) == (1, 1)
