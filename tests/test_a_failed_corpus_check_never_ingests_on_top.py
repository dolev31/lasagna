"""A benchmark corpus is never ingested on top of itself.

When the harness could not read which corpus documents were already in the
store, it ingested the whole corpus again on top of what was there. Milvus
``insert`` keeps both rows of an id, so every chunk was stored twice and a
dense ranked list held each one twice: a repeat took a place in the cut to
``k`` and the scored list came back short. The session's vectors are now
removed first, and a run that cannot remove them stops instead.

With that, a ranked list the harness scores cannot repeat a chunk id: its
corpora store each chunk once (``<document>_chunk_<i>``, ids unique per
corpus), and every fusion it can run (hybrid RRF, query fusion, BM25 and
SPLADE top-k) merges by chunk id.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import lasagna.api.rag_backend as rag_backend_mod
from lasagna.knowledge_evaluation_sdk import _fast_ingest as fast_ingest_mod
from lasagna.knowledge_evaluation_sdk import experiment as exp


@pytest.fixture
def world(monkeypatch, tmp_path):
    exp._corpus_sessions.clear()
    monkeypatch.setattr(exp, "_results_root", lambda: str(tmp_path))
    events = []
    store = SimpleNamespace(
        exists=True,
        delete_fails=False,
        session_filter=lambda c, s: ("session", s),
    )
    store.has_collection = lambda name: store.exists

    def delete(**kw):
        if store.delete_fails:
            raise ConnectionError("the store went away")
        events.append("delete")

    store.delete = delete
    backend = SimpleNamespace(
        _collection_manager=SimpleNamespace(resolve_backend=lambda s: store),
        hybrid_searcher=SimpleNamespace(clear_session=lambda s: events.append("clear")),
    )
    monkeypatch.setattr(rag_backend_mod, "get_rag_backend", lambda *a, **k: backend)

    def unreadable(*a):
        raise RuntimeError("query timed out")

    monkeypatch.setattr(exp, "_present_document_ids", unreadable)

    async def fast_ingest(corpus, corpus_to_valid, ready_ids, **kw):
        events.append("ingest")
        ingested = set(corpus_to_valid.values())
        return 3 * len(ingested), ingested, set()

    monkeypatch.setattr(fast_ingest_mod, "fast_ingest_corpus", fast_ingest)
    yield SimpleNamespace(events=events, store=store)
    exp._corpus_sessions.clear()


def test_what_is_there_is_removed_before_the_corpus_is_ingested(world):
    asyncio.run(exp._ensure_corpus_ingested("smoke", []))
    assert world.events[:3] == ["delete", "clear", "ingest"]


def test_a_new_collection_is_simply_ingested(world):
    world.store.exists = False
    asyncio.run(exp._ensure_corpus_ingested("smoke", []))
    assert world.events == ["clear", "ingest"]


def test_a_run_that_cannot_remove_them_stops(world):
    world.store.delete_fails = True
    with pytest.raises(Exception, match="not ingested on top"):
        asyncio.run(exp._ensure_corpus_ingested("smoke", []))
    assert "ingest" not in world.events
