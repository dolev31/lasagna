"""A worker's search cache stops serving results once another process changes the session.

Found on the live stack (two gunicorn workers, Celery): after a document was
deleted, a search kept returning it, a cache hit, for up to the cache's TTL
(300 s). The search cache is per process, and ``invalidate_session`` runs only
in the process that took the upload or the delete. An entry now records the
session's content generation (bumped by every upload and delete) that its
search started from, and a lookup that finds another one is a miss.
"""

from __future__ import annotations

import pytest

from lasagna.core import database
from lasagna.core.database import RAGDatabase
from lasagna.infrastructure.cache.search_cache import SearchCache

SESSION = "s-shared"
HIT = [{"chunk_id": "d1_chunk_0", "document_id": "d1", "score": 0.9}]


@pytest.fixture
def db(tmp_path, monkeypatch):
    store = RAGDatabase(str(tmp_path / "metadata.db"))
    monkeypatch.setattr(database, "_rag_database", store)
    return store


def _search(cache):
    """What a search does with the cache: look up, and on a miss store its result."""
    found = cache.get("tides", SESSION)
    if found is None:
        cache.put("tides", SESSION, HIT, [0.1, 0.2])
        return "miss"
    return "hit"


def test_a_delete_in_another_process_ends_this_workers_cached_result(db):
    worker = SearchCache()
    assert [_search(worker), _search(worker)] == ["miss", "hit"]

    db.bump_content_generation(SESSION)  # another worker deleted a document

    assert _search(worker) == "miss"
    assert _search(worker) == "hit"


def test_a_change_during_the_search_is_not_hidden_by_its_result(db):
    worker = SearchCache()
    assert worker.get("tides", SESSION) is None  # the search starts
    db.bump_content_generation(SESSION)  # an upload lands while it runs
    worker.put("tides", SESSION, HIT, [0.1, 0.2])  # its result predates the upload

    assert worker.get("tides", SESSION) is None


def test_an_unchanged_session_keeps_hitting(db):
    worker = SearchCache()
    _search(worker)
    assert [_search(worker) for _ in range(3)] == ["hit"] * 3


def test_an_unreadable_database_keeps_the_cache_as_it_was(db, monkeypatch):
    worker = SearchCache()
    _search(worker)

    def unreadable(session_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "get_content_generation", unreadable)
    assert _search(worker) == "hit"
