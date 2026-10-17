"""The check a search makes on where a session's documents are is one count, and rarely.

Every search asked whether some of the session's documents sit in another
model's collection by reading every row of the session, once every 3 seconds
per process: 0.73 s at 100,000 documents. It is now one ``GROUP BY
collection_name`` count, made again only when the session's documents changed
(its content generation moved), and never for the evaluation harness's
sessions, whose recorded latency it would have been part of.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.api import _profile_guard
from lasagna.core.database import RAGDatabase, get_rag_database

MINILM = "all-MiniLM-L6-v2"


@pytest.fixture
def calls(monkeypatch):
    seen = {"rows": 0, "counts": 0}
    rows = RAGDatabase.get_documents
    counts = getattr(RAGDatabase, "count_searched_documents", None)

    def get_documents(self, *a, **k):
        seen["rows"] += 1
        return rows(self, *a, **k)

    def count_searched_documents(self, *a, **k):
        seen["counts"] += 1
        return counts(self, *a, **k)

    monkeypatch.setattr(RAGDatabase, "get_documents", get_documents)
    if counts is not None:
        monkeypatch.setattr(RAGDatabase, "count_searched_documents", count_searched_documents)
    monkeypatch.setattr(_profile_guard, "_ELSEWHERE_TTL_SECONDS", 0.0)
    return seen


def _documents(sid: str, collection: str, n: int) -> None:
    for i in range(n):
        get_rag_database().save_document(
            {
                "document_id": f"d{i}_{sid}",
                "session_id": sid,
                "filename": f"{i}.md",
                "file_size": 1,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
            },
            "/nonexistent",
            collection_name=collection,
        )


def test_the_check_is_one_count_per_change_of_the_sessions_documents(calls):
    sid = f"count-{uuid.uuid4().hex[:8]}"
    _documents(sid, "elsewhere_collection", 3)

    first = _profile_guard.documents_elsewhere(sid, "searched_collection", MINILM)
    assert first.startswith(f"3 of 3 documents in session {sid!r}")
    for _ in range(5):
        assert _profile_guard.documents_elsewhere(sid, "searched_collection", MINILM) == first
    assert calls == {"rows": 0, "counts": 1}

    get_rag_database().bump_content_generation(sid)  # an upload, reindex or delete
    _profile_guard.documents_elsewhere(sid, "searched_collection", MINILM)
    assert calls == {"rows": 0, "counts": 2}


def test_the_count_groups_by_collection_and_skips_failed_documents():
    sid = f"count-{uuid.uuid4().hex[:8]}"
    _documents(sid, "a", 2)
    _documents(f"{sid}-other", "a", 1)
    get_rag_database().save_document(
        {
            "document_id": f"failed_{sid}",
            "session_id": sid,
            "filename": "f.md",
            "file_size": 1,
            "file_type": "md",
            "status": "error",
            "chunks_count": 0,
        },
        "/nonexistent",
        collection_name="b",
    )
    counts = get_rag_database().count_searched_documents(
        sid, sorted(_profile_guard._SEARCHED_STATUSES)
    )
    assert counts == {"a": 2}


def test_the_harness_sessions_are_not_checked(calls):
    from lasagna.api._search_orchestrator import _note_documents_elsewhere
    from lasagna.core.session_config import keep_in_process

    sid = f"bench_corpus_{uuid.uuid4().hex[:12]}"
    _documents(sid, "elsewhere_collection", 2)
    keep_in_process(sid)

    class _Config:
        collection_name, embedding_model = "searched_collection", "all-MiniLM-L6-v2"

    _note_documents_elsewhere(sid, _Config())
    assert calls == {"rows": 0, "counts": 0}
