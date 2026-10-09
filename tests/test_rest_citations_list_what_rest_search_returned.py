"""``GET /api/v1/citations`` lists what a REST search just returned.

The list read only the in-memory results that MCP search stored, so for a
REST-only client it was always ``total: 0``, even right after
``POST /search/citations`` had returned citations.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lasagna.api.rest.dependencies import get_backend

_CITATION = SimpleNamespace(
    citation_id="c-1",
    document_id="d-1",
    document_name="tides.txt",
    chunk_id="k-1",
    chunk_index=0,
    page_number=None,
    score=0.9,
    content="Tides are caused by the moon.",
    highlights=["the moon"],
    highlight_ranges=[(21, 29)],
)


class _Backend:
    async def async_search_with_citations(self, **kwargs):
        return [], [_CITATION], "context"


def test_the_citations_list_shows_a_rest_searchs_citations():
    from lasagna.api.rest.routes import citations, search

    app = FastAPI()
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(citations.router, prefix="/api/v1")
    app.dependency_overrides[get_backend] = lambda: _Backend()
    client = TestClient(app)
    headers = {"X-Session-ID": "rest-citations"}

    searched = client.post("/api/v1/search/citations", json={"query": "tides"}, headers=headers)
    assert searched.status_code == 200, searched.text

    listed = client.get("/api/v1/citations", headers=headers).json()
    assert listed["total"] == 1
    cited = listed["citations"][0]
    assert cited["citation_id"] == "c-1"
    assert cited["content"] == "Tides are caused by the moon."
    assert cited == searched.json()["citations"][0]
