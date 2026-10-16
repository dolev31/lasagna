"""The text log shows each request's id, and rate limits are counted by route.

The request id reached ``record.extra`` but only the JSON format printed it:
the deployed server logs text by default through loguru's own handler, whose
format has no extra fields, so one request's lines could not be told apart
from another's. And ``rag.http.rate_limited`` was labelled with the raw path,
because the middleware runs before routing sets ``scope["route"]``: one time
series per document id, and per URL a scanner tried.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from lasagna.api.rest import limits
from lasagna.api.rest.app import create_app
from lasagna.utils.monitoring import METRICS


def test_the_text_format_shows_the_request_id(monkeypatch):
    added = []
    monkeypatch.delenv("LASAGNA_LOG_FORMAT", raising=False)
    monkeypatch.setattr(
        limits,
        "logger",
        SimpleNamespace(remove=lambda handler: None, add=lambda sink, **kw: added.append(kw)),
    )

    assert limits.apply_log_format() is False
    (kwargs,) = added
    lines = []
    sink = logger.add(lines.append, format=kwargs["format"], colorize=False)
    try:
        with logger.contextualize(request_id="r-9"):
            logger.info("inside a request")
        logger.info("outside any request")
    finally:
        logger.remove(sink)

    assert "[r-9] inside a request" in lines[0]
    assert "[" not in lines[1].split(" - ", 1)[1]


def test_a_sink_an_entry_point_set_up_is_kept(monkeypatch):
    calls = []

    def remove(handler):
        calls.append(handler)
        raise ValueError("There is no existing handler with id 0")

    monkeypatch.delenv("LASAGNA_LOG_FORMAT", raising=False)
    monkeypatch.setattr(
        limits, "logger", SimpleNamespace(remove=remove, add=lambda *a, **k: calls.append("add"))
    )

    assert limits.apply_log_format() is False
    assert calls == [0]


@pytest.fixture
def refused(monkeypatch):
    """A rate limiter that refuses every request, and the counters it causes."""
    import lasagna.core.rate_limiter as rate_limiter

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    limiter = SimpleNamespace(
        is_enabled=True,
        try_acquire=lambda key, path: (False, 1.0),
        record_request_latency=lambda ms: None,
    )
    monkeypatch.setattr(rate_limiter, "get_rate_limiter", lambda: limiter)
    counted = []

    class _Recorder:
        def counter(self, name, value=1, tags=None):
            counted.append((name, dict(tags or {})))

        def histogram(self, name, value, tags=None):
            pass

    monkeypatch.setattr(METRICS, "_backend", _Recorder())
    return counted


def test_a_refused_request_is_counted_under_its_route_template(refused):
    client = TestClient(create_app())
    for doc_id in ("doc_a1", "doc_b2"):
        assert client.get(f"/api/v1/documents/{doc_id}").status_code == 429
    client.get("/wp-admin/setup.php")

    paths = [tags["path"] for name, tags in refused if name == "rag.http.rate_limited"]
    assert paths == ["/api/v1/documents/{document_id}"] * 2 + ["unmatched"]
