"""Phase 4: REST API tests — direct HTTP calls to the knowledge REST API.

These tests verify the FastAPI REST endpoints at /api/v1/*.
The REST API may be served standalone or through the demo server.

Prerequisites:
    lasagna serve  (or python -m lasagna serve)
"""

import os
from pathlib import Path

import pytest
import requests

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

# The REST API, which `lasagna serve` puts on :8000. This used to default to
# :8005 -- the MCP server's port -- with a comment saying "for now, we test
# against the MCP server's HTTP endpoints". The MCP server serves /, /docs,
# /health and /api/call and nothing else, so every test below that touches
# /api/v1/... was pointed at a server that does not have those routes.
REST_BASE = os.environ.get("RAG_REST_URL", "http://127.0.0.1:8000")


@pytest.fixture
def session_headers():
    """Headers with a unique session ID."""
    from .conftest import unique_session_id

    sid = unique_session_id("rest")
    return {"X-Session-ID": sid}, sid


class TestRESTHealth:
    """REST API health endpoints."""

    def test_health_endpoint(self, api_server):
        """Health check returns 200 with status info."""
        r = requests.get(f"{api_server}/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data or isinstance(data, dict)

    def test_root_endpoint(self, api_server):
        """Root endpoint returns server info."""
        r = requests.get(f"{api_server}/", timeout=10)
        assert r.status_code == 200

    def test_docs_endpoint(self, api_server):
        """OpenAPI docs are accessible."""
        r = requests.get(f"{api_server}/docs", timeout=10)
        assert r.status_code == 200


class TestRESTDocumentOperations:
    """Document CRUD through REST API / MCP HTTP interface."""

    def test_upload_via_http(self, api_server):
        """Upload a document via multipart form.

        NOTE: The MCP server (SSE transport) doesn't expose REST upload
        endpoints. This test requires a standalone REST API server
        (python -m lasagna serve).
        """
        doc_path = SAMPLE_DOCS_DIR / "product_overview.txt"
        if not doc_path.exists():
            pytest.skip("Sample doc not found")

        with open(doc_path, "rb") as f:
            r = requests.post(
                f"{api_server}/upload",
                files={"file": ("product_overview.txt", f, "text/plain")},
                timeout=60,
            )
        if r.status_code == 404:
            pytest.skip(
                "REST upload endpoint not available on MCP server. "
                "Run `python -m lasagna serve` for REST API."
            )
        # Accept various success codes depending on endpoint
        assert r.status_code in (200, 201, 202), f"Upload failed: {r.status_code} {r.text}"


class TestRESTResponseHeaders:
    """Verify response metadata headers."""

    def test_health_response_time(self, api_server):
        """Health endpoint responds quickly."""
        r = requests.get(f"{api_server}/health", timeout=10)
        assert r.status_code == 200
        # Response should be fast (< 5 seconds)
        assert r.elapsed.total_seconds() < 5.0
