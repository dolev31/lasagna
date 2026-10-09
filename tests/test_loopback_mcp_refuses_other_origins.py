"""The MCP HTTP surfaces on loopback refuse requests a web page could forge.

With ``--enable-docs-api``, a cross-origin ``text/plain`` POST (a CORS "simple"
request, so no preflight) to ``/api/call`` with a foreign ``Host`` ran
``clear_session``: "Session 'm1' cleared: 2 documents deleted", HTTP 200. The
embedded ``/mcp`` mount answered a foreign ``Host`` with 200 where the
standalone ``/sse`` returns 421.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from starlette.testclient import TestClient  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402

_CALL = {"tool": "get_profile_stats", "arguments": {"session_id": "default"}}


@pytest.fixture
def docs_api(monkeypatch):
    monkeypatch.setenv("LASAGNA_ENABLE_DOCS_API", "1")
    srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8005)
    return TestClient(srv.sse_app(), base_url="http://127.0.0.1:8005")


def test_the_docs_tester_still_works_from_its_own_page(docs_api):
    response = docs_api.post("/api/call", json=_CALL, headers={"Origin": "http://127.0.0.1:8005"})
    assert response.status_code == 200, response.text
    assert "result" in response.json()


@pytest.mark.parametrize(
    "headers, status",
    [
        ({"Content-Type": "text/plain"}, 400),  # the no-preflight form
        ({"Host": "attacker.example:8005"}, 421),  # DNS rebinding
        ({"Origin": "https://attacker.example"}, 403),  # cross-origin
    ],
)
def test_the_docs_tester_refuses_what_a_web_page_could_send(docs_api, headers, status):
    import json

    body = json.dumps(_CALL)
    headers = {"Content-Type": "application/json", **headers}
    response = docs_api.post("/api/call", content=body, headers=headers)
    assert response.status_code == status, response.text


def _embedded(monkeypatch):
    monkeypatch.setenv("RAG_EMBED_MCP", "true")
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    from lasagna.api.rest.app import create_app

    return TestClient(create_app(), base_url="http://127.0.0.1:8000")


def test_the_embedded_mount_refuses_a_foreign_host(monkeypatch):
    client = _embedded(monkeypatch)
    response = client.post(
        "/mcp/messages/?session_id=0123456789abcdef0123456789abcdef",
        json={},
        headers={"Host": "attacker.example:8000"},
    )
    assert response.status_code == 421


def test_the_embedded_mount_still_answers_its_own_host(monkeypatch):
    client = _embedded(monkeypatch)
    response = client.post("/mcp/messages/?session_id=0123456789abcdef0123456789abcdef", json={})
    # Past the Host check: the session simply does not exist.
    assert response.status_code == 404
