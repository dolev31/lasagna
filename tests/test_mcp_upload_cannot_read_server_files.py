"""On a networked, authenticated MCP surface, upload_document must not read
arbitrary server files.

Any caller can mint a session token on such a surface, and upload_document
takes a path on the server's disk; ingesting it and reading the chunks back
returned the contents of any file the process could open.
"""

from __future__ import annotations

import pytest

from lasagna.mcp import transport_guard


@pytest.fixture
def networked_with_auth(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    previous = transport_guard.get_transport()
    transport_guard.set_transport("sse")
    yield
    transport_guard.set_transport(previous)


def test_a_path_is_refused_without_an_upload_root(networked_with_auth, monkeypatch, tmp_path):
    monkeypatch.delenv(transport_guard.UPLOAD_ROOT_ENV, raising=False)
    secret = tmp_path / "credentials.json"
    secret.write_text("{}")
    with pytest.raises(transport_guard.MCPAuthError):
        transport_guard.guard_server_path(str(secret))


def test_only_paths_inside_the_upload_root_are_allowed(networked_with_auth, monkeypatch, tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    inside = root / "doc.md"
    inside.write_text("ok")
    outside = tmp_path / "credentials.json"
    outside.write_text("{}")
    monkeypatch.setenv(transport_guard.UPLOAD_ROOT_ENV, str(root))

    assert transport_guard.guard_server_path(str(inside)) == str(inside.resolve())
    with pytest.raises(transport_guard.MCPAuthError):
        transport_guard.guard_server_path(str(outside))
    with pytest.raises(transport_guard.MCPAuthError):
        transport_guard.guard_server_path(str(root / ".." / "credentials.json"))


def test_stdio_keeps_reading_local_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    previous = transport_guard.get_transport()
    transport_guard.set_transport("stdio")
    try:
        f = tmp_path / "a.md"
        f.write_text("x")
        assert transport_guard.guard_server_path(str(f)) == str(f.resolve())
    finally:
        transport_guard.set_transport(previous)
