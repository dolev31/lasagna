"""`lasagna serve` trusts X-Forwarded-For only from a proxy the operator names.

It trusted 127.0.0.1 by default, so a client on the same host could send a new
X-Forwarded-For with every request and get a fresh per-IP rate-limit bucket
each time: with auth on, 40 of 40 session mints succeeded where 5 should have.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lasagna import cli


@pytest.fixture
def served(monkeypatch):
    ran = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.update(k))

    def serve():
        result = CliRunner().invoke(cli.app, ["serve", "--no-open", "--port", "8931"])
        assert result.exit_code == 0, result.output
        return ran

    return serve


def test_forwarding_headers_are_ignored_by_default(served, monkeypatch):
    monkeypatch.delenv("RAG_FORWARDED_ALLOW_IPS", raising=False)
    assert served()["proxy_headers"] is False


def test_a_named_proxy_is_trusted(served, monkeypatch):
    monkeypatch.setenv("RAG_FORWARDED_ALLOW_IPS", "127.0.0.1")
    ran = served()
    assert ran["proxy_headers"] is True
    assert ran["forwarded_allow_ips"] == "127.0.0.1"
