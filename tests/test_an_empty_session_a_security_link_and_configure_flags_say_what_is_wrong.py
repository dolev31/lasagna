"""Three answers that led with the wrong thing now say what is wrong.

* A REST search of a session with no documents (an `X-Session-ID` typo) was
  answered 503 "BAAI/bge-m3 is not downloaded", while the CLI said the session
  has no documents. It now answers 200 with no results and a note naming the
  sessions that have documents, before any model loads.
* The bind refusal and `serve --help` said "See SECURITY.md", a file that is in
  the repository and not in the wheel. They link it on GitHub.
* `lasagna configure --reindex` without `--profile` listed the profiles, and
  `--force --reindex` was accepted with `--force` dropped. Both are refused with
  a sentence that says what is missing or why the two do not go together.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lasagna.api.rest.app import create_app
from lasagna.api.rest.dependencies import get_backend
from lasagna.core import model_downloads

SECURITY_URL = "https://github.com/dolev31/lasagna/blob/main/SECURITY.md"


class _NoSearch:
    """A backend a search of an empty session must never reach."""

    def __getattr__(self, name):
        raise AssertionError(f"the backend's {name} ran for a session with no documents")


@pytest.fixture
def rest(monkeypatch):
    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"my-corpus": 3})
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: _NoSearch()
    return TestClient(app, base_url="http://localhost")


@pytest.mark.parametrize("route", ["/api/v1/search", "/api/v1/search/citations"])
def test_a_misspelled_session_gets_no_results_and_the_sessions_that_have_documents(rest, route):
    reply = rest.post(route, json={"query": "ablation"}, headers={"X-Session-ID": "my-corpsu"})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["results"] == [] and body["total_results"] == 0
    assert body["note"] == (
        "Session 'my-corpsu' has no documents, so nothing was searched. Sessions with "
        "documents: my-corpus (3); name one with the X-Session-ID header (or a session "
        "token), or add a document with POST /api/v1/documents, sending the same session."
    )
    assert "not downloaded" not in reply.text


def test_the_multi_query_route_answers_the_same(rest):
    reply = rest.post(
        "/api/v1/search/multi-query", json={"query": "ablation"}, headers={"X-Session-ID": "x"}
    )
    assert reply.status_code == 200, reply.text
    assert reply.json()["note"].startswith("Session 'x' has no documents")


def test_a_session_with_documents_is_searched(monkeypatch):
    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"my-corpus": 3})

    async def async_search(**kw):
        return []

    app = create_app()
    app.dependency_overrides[get_backend] = lambda: SimpleNamespace(
        async_search=async_search, _last_routing_decision=None
    )
    reply = TestClient(app, base_url="http://localhost").post(
        "/api/v1/search", json={"query": "ablation"}, headers={"X-Session-ID": "my-corpus"}
    )
    assert reply.status_code == 200, reply.text
    assert reply.json()["note"] is None


def test_where_sessions_belong_to_callers_no_other_session_is_named(monkeypatch):
    from lasagna.api.rest.routes import search

    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"theirs": 9})
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    note = search._nothing_to_search("mine")
    assert "theirs" not in note
    assert note == (
        "Session 'mine' has no documents, so nothing was searched. Add one with "
        "POST /api/v1/documents, sending the same session."
    )


def test_the_bind_refusal_links_the_security_policy(monkeypatch):
    from lasagna.api.rest import auth_mode

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("RAG_BIND_IS_PROTECTED", raising=False)
    with pytest.raises(auth_mode.UnsafeBindError) as refused:
        auth_mode.require_auth_or_refuse_to_start("0.0.0.0")
    assert str(refused.value).endswith(f"See {SECURITY_URL}.")
    assert auth_mode.SECURITY_URL == SECURITY_URL


def test_serve_help_links_the_security_policy():
    from lasagna.cli import app

    out = CliRunner().invoke(app, ["serve", "--help"], env={"COLUMNS": "400"})
    words = " ".join(out.output.replace("│", " ").split())
    assert SECURITY_URL in words and "see SECURITY.md" not in words


def _configure(*args):
    from lasagna.cli import app

    return CliRunner().invoke(app, ["configure", *args])


@pytest.mark.parametrize("flag", ["--reindex", "--force"])
def test_a_switch_flag_without_a_profile_says_the_profile_is_missing(flag):
    out = _configure(flag)
    words = " ".join(out.output.split())
    assert out.exit_code == 1, words
    assert f"{flag} applies to a profile switch, and no --profile was given" in words
    assert f"lasagna configure --profile <name> -s default {flag}" in words
    assert "Available Profiles" not in words


def test_force_and_reindex_together_are_refused():
    out = _configure("--profile", "speed", "--force", "--reindex")
    words = " ".join(out.output.split())
    assert out.exit_code == 1, words
    assert "--force and --reindex cannot be combined" in words
    assert "Applied profile" not in words
