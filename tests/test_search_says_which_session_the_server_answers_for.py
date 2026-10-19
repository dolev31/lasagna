"""``lasagna search`` without -s names the session it searched when the server answers for another.

``lasagna serve -s my-corpus`` answers requests that name no session with
my-corpus, but ``lasagna search "q"`` searched 'default' and said nothing, so
the results looked like the corpus the server was started for. ``status`` shows
'default' too, so the session is not switched: the search says in one line
which session it searched and the command that searches the other.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import local_server, model_downloads


class _Remote:
    async def search_documents(self, query, top_k=5):
        return []

    async def close(self):
        pass


@pytest.fixture
def served(tmp_path, monkeypatch, live_pid):
    """A server (another live process) started with ``-s my-corpus``."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(data))
    record = data / "serve.json"
    record.write_text(
        json.dumps({"pid": live_pid, "url": "http://127.0.0.1:8765", "session": "my-corpus"})
    )
    record.chmod(0o600)
    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(
        model_downloads, "sessions_with_documents", lambda: {"default": 1, "my-corpus": 3}
    )
    from lasagna.sdk import KnowledgeClient

    monkeypatch.setattr(KnowledgeClient, "from_url", classmethod(lambda cls, url, **kw: _Remote()))
    return record


def _search(*args):
    result = CliRunner().invoke(cli.app, ["search", "tides", *args], terminal_width=200)
    return result, " ".join(result.output.split())


def test_the_server_records_its_session(tmp_path, monkeypatch):
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path))
    local_server.announce("http://127.0.0.1:8765", session="my-corpus")
    assert json.loads((tmp_path / "serve.json").read_text())["session"] == "my-corpus"


def test_a_search_without_s_names_the_session_it_searched(served):
    result, out = _search()
    assert result.exit_code == 0, out
    assert "Searching session 'default'; the server answers for 'my-corpus'." in out
    assert "To search that one: lasagna search tides -s my-corpus" in out


def test_a_search_that_names_a_session_says_nothing_extra(served):
    for args in (["-s", "my-corpus"], ["-s", "default"]):
        result, out = _search(*args)
        assert result.exit_code == 0, out
        assert "the server answers for" not in out


def test_a_server_that_answers_for_default_says_nothing_extra(served, live_pid):
    served.write_text(
        json.dumps({"pid": live_pid, "url": "http://127.0.0.1:8765", "session": "default"})
    )
    result, out = _search()
    assert result.exit_code == 0, out
    assert "the server answers for" not in out
