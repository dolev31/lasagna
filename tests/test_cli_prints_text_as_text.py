"""CLI output must print chunk text and error text as text, not as Rich markup.

`lasagna search` passed chunk content to Rich unescaped, so
"see [metrics](../x)" printed as "see (../x)" and "[copy.deepcopy(d) ...]"
vanished; text containing "[/x]" raised MarkupError.
"""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from lasagna import cli


def _hit(text):
    return SimpleNamespace(
        score=0.9,
        chunk=SimpleNamespace(content=text, document_id="doc_0123456789abcdef"),
        document=SimpleNamespace(filename="notes.md"),
    )


def test_search_prints_brackets_and_the_filename(monkeypatch):
    text = "see [metrics](../x) and out = [copy.deepcopy(d) for d in docs] [/x]"
    monkeypatch.setattr(cli, "_run", lambda coro: (coro.close(), [_hit(text)])[1])
    monkeypatch.setattr(cli, "_get_client", lambda *a, **k: None)
    out = CliRunner().invoke(cli.app, ["search", "q"], terminal_width=400).output
    assert "[metrics]" in out and "[copy.deepcopy(d)" in out and "[/x]" in out
    assert "notes.md" in out


def test_error_text_with_brackets_does_not_crash(monkeypatch):
    def boom(coro):
        coro.close()
        raise RuntimeError("bad filter [/x] in [session]")

    monkeypatch.setattr(cli, "_run", boom)
    monkeypatch.setattr(cli, "_get_client", lambda *a, **k: None)
    r = CliRunner().invoke(cli.app, ["search", "q"], terminal_width=400)
    assert r.exit_code == 1
    assert "[/x]" in r.output and "[session]" in r.output
