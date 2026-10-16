"""`lasagna upload` takes several files and glob patterns, through one client.

It took one path, so the README looped `lasagna upload` over a folder: one
process, one model load and one Milvus Lite start per file.
"""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from lasagna import cli
from lasagna.sdk.types import DocumentStatus


def _client(clients):
    uploaded, waited = [], []

    async def upload(path, **kwargs):
        uploaded.append(path)
        name = path.rsplit("/", 1)[-1]
        if name == "same.md":
            return SimpleNamespace(document_id="doc-a", duplicate_of="doc-a", chunk_count=1)
        return SimpleNamespace(
            document_id=f"doc-{name}", duplicate_of=None, chunk_count=0, status=None
        )

    async def wait_for_processing(document_id):
        waited.append(document_id)
        failed = document_id.endswith("bad.md")
        return SimpleNamespace(
            status=DocumentStatus.FAILED if failed else DocumentStatus.READY,
            chunk_count=0 if failed else 3,
            error_message="no text extracted" if failed else None,
        )

    async def close():
        pass

    client = SimpleNamespace(
        upload=upload,
        documents=SimpleNamespace(wait_for_processing=wait_for_processing),
        close=close,
        uploaded=uploaded,
        waited=waited,
    )
    clients.append(client)
    return client


def _invoke(monkeypatch, tmp_path, *args):
    clients = []
    monkeypatch.setattr(cli, "_get_client", lambda session_id: _client(clients))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["upload", *args], terminal_width=200)
    return result, clients


def test_a_glob_uploads_every_match_through_one_client(monkeypatch, tmp_path):
    (tmp_path / "docs").mkdir()
    for name in ("a.md", "b.md", "notes.txt"):
        (tmp_path / "docs" / name).write_text("text")
    result, clients = _invoke(monkeypatch, tmp_path, "docs/*.md", "-s", "my-corpus")
    assert result.exit_code == 0, result.output
    assert len(clients) == 1
    assert [p.rsplit("/", 1)[-1] for p in clients[0].uploaded] == ["a.md", "b.md"]
    assert "Uploaded to session my-corpus" in result.output
    assert "doc-a.md" in result.output and "doc-b.md" in result.output


def test_several_paths_report_each_and_fail_if_any_did(monkeypatch, tmp_path):
    for name in ("good.md", "bad.md", "same.md"):
        (tmp_path / name).write_text("text")
    result, clients = _invoke(monkeypatch, tmp_path, "good.md", "bad.md", "same.md")
    assert result.exit_code == 1
    out = " ".join(result.output.split())
    assert "already uploaded" in out
    assert "Upload failed: bad.md could not be indexed (no text extracted)" in out
    assert clients[0].waited == ["doc-good.md", "doc-bad.md"]


def test_a_pattern_that_matches_nothing_is_named(monkeypatch, tmp_path):
    result, _ = _invoke(monkeypatch, tmp_path, "docs/*.pdf")
    assert result.exit_code == 1
    assert "No file matches docs/*.pdf" in result.output
