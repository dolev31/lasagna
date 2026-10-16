"""`lasagna upload` exits non-zero when the document failed to index.

`lasagna upload bad.pdf` and `empty.txt` printed the green "Uploaded
successfully ... Status: DocumentStatus.FAILED" panel and exited 0, so a
script uploading a corpus could not tell a failed document from an indexed one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.sdk.types import DocumentStatus


def _client(status, error_message=None):
    doc = SimpleNamespace(status=status, chunk_count=0, error_message=error_message)

    async def upload(path, **kwargs):
        return SimpleNamespace(document_id="doc-1", chunk_count=0, status=DocumentStatus.PENDING)

    async def wait_for_processing(document_id):
        return doc

    async def close():
        pass

    return SimpleNamespace(
        upload=upload,
        documents=SimpleNamespace(wait_for_processing=wait_for_processing),
        close=close,
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    return str(path)


def test_a_failed_document_is_an_error_with_its_reason(monkeypatch, source):
    monkeypatch.setattr(
        cli, "_get_client", lambda session_id: _client(DocumentStatus.FAILED, "no text extracted")
    )
    result = CliRunner().invoke(cli.app, ["upload", source])
    assert result.exit_code == 1
    assert "Uploaded successfully" not in result.output
    assert "no text extracted" in result.output


def test_an_indexed_document_is_a_success(monkeypatch, source):
    monkeypatch.setattr(cli, "_get_client", lambda session_id: _client(DocumentStatus.READY))
    result = CliRunner().invoke(cli.app, ["upload", source])
    assert result.exit_code == 0, result.output
    assert "Uploaded successfully" in result.output
    assert "Status: ready" in result.output
