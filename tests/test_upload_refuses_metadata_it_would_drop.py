"""``upload(metadata=...)`` is refused rather than accepted and dropped.

The SDK's ``upload`` and ``upload_batch`` took ``metadata`` and its docstring
showed ``metadata={"department": "finance"}``, but nothing stored it (the REST
upload route takes none either): the upload succeeded and the metadata was
gone. A non-empty ``metadata`` now raises ``ValueError`` saying so, before
anything is uploaded.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    from lasagna.sdk import KnowledgeClient

    uploads = []

    class _Backend:
        def upload_document(self, **kwargs):
            uploads.append(kwargs)
            raise AssertionError("the upload ran")

    instance = KnowledgeClient(session_id="no-metadata")

    async def backend():
        return _Backend()

    monkeypatch.setattr(instance, "_get_backend", backend)
    monkeypatch.setattr(instance, "_remember_session_config", lambda: None)
    path = tmp_path / "a.md"
    path.write_text("# A\n\nalpha\n")
    return instance, str(path), uploads


def test_upload_with_metadata_is_refused_before_anything_is_uploaded(client):
    instance, path, uploads = client
    for upload in (
        instance.documents.upload(path, metadata={"department": "finance"}),
        instance.upload(path, metadata={"department": "finance"}),
        instance.documents.upload_batch([path, path], metadata={"department": "finance"}),
    ):
        with pytest.raises(ValueError, match="does not store metadata .department given."):
            asyncio.run(upload)
    assert uploads == []


def test_no_metadata_or_an_empty_one_uploads(client):
    instance, path, uploads = client
    for metadata in (None, {}):
        with pytest.raises(Exception, match="the upload ran"):
            asyncio.run(instance.documents.upload(path, metadata=metadata))
    assert len(uploads) == 2
