"""Every status the backend can report has an SDK status.

The SDK's DocumentStatus had no SOFT_DELETED, so ``documents.archive()``
archived the document and then raised DocumentError ("'soft_deleted' is not a
valid DocumentStatus") while converting the result, and any listing that
included an archived document would fail the same way.
"""

from __future__ import annotations

import pytest

from lasagna.core.schema import DocumentStatus as CoreStatus
from lasagna.sdk.managers.document_manager import _resolve_status
from lasagna.sdk.types import DocumentStatus


@pytest.mark.parametrize("core", list(CoreStatus), ids=lambda s: s.value)
def test_every_core_status_resolves(core):
    assert isinstance(_resolve_status(core), DocumentStatus)
    assert isinstance(_resolve_status(core.value), DocumentStatus)


def test_an_archived_document_reads_as_soft_deleted():
    assert _resolve_status("soft_deleted") is DocumentStatus.SOFT_DELETED
