"""Every setting the configuration UI shows is one the configuration reads.

The UI offered S3 bucket, endpoint and region and the fsspec protocol and base
path. None is a configuration key, so saving them in the UI did nothing: the
storage registry reads those details from storage_backend_config or the
RAG_STORAGE_* environment variables.
"""

from __future__ import annotations

from lasagna.configurator.schema import PARAMETERS
from lasagna.core.config import RAGConfig


def test_every_visible_parameter_is_read_by_the_configuration():
    unread = [
        p.key
        for p in PARAMETERS
        if not p.hidden
        and not p.parent_key
        and not isinstance(getattr(RAGConfig, p.key, None), property)
    ]
    assert not unread, f"shown in the UI but read by nothing: {unread}"


def test_the_storage_backend_help_says_where_its_settings_come_from():
    storage = next(p for p in PARAMETERS if p.key == "storage_backend")
    assert "storage_backend_config" in storage.help_text
    assert "RAG_STORAGE_S3_BUCKET" in storage.help_text
    assert "RAG_STORAGE_FSSPEC_PROTOCOL" in storage.help_text
