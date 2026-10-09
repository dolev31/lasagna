"""Settings the docs and the configurator promise must do something.

docling_timeout, storage_backend_config and the citation retention limits were
documented and read by nothing: a user who set a bucket there, or a timeout
for a pathological PDF, got no effect and no warning.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from lasagna.processing import docling_processor


def test_a_slow_conversion_times_out():
    gate = threading.Event()

    class Slow:
        def convert(self, path):
            gate.wait(5)
            return "late"

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="docling_timeout"):
        docling_processor._convert_with_timeout(Slow(), "/x/paper.pdf", 0.2)
    assert time.monotonic() - started < 2
    gate.set()


def test_a_fast_conversion_returns_and_errors_propagate():
    ok = SimpleNamespace(convert=lambda p: "doc")
    assert docling_processor._convert_with_timeout(ok, "p", 5) == "doc"

    def boom(p):
        raise ValueError("bad pdf")

    with pytest.raises(ValueError, match="bad pdf"):
        docling_processor._convert_with_timeout(SimpleNamespace(convert=boom), "p", 5)


def test_storage_backend_config_reaches_the_backend(monkeypatch):
    from lasagna.storage import registry as storage_registry
    from lasagna.core.config import rag_config

    monkeypatch.setattr(
        type(rag_config),
        "storage_backend_config",
        property(lambda self: '{"bucket": "papers", "prefix": "runs"}'),
    )
    seen = {}
    fake = SimpleNamespace(create=lambda name, singleton=True, **kw: seen.update(kw) or "made")
    monkeypatch.setattr(storage_registry, "get_storage_registry", lambda: fake)
    storage_registry.get_storage_backend("s3")
    assert seen == {"bucket": "papers", "prefix": "runs"}


def test_citation_retention_comes_from_config(monkeypatch):
    from lasagna.agentic import citation_manager
    from lasagna.core.config import rag_config

    monkeypatch.setattr(type(rag_config), "citation_db_max_age_hours", property(lambda s: 12))
    monkeypatch.setattr(type(rag_config), "citation_db_max_per_session", property(lambda s: 34))
    import lasagna.core.database as db

    monkeypatch.setattr(db, "get_rag_database", lambda: SimpleNamespace(db_path="x"))
    mgr = citation_manager.CitationDatabase(cleanup_on_init=False)
    assert (mgr.max_age_hours, mgr.max_per_session) == (12, 34)


def test_the_removed_no_op_toggles_are_gone_from_the_ui():
    from lasagna.configurator import schema

    keys = {p.key for p in schema.PARAMETERS} if hasattr(schema, "PARAMETERS") else set()
    text = open(schema.__file__).read()
    for key in ("enable_metrics", "track_retrieval_quality", "track_latency_breakdown"):
        assert key not in keys and f'key="{key}"' not in text
