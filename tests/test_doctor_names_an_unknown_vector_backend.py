"""`lasagna doctor` refuses a vector backend nothing registers, naming the likely one.

``RAG_VECTOR_BACKEND=milvuss`` was reported as "ok vector_backend: configured:
milvuss", and the first upload failed.
"""

from __future__ import annotations

from lasagna.core import diagnostics


def _check(monkeypatch, name):
    from lasagna.core.config import rag_config

    monkeypatch.setattr(type(rag_config), "vector_backend", property(lambda self: name))
    return diagnostics._check_vector_backend()


def test_a_typo_fails_with_the_likely_name(monkeypatch):
    check = _check(monkeypatch, "milvuss")
    assert check.status == diagnostics._FAIL
    assert "'milvuss'" in check.detail and "did you mean 'milvus'?" in check.detail
    assert "pgvector" in check.advice


def test_a_registered_backend_passes(monkeypatch):
    check = _check(monkeypatch, "milvus")
    assert check.status != diagnostics._FAIL
