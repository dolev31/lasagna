"""No legacy `lasagna evaluate`, and a core install is not warned about Docling.

`lasagna evaluate` was an undocumented command from before the benchmark
harness: it checked for no key, retried for 69 s and failed, and on the way
warned "Docling enabled in config but not installed" for every document. The
settings file turns Docling on and falls back without it by design, so the
warning appeared on every core install.
"""

from __future__ import annotations

from types import SimpleNamespace

from loguru import logger
from typer.testing import CliRunner

from lasagna import cli
from lasagna.api import _document_pipeline as dp


def test_evaluate_is_gone():
    result = CliRunner().invoke(cli.app, ["evaluate", "paper.md", "gt.jsonl"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    assert " evaluate " not in CliRunner().invoke(cli.app, ["--help"]).output


def test_a_missing_docling_is_said_once_and_not_as_a_warning(monkeypatch):
    monkeypatch.setattr(dp, "is_docling_available", lambda: False)
    monkeypatch.setattr(dp, "_docling_fallback_said", False)
    config = SimpleNamespace(docling_enabled=True)
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        for _ in range(3):
            assert dp.DocumentPipeline._should_use_docling("paper.pdf", config) is False
    finally:
        logger.remove(sink)
    said = [r for r in records if "Docling" in r["message"]]
    assert [r["level"].name for r in said] == ["INFO"]
    assert "lasagna-rag[docling]" in said[0]["message"]
