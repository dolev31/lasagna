"""A long corpus ingest must say how far it has got, at the CLI's default level.

A 5,183-document SciFact ingest ran 33 minutes on CPU printing nothing, so a
healthy run looked hung and was killed.
"""

from __future__ import annotations

import io
import sys

from loguru import logger

from lasagna.knowledge_evaluation_sdk import _fast_ingest


def test_progress_lines_report_count_and_eta(monkeypatch):
    lines = []
    sink = logger.add(lambda m: lines.append(m.record), level="INFO")
    try:
        clock = iter([0.0, 40.0, 80.0, 81.0])
        monkeypatch.setattr(_fast_ingest.time, "perf_counter", lambda: next(clock))
        p = _fast_ingest._Progress(total=100, every_s=30.0)
        p.update(25)  # t=40: due
        p.update(30)  # t=80: due
        p.update(100)  # t=81: last, always reported
    finally:
        logger.remove(sink)
    msgs = [r["message"] for r in lines if r["extra"].get("progress")]
    assert msgs[0].startswith("Ingesting corpus: 25/100") and "min left" in msgs[0]
    assert msgs[-1].startswith("Ingesting corpus: 100/100") and "embedded in" in msgs[-1]


def test_the_cli_default_shows_progress_but_not_other_info(monkeypatch):
    from lasagna import cli

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.delenv("LASAGNA_LOG_LEVEL", raising=False)
    logger.enable("lasagna")
    cli._configure_cli_logging()
    try:
        logger.info("ordinary info")
        logger.bind(progress=True).info("Ingesting corpus: 1/2")
        logger.warning("a warning")
    finally:
        logger.remove()
        logger.add(sys.__stderr__)
    out = buf.getvalue()
    assert "Ingesting corpus: 1/2" in out and "a warning" in out
    assert "ordinary info" not in out
