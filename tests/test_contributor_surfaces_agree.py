"""What a contributor reads agrees with what the code does.

* CONTRIBUTING said a reranker without ``**kwargs`` raises TypeError; the
  pipeline passes it only ``top_k``, ``component validate`` passed such a class
  and a sweep ran it.
* ``LASAGNA_CHUNK_SISE`` was ignored without a word while ``RAG_CHUNK_SISE``
  warned with the setting it was close to.
* A typo in a factor or a track name got no "did you mean".
* ``log_level`` accepted TRACE, SUCCESS and CRITICAL while ``lasagna-mcp
  --log-level`` took four levels.
* ``config_for_standalone`` said "standalone usage standalone environment".
* The deploy Makefile said ``make logs`` follows the API's logs; it follows
  every service's.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from loguru import logger

from tests._paths import DEPLOY_DIR, PACKAGE_DIR

REPO = PACKAGE_DIR.parents[1]


def test_a_reranker_without_kwargs_runs_and_the_docs_say_so():
    from lasagna.core.base_components import BaseReranker, ScoredDocument
    from lasagna.retrieval.search.ranker import adapt_reranker_for_pipeline

    class Plain(BaseReranker):
        def rerank(self, query, documents, top_k=5):
            return list(reversed(documents))[:top_k]

    docs = [
        {"id": f"c{i}", "chunk_id": f"c{i}", "document_id": "d", "content": f"t{i}", "score": 1.0}
        for i in range(3)
    ]
    out = adapt_reranker_for_pipeline(Plain()).rerank("q", docs, 2, config=object())
    assert [r.chunk_id for r in out] == ["c2", "c1"]
    assert isinstance(ScoredDocument, type)

    contributing = " ".join((REPO / "CONTRIBUTING.md").read_text().split())
    assert "raises `TypeError`" not in contributing
    assert "a `rerank` without `**kwargs` runs too" in contributing


def _warnings_for(monkeypatch, **env) -> str:
    from lasagna.core import config as config_module

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(config_module, "_warned_unknown_env", set())
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        config_module.RAGConfig._load_env_overrides(config_module.RAGConfig()._config)
    finally:
        logger.remove(sink)
    return " ".join(seen)


@pytest.mark.parametrize("prefix", ["RAG_", "LASAGNA_"])
def test_a_misspelled_setting_warns_under_either_prefix(monkeypatch, prefix):
    said = _warnings_for(monkeypatch, **{f"{prefix}CHUNK_SISE": "500"})
    assert f"{prefix}CHUNK_SISE is not a Lasagna setting" in said
    assert f"Did you mean {prefix}CHUNK_SIZE?" in said


def test_a_lasagna_variable_that_is_no_setting_is_left_alone(monkeypatch):
    said = _warnings_for(
        monkeypatch, LASAGNA_LOG_FORMAT="json", LASAGNA_MCP_TOOLS="all", LASAGNA_WIDGET="1"
    )
    assert "LASAGNA_LOG_FORMAT" not in said and "LASAGNA_MCP_TOOLS" not in said
    assert "LASAGNA_WIDGET" not in said


def test_a_misspelled_track_names_the_track():
    from lasagna.knowledge_evaluation_sdk.experiment import _parse_tracks

    with pytest.raises(ValueError) as refused:
        _parse_tracks(["retreival"])
    assert "(did you mean 'retrieval'?)" in str(refused.value)


def test_a_misspelled_factor_names_the_setting():
    from lasagna.knowledge_evaluation_sdk.experiment import FactorialSpec

    design = FactorialSpec(factor_a="enable_reranknig", factor_b="enable_hyde")
    with pytest.raises(ValueError) as refused:
        design.validate()
    assert "(did you mean 'enable_reranking'?)" in str(refused.value)


def test_every_surface_takes_the_same_log_levels():
    from lasagna.core._text import LOG_LEVELS
    from lasagna.core.config import _CHOICES

    assert _CHOICES["log_level"] == LOG_LEVELS
    help_text = subprocess.run(
        [sys.executable, "-m", "lasagna.mcp.server", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    words = " ".join(help_text.split())
    for level in LOG_LEVELS:
        assert level in words, level


def test_the_standalone_preset_says_what_it_is_for():
    from lasagna.sdk import config_for_standalone

    doc = " ".join((config_for_standalone.__doc__ or "").split())
    assert "standalone usage standalone" not in doc
    assert "folders you name" in doc


def test_the_makefile_says_what_make_logs_follows():
    header = (DEPLOY_DIR / "Makefile").read_text()
    assert "Follow API logs" not in header
    assert "make logs                       Follow every service's logs" in header
