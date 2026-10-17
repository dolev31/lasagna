"""Documents that fail to ingest are visible in the results, and fail a strict run.

Fast ingestion tolerated up to 10% of a corpus failing to chunk (a 120 s
timeout included), saved the failures to ``_ingest_allowlist.json``, and later
runs reused the partial corpus. Nothing in the result said so: the corpus
fingerprint and the comparability key were those of the complete corpus, so a
run that scored 90% of a corpus shared a leaderboard group with runs that
scored all of it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from loguru import logger

import lasagna.api.rag_backend as rag_backend_mod
from lasagna.knowledge_evaluation_sdk import _fast_ingest as fast_ingest_mod
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters import smoke
from lasagna.knowledge_evaluation_sdk.bundle import (
    METRICS_VERSION,
    comparability_key,
    held_fixed_from,
)

# The live-run fixtures of the ranked-order test: real profiles, the real
# orchestrator, a stub retriever over the smoke corpus.
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401
    live,
    orchestrator,
)

FAILING = sorted(smoke._CORPUS)[:2]


@pytest.fixture
def ingest(monkeypatch, tmp_path):
    """`_ensure_corpus_ingested` over stubs: two documents always fail to chunk."""
    exp._corpus_sessions.clear()
    monkeypatch.setattr(exp, "_results_root", lambda: str(tmp_path))
    store = SimpleNamespace(session_filter=lambda c, s: None, delete=lambda **kw: None)
    backend = SimpleNamespace(
        _collection_manager=SimpleNamespace(resolve_backend=lambda s: store),
        hybrid_searcher=SimpleNamespace(clear_session=lambda s: None),
    )
    monkeypatch.setattr(rag_backend_mod, "get_rag_backend", lambda *a, **k: backend)
    stored: set = set()
    monkeypatch.setattr(exp, "_present_document_ids", lambda *a: set(stored))
    runs = []

    async def fast_ingest(corpus, corpus_to_valid, ready_ids, **kw):
        runs.append(len(corpus))
        failing = {corpus_to_valid[c] for c in FAILING}
        ingested = set(corpus_to_valid.values()) - failing - set(ready_ids)
        stored.update(ingested)
        return 3 * len(ingested), ingested, failing

    monkeypatch.setattr(fast_ingest_mod, "fast_ingest_corpus", fast_ingest)
    warnings = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING", format="{message}")
    yield SimpleNamespace(runs=runs, warnings=warnings)
    logger.remove(sink)
    exp._corpus_sessions.clear()


def _gap():
    return asyncio.run(exp._ensure_corpus_ingested("smoke", []))["corpus_ingest_gap"]


def test_the_corpus_records_what_failed(ingest):
    gap = _gap()
    assert gap["documents"] == len(smoke._CORPUS) == 24
    assert (gap["failed"], gap["ingested"]) == (2, 22)
    assert gap["first_failed"] == FAILING
    assert len(gap["ingested_digest"]) == 16
    said = [w for w in ingest.warnings if "could not be ingested" in w]
    assert len(said) == 1 and "2 of 24" in said[0] and FAILING[0] in said[0]


def test_a_later_run_reusing_the_partial_corpus_records_it_too(ingest):
    first = _gap()
    exp._corpus_sessions.clear()
    assert _gap() == first
    assert ingest.runs == [24]  # reused, not ingested again


def _result(gap):
    return {
        "benchmark_name": "smoke",
        "split": "test",
        "n_samples": 8,
        "sampling_seed": 42,
        "sample_id_digest": "d1",
        "corpus_fingerprint": "fp1",
        "corpus_provenance": "native",
        "resolved_mode": "live",
        "corpus_ingest_gap": gap,
        "system": {
            "embedder_model": "m",
            "embedder_dim": 8,
            "chunker": "fixed",
            "retriever_top_k": 10,
            "vector_backend": "milvus",
            "bm25_stemming": True,
        },
    }


META = {"metrics_version": METRICS_VERSION, "judge_model": None, "similarity_model": None}


def test_a_complete_corpus_keeps_its_comparability_key():
    # Computed with the code before corpus_ingest_gap existed.
    assert comparability_key(held_fixed_from(_result(None), META)) == "1c54f5025cbc079d"


def test_a_partial_corpus_never_shares_a_key_with_the_complete_one(ingest):
    gap = _gap()
    partial = held_fixed_from(_result(gap), META)
    assert partial["corpus_ingest_gap"] == {
        k: gap[k] for k in ("documents", "ingested", "failed", "ingested_digest")
    }
    assert comparability_key(partial) != comparability_key(held_fixed_from(_result(None), META))
    other = dict(gap, ingested_digest="0" * 16)
    assert comparability_key(held_fixed_from(_result(other), META)) != comparability_key(partial)


def _run(tmp_path, monkeypatch, gap, **settings):
    async def partial(benchmark_name, samples, **kw):
        ids = {c: c for c in smoke._CORPUS}
        return {
            "session_id": "stub-session",
            "corpus_to_valid": ids,
            "valid_to_corpus": ids,
            "corpus_fingerprint": "stub",
            "corpus_provenance": "native",
            "corpus_ingest_gap": gap,
        }

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", partial)
    data = {
        "name": "partial",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": ["retrieval"],
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {"base_profile": "speed"},
        **settings,
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())._results[0]


GAP = {
    "documents": 24,
    "ingested": 22,
    "failed": 2,
    "ingested_digest": "abcd" * 4,
    "first_failed": FAILING,
}


@pytest.mark.usefixtures("live")
def test_an_evaluation_on_a_partial_corpus_fails_and_names_the_setting(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, GAP)
    assert result.error and "2 of 24 corpus documents could not be ingested" in result.error
    assert "ingest_failure_tolerance: 0.084" in result.error
    assert result.metrics == {}


@pytest.mark.usefixtures("live")
def test_a_tolerance_lets_it_run_and_the_result_says_so(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, GAP, ingest_failure_tolerance=0.1)
    assert result.error is None, result.error
    assert result.corpus_ingest_gap == GAP
    assert result.to_dict()["corpus_ingest_gap"] == GAP


@pytest.mark.usefixtures("live")
def test_a_complete_corpus_runs_as_before(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, None)
    assert result.error is None and result.corpus_ingest_gap is None


@pytest.mark.parametrize("value", [0.2, -0.1, "some", True])
def test_a_tolerance_outside_zero_to_a_tenth_is_refused(value):
    with pytest.raises(ValueError, match="ingest_failure_tolerance"):
        exp._parse_ingest_tolerance(value)
