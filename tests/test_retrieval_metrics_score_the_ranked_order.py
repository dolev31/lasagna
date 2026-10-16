"""Retrieval metrics are computed on the ranked list, not on the order the generator reads.

The standard, balanced and max_quality profiles turn on context reordering
("lost in the middle"): even ranks first, odd ranks reversed at the end, so the
generator reads the best passages at both ends of its context. The retrieval
track scored that order. The second-ranked document moved to the last place,
the cut to k documents dropped it, and NDCG and MRR measured a presentation
choice instead of the ranking. Standard is the default ``base_profile``, so a
sweep that named no profile was affected.

The retrieval track now searches with reordering off; the generation track
keeps it, since what the generator reads is the system under test there. The
stage manifest records which order each track used.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseReranker, BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters import smoke

CORPUS = sorted(smoke._CORPUS)
_RELEVANT = {q["question"]: list(q["expected_docs"]) for q in smoke._QUERIES}


def ranked(question):
    """The retriever's list: one non-relevant document, then the relevant ones."""
    relevant = _RELEVANT.get(question, [])
    filler = [d for d in CORPUS if d not in relevant]
    return filler[:1] + relevant + filler[1:]


class _Ranked(BaseRetriever):
    def retrieve(self, query, collection, top_k=5, **kwargs):
        docs = ranked(query)[:top_k]
        return [
            ScoredDocument(
                content=smoke._CORPUS[d], score=1.0 - i / 100, document_id=d, chunk_id=f"c-{d}"
            )
            for i, d in enumerate(docs)
        ]


class _KeepOrder(BaseReranker):
    """The profile's reranking step, without a model: candidates keep their order."""

    def rerank(self, query, documents, top_k=5, **kwargs):
        return list(documents)[:top_k]


class _Store:
    backend_name = "fake"

    def search(self, **kw):
        raise AssertionError("the store's own search ran instead of the retriever")


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

    _Store.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    store = _Store()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: store)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = store
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = store
    return o


def _search(orchestrator, question, top_k, config):
    out = orchestrator.search(
        question,
        "stub-session",
        top_k=top_k,
        _retriever_override=_Ranked(),
        _reranker_override=_KeepOrder(),
        query_config_overrides=config,
    )
    return [r.chunk.document_id for r in out]


def _profile(name):
    return dict(exp._resolve_profile_config(name))


QUESTION = next(iter(_RELEVANT))


@pytest.mark.parametrize("profile", ["standard", "balanced", "max_quality"])
def test_these_profiles_reorder_what_the_generator_reads(profile):
    assert _profile(profile)["enable_context_reordering"] is True


def test_the_generator_still_reads_the_reordered_list(orchestrator):
    config = {**_profile("standard"), "use_agentic_rag": False, "enable_search_cache": False}
    docs = ranked(QUESTION)
    assert _search(orchestrator, QUESTION, 6, config) == [
        docs[0],
        docs[2],
        docs[4],
        docs[5],
        docs[3],
        docs[1],
    ]


def test_the_retrieval_settings_turn_reordering_off():
    config = _profile("standard")
    exp._fix_retrieval_depth(config, depth=40)
    assert config["enable_context_reordering"] is False


# -- the retrieval track, end to end through the real orchestrator --------------


class _Pipeline:
    """KnowledgePipeline with the real profile config; queries run the real orchestrator."""

    orchestrator = None
    configs: list = []

    def __init__(self, profile):
        self._config = _profile(profile)
        self._component_names, self._instances = {}, {}

    def replace(self, slot, name):
        self._component_names[slot] = name
        return self

    def _apply_session_config(self, session_id):
        pass

    def get_component(self, slot):
        return None

    def query(self, question, k, session_id, config):
        _Pipeline.configs.append(dict(config))
        out = _Pipeline.orchestrator.search(
            question,
            session_id,
            top_k=k,
            _retriever_override=_Ranked(),
            _reranker_override=_KeepOrder(),
            query_config_overrides=config,
        )
        return out


@pytest.fixture
def live(monkeypatch, orchestrator):
    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.pipeline import KnowledgePipeline

    async def ingest(benchmark_name, samples, **kw):
        ids = {c: c for c in CORPUS}
        return {
            "session_id": "stub-session",
            "corpus_to_valid": ids,
            "valid_to_corpus": ids,
            "corpus_fingerprint": "stub",
            "corpus_provenance": "native",
        }

    _Pipeline.orchestrator = orchestrator
    _Pipeline.configs = []
    monkeypatch.setattr(exp, "_ensure_corpus_ingested", ingest)
    monkeypatch.setattr(
        KnowledgePipeline, "from_profile", classmethod(lambda c, p=None: _Pipeline(p))
    )
    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="stub-embedder", dimension=8),
        hybrid_searcher=SimpleNamespace(
            is_native_mode_available=lambda: True, _backend_supports_native_hybrid=lambda s: True
        ),
    )
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda *a, **k: backend)


def _run(tmp_path, profile):
    data = {
        "name": "order",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": ["retrieval"],
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {"base_profile": profile},
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())._results[0]


@pytest.mark.parametrize("profile", ["standard", "speed"])
def test_the_retrieval_track_scores_the_ranked_list(live, tmp_path, profile):
    # Every query's relevant documents start at rank 2, so MRR is exactly 0.5.
    # Scored on the reordered list it was 0.0: rank 2 went to the end.
    result = _run(tmp_path, profile)
    assert result.error is None, result.error
    assert result.metrics["mrr@10"] == pytest.approx(0.5)
    for row in result.per_sample_scores:
        question = next(q for q in _RELEVANT if sorted(_RELEVANT[q]) == row["_expected_docs"])
        assert row["_retrieved_ids"] == ranked(question)[:10]


def test_the_manifest_records_the_order_each_track_used(live, tmp_path):
    result = _run(tmp_path, "standard")
    order = result.system["stages"]["reranker"]["settings"]["context_order"]
    assert order == {"retrieval": "ranked", "generation": "lost_in_the_middle"}
    assert result.system["stages"]["reranker"]["settings"]["enable_context_reordering"] is False


def test_a_retrieval_sweep_says_reordering_changed_nothing(live, tmp_path):
    # The manifest files the arm's change under the reranker (the generator's
    # order), and the ranked lists show it does not reach retrieval.
    from lasagna.knowledge_evaluation_sdk.stages import isolation_report

    data = {
        "name": "order",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": ["retrieval"],
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {
            "base_profile": "speed",
            "config_overrides": [
                {"name": "control"},
                {"name": "reordered", "enable_context_reordering": True},
            ],
        },
    }
    rows = asyncio.run(exp.Experiment.from_dict(data).run())._results
    a, b = (r.system["stages"] for r in rows)
    assert [s for s in a if a[s]["digest"] != b[s]["digest"]] == ["reranker"]
    report = isolation_report([r.to_dict() for r in rows])
    assert report["undeclared"] == []
    assert any("enable_context_reordering) changed nothing" in m for m in report["inert"])
    assert {r.config_name: r.metrics["mrr@10"] for r in rows} == {
        "control": pytest.approx(0.5),
        "reordered": pytest.approx(0.5),
    }
