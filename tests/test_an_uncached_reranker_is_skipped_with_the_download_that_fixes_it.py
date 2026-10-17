"""A reranker that is not downloaded is skipped, named, and its download given.

`lasagna search ablation -s own` on the default profile, offline with the
embedder cached and the reranker not, failed with transformers' "We couldn't
connect to 'https://huggingface.co' ... Check your internet connection". And
the hint for a missing embedder named only BGE-M3, 2.3 of the profile's 3.2 GB,
so the next search failed on the reranker.

Now an offline load failure of a reranker that is not on disk is reported as
the embedder's is: not downloaded, with the fix. Like HyDE and GraphRAG, an
optional stage that cannot run falls back: the search returns its retrieval
order, names ``reranker`` in ``degraded_stages`` and warns with the fix; under
evaluation it raises. Every download hint names every model the session still
needs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core import model_downloads
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.model_downloads import ModelNotDownloaded
from lasagna.core.strict import strict
from lasagna.retrieval.search import ranker
from lasagna.utils import degraded

RERANKER = "mixedbread-ai/mxbai-rerank-large-v1"
EMBEDDER = "BAAI/bge-m3"


#: What transformers raises offline for a model it does not have.
OFFLINE_TEXT = (
    "We couldn't connect to 'https://huggingface.co' to load the files, and couldn't "
    "find them in the cached files. Check your internet connection"
)


@pytest.fixture
def nothing_cached(monkeypatch):
    """Offline, with no model on disk: the cross-encoder fails as transformers does."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_downloads, "is_cached", lambda model, filename="config.json": False)
    monkeypatch.setattr(ranker, "_refuse_unscorable_reranker", lambda name, trust: None)

    def offline(*a, **k):
        raise OSError(OFFLINE_TEXT)

    monkeypatch.setattr(ranker, "CrossEncoder", offline)


def test_the_loaders_offline_error_becomes_the_refusal(nothing_cached, monkeypatch):
    reranker = ranker.CrossEncoderReranker(model_name=RERANKER)
    with pytest.raises(ModelNotDownloaded) as refused:
        reranker._model_for(RERANKER)
    assert str(refused.value) == (
        f"The reranker model {RERANKER} is not downloaded, and Hugging Face offline "
        f"mode is on (HF_HUB_OFFLINE is set)."
    )
    # Not remembered as a failure: it loads once the model is on disk.
    assert reranker._failed == {}
    monkeypatch.setattr(ranker, "CrossEncoder", lambda name, device=None, **kw: "loaded")
    assert reranker._model_for(RERANKER) == "loaded"


def test_a_model_that_is_on_disk_and_fails_keeps_its_own_error(nothing_cached, monkeypatch):
    monkeypatch.setattr(model_downloads, "is_cached", lambda model, filename="config.json": True)
    with pytest.raises(RuntimeError, match="Cannot load reranker model") as failed:
        ranker.CrossEncoderReranker(model_name=RERANKER)._model_for(RERANKER)
    assert not isinstance(failed.value, ModelNotDownloaded)


class _Retriever(BaseRetriever):
    _forwards_session_config = True

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(content=c, score=s, document_id="d", chunk_id=c)
            for c, s in (("c0", 0.9), ("c1", 0.8), ("c2", 0.7))
        ]


class _Backend:
    backend_name = "fake"


class _Cache:
    def __init__(self):
        self.stored = 0

    def get(self, **kw):
        return None

    def put(self, **kw):
        self.stored += 1


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

    _Backend.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    cache = _Cache()
    monkeypatch.setattr(orch_mod, "get_search_cache", lambda config: cache)
    monkeypatch.setattr(orch_mod, "_note_documents_elsewhere", lambda session_id, config: None)
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = ranker.CombinedReranker()
    o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = backend
    o.cache = cache
    return o


SETTINGS = dict(
    enable_hybrid_search=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=True,
    enable_hyde=False,
    enable_mmr=False,
    enable_context_reordering=False,
    enable_hierarchical=False,
    enable_rag_fusion=False,
    enable_query_expansion=False,
    enable_reranking=True,
    reranker_model=RERANKER,
)


def _search(o):
    return o.search(
        "ablation",
        "own",
        top_k=2,
        min_score=None,
        _retriever_override=_Retriever(),
        query_config_overrides=SETTINGS,
    )


def test_a_search_falls_back_to_retrieval_order_and_says_so(
    nothing_cached, orchestrator, monkeypatch
):
    warnings = []
    monkeypatch.setattr(orch_mod.logger, "warning", lambda message: warnings.append(message))
    with degraded.collecting() as stages:
        out = _search(orchestrator)
        reasons = degraded.reasons()
        mcp_fix = degraded.fix("reranker", "mcp")

    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
    assert stages == ["reranker"]
    assert reasons["reranker"].startswith(f"The reranker model {RERANKER} is not downloaded")
    assert "These results are in retrieval order, not reranked." in reasons["reranker"]
    assert "lasagna configure --profile speed -s own" in mcp_fix
    (warning,) = [w for w in warnings if RERANKER in w]
    # The exact command, with what loading the model reads and its size.
    assert f"hf download {RERANKER} --include added_tokens.json" in warning
    assert "(881 MB)" in warning and "unset HF_HUB_OFFLINE" in warning
    assert "couldn't connect" not in warning
    # Not cached: once the model is on disk the next search reranks.
    assert orchestrator.cache.stored == 0


def test_under_evaluation_a_missing_reranker_raises(nothing_cached, orchestrator):
    with strict(), pytest.raises(ModelNotDownloaded, match=RERANKER):
        _search(orchestrator)


def test_every_download_hint_names_every_model_the_session_needs(nothing_cached, monkeypatch):
    from lasagna.core import setup_text

    monkeypatch.setattr(setup_text, "session_models", lambda session_id: [EMBEDDER, RERANKER])
    fix = model_downloads.fix_for_model("own", EMBEDDER)
    assert f"hf download {EMBEDDER} --include modules.json" in fix
    assert f"hf download {RERANKER} --include added_tokens.json" in fix
    assert "(2.3 GB)" in fix and "(881 MB)" in fix and "3.2 GB in all" in fix
    assert "download them once with network access" in fix
    assert "so they download on first use" in fix

    # The reranker that failed first, then whatever else is still missing.
    assert model_downloads.still_to_download("own", RERANKER) == [RERANKER, EMBEDDER]


def test_doctor_offline_names_both_models_in_its_fix(nothing_cached, monkeypatch):
    from lasagna.core import diagnostics, setup_text

    monkeypatch.setattr(setup_text, "session_models", lambda session_id: [EMBEDDER, RERANKER])
    check = diagnostics._check_model_cache("default")
    assert f"{EMBEDDER}, {RERANKER}" in check.detail
    assert f"hf download {RERANKER}" in check.advice and f"hf download {EMBEDDER}" in check.advice
