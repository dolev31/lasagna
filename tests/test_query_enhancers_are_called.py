"""A query enhancer arm must run the enhancer, and fusion must not score 0.0.

The orchestrator only knew the internal ``generate_query_variations``. A
component implementing the documented ``BaseQueryEnhancer.enhance`` was never
called with expansion off, and with expansion on it raised AttributeError and
the arm scored as plain search. Each variation also searched without the arm's
config overrides, so under a sweep's embedder it failed and the fused list came
back empty: ndcg@10 0.0 with no error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseQueryEnhancer


class _Config(SimpleNamespace):
    def with_query_overrides(self, overrides):
        return _Config(**{**self.__dict__, **overrides})


def _hit(chunk_id):
    return SimpleNamespace(chunk=SimpleNamespace(chunk_id=chunk_id), score=1.0, document=None)


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.query_enhancer = None
    o._collection_manager = SimpleNamespace(
        resolve_backend=lambda sid: SimpleNamespace(backend_name="milvus")
    )
    base = _Config(enable_query_expansion=False, rerank_top_k=5, rrf_k=60)
    o._get_session_config = lambda sid: base
    o.calls = []

    def fake_search(query, session_id=None, top_k=None, **kw):
        o.calls.append((query, kw.get("query_config_overrides")))
        return [_hit(f"{query}-1"), _hit("shared")]

    o.search = fake_search
    return o


class Keywords(BaseQueryEnhancer):
    def __init__(self):
        self.n = 0

    def enhance(self, query, **kwargs):
        self.n += 1
        return [query, f"{query} keywords", f"{query} synonyms"]


def test_a_public_enhancer_is_called_and_its_queries_are_searched(orchestrator):
    enh = Keywords()
    out = orchestrator.search_multi_query("q", "s", top_k=5, n_variations=3, _enhancer_override=enh)
    assert enh.n == 1
    assert sorted(q for q, _ in orchestrator.calls) == ["q", "q keywords", "q synonyms"]
    assert out and out[0].chunk.chunk_id == "shared"


def test_every_variation_searches_under_the_arms_overrides(orchestrator):
    arm = {"embedding_model": "small-384", "enable_query_expansion": True}
    orchestrator.search_multi_query(
        "q",
        "s",
        top_k=5,
        n_variations=3,
        _enhancer_override=Keywords(),
        query_config_overrides=arm,
    )
    assert orchestrator.calls and all(o == arm for _, o in orchestrator.calls)


def test_no_variations_falls_back_to_the_plain_search(orchestrator):
    class Nothing:
        def generate_query_variations(self, query, n):
            return []

    arm = {"enable_query_expansion": True}
    out = orchestrator.search_multi_query(
        "q",
        "s",
        top_k=5,
        n_variations=3,
        _enhancer_override=Nothing(),
        query_config_overrides=arm,
    )
    assert [q for q, _ in orchestrator.calls] == ["q"]
    assert out, "a fusion with nothing to fuse must return the plain results, not []"


def test_a_failing_enhancer_falls_back_too(orchestrator):
    class Broken(BaseQueryEnhancer):
        def enhance(self, query, **kwargs):
            raise RuntimeError("no LLM")

    out = orchestrator.search_multi_query(
        "q", "s", top_k=5, n_variations=3, _enhancer_override=Broken()
    )
    assert out and [q for q, _ in orchestrator.calls] == ["q"]
