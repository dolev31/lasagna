"""Under evaluation, a stage that cannot run fails the task.

Serving falls back to plain search when query expansion or HyDE cannot run.
In a sweep the same fallback scored an expansion or HyDE arm exactly like the
control, with Failed: 0. The harness turns strict stages on for its tasks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.strict import strict, strict_stages


class _Config(SimpleNamespace):
    def with_query_overrides(self, overrides):
        return _Config(**{**self.__dict__, **overrides})


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.query_enhancer = None
    o._collection_manager = SimpleNamespace(
        resolve_backend=lambda sid: SimpleNamespace(backend_name="milvus")
    )
    o._get_session_config = lambda sid: _Config(
        enable_query_expansion=True, rerank_top_k=5, rrf_k=60
    )
    o.search = lambda query, *a, **k: [
        SimpleNamespace(chunk=SimpleNamespace(chunk_id="c"), score=1.0, document=None)
    ]
    return o


class NoLLM:
    def generate_query_variations(self, query, n):
        # What the internal enhancer does when no LLM is configured.
        raise RuntimeError("query variations need an LLM, and none is configured")


class NothingToAdd:
    def enhance(self, query):
        return [query]  # a technique with nothing to change for this query


def test_serving_falls_back_to_plain_search(orchestrator):
    assert not strict_stages()
    out = orchestrator.search_multi_query(
        "q", "s", top_k=5, n_variations=3, _enhancer_override=NoLLM()
    )
    assert out


def test_evaluation_fails_instead(orchestrator):
    with strict():
        with pytest.raises(RuntimeError, match="needs? an LLM"):
            orchestrator.search_multi_query(
                "q", "s", top_k=5, n_variations=3, _enhancer_override=NoLLM()
            )
    assert not strict_stages(), "strict mode must not outlive the evaluation"


def test_an_enhancer_with_nothing_to_add_is_not_a_failure(orchestrator):
    # A keyless function enhancer (drop stopwords) failed its whole task on
    # the first query that had no stopwords.
    with strict():
        out = orchestrator.search_multi_query(
            "photosynthesis", "s", top_k=5, n_variations=3, _enhancer_override=NothingToAdd()
        )
    assert out


def test_the_internal_enhancer_says_when_it_has_no_llm(monkeypatch):
    from lasagna.retrieval.enhancement import enhancer

    monkeypatch.setattr(enhancer.QueryExpander, "_get_llm", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError, match="LLM"):
        enhancer.QueryEnhancer().generate_query_variations("what is rag", 3)
