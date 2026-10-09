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
        return []  # what the internal enhancer returns when no LLM is configured


def test_serving_falls_back_to_plain_search(orchestrator):
    assert not strict_stages()
    out = orchestrator.search_multi_query(
        "q", "s", top_k=5, n_variations=3, _enhancer_override=NoLLM()
    )
    assert out


def test_evaluation_fails_instead(orchestrator):
    with strict():
        with pytest.raises(RuntimeError, match="no variations"):
            orchestrator.search_multi_query(
                "q", "s", top_k=5, n_variations=3, _enhancer_override=NoLLM()
            )
    assert not strict_stages(), "strict mode must not outlive the evaluation"
