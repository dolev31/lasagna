"""Generation-track arms under ``speed`` must execute explicitly enabled features.

The ``speed`` profile sets ``enable_query_routing=False`` and
``routing_default_tier="fast"``. The retrieval track overrides
``routing_default_tier`` to ``"standard"`` via ``_fix_retrieval_routing``, but
the generation track used the profile's ``"fast"`` tier for non-agentic arms,
causing ``QueryRouter`` to apply ``_FAST_OVERRIDES`` and silently disable
explicitly enabled features such as reranking.
"""

from __future__ import annotations

import asyncio
import re

from lasagna.core.base_components import BaseReranker
from lasagna.knowledge_evaluation_sdk import experiment as exp
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401 - pytest fixtures
    live,
    orchestrator,
)
from tests.test_the_generator_reads_the_top_passages_reordered import (  # noqa: F401 - pytest fixture
    llm,
)


class _RecordingReversingReranker(BaseReranker):
    """Deterministic hermetic reranker that records calls and reverses order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rerank(self, query, documents, top_k=5, **kwargs):
        self.calls.append(query)
        return list(reversed(documents))[:top_k]


def _ranks(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"PASSAGE(\d+):", text)]


def _run_experiment(tmp_path, tracks, arms):
    data = {
        "name": "speed-gen-routing",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": tracks,
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {
            "base_profile": "speed",
            "config_overrides": arms,
        },
    }
    results = asyncio.run(exp.Experiment.from_dict(data).run())._results
    for r in results:
        assert r.error is None, r.error
    return results


def _run_generation(tmp_path, arms):
    return _run_experiment(tmp_path, ["generation"], arms)


def test_reranker_arm_executes_reranking_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    reranker = _RecordingReversingReranker()
    real_search = orchestrator.search

    def _search_with_reranker(*args, **kwargs):
        kwargs["_reranker_override"] = reranker
        return real_search(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "search", _search_with_reranker)

    _run_generation(
        tmp_path,
        [
            {"name": "control"},
            {"name": "with_reranker", "enable_reranking": True},
        ],
    )

    assert len(reranker.calls) > 0, (
        "Reranker was never invoked on the generation track despite enable_reranking=True"
    )
    # Control arm runs first without reranking (ranks 0..4); reranker arm reverses them.
    prompts_by_order = {tuple(_ranks(p)) for p in llm.prompts}
    assert any(order != (0, 1, 2, 3, 4) for order in prompts_by_order), (
        f"Generator only saw un-reranked passages: {prompts_by_order}"
    )


def test_explicit_enable_query_routing_is_preserved_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.retrieval.routing.router import get_query_router

    seen: list[bool] = []
    router = get_query_router()
    real_route = router.route

    def _spy_route(*args, **kwargs):
        seen.append(bool(getattr(kwargs["config"], "enable_query_routing", False)))
        return real_route(*args, **kwargs)

    monkeypatch.setattr(router, "route", _spy_route)

    _run_generation(
        tmp_path,
        [
            {"name": "routing_on", "enable_query_routing": True},
        ],
    )

    assert seen and all(seen), f"Expected enable_query_routing=True, saw: {seen}"


def test_explicit_routing_default_tier_fast_is_preserved_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    reranker = _RecordingReversingReranker()
    real_search = orchestrator.search

    def _search_with_reranker(*args, **kwargs):
        kwargs["_reranker_override"] = reranker
        return real_search(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "search", _search_with_reranker)

    _run_generation(
        tmp_path,
        [
            {
                "name": "explicit_fast",
                "routing_default_tier": "fast",
                "enable_reranking": True,
            },
        ],
    )

    # Explicit routing_default_tier="fast" must be honored and keep fast-tier overrides.
    assert reranker.calls == []


def test_agentic_generation_routing_still_forces_comprehensive_tier(monkeypatch):
    from unittest.mock import MagicMock

    from lasagna.core.pipeline import KnowledgePipeline

    pipeline = KnowledgePipeline.from_profile("speed")
    exp._apply_overrides_to_pipeline(pipeline, {"use_agentic_rag": True}, "s-agentic")
    gen_config = dict(pipeline._config)
    exp._fix_generation_routing(gen_config, {"use_agentic_rag": True})

    backend = MagicMock()
    backend.search.return_value = []
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: backend)

    pipeline.query(
        "multi-hop question", top_k=5, session_id="s-agentic", config_overrides=gen_config
    )

    backend.search.assert_called_once()
    assert backend.search.call_args.kwargs["routing_tier"] == "comprehensive"


def test_speed_profile_outside_harness_remains_unchanged():
    speed_spec = exp._resolve_profile_config("speed")
    assert speed_spec["enable_query_routing"] is False
    assert speed_spec["routing_default_tier"] == "fast"


def test_run_record_reports_effective_routing_tiers_for_both_tracks(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.knowledge_evaluation_sdk.stages import isolation_report

    reranker = _RecordingReversingReranker()
    real_search = orchestrator.search

    def _search_with_reranker(*args, **kwargs):
        kwargs["_reranker_override"] = reranker
        return real_search(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "search", _search_with_reranker)

    results = _run_experiment(
        tmp_path,
        ["retrieval", "generation"],
        [
            {"name": "control"},
            {"name": "with_reranker", "enable_reranking": True},
        ],
    )
    by_name = {r.config_name: r for r in results}
    assert by_name["control"].system["routing_tiers"] == {
        "retrieval": {"standard": 12},
        "generation": {"fast": 12},
    }
    assert by_name["with_reranker"].system["routing_tiers"] == {
        "retrieval": {"standard": 12},
        "generation": {"standard": 12},
    }
    report = isolation_report([r.to_dict() for r in results])
    assert report["undeclared"] == []
    assert report["inert"] == []


def test_run_record_reports_forced_comprehensive_tier_for_agentic_generation(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from types import SimpleNamespace

    from lasagna.agentic import agentic_rag
    from lasagna.core.schema import SearchResult

    async def _fake_retrieve(self, **kwargs):
        return SimpleNamespace(
            chunks=[
                {
                    "chunk_id": "c1",
                    "content": "PASSAGE0: trehalose",
                    "document_id": "d1",
                    "score": 1.0,
                    "metadata": {},
                }
            ],
            agent_summary="trehalose",
            iterations=1,
            quality_score=1.0,
            total_tokens=0,
        )

    monkeypatch.setattr("lasagna.api._search_orchestrator.SearchResult", SearchResult)
    monkeypatch.setattr(agentic_rag.AgenticRAG, "retrieve", _fake_retrieve)
    real_search = orchestrator.search

    def _search_forcing_comprehensive(*args, **kwargs):
        cfg = kwargs.get("query_config_overrides") or {}
        if cfg.get("use_agentic_rag"):
            kwargs["routing_tier"] = "comprehensive"
        return real_search(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "search", _search_forcing_comprehensive)

    (result,) = _run_generation(
        tmp_path,
        [{"name": "agentic", "use_agentic_rag": True}],
    )
    assert result.system["routing_tiers"] == {"generation": {"comprehensive": 12}}


def test_run_record_reports_mixed_tiers_for_explicit_routing_experiment(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.retrieval.routing.models import RoutingDecision, RoutingTier
    from lasagna.retrieval.routing.router import get_query_router

    router = get_query_router()
    call_idx = 0

    def _alternating_route(query, document_ids, classification, config, forced_tier=None):
        nonlocal call_idx
        tier = RoutingTier.FAST if call_idx % 2 == 0 else RoutingTier.STANDARD
        call_idx += 1
        return RoutingDecision(
            tier=tier,
            signals=None,
            confidence=1.0,
            rationale="alternating test route",
            strategy_used="rule_based",
            overrides=router.get_tier_overrides(tier),
        )

    monkeypatch.setattr(router, "route", _alternating_route)

    (result,) = _run_generation(
        tmp_path,
        [{"name": "explicit_routing", "enable_query_routing": True}],
    )
    assert result.system["routing_tiers"] == {"generation": {"fast": 6, "standard": 6}}


def test_comprehensive_fallback_records_single_standard_fallback_entry(
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.agentic import agentic_rag
    from lasagna.core.strict import observe_routing

    async def _failing_retrieve(self, **kwargs):
        raise RuntimeError("agentic loop failed")

    monkeypatch.setattr(agentic_rag.AgenticRAG, "retrieve", _failing_retrieve)
    monkeypatch.setattr(orchestrator._default_backend, "search", lambda **kw: [])

    with observe_routing() as tiers:
        orchestrator.search(
            "who wrote hamlet",
            "stub-session",
            top_k=3,
            routing_tier="comprehensive",
            query_config_overrides={"use_agentic_rag": True, "enable_search_cache": False},
        )

    assert tiers == ["standard_fallback"]


def test_routing_decision_failure_records_single_standard_entry(
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.core.strict import observe_routing
    from lasagna.retrieval.routing.router import get_query_router
    from tests.test_retrieval_metrics_score_the_ranked_order import _Ranked

    router = get_query_router()

    def _failing_route(*args, **kwargs):
        raise RuntimeError("router boom")

    monkeypatch.setattr(router, "route", _failing_route)

    with observe_routing() as tiers:
        results = orchestrator.search(
            "who wrote hamlet",
            "stub-session",
            top_k=3,
            _retriever_override=_Ranked(),
            query_config_overrides={
                **exp._resolve_profile_config("speed"),
                "enable_search_cache": False,
            },
        )

    assert tiers == ["standard"]
    assert results


def test_context_reordering_executes_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
):
    _run_generation(
        tmp_path,
        [{"name": "reordered", "enable_context_reordering": True}],
    )
    assert llm.prompts and {tuple(_ranks(p)) for p in llm.prompts} == {(0, 2, 4, 3, 1)}


def test_hierarchical_auto_merge_executes_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from types import SimpleNamespace

    merge_calls: list[int] = []

    def _merge_results(results, config=None):
        merge_calls.append(len(results))
        return results

    monkeypatch.setattr(orchestrator, "_auto_merger", SimpleNamespace(merge_results=_merge_results))

    _run_generation(
        tmp_path,
        [{"name": "hierarchical", "enable_hierarchical": True}],
    )
    assert len(merge_calls) > 0


def test_hyde_executes_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from types import SimpleNamespace

    import numpy as np

    hyde_calls: list[str] = []

    def _get_hyde_embedding(query, **kwargs):
        hyde_calls.append(query)
        return np.ones(3)

    monkeypatch.setattr(
        orchestrator,
        "query_enhancer",
        SimpleNamespace(get_hyde_embedding=_get_hyde_embedding),
    )

    _run_generation(
        tmp_path,
        [{"name": "with_hyde", "enable_hyde": True}],
    )
    assert len(hyde_calls) > 0


def test_plain_speed_control_retains_fast_routing_tier(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
):
    (result,) = _run_generation(tmp_path, [{"name": "control"}])
    assert result.system["routing_tiers"] == {"generation": {"fast": 12}}
    assert result.system["stages"]["query_enhancer"]["settings"]["routing_default_tier"] == {
        "retrieval": "standard",
        "generation": "fast",
    }


def test_hybrid_search_executes_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from types import SimpleNamespace

    from tests.test_retrieval_metrics_score_the_ranked_order import _Ranked

    hybrid_calls: list[str] = []
    real_search = orchestrator.search
    ranked_retriever = _Ranked()

    def _search_without_retriever_override(*args, **kwargs):
        kwargs["_retriever_override"] = None
        return real_search(*args, **kwargs)

    def _backend_search(**kwargs):
        docs = ranked_retriever.retrieve("who", "", top_k=kwargs.get("top_k", 10))
        return [
            SimpleNamespace(
                id=d.chunk_id,
                score=d.score,
                distance=d.score,
                entity={
                    "content": d.content,
                    "document_id": d.document_id,
                    "metadata": {},
                },
            )
            for d in docs
        ]

    def _search_hybrid(query, dense_results, **kwargs):
        hybrid_calls.append(query)
        converted = [
            SimpleNamespace(
                chunk_id=h["id"],
                document_id=h["entity"]["document_id"],
                content=h["entity"]["content"],
                score=h["distance"],
                combined_score=h["distance"],
                metadata=h["entity"].get("metadata", {}),
                highlights=None,
                embedding=None,
            )
            for h in dense_results
        ]
        return converted, {"sparse_ms": 1.0, "merge_ms": 0.5, "hybrid_mode": "hybrid"}

    monkeypatch.setattr(orchestrator, "search", _search_without_retriever_override)
    monkeypatch.setattr(orchestrator._default_backend, "search", _backend_search)
    monkeypatch.setattr(
        orchestrator,
        "hybrid_searcher",
        SimpleNamespace(
            hydrate_session_index=lambda *a, **kw: None,
            search_hybrid=_search_hybrid,
        ),
    )

    (result,) = _run_generation(
        tmp_path,
        [{"name": "with_hybrid", "enable_hybrid_search": True}],
    )
    assert len(hybrid_calls) > 0
    assert result.system["routing_tiers"] == {"generation": {"standard": 12}}


def test_mmr_is_preserved_on_generation_track_under_speed(
    tmp_path,
    llm,  # noqa: F811 - pytest fixture
    orchestrator,  # noqa: F811 - pytest fixture
    monkeypatch,
):
    from lasagna.api import _search_orchestrator as orch_mod

    reranker = _RecordingReversingReranker()
    real_search = orchestrator.search
    mmr_lambdas: list[object] = []
    orig_log_json = orch_mod.log_json

    def _search_with_reranker(*args, **kwargs):
        kwargs["_reranker_override"] = reranker
        return real_search(*args, **kwargs)

    def _spy_log_json(event, payload):
        if event == "rag_search_metrics":
            mmr_lambdas.append(payload.get("mmr_lambda"))
        return orig_log_json(event, payload)

    monkeypatch.setattr(orchestrator, "search", _search_with_reranker)
    monkeypatch.setattr(orch_mod, "log_json", _spy_log_json)

    (result,) = _run_generation(
        tmp_path,
        [{"name": "with_mmr", "enable_mmr": True, "enable_reranking": True}],
    )
    assert result.system["routing_tiers"] == {"generation": {"standard": 12}}
    assert mmr_lambdas and all(v is not None for v in mmr_lambdas)
