from __future__ import annotations

from types import SimpleNamespace

import pytest
from lasagna.agentic.agentic_rag import (
    RetrievedChunkData,
    _grade_retrieval_quality,
    search_knowledge_base,
)
from lasagna.agentic.loops._base import loop_llm
from lasagna.agentic.nodes.refine_chunks import refine_chunks_node
from tests._loop_stubs import ScriptedLLM, hit, install_backend, pipeline_input


class RecordingProvider:
    def __init__(self, replies=None):
        self.llm_calls = []
        self.chat_calls = []
        self.replies = replies

    def get_llm(self, model_name=None, temperature=0.0, **kwargs):
        self.llm_calls.append({"model_name": model_name, "temperature": temperature, **kwargs})
        return ScriptedLLM(self.replies or ["No.\nSo the final answer is: done"])

    def get_chat_model(self, model_settings, temperature=0.0, **kwargs):
        self.chat_calls.append(
            {"model_settings": model_settings, "temperature": temperature, **kwargs}
        )
        return ScriptedLLM(self.replies or ["refined"])


def _config(**overrides):
    values = {
        "agent_temperature": 0.9,
        "agentic_models": {},
        "refinement_model": {"platform": "fake", "model_name": "refiner"},
        "chunk_refinement_method": "refined",
        "enable_hyde": False,
        "agentic_hyde_strategy": "off",
        "max_query_length": 12,
        "chunk_cite_retrieval": {"min_chunks": 2},
        "enable_crag": False,
        "crag_web_search_fallback": False,
        "auto_refine_avg_length_threshold": 1000,
        "auto_refine_chunk_count_threshold": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_shared_loop_model_is_bound_to_agent_temperature(monkeypatch):
    llm = ScriptedLLM(["No.\nSo the final answer is: done"])
    config = _config()

    from lasagna.agentic.loops.self_ask import SelfAskPipeline

    result = await SelfAskPipeline().run(pipeline_input("q", llm, config))

    assert result.agent_summary == "done"
    assert llm.bound == [{"temperature": 0.9}]


def test_loop_default_temperature_is_zero(monkeypatch):
    from lasagna.core import config as config_module

    monkeypatch.setattr(config_module, "rag_config", SimpleNamespace(agent_temperature=0.0))
    llm = ScriptedLLM([])

    resolved = loop_llm("self_ask", pipeline_input("q", llm), SimpleNamespace())

    assert resolved is llm


def test_pinned_loop_model_is_constructed_with_agent_temperature(monkeypatch):
    import lasagna.providers

    provider = RecordingProvider(["A", "answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(agentic_models={"self_ask": "pinned-model"})

    loop_llm("self_ask", SimpleNamespace(llm=ScriptedLLM([])), config)

    assert provider.llm_calls == [{"model_name": "pinned-model", "temperature": 0.9}]


@pytest.mark.asyncio
async def test_legacy_refinement_model_uses_agent_temperature(monkeypatch):
    from lasagna.agentic import agentic_rag

    provider = RecordingProvider(["A", "answer"])
    monkeypatch.setattr(agentic_rag, "get_llm_provider", lambda: provider)
    install_backend(monkeypatch, {"q": [hit("c1", "original")]})
    config = _config()
    monkeypatch.setattr(
        agentic_rag,
        "rag_config",
        SimpleNamespace(with_session=lambda session_id: config),
    )
    monkeypatch.setattr(
        agentic_rag,
        "get_coverage_scorer",
        lambda: SimpleNamespace(
            score_coverage=lambda query, chunks: SimpleNamespace(
                coverage_ratio=0.0,
                confidence=0.0,
                uncovered_aspects=[],
            )
        ),
    )
    monkeypatch.setattr(
        agentic_rag,
        "_find_semantically_similar_query",
        lambda *args, **kwargs: None,
    )

    await search_knowledge_base.coroutine(
        "q",
        {
            "configurable": {
                "session_id": "session",
                "top_k": 1,
                "hop_data": [],
                "previous_queries": [],
            }
        },
    )

    assert provider.chat_calls[-1]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_refinement_node_uses_agent_temperature(monkeypatch):
    from lasagna.agentic import nodes

    provider = RecordingProvider(["A", "answer"])
    monkeypatch.setattr(nodes.refine_chunks, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(
        nodes.refine_chunks,
        "refine_chunk",
        lambda **kwargs: _refined_chunk(kwargs["chunk_index"], kwargs["chunk_content"]),
    )

    chunk = RetrievedChunkData(
        chunk_id="c1",
        document_id="d1",
        document_name="doc",
        content="original",
        score=1.0,
        chunk_index=0,
        page_number=None,
        metadata={},
    )
    await refine_chunks_node(
        {
            "retrieved_chunks": [chunk],
            "refinement_method": "refined",
            "current_query": "q",
            "session_config": _config(),
        }
    )

    assert provider.chat_calls[-1]["temperature"] == 0.9


async def _refined_chunk(index, content):
    return index, "refined", content


@pytest.mark.asyncio
async def test_adaptive_classifier_stays_deterministic(monkeypatch):
    from lasagna.agentic.loops.adaptive import AdaptiveRAGPipeline
    import lasagna.providers

    provider = RecordingProvider(["A", "answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(adaptive_rag_classifier_model="classifier")

    await AdaptiveRAGPipeline().run(pipeline_input("q", ScriptedLLM(["A", "answer"]), config))

    assert provider.llm_calls == [{"model_name": "classifier", "temperature": 0.0}]


@pytest.mark.asyncio
async def test_crag_grading_stays_deterministic(monkeypatch):
    from lasagna.agentic import agentic_rag

    provider = RecordingProvider(["correct"])
    monkeypatch.setattr(agentic_rag, "get_llm_provider", lambda: provider)
    config = _config()
    chunk = RetrievedChunkData(
        chunk_id="c1",
        document_id="d1",
        document_name="doc",
        content="content",
        score=1.0,
        chunk_index=0,
        page_number=None,
        metadata={},
    )

    await _grade_retrieval_quality("q", [chunk], config)

    assert provider.chat_calls[-1]["temperature"] == 0.0
