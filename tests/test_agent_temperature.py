from __future__ import annotations

from types import SimpleNamespace

import pytest
from lasagna.agentic.agentic_rag import (
    AgenticRAG,
    LegacyReActPipeline,
    RetrievedChunkData,
    _grade_retrieval_quality,
    search_knowledge_base,
)
from lasagna.agentic.graph_builder import BuiltinStructuredPipeline
from lasagna.agentic.loops._base import loop_llm
from lasagna.agentic.nodes.refine_chunks import refine_chunks_node
from lasagna.providers.llm_provider import SimpleLLMProvider
from tests._loop_stubs import ScriptedLLM, hit, install_backend, pipeline_input


class RecordingProvider:
    def __init__(self, replies=None):
        self.llm_calls = []
        self.chat_calls = []
        self.shared_llm = ScriptedLLM(replies or ["No.\nSo the final answer is: done"])

    def get_llm(self, model_name=None, temperature=0.0, **kwargs):
        self.llm_calls.append({"model_name": model_name, "temperature": temperature, **kwargs})
        llm = ScriptedLLM([])
        llm.replies = self.shared_llm.replies
        llm.prompts = self.shared_llm.prompts
        llm.stops = self.shared_llm.stops
        llm.bound = self.shared_llm.bound
        llm.temperature = temperature
        return llm

    def get_chat_model(self, model_settings, temperature=0.0, **kwargs):
        self.chat_calls.append(
            {"model_settings": model_settings, "temperature": temperature, **kwargs}
        )
        llm = ScriptedLLM(self.shared_llm.replies or ["refined"])
        llm.temperature = temperature
        return llm


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
async def test_shared_loop_model_is_constructed_with_agent_temperature(monkeypatch):
    import lasagna.providers
    from lasagna.agentic.loops.self_ask import SelfAskPipeline

    provider = RecordingProvider(["No.\nSo the final answer is: done"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(agent_temperature=0.9)

    result = await SelfAskPipeline().run(pipeline_input("q", None, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]


@pytest.mark.asyncio
async def test_shared_loop_model_rebuilds_when_handed_mismatched_temperature(monkeypatch):
    import lasagna.providers
    from lasagna.agentic.loops.self_ask import SelfAskPipeline

    provider = RecordingProvider(["No.\nSo the final answer is: done"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    wrong_temp_llm = ScriptedLLM(["unused"])
    wrong_temp_llm.temperature = 0.0
    config = _config(agent_temperature=0.9)

    result = await SelfAskPipeline().run(pipeline_input("q", wrong_temp_llm, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert wrong_temp_llm.bound == []


def test_loop_default_temperature_is_zero(monkeypatch):
    import lasagna.providers
    from lasagna.core import config as config_module

    provider = RecordingProvider(["No.\nSo the final answer is: done"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(
        config_module,
        "rag_config",
        SimpleNamespace(agent_temperature=0.0, agentic_models={}),
    )

    resolved = loop_llm("self_ask", pipeline_input("q", None), SimpleNamespace())

    assert resolved.temperature == 0.0
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.0}]


def test_pinned_loop_model_is_constructed_with_agent_temperature(monkeypatch):
    import lasagna.providers

    provider = RecordingProvider(["A", "answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(agentic_models={"self_ask": "pinned-model"})

    loop_llm("self_ask", SimpleNamespace(llm=ScriptedLLM([])), config)

    assert provider.llm_calls == [{"model_name": "pinned-model", "temperature": 0.9}]


def test_agentic_rag_create_uses_session_agent_temperature(monkeypatch):
    from lasagna.agentic import agentic_rag

    provider = RecordingProvider(["answer"])
    monkeypatch.setattr(agentic_rag, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(agentic_rag, "is_langfuse_tracing_enabled", lambda: False)
    monkeypatch.setattr(agentic_rag, "create_agent", lambda llm, tools, system_prompt: "graph")
    config = _config(agent_temperature=0.9, use_structured_graph=False)

    agent = AgenticRAG.create(session_config=config)

    assert len(provider.chat_calls) == 1
    assert provider.chat_calls[0]["temperature"] == 0.9
    assert agent.llm.temperature == 0.9


@pytest.mark.asyncio
async def test_legacy_react_pipeline_constructs_llm_with_agent_temperature(monkeypatch):
    from lasagna.agentic import agentic_rag
    import lasagna.providers

    provider = RecordingProvider(["answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(agentic_rag, "create_agent", lambda llm, tools, system_prompt: "graph")
    captured = {}

    async def _fake_retrieve(self, **kwargs):
        captured["llm_temp"] = getattr(self.llm, "temperature", None)
        return "ok"

    monkeypatch.setattr(AgenticRAG, "retrieve", _fake_retrieve)
    wrong_temp_llm = ScriptedLLM(["unused"])
    wrong_temp_llm.temperature = 0.0
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    await LegacyReActPipeline().run(pipeline_input("q", wrong_temp_llm, config))

    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert captured["llm_temp"] == 0.9
    assert wrong_temp_llm.bound == []


@pytest.mark.asyncio
async def test_structured_graph_pipeline_constructs_llm_with_agent_temperature(monkeypatch):
    from lasagna.agentic import graph_builder
    import lasagna.providers

    provider = RecordingProvider(["answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    captured = {}

    class _FakeGraph:
        async def ainvoke(self, state):
            captured["llm_temp"] = getattr(state["llm"], "temperature", None)
            return {"agent_summary": "done"}

    monkeypatch.setattr(graph_builder, "build_rag_graph", lambda **kwargs: _FakeGraph())
    wrong_temp_llm = ScriptedLLM(["unused"])
    wrong_temp_llm.temperature = 0.0
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    result = await BuiltinStructuredPipeline().run(pipeline_input("q", wrong_temp_llm, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert captured["llm_temp"] == 0.9
    assert wrong_temp_llm.bound == []


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
async def test_pinned_adaptive_classifier_stays_deterministic(monkeypatch):
    from lasagna.agentic.loops.adaptive import AdaptiveRAGPipeline
    import lasagna.providers

    provider = RecordingProvider(["A", "So the answer is: done."])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(agent_temperature=0.9, adaptive_rag_classifier_model="classifier")

    result = await AdaptiveRAGPipeline().run(pipeline_input("q", None, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [
        {"model_name": None, "temperature": 0.9},
        {"model_name": "classifier", "temperature": 0.0},
    ]


@pytest.mark.asyncio
async def test_unpinned_adaptive_classifier_stays_deterministic_while_reasoning_uses_agent_temperature(
    monkeypatch,
):
    from lasagna.agentic.loops.adaptive import AdaptiveRAGPipeline
    import lasagna.providers

    provider = RecordingProvider(["A", "So the answer is: done."])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    config = _config(agent_temperature=0.9, adaptive_rag_classifier_model="")

    result = await AdaptiveRAGPipeline().run(pipeline_input("q", None, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [
        {"model_name": None, "temperature": 0.9},
        {"model_name": None, "temperature": 0.0},
    ]


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


def test_ollama_model_constructed_with_temperature_without_binding():
    provider = SimpleLLMProvider(default_provider="ollama")
    llm = provider.get_llm(model_name="llama3.2", temperature=0.9)

    assert getattr(llm, "temperature", None) == 0.9
    assert not hasattr(llm, "kwargs") or "temperature" not in getattr(llm, "kwargs", {})
