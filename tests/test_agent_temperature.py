from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from lasagna.agentic.agentic_rag import (
    AgenticRAG,
    LegacyReActPipeline,
    RetrievedChunkData,
    _grade_retrieval_quality,
    search_knowledge_base,
)
from lasagna.agentic.graph_builder import BuiltinStructuredPipeline
from lasagna.agentic.loops._base import llm_with_temperature, loop_llm
from lasagna.agentic.nodes.refine_chunks import refine_chunks_node
from lasagna.providers.llm_provider import SimpleLLMProvider
from tests._loop_stubs import ScriptedLLM, hit, install_backend, pipeline_input


class CopyableScriptedLLM(ScriptedLLM):
    def __init__(
        self,
        replies=None,
        *,
        model_name: str = "provider-default",
        provider: str = "default-provider",
        temperature: float = 0.0,
    ):
        super().__init__(replies or [])
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.used_models: list[tuple[str, str, float]] = []

    async def ainvoke(self, messages, stop=None, **kwargs):
        self.used_models.append((self.model_name, self.provider, self.temperature))
        return await super().ainvoke(messages, stop=stop, **kwargs)

    def model_copy(self, *, update=None, **kwargs):
        copied = CopyableScriptedLLM(
            model_name=self.model_name,
            provider=self.provider,
            temperature=self.temperature,
        )
        copied.replies = self.replies
        copied.prompts = self.prompts
        copied.stops = self.stops
        copied.bound = self.bound
        copied.used_models = self.used_models
        if update:
            for key, value in update.items():
                setattr(copied, key, value)
        return copied


class RecordingProvider:
    def __init__(self, replies=None):
        self.llm_calls = []
        self.chat_calls = []
        self.shared_llm = CopyableScriptedLLM(replies or ["No.\nSo the final answer is: done"])

    def get_llm(self, model_name=None, temperature=0.0, **kwargs):
        self.llm_calls.append({"model_name": model_name, "temperature": temperature, **kwargs})
        llm = CopyableScriptedLLM(
            model_name=model_name or "provider-default",
            provider="default-provider",
            temperature=temperature,
        )
        llm.replies = self.shared_llm.replies
        llm.prompts = self.shared_llm.prompts
        llm.stops = self.shared_llm.stops
        llm.bound = self.shared_llm.bound
        llm.used_models = self.shared_llm.used_models
        return llm

    def get_chat_model(self, model_settings, temperature=0.0, **kwargs):
        self.chat_calls.append(
            {"model_settings": model_settings, "temperature": temperature, **kwargs}
        )
        llm = CopyableScriptedLLM(
            self.shared_llm.replies or ["refined"],
            model_name=model_settings.get("model_name", "provider-default"),
            provider=model_settings.get("platform", "default-provider"),
            temperature=temperature,
        )
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
async def test_shared_loop_model_preserves_caller_model_when_temperature_differs(monkeypatch):
    import lasagna.providers
    from lasagna.agentic.loops.self_ask import SelfAskPipeline

    provider = RecordingProvider(["unused"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    caller_llm = CopyableScriptedLLM(
        ["No.\nSo the final answer is: done"],
        model_name="gpt-4o",
        provider="custom-openai",
        temperature=0.2,
    )
    config = _config(agent_temperature=0.9)

    resolved = loop_llm("self_ask", pipeline_input("q", caller_llm, config), config)
    result = await SelfAskPipeline().run(pipeline_input("q", caller_llm, config))

    assert result.agent_summary == "done"
    assert resolved is not caller_llm
    assert resolved.model_name == "gpt-4o"
    assert resolved.provider == "custom-openai"
    assert resolved.temperature == 0.9
    assert caller_llm.temperature == 0.2
    assert caller_llm.used_models == [("gpt-4o", "custom-openai", 0.9)]
    assert caller_llm.bound == []
    assert provider.llm_calls == []


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
async def test_legacy_react_pipeline_preserves_caller_model_and_updates_temperature(monkeypatch):
    from lasagna.agentic import agentic_rag
    import lasagna.providers

    provider = RecordingProvider(["answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(agentic_rag, "create_agent", lambda llm, tools, system_prompt: "graph")
    captured = {}

    async def _fake_retrieve(self, **kwargs):
        captured["llm"] = self.llm
        return "ok"

    monkeypatch.setattr(AgenticRAG, "retrieve", _fake_retrieve)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-request-is-made")  # pragma: allowlist secret
    caller_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        base_url="https://custom.example.com/v1",
    )
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    await LegacyReActPipeline().run(pipeline_input("q", caller_llm, config))

    assert provider.llm_calls == []
    assert isinstance(captured["llm"], ChatOpenAI)
    assert captured["llm"].model_name == "gpt-4o"
    assert captured["llm"].openai_api_base == "https://custom.example.com/v1"
    assert captured["llm"].temperature == 0.9
    assert caller_llm.temperature == 0.2


@pytest.mark.asyncio
async def test_legacy_react_pipeline_constructs_llm_when_missing(monkeypatch):
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
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    await LegacyReActPipeline().run(pipeline_input("q", None, config))

    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert captured["llm_temp"] == 0.9


@pytest.mark.asyncio
async def test_structured_graph_pipeline_preserves_caller_model_and_updates_temperature(
    monkeypatch,
):
    from lasagna.agentic import graph_builder
    import lasagna.providers

    provider = RecordingProvider(["answer"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    captured = {}

    class _FakeGraph:
        async def ainvoke(self, state):
            captured["llm"] = state["llm"]
            return {"agent_summary": "done"}

    monkeypatch.setattr(graph_builder, "build_rag_graph", lambda **kwargs: _FakeGraph())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-request-is-made")  # pragma: allowlist secret
    caller_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        base_url="https://custom.example.com/v1",
    )
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    result = await BuiltinStructuredPipeline().run(pipeline_input("q", caller_llm, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == []
    assert isinstance(captured["llm"], ChatOpenAI)
    assert captured["llm"].model_name == "gpt-4o"
    assert captured["llm"].openai_api_base == "https://custom.example.com/v1"
    assert captured["llm"].temperature == 0.9
    assert caller_llm.temperature == 0.2


@pytest.mark.asyncio
async def test_structured_graph_pipeline_constructs_llm_when_missing(monkeypatch):
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
    config = _config(agent_temperature=0.9, max_agent_iterations=2)

    result = await BuiltinStructuredPipeline().run(pipeline_input("q", None, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert captured["llm_temp"] == 0.9


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
    assert provider.shared_llm.used_models == [
        ("classifier", "default-provider", 0.0),
        ("provider-default", "default-provider", 0.9),
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
    assert provider.llm_calls == [{"model_name": None, "temperature": 0.9}]
    assert provider.shared_llm.used_models == [
        ("provider-default", "default-provider", 0.0),
        ("provider-default", "default-provider", 0.9),
    ]


@pytest.mark.asyncio
async def test_unpinned_adaptive_classifier_preserves_caller_model_at_zero_while_reasoning_uses_agent_temperature(
    monkeypatch,
):
    from lasagna.agentic.loops.adaptive import AdaptiveRAGPipeline
    import lasagna.providers

    provider = RecordingProvider(["unused"])
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    caller_llm = CopyableScriptedLLM(
        ["A", "So the answer is: done."],
        model_name="gpt-4o",
        provider="custom-openai",
        temperature=0.2,
    )
    config = _config(agent_temperature=0.9, adaptive_rag_classifier_model="")

    result = await AdaptiveRAGPipeline().run(pipeline_input("q", caller_llm, config))

    assert result.agent_summary == "done"
    assert provider.llm_calls == []
    assert caller_llm.temperature == 0.2
    assert caller_llm.used_models == [
        ("gpt-4o", "custom-openai", 0.0),
        ("gpt-4o", "custom-openai", 0.9),
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


def test_chat_openai_model_copy_preserves_model_and_updates_temperature(monkeypatch):
    import lasagna.providers

    provider = RecordingProvider()
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-request-is-made")  # pragma: allowlist secret
    original = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        base_url="https://custom.example.com/v1",
    )

    updated = llm_with_temperature(original, 0.9)

    assert provider.llm_calls == []
    assert isinstance(updated, ChatOpenAI)
    assert updated.model_name == "gpt-4o"
    assert updated.openai_api_base == "https://custom.example.com/v1"
    assert updated.temperature == 0.9
    assert updated._default_params["model"] == "gpt-4o"
    assert updated._default_params["temperature"] == 0.9
    assert original.temperature == 0.2

    bound = original.bind(stop=["END"])
    bound_updated = llm_with_temperature(bound, 0.9)
    assert bound_updated.kwargs == {"stop": ["END"]}
    assert bound_updated.bound.model_name == "gpt-4o"
    assert bound_updated.bound.openai_api_base == "https://custom.example.com/v1"
    assert bound_updated.bound.temperature == 0.9
    assert bound.bound.temperature == 0.2


def test_ollama_model_constructed_and_copied_with_temperature_without_binding(monkeypatch):
    import lasagna.providers

    provider = SimpleLLMProvider(default_provider="ollama")
    llm = provider.get_llm(model_name="llama3.2", temperature=0.2)

    assert getattr(llm, "temperature", None) == 0.2
    assert llm.cache.deterministic is False
    assert not hasattr(llm, "kwargs") or "temperature" not in getattr(llm, "kwargs", {})

    recording = RecordingProvider()
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: recording)
    copied = llm_with_temperature(llm, 0.9)
    copied_zero = llm_with_temperature(copied, 0.0)

    assert recording.llm_calls == []
    assert getattr(copied, "model", getattr(copied, "model_name", None)) == "llama3.2"
    assert getattr(copied, "temperature", None) == 0.9
    assert getattr(llm, "temperature", None) == 0.2
    assert copied.cache.deterministic is False
    assert copied_zero.cache.deterministic is True
    assert copied.cache is not copied_zero.cache
    assert not hasattr(copied, "kwargs") or "temperature" not in getattr(copied, "kwargs", {})

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        ChatOllama = None

    if ChatOllama is not None:
        native = ChatOllama(
            model="llama3.2",
            temperature=0.2,
            base_url="http://custom-ollama:11434",
        )
        native_copied = llm_with_temperature(native, 0.9)
        assert native_copied.model == "llama3.2"
        assert native_copied.base_url == "http://custom-ollama:11434"
        assert native_copied.temperature == 0.9
        assert native.temperature == 0.2
        params = native_copied._chat_params([HumanMessage(content="hi")])
        assert params["model"] == "llama3.2"
        assert params["options"]["temperature"] == 0.9


def test_non_copyable_model_raises_type_error_instead_of_replacing(monkeypatch):
    import lasagna.providers

    provider = RecordingProvider()
    monkeypatch.setattr(lasagna.providers, "get_llm_provider", lambda: provider)
    non_copyable = SimpleNamespace(model_name="custom", temperature=0.2)

    with pytest.raises(TypeError, match="does not support model_copy"):
        llm_with_temperature(non_copyable, 0.9)

    assert provider.llm_calls == []
