"""The published agentic loops run their papers' control flow inside the pipeline.

Each loop is driven by a scripted LLM, so a test fixes what the model says and
checks what the loop does with it: when it searches, with which query, when it
stops, and which passages it hands back with its answer. Every search goes
through ``RAGBackend.search`` (replaced here by a recorder), the pipeline's own
retrieval, with the arm's settings.
"""

from __future__ import annotations


import pytest

from lasagna.core.base_components import BaseAgenticPipeline
from lasagna.core.registry import registry

from tests._loop_stubs import (
    ArmConfig,
    ScriptedLLM,
    hit,
    install_backend,
    pipeline_input,
)

#: Each loop's settings and one short scripted run that reaches a search;
#: every loop's section below adds its own entries.
LOOP_SETTINGS = {}
SHORT_RUNS = {}


def _loop(name):
    return registry.get("agentic_pipeline", name)


# ---------------------------------------------------------------------------
# Self-Ask
# ---------------------------------------------------------------------------


async def test_self_ask_searches_each_follow_up_until_the_final_answer(monkeypatch):
    backend = install_backend(
        monkeypatch,
        {
            "Who was the founder of craigslist?": [hit("c1", "Craigslist was founded by Craig.")],
            "When was Craig Newmark born?": [hit("c2", "Craig Newmark, born December 6, 1952.")],
        },
    )
    llm = ScriptedLLM(
        [
            "Yes.\nFollow up: Who was the founder of craigslist?",
            "Craig Newmark.",  # read off the first search's passages
            "Follow up: When was Craig Newmark born?",
            "December 6, 1952.",  # read off the second
            "So the final answer is: December 6, 1952",
        ]
    )
    result = await _loop("self_ask").run(
        pipeline_input("When was the founder of craigslist born?", llm)
    )

    assert [c["query"] for c in backend.calls] == [
        "Who was the founder of craigslist?",
        "When was Craig Newmark born?",
    ]
    assert result.agent_summary == "December 6, 1952"
    assert [c["content"] for c in result.chunks] == [
        "Craigslist was founded by Craig.",
        "Craig Newmark, born December 6, 1952.",
    ]
    # Generation stops where the search must answer, and the search's answer
    # is what the model continues from.
    assert "\nIntermediate answer:" in llm.stops[0]
    assert "Craigslist was founded by Craig." in llm.prompts[1]
    assert llm.prompts[2].endswith(
        "Follow up: Who was the founder of craigslist?\nIntermediate answer: Craig Newmark."
    )
    assert result.metrics.stop_reason == "final_answer"


async def test_self_ask_does_not_search_when_no_follow_up_is_needed(monkeypatch):
    backend = install_backend(monkeypatch, {})
    llm = ScriptedLLM(["No.\nSo the final answer is: Paris"])
    result = await _loop("self_ask").run(pipeline_input("What is the capital of France?", llm))
    assert backend.calls == [] and result.agent_summary == "Paris" and result.chunks == []


async def test_self_ask_concludes_when_out_of_follow_ups(monkeypatch):
    backend = install_backend(monkeypatch, {"A?": [hit("a", "alpha")]})
    llm = ScriptedLLM(
        [
            "Yes.\nFollow up: A?",
            "alpha",
            "Follow up: B?",  # over the budget of one: never searched
            "alpha-beta",
        ]
    )
    config = ArmConfig(self_ask_max_follow_ups=1)
    result = await _loop("self_ask").run(pipeline_input("A and B?", llm, config))

    assert [c["query"] for c in backend.calls] == ["A?"]
    assert llm.prompts[-1].endswith("Intermediate answer: alpha.\nSo the final answer is:")
    assert result.agent_summary == "alpha-beta"
    assert result.metrics.stop_reason == "max_follow_ups"


LOOP_SETTINGS["self_ask"] = ["self_ask_max_follow_ups"]
SHORT_RUNS["self_ask"] = ["Yes.\nFollow up: Who?", "Ada.", "So the final answer is: Ada"]


@pytest.mark.parametrize(
    "text,first",
    [
        (
            "Craigslist was founded by Craig Newmark. He was born in 1952.",
            "Craigslist was founded by Craig Newmark.",
        ),
        (
            "Nosferatu was directed by F.W. Murnau. So the answer is: x.",
            "Nosferatu was directed by F.W. Murnau.",
        ),
        ("Dr. Caligari is a 1920 film. It is silent.", "Dr. Caligari is a 1920 film."),
        ("It opened in 1999. Then it closed.", "It opened in 1999."),
        ("  One sentence only  ", "One sentence only"),
    ],
)
def test_a_step_is_the_first_sentence_whatever_is_installed(text, first):
    from lasagna.agentic.loops._base import first_sentence

    assert first_sentence(text) == first


# ---------------------------------------------------------------------------
# Every loop: registration, the stage contract, search and settings
# ---------------------------------------------------------------------------

LOOPS = list(LOOP_SETTINGS)


@pytest.mark.parametrize("name", LOOPS)
def test_each_loop_is_a_listed_built_in_that_implements_the_stage(name):
    listed = {e["name"]: e for e in registry.list_pipelines()}
    assert name in listed and registry.is_builtin("agentic_pipeline", name)
    assert issubclass(registry.get_class("agentic_pipeline", name), BaseAgenticPipeline)
    assert listed[name]["paper"].startswith("arxiv:")


@pytest.mark.parametrize("name", LOOPS)
async def test_each_loop_without_an_llm_says_so(name, monkeypatch):
    import lasagna.providers as providers

    def no_provider():
        raise providers.LLMProviderNotConfigured("no key")

    monkeypatch.setattr(providers, "get_llm_provider", no_provider)
    backend = install_backend(monkeypatch, {})
    with pytest.raises(RuntimeError, match=f"{name} needs an LLM, and none is configured"):
        await _loop(name).run(pipeline_input("q", llm=None))
    assert backend.calls == [], "a loop with no LLM must not retrieve and carry on"


@pytest.mark.parametrize("name", LOOPS)
async def test_each_loop_searches_through_the_pipeline_under_the_arms_settings(name, monkeypatch):
    backend = install_backend(monkeypatch, {}, default=[hit("a", "Ada wrote the first program.")])
    config = ArmConfig(reranker_model="arm-reranker", hybrid_alpha=0.3)
    inp = pipeline_input("Who wrote the first program?", ScriptedLLM(SHORT_RUNS[name]), config)
    before = (inp.question, inp.session_id, inp.top_k, dict(config._query_overrides))

    result = await _loop(name).run(inp)

    assert backend.calls, f"{name} never searched"
    for call in backend.calls:
        assert call["_from_routing"] is True  # the standard tier, not the agentic one again
        assert call["session_id"] == "arm-session"
        assert call["query_config_overrides"] == {
            "reranker_model": "arm-reranker",
            "hybrid_alpha": 0.3,
        }
    # The answer travels with the passages it was read from, and what the loop did.
    assert result.agent_summary.startswith("Ada")
    assert result.chunks and result.chunks[0]["content"] == "Ada wrote the first program."
    assert result.chunks[0]["metadata"]["_agentic_loop"]["method"] == name
    assert (inp.question, inp.session_id, inp.top_k, dict(config._query_overrides)) == before


@pytest.mark.parametrize("name", LOOPS)
def test_each_loops_settings_are_sweep_keys_of_the_agentic_stage(name):
    from lasagna.core.config import rag_config
    from lasagna.knowledge_evaluation_sdk.stages import stage_of

    for key in LOOP_SETTINGS[name]:
        assert key in rag_config._config, key  # a sweep accepts it
        assert stage_of(key) == "agentic_pipeline", key  # and files it under this stage
        assert getattr(rag_config, key) == rag_config._config[key], key
