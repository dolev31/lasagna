"""A loop's stop sequences do not end a reasoning model's reply before it answers.

Self-Ask stops generation where the search must answer ("Intermediate answer:")
and asks for its final answer with a stop at the newline. A reasoning model
served with its thinking split off (Ollama's ``think``, vLLM's reasoning
parser) is stopped on its reasoning too. Qwen3's reasoning opens with
``<think>\\n``, so the newline stop ended the reply after two tokens with
nothing said: in Ollama 0.40.13 with qwen3:0.6b, ``content`` and ``thinking``
both came back empty. Every self_ask question, in the Compose stack and on
Qwen3-8B under vLLM, ended with an empty answer, and the harness had no answer
of the loop's to score.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from lasagna.core.registry import registry

from tests._loop_stubs import ScriptedLLM, install_backend, pipeline_input


class Reasons(ScriptedLLM):
    """A reasoning model: any stop ends it inside its reasoning, so a call with
    stops says nothing. Asked without, it gives its scripted reply, which opens
    with the blank lines a reasoning model's answer starts with."""

    async def ainvoke(self, messages, stop=None, **kwargs):
        if stop:
            self.prompts.append(messages[-1].content)
            self.stops.append(stop)
            return AIMessage(content="")
        return await super().ainvoke(messages, **kwargs)


async def _self_ask(question, llm, monkeypatch):
    install_backend(monkeypatch, {})
    loop = registry.get("agentic_pipeline", "self_ask")
    return await loop.run(pipeline_input(question, llm))


async def test_a_reasoning_model_gets_its_answer_through(monkeypatch):
    llm = Reasons(["\n\nNo.\nSo the final answer is: Paris"])
    result = await _self_ask("What is the capital of France?", llm, monkeypatch)
    assert result.agent_summary == "Paris"
    assert result.metrics.stop_reason == "final_answer"


async def test_the_final_answer_is_its_first_line_after_the_blank_ones(monkeypatch):
    # No marker in the chain, so the loop asks for the final answer with a
    # newline stop: the call that came back empty for every question.
    llm = Reasons(["\n\nThe capital of France is Paris.", "\n\nParis\nIt has been since 987."])
    result = await _self_ask("What is the capital of France?", llm, monkeypatch)
    assert llm.stops[-2] == ["\n"] and llm.stops[-1] is None
    assert result.agent_summary == "Paris"
    assert result.metrics.stop_reason == "concluded_without_marker"


async def test_a_model_that_honors_stops_is_asked_once(monkeypatch):
    llm = ScriptedLLM(["No.\nSo the final answer is: Paris"])
    result = await _self_ask("What is the capital of France?", llm, monkeypatch)
    assert result.agent_summary == "Paris"
    assert len(llm.prompts) == 1 and llm.stops[0]


async def test_an_empty_reply_is_asked_again_once_not_more(monkeypatch):
    llm = ScriptedLLM(["", "", "", ""])
    result = await _self_ask("What is the capital of France?", llm, monkeypatch)
    assert result.agent_summary == ""
    assert [bool(s) for s in llm.stops] == [True, False, True, False]
