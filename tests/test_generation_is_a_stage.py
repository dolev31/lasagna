"""Answer generation is a swappable stage, like the other seven.

The answer prompt was hard-coded in the harness, so "how the answer is
written" (citations, chain of thought, reading notes) could not be compared
with everything else held fixed. A run that names no generator keeps the old
prompt word for word, so earlier results stay reproducible.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import lasagna
from lasagna.core.base_components import BaseGenerator
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness

CONTEXTS = ["Paris is the capital of France.", "The Seine flows through Paris."]


class _ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


@pytest.mark.parametrize("name", ["concise", "cited", "chain_of_thought", "chain_of_note"])
def test_every_built_in_generator_passes_its_contract(name):
    gen = registry.get("generator", name)
    assert isinstance(gen, BaseGenerator)
    assert ComponentTestHarness().validate(gen).passed


@pytest.mark.parametrize("name", ["concise", "cited", "chain_of_thought", "chain_of_note"])
def test_a_generator_without_a_model_says_so(name):
    with pytest.raises(RuntimeError, match="needs an LLM"):
        registry.get("generator", name).generate("q?", CONTEXTS, llm=None)


def test_concise_is_the_harness_prompt_word_for_word():
    llm = _ScriptedLLM("Paris.")
    registry.get("generator", "concise").generate("Capital?", CONTEXTS, llm=llm)
    assert llm.prompts[0] == (
        "Answer the following question in ONE concise sentence based only on the "
        "provided context. Do not explain or elaborate. \n\n"
        "Context:\nParis is the capital of France.\n\nThe Seine flows through Paris.\n\n"
        "Question: Capital?\n\nAnswer:"
    )


def test_cited_numbers_the_documents_it_asks_to_cite():
    llm = _ScriptedLLM("Paris is the capital [1].")
    out = registry.get("generator", "cited").generate("Capital?", CONTEXTS, llm=llm)
    assert out == "Paris is the capital [1]."
    assert "Document [1]: Paris" in llm.prompts[0] and "Document [2]: The Seine" in llm.prompts[0]
    assert "cite" in llm.prompts[0].lower()


def test_reasoning_strategies_return_only_the_final_answer():
    reasoning = "The first document says Paris.\nSo the capital is Paris.\nAnswer: Paris"
    for name in ("chain_of_thought", "chain_of_note"):
        out = registry.get("generator", name).generate(
            "Capital?", CONTEXTS, llm=_ScriptedLLM(reasoning)
        )
        assert out == "Paris", name


def test_the_abstention_instruction_reaches_the_prompt():
    llm = _ScriptedLLM("unanswerable")
    registry.get("generator", "cited").generate(
        "q?", CONTEXTS, llm=llm, instructions="MAY ABSTAIN."
    )
    assert "MAY ABSTAIN." in llm.prompts[0]


def test_a_function_is_a_generator_in_one_line():
    @lasagna.generator("t_first_sentence")
    def first_sentence(query, contexts):
        return contexts[0].split(".")[0] if contexts else ""

    gen = registry.get("generator", "t_first_sentence")
    assert isinstance(gen, BaseGenerator)
    assert gen.generate("q", CONTEXTS, llm=None) == "Paris is the capital of France"
    assert ComponentTestHarness().validate(gen).passed

    @lasagna.generator("t_uses_the_model")
    def with_model(query, contexts, llm):
        return llm.invoke(query).content

    assert (
        registry.get("generator", "t_uses_the_model").generate("q", [], llm=_ScriptedLLM("ok"))
        == "ok"
    )


def test_the_pipeline_answers_with_its_generator():
    from lasagna.core.pipeline import KnowledgePipeline

    p = KnowledgePipeline()
    assert p.generate("Capital?", CONTEXTS, llm=_ScriptedLLM("Paris.")) == "Paris."
    p.replace("generator", "chain_of_thought")
    assert p.generate("Capital?", CONTEXTS, llm=_ScriptedLLM("x\nAnswer: Paris")) == "Paris"


def test_a_sweep_can_vary_the_generator():
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _parse_config

    cfg = _parse_config(
        {
            "name": "gen",
            "benchmarks": [{"name": "smoke"}],
            "pipeline": {"base_profile": "speed", "generators": ["concise", "cited"]},
        }
    )
    tasks = Experiment(cfg).generate_tasks()
    assert {t.config_overrides.get("generator") for t in tasks} == {"concise", "cited"}


def test_offline_scoring_uses_the_named_generator(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")

    class _Judge:
        def __init__(self, llm=None):
            pass

        def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
            return ragas.DetailedEvalResult(aggregate={}, per_sample=[{} for _ in questions])

    llm = _ScriptedLLM("reasoning\nAnswer: blue")
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda: None)
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: llm)
    )
    task = BenchmarkTask(
        benchmark_name="b",
        config_name="c",
        run_index=0,
        mode="offline",
        config_overrides={"generator": "chain_of_thought"},
    )
    sample = BenchmarkSample(
        query_id="q", question="sky?", ground_truth="blue", context="The sky is blue."
    )
    result = asyncio.run(exp._eval_offline(task, [sample]))
    assert result.error is None, result.error
    assert result.metrics["exact_match"] == 1.0, "the final answer line was scored"
    assert "think step by step" in llm.prompts[0].lower()
