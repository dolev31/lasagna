"""The published query-augmentation enhancers do what their papers do, in the pipeline.

Each enhancer is checked three ways: it satisfies the query_enhancer contract
(the same harness a sweep runs before ingesting anything), its output follows
from what the LLM wrote the way its paper says, and without an LLM it says so
instead of handing back the bare query, which would score as the control.
The last test runs one through the pipeline's own multi-query search, so the
variations are shown to be searched and fused there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.retrieval.enhancement import augmentation

NEW = ["step_back", "query2doc", "rewrite", "multi_query"]


class SyncLLM:
    """A chat model that answers every prompt with one scripted reply."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        self.prompts.append(messages[-1]["content"])
        reply = self.reply(self.prompts[-1]) if callable(self.reply) else self.reply
        return AIMessage(content=reply)


def _with_llm(monkeypatch, name, reply):
    """The registered enhancer, its LLM replaced by a scripted one."""
    component = registry.get("query_enhancer", name)
    llm = SyncLLM(reply)
    monkeypatch.setattr(type(component), "_llm_for", lambda self, *a, **k: llm)
    return component, llm


@pytest.mark.parametrize("name", NEW)
def test_each_is_a_listed_built_in_with_its_paper(name):
    entry = next(e for e in registry.list_enhancers() if e["name"] == name)
    assert entry["implements_contract"] and entry["paper"].startswith("arxiv:")
    assert registry.get_class("query_enhancer", name)._forwards_session_config


@pytest.mark.parametrize("name", NEW)
def test_each_passes_the_query_enhancer_contract(name, monkeypatch):
    component, _ = _with_llm(monkeypatch, name, "1. first generated query\n2. second one here")
    report = ComponentTestHarness().validate(component)
    assert report.passed, report.summary()


@pytest.mark.parametrize("name", NEW)
def test_each_without_an_llm_says_so(name, monkeypatch):
    def no_provider():
        raise RuntimeError("no LLMProvider configured")

    monkeypatch.setattr(augmentation, "get_llm_provider", no_provider)
    monkeypatch.setattr(augmentation, "resolve_model_settings", lambda **kw: {})
    with pytest.raises(RuntimeError, match=f"{name} needs an LLM, and none is configured"):
        registry.get("query_enhancer", name).enhance("who wrote hamlet")


def test_step_back_searches_the_question_and_its_abstraction(monkeypatch):
    component, llm = _with_llm(
        monkeypatch, "step_back", "Which positions have Ada Lovelace held in her career?"
    )
    out = component.enhance("Which position did Ada Lovelace hold in 1843?")
    assert out == [
        "Which position did Ada Lovelace hold in 1843?",
        "Which positions have Ada Lovelace held in her career?",
    ]
    # The paper's few-shot prompt, ending on the question to abstract.
    assert "step back and paraphrase a question" in llm.prompts[0]
    assert llm.prompts[0].endswith(
        "Original Question: Which position did Ada Lovelace hold in 1843?\nStepback Question:"
    )


def test_query2doc_puts_the_query_in_front_of_its_pseudo_document(monkeypatch):
    component, llm = _with_llm(monkeypatch, "query2doc", "Hamlet is a tragedy by Shakespeare.")
    dense = component.enhance("who wrote hamlet", config=SimpleNamespace(query2doc_format="dense"))
    # The paper's dense form, q [SEP] d'; HyDE would search d' alone.
    assert dense == [
        "who wrote hamlet",
        "who wrote hamlet [SEP] Hamlet is a tragedy by Shakespeare.",
    ]
    assert llm.prompts[0].startswith("Write a passage that answers the given query:\nQuery: ")
    assert llm.prompts[0].endswith("Query: who wrote hamlet\nPassage:")

    # The sparse form repeats the query (n = 5 in the paper) to keep its terms' weight.
    sparse = component.enhance(
        "who wrote hamlet",
        config=SimpleNamespace(query2doc_format="sparse", query2doc_query_repeats=5),
    )
    assert sparse[1] == "who wrote hamlet " * 5 + "Hamlet is a tragedy by Shakespeare."


def test_rewrite_searches_each_query_the_rewriter_wrote(monkeypatch):
    component, llm = _with_llm(
        monkeypatch, "rewrite", "Nicholas Ray profession; Elia Kazan profession**"
    )
    q = "What profession does Nicholas Ray and Elia Kazan have in common?"
    assert component.enhance(q, n=3) == [q, "Nicholas Ray profession", "Elia Kazan profession"]
    assert "Split the queries with ';' and end the queries with '**'" in llm.prompts[0]
    # At most the pipeline's variation count.
    assert component.enhance(q, n=1) == [q, "Nicholas Ray profession"]


def test_multi_query_asks_for_the_pipelines_count_and_returns_that_many(monkeypatch):
    reply = "1. rag fusion explained\n2. what is reciprocal rank fusion\n3. rag vs rag fusion\n4. fusion retrieval"
    component, llm = _with_llm(monkeypatch, "multi_query", reply)
    out = component.enhance("what is rag fusion", n=4)
    assert out == [
        "what is rag fusion",
        "rag fusion explained",
        "what is reciprocal rank fusion",
        "rag vs rag fusion",
        "fusion retrieval",
    ]
    # RAG-Fusion's messages: its system prompt, the query, then the count.
    assert [m["role"] for m in llm.messages[0]] == ["system", "user", "user"]
    assert (
        "generates multiple search queries based on a single input query"
        in llm.messages[0][0]["content"]
    )
    assert (
        llm.messages[0][1]["content"]
        == "Generate multiple search queries related to: what is rag fusion"
    )
    assert llm.messages[0][2]["content"] == "OUTPUT (4 queries):"


def test_generated_queries_are_searched_and_fused_by_the_pipeline(monkeypatch):
    from lasagna.api import _search_orchestrator as orch_mod

    class _Config(SimpleNamespace):
        def with_query_overrides(self, overrides):
            return _Config(**{**self.__dict__, **overrides})

    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.query_enhancer = None
    o._collection_manager = SimpleNamespace(
        resolve_backend=lambda sid: SimpleNamespace(backend_name="milvus")
    )
    o._get_session_config = lambda sid: _Config(
        enable_query_expansion=False, rerank_top_k=5, rrf_k=60, max_query_variations=2
    )
    searched = []

    def fake_search(query, session_id=None, top_k=None, **kw):
        searched.append(query)
        ids = {"q": ["a", "b"], "q one": ["b", "c"], "q two": ["b", "d"]}[query]
        return [
            SimpleNamespace(chunk=SimpleNamespace(chunk_id=i), score=1.0, document=None)
            for i in ids
        ]

    o.search = fake_search
    component, _ = _with_llm(monkeypatch, "multi_query", "1. q one\n2. q two\n3. q three")
    out = o.search_multi_query("q", "s", top_k=3, n_variations=2, _enhancer_override=component)

    assert sorted(searched) == ["q", "q one", "q two"]  # capped at the pipeline's count
    # "b" is in all three lists, so reciprocal rank fusion puts it first.
    assert [r.chunk.chunk_id for r in out][0] == "b"
