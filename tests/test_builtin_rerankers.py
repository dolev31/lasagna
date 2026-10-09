"""monoT5, LLM grading, FlashRank and Cohere do what they are named for, or fail.

Each is checked against its method on passages small enough to order by hand
(monoT5's softmax over "true"/"false", a 0-3 relevance grade), and each fails
loudly when its model, LLM, package or key is missing: returning the retrieval
order would put the control's number under the reranker's name. Every model
and LLM is stubbed except in the real-model test, which skips when the model is
not cached. No test touches the network.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.core.base_components import ScoredDocument
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.knowledge_evaluation_sdk.stages import stage_of
from lasagna.retrieval.search import rerankers
from lasagna.retrieval.search.ranker import adapt_reranker_for_pipeline


def _docs(*texts):
    return [
        ScoredDocument(
            content=t, score=1.0 - i / 10, document_id=f"d{i}", chunk_id=f"c{i}", metadata={"i": i}
        )
        for i, t in enumerate(texts)
    ]


def _order(out):
    return [d.chunk_id for d in out]


class _LLM:
    """Answers with ``reply(prompt)``; records every prompt."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply(prompt))


@pytest.fixture
def llm_provider(monkeypatch):
    """Install a provider whose LLM is ``set(reply)``; records the model asked for."""
    state = SimpleNamespace(llm=None, models=[])

    def install(reply):
        state.llm = _LLM(reply)
        return state.llm

    provider = SimpleNamespace(
        get_llm=lambda model_name=None, temperature=0.0, **kw: state.models.append(model_name)
        or state.llm
    )
    monkeypatch.setattr("lasagna.providers.get_llm_provider", lambda: provider)
    state.set = install
    return state


@pytest.fixture
def no_llm(monkeypatch):
    from lasagna.providers import LLMProviderNotConfigured

    def missing():
        raise LLMProviderNotConfigured("No LLMProvider configured")

    monkeypatch.setattr("lasagna.providers.get_llm_provider", missing)


def test_their_settings_belong_to_the_reranker_stage():
    for key in (
        "monot5_model",
        "flashrank_model",
        "cohere_rerank_model",
    ):
        assert stage_of(key) == "reranker", key


def test_the_new_rerankers_are_listed_as_conforming_with_their_sources():
    listed = {e["name"]: e for e in registry.list_rerankers()}
    for name in ("monot5", "llm_pointwise", "flashrank", "cohere"):
        assert listed[name]["implements_contract"], name
    assert listed["monot5"]["paper"] == "arxiv:2003.06713"
    assert listed["llm_pointwise"]["paper"] == "arxiv:2309.10621"


# ---------------------------------------------------------------------------
# monoT5
# ---------------------------------------------------------------------------


class _FakeMonoT5:
    """[false, true] logits chosen per passage; records the inputs it was given."""

    def __init__(self, logits):
        self.logits, self.inputs = logits, []

    def true_false_logits(self, texts, batch_size):
        self.inputs.extend(texts)
        return np.array(
            [self.logits[t.split("Document: ")[1][: -len(" Relevant:")]] for t in texts]
        )


@pytest.fixture
def monot5(monkeypatch):
    monkeypatch.setattr(rerankers.MonoT5Reranker, "_models", {})
    monkeypatch.setattr(rerankers.MonoT5Reranker, "_failed", {})
    fake = _FakeMonoT5({"a": [5.0, 4.0], "b": [0.0, 1.0], "c": [0.0, 0.0]})
    loaded = []
    monkeypatch.setattr(
        rerankers.MonoT5Reranker,
        "_load",
        classmethod(lambda cls, name: loaded.append(name) or fake),
    )
    fake.loaded = loaded
    return fake


class TestMonoT5:
    def test_it_passes_the_reranker_contract(self, monot5):
        monot5.logits.update(
            {f"Sample document {i} with some content for testing.": [0.0, -i] for i in range(5)}
        )
        report = ComponentTestHarness().validate(registry.get("reranker", "monot5"))
        assert report.passed, report.failures

    def test_a_passage_scores_the_log_probability_of_true_against_false(self, monot5):
        out = registry.get("reranker", "monot5").rerank("q", _docs("a", "b", "c"), top_k=3)
        # Raw "true" logits order a (4) > b (1) > c (0); P(true) orders b > c > a.
        assert _order(out) == ["c1", "c2", "c0"]
        expected = [1 - np.logaddexp(0, 1), 0 - np.logaddexp(0, 0), 4 - np.logaddexp(5, 4)]
        assert [d.score for d in out] == pytest.approx(expected)

    def test_the_input_is_monot5s_template(self, monot5):
        registry.get("reranker", "monot5").rerank("is it", _docs("a"), top_k=1)
        assert monot5.inputs == ["Query: is it Document: a Relevant:"]

    def test_the_arms_model_is_the_one_loaded(self, monot5):
        adapted = adapt_reranker_for_pipeline(registry.get("reranker", "monot5"))
        rows = [{"id": "c0", "content": "a", "score": 1.0, "document_id": "d0", "metadata": {}}]
        adapted.rerank(query="q", results=rows, top_k=1, config=SimpleNamespace(monot5_model="m3"))
        assert monot5.loaded == ["m3"]

    def test_a_model_that_cannot_load_raises_and_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(rerankers.MonoT5Reranker, "_models", {})
        monkeypatch.setattr(rerankers.MonoT5Reranker, "_failed", {})
        attempts = []

        def broken(name):
            attempts.append(name)
            raise OSError(f"{name} is not on the Hub")

        monkeypatch.setattr(rerankers, "_MonoT5", broken)
        component = rerankers.MonoT5Reranker(model_name="nope/monot5")
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Cannot load monoT5 model 'nope/monot5'"):
                component.rerank("q", _docs("a", "b"), top_k=2)
        assert attempts == ["nope/monot5"]


# ---------------------------------------------------------------------------
# LLM pointwise grading
# ---------------------------------------------------------------------------


def _grader(prompt):
    passage = prompt.split("Passage: ")[1].split("\n")[0]
    return {"exact answer": "##final score: 3", "some answer": "##final score: 2"}.get(
        passage, "##final score: 0"
    )


class TestLLMPointwise:
    def test_it_passes_the_reranker_contract(self, llm_provider):
        llm_provider.set(_grader)
        report = ComponentTestHarness().validate(registry.get("reranker", "llm_pointwise"))
        assert report.passed, report.failures

    def test_passages_are_ordered_by_grade_ties_by_retrieval_order(self, llm_provider):
        llm_provider.set(_grader)
        docs = _docs("unrelated", "some answer", "exact answer", "also unrelated", "some answer")
        out = registry.get("reranker", "llm_pointwise").rerank("q", docs, top_k=5)
        assert _order(out) == ["c2", "c1", "c4", "c0", "c3"]
        assert [d.score for d in out] == [3.0, 2.0, 2.0, 0.0, 0.0]

    def test_each_prompt_holds_the_scale_the_query_and_one_passage(self, llm_provider):
        llm = llm_provider.set(_grader)
        registry.get("reranker", "llm_pointwise").rerank("why is the sky blue", _docs("p"), top_k=1)
        (prompt,) = llm.prompts
        assert "integer scale of 0 to 3" in prompt
        assert "Query: why is the sky blue\nPassage: p\n" in prompt

    def test_the_arms_llm_is_the_one_asked(self, llm_provider):
        llm_provider.set(_grader)
        registry.get("reranker", "llm_pointwise").rerank(
            "q", _docs("p"), top_k=1, config=SimpleNamespace(llm_reranker_model="judge-x")
        )
        assert llm_provider.models == ["judge-x"]

    @pytest.mark.parametrize(
        "reply,grade",
        [
            ("##final score: 2", 2),
            ("**Final score:** 3", 3),
            ('[{"M": 2, "T": 1, "O": 1}]', 1),
            ("<think>maybe 3, maybe 1</think>##final score: 1", 1),
            ("2", 2),
            ("##final score: 7", None),
            ("I cannot judge this.", None),
        ],
    )
    def test_grades_are_read_from_the_shapes_llms_answer_in(self, reply, grade):
        assert rerankers.LLMPointwiseReranker.parse_grade(reply) == grade

    def test_an_unreadable_grade_counts_as_zero(self, llm_provider):
        llm_provider.set(lambda p: "no idea" if "first" in p else "##final score: 1")
        out = registry.get("reranker", "llm_pointwise").rerank(
            "q", _docs("first", "second"), top_k=2
        )
        assert _order(out) == ["c1", "c0"]

    def test_no_readable_grade_at_all_is_an_error(self, llm_provider):
        llm_provider.set(lambda p: "I'd rather not say.")
        with pytest.raises(RuntimeError, match="no response .* contained a 0-3 grade"):
            registry.get("reranker", "llm_pointwise").rerank("q", _docs("a", "b"), top_k=2)

    def test_without_an_llm_it_says_so(self, no_llm):
        with pytest.raises(
            RuntimeError, match="llm_pointwise needs an LLM, and none is configured"
        ):
            registry.get("reranker", "llm_pointwise").rerank("q", _docs("a"), top_k=1)


# ---------------------------------------------------------------------------
# Optional dependencies: FlashRank and Cohere
# ---------------------------------------------------------------------------


def _flashrank_module(listwise=False):
    module = types.ModuleType("flashrank")

    class Ranker:
        built = []

        def __init__(self, model_name, cache_dir=None):
            Ranker.built.append(model_name)

        def rerank(self, request):
            ranked = sorted(request.passages, key=lambda p: -len(p["text"]))
            if listwise:
                return [dict(p) for p in ranked]
            return [dict(p, score=len(p["text"]) / 10) for p in ranked]

    class RerankRequest:
        def __init__(self, query=None, passages=None):
            self.query, self.passages = query, passages or []

    module.Ranker, module.RerankRequest = Ranker, RerankRequest
    return module


class TestFlashRank:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(rerankers.FlashRankReranker, "_rankers", {})

    def test_without_the_package_it_names_the_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flashrank", None)
        with pytest.raises(ImportError, match="pip install flashrank"):
            registry.get("reranker", "flashrank").rerank("q", _docs("a"), top_k=1)

    def test_scores_come_from_flashrank_and_the_arms_model_loads(self, monkeypatch):
        module = _flashrank_module()
        monkeypatch.setitem(sys.modules, "flashrank", module)
        out = registry.get("reranker", "flashrank").rerank(
            "q", _docs("aa", "aaaa", "a"), top_k=3, config=SimpleNamespace(flashrank_model="tiny")
        )
        assert _order(out) == ["c1", "c0", "c2"]
        assert [d.score for d in out] == pytest.approx([0.4, 0.2, 0.1])
        assert module.Ranker.built == ["tiny"]

    def test_a_listwise_model_order_is_kept(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flashrank", _flashrank_module(listwise=True))
        out = registry.get("reranker", "flashrank").rerank("q", _docs("aa", "aaaa", "a"), top_k=3)
        assert _order(out) == ["c1", "c0", "c2"]

    def test_it_passes_the_reranker_contract(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flashrank", _flashrank_module())
        report = ComponentTestHarness().validate(registry.get("reranker", "flashrank"))
        assert report.passed, report.failures


def _cohere_module(fail=False):
    module = types.ModuleType("cohere")

    class ClientV2:
        calls = []

        def __init__(self, api_key):
            self.api_key = api_key

        def rerank(self, model, query, documents, top_n):
            if fail:
                raise ConnectionError("503")
            ClientV2.calls.append((model, query, list(documents), top_n))
            ranked = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=i, relevance_score=len(documents[i]) / 10) for i in ranked
                ]
            )

    module.ClientV2 = ClientV2
    return module


class TestCohere:
    def test_without_the_package_it_names_the_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cohere", None)
        with pytest.raises(ImportError, match="pip install cohere"):
            registry.get("reranker", "cohere").rerank("q", _docs("a"), top_k=1)

    def test_without_a_key_it_names_the_variable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cohere", _cohere_module())
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
            registry.get("reranker", "cohere").rerank("q", _docs("a"), top_k=1)

    def test_scores_are_coheres_relevance_scores(self, monkeypatch):
        module = _cohere_module()
        monkeypatch.setitem(sys.modules, "cohere", module)
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        out = registry.get("reranker", "cohere").rerank(
            "q", _docs("aa", "aaaa", "a"), top_k=2, config=SimpleNamespace(cohere_rerank_model="r1")
        )
        assert _order(out) == ["c1", "c0"]
        assert [d.score for d in out] == pytest.approx([0.4, 0.2])
        assert module.ClientV2.calls == [("r1", "q", ["aa", "aaaa", "a"], 3)]

    def test_a_failed_call_is_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cohere", _cohere_module(fail=True))
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        with pytest.raises(RuntimeError, match="Rerank call failed"):
            registry.get("reranker", "cohere").rerank("q", _docs("a", "b"), top_k=2)

    def test_it_passes_the_reranker_contract(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cohere", _cohere_module())
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        report = ComponentTestHarness().validate(registry.get("reranker", "cohere"))
        assert report.passed, report.failures


# ---------------------------------------------------------------------------
# Real models, when cached
# ---------------------------------------------------------------------------


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


QUESTION = "what is the capital of france"
PASSAGES = ("Bananas are rich in potassium.", "Paris is the capital and largest city of France.")


@pytest.mark.skipif(
    not _cached("castorini/monot5-base-msmarco-10k"), reason="monoT5 checkpoint not cached"
)
def test_monot5_prefers_the_passage_that_answers(monkeypatch):
    monkeypatch.setattr(rerankers.MonoT5Reranker, "_models", {})
    out = rerankers.MonoT5Reranker().rerank(QUESTION, _docs(*PASSAGES), top_k=2)
    assert _order(out) == ["c1", "c0"] and out[0].score > out[1].score
