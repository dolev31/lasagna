"""The listwise LLM reranker runs RankGPT, not a look-alike.

Sun et al. 2023 ("Is ChatGPT Good at Search?") slide a window of 20 passages
from the bottom of the list to the top in steps of 10, and each window's
permutation reorders it in place, so a relevant passage can climb across
windows. The reranker called itself RankGPT-style but slid front to back with
a fixed step of half the window and summed Borda counts, kept the retrieval
order when the LLM failed, and read any number in a response as a passage.
The LLM is a stub; nothing here calls one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.knowledge_evaluation_sdk.stages import stage_of
from lasagna.retrieval.search import ranker
from lasagna.retrieval.search.ranker import parse_rankgpt_permutation


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


def test_window_and_step_are_reranker_settings():
    assert stage_of("llm_reranker_window_size") == "reranker"
    assert stage_of("llm_reranker_step_size") == "reranker"


def _windows(llm):
    """The passages shown in each call, as the LLM saw them."""
    out = []
    for messages in llm.prompts:
        shown = [m.content for m in messages if m.type == "human" and m.content.startswith("[")]
        out.append([s.split("] ", 1)[1] for s in shown])
    return out


def _gold_first(messages):
    """Put any passage containing "gold" first; otherwise keep the window's order."""
    shown = [m.content for m in messages if m.type == "human" and m.content.startswith("[")]
    ids = list(range(1, len(shown) + 1))
    ids.sort(key=lambda i: "gold" not in shown[i - 1])
    return " > ".join(f"[{i}]" for i in ids)


def _rows(texts):
    return [
        {"id": f"c{i}", "content": t, "score": 1.0, "document_id": f"d{i}", "metadata": {}}
        for i, t in enumerate(texts)
    ]


def _listwise(**settings):
    config = SimpleNamespace(
        llm_reranker_window_size=20, llm_reranker_step_size=10, llm_reranker_model="m", **settings
    )
    return ranker.LLMListwiseReranker(config=config), config


class TestRankGPT:
    def test_windows_slide_from_the_bottom_and_carry_a_passage_to_the_top(self, llm_provider):
        llm = llm_provider.set(_gold_first)
        texts = [f"p{i}" for i in range(29)] + ["gold"]
        reranker, config = _listwise()
        out = reranker.rerank("q", _rows(texts), top_k=30, config=config)
        windows = _windows(llm)
        # 30 passages, window 20, step 10: positions 10-29, then 0-19.
        assert windows[0] == texts[10:]
        assert windows[1] == texts[:10] + ["gold"] + texts[10:19]
        assert [r.content for r in out][:3] == ["gold", "p0", "p1"]

    def test_window_and_step_are_the_arms_settings(self, llm_provider):
        llm = llm_provider.set(_gold_first)
        reranker, config = _listwise(**{})
        config.llm_reranker_window_size, config.llm_reranker_step_size = 4, 2
        reranker.rerank("q", _rows([f"p{i}" for i in range(8)]), top_k=8, config=config)
        # Windows 4-7, 2-5, 0-3.
        assert [w[0] for w in _windows(llm)] == ["p4", "p2", "p0"]

    def test_a_list_shorter_than_a_window_is_still_ranked(self, llm_provider):
        llm = llm_provider.set(_gold_first)
        reranker, config = _listwise()
        out = reranker.rerank("q", _rows(["a", "b", "gold"]), top_k=3, config=config)
        assert len(llm.prompts) == 1 and out[0].content == "gold"

    def test_the_prompt_is_rankgpts(self, llm_provider):
        llm = llm_provider.set(_gold_first)
        reranker, config = _listwise()
        long_passage = " ".join(["word"] * 400)
        reranker.rerank("q", _rows(["a", long_passage]), top_k=2, config=config)
        messages = llm.prompts[0]
        assert messages[0].type == "system" and "RankGPT" in messages[0].content
        assert messages[-1].content.startswith("Search Query: q.")
        assert "[] > []" in messages[-1].content
        assert len(messages[5].content.split()) == 1 + 300, "a passage is cut to 300 words"

    @pytest.mark.parametrize(
        "reply,order,named",
        [
            ("[3] > [1] > [3] > [9]", [2, 0, 1], True),
            ("Here are the 20 passages ranked: [2] > [1]", [1, 0, 2], True),
            ("<think>3 then 1</think>2 > 1", [1, 0, 2], True),
            ("I cannot rank these.", [0, 1, 2], False),
        ],
    )
    def test_permutations_are_parsed_robustly(self, reply, order, named):
        assert parse_rankgpt_permutation(reply, 3) == (order, named)

    def test_no_response_naming_a_passage_is_an_error(self, llm_provider):
        llm_provider.set(lambda messages: "Sorry, I can't help with that.")
        reranker, config = _listwise()
        with pytest.raises(RuntimeError, match="named a passage"):
            reranker.rerank("q", _rows(["a", "b"]), top_k=2, config=config)

    def test_without_an_llm_it_says_so(self, no_llm):
        reranker, config = _listwise()
        with pytest.raises(RuntimeError, match="llm_listwise needs an LLM, and none is configured"):
            reranker.rerank("q", _rows(["a", "b"]), top_k=2, config=config)

    def test_the_component_passes_the_reranker_contract(self, llm_provider):
        llm_provider.set(_gold_first)
        report = ComponentTestHarness().validate(registry.get("reranker", "llm_listwise"))
        assert report.passed, report.failures
