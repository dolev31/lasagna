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
    with_logprobs,
)

#: Each loop's settings and one short scripted run that reaches a search;
#: every loop's section below adds its own entries.
LOOP_SETTINGS = {}
SHORT_RUNS = {}

Q_BIDEN = "Where did Joe Biden go to law school?"


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
# IRCoT
# ---------------------------------------------------------------------------

Q_CRAIG = "When was the founder of craigslist born?"
S1 = "Craigslist was founded by Craig Newmark."
S2 = "Craig Newmark was born on December 6, 1952."
READER = "Craigslist was founded by Craig Newmark, born December 6, 1952. So the answer is: December 6, 1952."


async def test_ircot_alternates_a_reasoning_sentence_and_a_search_until_the_answer(monkeypatch):
    backend = install_backend(
        monkeypatch,
        {
            Q_CRAIG: [hit("p1", "craigslist is a classifieds site founded by Craig Newmark.")],
            S1: [hit("p2", "Craig Newmark (born December 6, 1952) is an American.")],
            S2: [hit("p3", "December 6 is the 340th day of the year.")],
        },
    )
    llm = ScriptedLLM(
        [
            S1 + " He later moved to San Francisco.",  # only the first sentence is a step
            S2,
            "So the answer is: December 6, 1952.",  # the chain states the answer: stop
            READER,  # the separate reader, over every collected paragraph
        ]
    )
    result = await _loop("ircot").run(pipeline_input(Q_CRAIG, llm))

    # The question first, then each step's sentence as the next query.
    assert [c["query"] for c in backend.calls] == [Q_CRAIG, S1, S2]
    # Each reason step sees every paragraph so far and the chain so far.
    assert "Wikipedia Title: p2.txt\nCraig Newmark (born December 6, 1952)" in llm.prompts[1]
    assert llm.prompts[2].endswith(f"Q: {Q_CRAIG}\nA: {S1} {S2}")
    # The reader starts a fresh chain over all three paragraphs.
    assert all(t in llm.prompts[3] for t in ("craigslist is", "Craig Newmark (born", "340th"))
    assert llm.prompts[3].endswith(f"Q: {Q_CRAIG}\nA: ")
    assert result.agent_summary == "December 6, 1952"
    assert {c["chunk_id"] for c in result.chunks} == {"p1", "p2", "p3"}
    assert result.metrics.stop_reason == "answer"


async def test_ircot_stops_at_its_step_limit_and_still_reads_an_answer(monkeypatch):
    backend = install_backend(monkeypatch, {Q_CRAIG: [hit("p1", "one")], S1: [hit("p2", "two")]})
    llm = ScriptedLLM([S1, S2, READER])
    config = ArmConfig(ircot_max_steps=2)
    result = await _loop("ircot").run(pipeline_input(Q_CRAIG, llm, config))

    assert len(llm.prompts) == 3  # two steps, then the reader
    assert [c["query"] for c in backend.calls] == [Q_CRAIG, S1, S2]
    assert result.agent_summary == "December 6, 1952"
    assert result.metrics.stop_reason == "max_steps"


async def test_ircot_caps_the_paragraphs_it_reasons_over(monkeypatch):
    install_backend(
        monkeypatch,
        {
            Q_CRAIG: [hit("p1", "one"), hit("p2", "two")],
            S1: [hit("p3", "three"), hit("p4", "four")],
        },
    )
    llm = ScriptedLLM([S1, "So the answer is: x.", "So the answer is: x."])
    config = ArmConfig(ircot_paragraphs_per_step=2, ircot_max_paragraphs=3)
    result = await _loop("ircot").run(pipeline_input(Q_CRAIG, llm, config))
    assert "three" in llm.prompts[1] and "four" not in llm.prompts[1]
    assert [c["chunk_id"] for c in result.chunks] == ["p1", "p3", "p2"]  # the paragraphs read


LOOP_SETTINGS["ircot"] = ["ircot_paragraphs_per_step", "ircot_max_steps", "ircot_max_paragraphs"]
SHORT_RUNS["ircot"] = ["So the answer is: Ada.", "So the answer is: Ada."]


# ---------------------------------------------------------------------------
# FLARE
# ---------------------------------------------------------------------------

UNSURE = "Joe Biden attended Penn Law School."
SURE = "Joe Biden attended Syracuse University College of Law."
DONE = "So the answer is Syracuse University College of Law."


async def test_flare_keeps_a_confident_sentence_without_searching_again(monkeypatch):
    backend = install_backend(monkeypatch, {Q_BIDEN: [hit("d1", "Biden studied law at Syracuse.")]})
    llm = ScriptedLLM([with_logprobs(SURE, [0.9] * 8), with_logprobs(DONE, [0.9] * 9)])
    result = await _loop("flare").run(pipeline_input(Q_BIDEN, llm))

    assert [c["query"] for c in backend.calls] == [Q_BIDEN]  # the question only
    assert result.agent_summary == "Syracuse University College of Law"
    assert result.metrics.loop_trace["confidence_source"] == "logprobs"
    assert result.metrics.stop_reason == "answer"
    # The paper's layout: exemplars, the search results, then the test case.
    assert "Search results:\n[1] Biden studied law at Syracuse.\nQuestion: " in llm.prompts[0]
    assert llm.prompts[1].endswith(f"Answer: {SURE}")
    assert llm.bound and llm.bound[0] == {"logprobs": True}


async def test_flare_searches_with_the_masked_sentence_when_a_token_is_unsure(monkeypatch):
    masked = "Joe Biden attended School."
    backend = install_backend(
        monkeypatch,
        {Q_BIDEN: [hit("d1", "Biden bio.")], masked: [hit("d2", "Biden: Syracuse Law, 1968.")]},
    )
    llm = ScriptedLLM(
        [
            # "Penn" and "Law" are below theta (0.8) and beta (0.4): retrieve, masking them.
            with_logprobs(UNSURE, [0.9, 0.9, 0.9, 0.1, 0.3, 0.9]),
            SURE,  # written again with what the masked query found
            with_logprobs(DONE, [0.9] * 9),
        ]
    )
    result = await _loop("flare").run(pipeline_input(Q_BIDEN, llm))

    assert [c["query"] for c in backend.calls] == [Q_BIDEN, masked]
    # The new documents replace the old ones.
    assert "Syracuse Law, 1968" in llm.prompts[1] and "Biden bio." not in llm.prompts[1]
    assert result.metrics.loop_trace["generation"] == f"{SURE} {DONE}"
    assert result.metrics.loop_trace["retrievals_triggered"] == 1


async def test_flare_without_logprobs_asks_the_model_which_spans_it_is_unsure_of(monkeypatch):
    masked = "Joe Biden attended Law School."
    backend = install_backend(
        monkeypatch,
        {Q_BIDEN: [hit("d1", "Biden bio.")], masked: [hit("d2", "Biden: Syracuse Law, 1968.")]},
    )
    llm = ScriptedLLM([UNSURE, "Penn", SURE, DONE, "NONE"])  # no logprobs on any reply
    result = await _loop("flare").run(pipeline_input(Q_BIDEN, llm))

    assert "not confident" in llm.prompts[1]
    assert [c["query"] for c in backend.calls] == [Q_BIDEN, masked]
    assert result.agent_summary == "Syracuse University College of Law"
    assert result.metrics.loop_trace["confidence_source"] == "self_reported"


async def test_flare_falls_back_when_the_model_refuses_the_logprobs_request(monkeypatch):
    install_backend(monkeypatch, {Q_BIDEN: [hit("d1", "Biden bio.")]})

    class Refuses(ScriptedLLM):
        def bind(self, **kwargs):
            outer = self

            class Bound:
                async def ainvoke(self, messages, stop=None, **kw):
                    raise ValueError("logprobs is not supported")

            outer.bound.append(kwargs)
            return Bound()

    llm = Refuses([DONE, "NONE"])
    result = await _loop("flare").run(pipeline_input(Q_BIDEN, llm))
    assert result.metrics.loop_trace["confidence_source"] == "self_reported"
    assert result.metrics.loop_trace["logprobs_refused"] == "ValueError"
    assert len(llm.bound) == 1  # asked once, not on every sentence


async def test_flare_stops_at_its_sentence_limit(monkeypatch):
    install_backend(monkeypatch, {Q_BIDEN: [hit("d1", "Biden bio.")]})
    llm = ScriptedLLM(
        [with_logprobs(SURE, [0.9] * 8), with_logprobs("He graduated in 1968.", [0.9] * 4)]
    )
    result = await _loop("flare").run(
        pipeline_input(Q_BIDEN, llm, ArmConfig(flare_max_sentences=2))
    )
    assert result.metrics.stop_reason == "max_sentences" and len(llm.prompts) == 2
    assert result.agent_summary == f"{SURE} He graduated in 1968."  # no stated answer: the text


LOOP_SETTINGS["flare"] = [
    "flare_retrieval_threshold",
    "flare_mask_threshold",
    "flare_max_sentences",
    "flare_docs_per_retrieval",
    "flare_confidence_source",
]
SHORT_RUNS["flare"] = ["So the answer is Ada.", "NONE"]


# ---------------------------------------------------------------------------
# Adaptive-RAG
# ---------------------------------------------------------------------------


async def test_adaptive_answers_a_simple_question_without_retrieval(monkeypatch):
    backend = install_backend(monkeypatch, {})
    llm = ScriptedLLM(["A", "Paris is the capital. So the answer is: Paris."])
    result = await _loop("adaptive").run(pipeline_input("What is the capital of France?", llm))
    assert backend.calls == [] and result.agent_summary == "Paris" and result.chunks == []
    assert "Wikipedia Title" not in llm.prompts[1]
    assert result.metrics.loop_trace["route"] == "no_retrieval"


async def test_adaptive_retrieves_once_for_a_moderate_question(monkeypatch):
    backend = install_backend(monkeypatch, {Q_BIDEN: [hit("d1", "Biden studied law at Syracuse.")]})
    llm = ScriptedLLM(["B", "He studied at Syracuse. So the answer is: Syracuse University."])
    result = await _loop("adaptive").run(pipeline_input(Q_BIDEN, llm))
    assert [c["query"] for c in backend.calls] == [Q_BIDEN]
    assert "Biden studied law at Syracuse." in llm.prompts[1]
    assert result.agent_summary == "Syracuse University"
    assert [c["chunk_id"] for c in result.chunks] == ["d1"]


async def test_adaptive_runs_ircot_for_a_complex_question(monkeypatch):
    backend = install_backend(monkeypatch, {Q_CRAIG: [hit("p1", "one")], S1: [hit("p2", "two")]})
    llm = ScriptedLLM(["C", S1, "So the answer is: December 6, 1952.", READER])
    result = await _loop("adaptive").run(pipeline_input(Q_CRAIG, llm))
    assert [c["query"] for c in backend.calls] == [Q_CRAIG, S1]
    assert result.agent_summary == "December 6, 1952"
    assert result.metrics.loop_trace["route"] == "multi_step"


async def test_adaptive_refuses_a_classifier_reply_that_names_no_class(monkeypatch):
    install_backend(monkeypatch, {})
    with pytest.raises(ValueError, match="named no class"):
        await _loop("adaptive").run(pipeline_input("q", ScriptedLLM(["It depends."])))


@pytest.mark.parametrize(
    "reply,label",
    [
        ("B", "B"),
        ("C, it needs several hops.", "C"),
        ("B (moderate). A single search.", "B"),
        ("Class (A, B or C): A", "A"),
        ("'C'", "C"),
    ],
)
def test_adaptive_reads_the_class_its_classifier_names(reply, label):
    from lasagna.agentic.loops.adaptive import complexity_label

    assert complexity_label(reply) == label


LOOP_SETTINGS["adaptive"] = ["adaptive_rag_classifier_model", "adaptive_rag_single_step_docs"]
SHORT_RUNS["adaptive"] = ["B", "So the answer is: Ada."]


# ---------------------------------------------------------------------------
# Self-RAG
# ---------------------------------------------------------------------------


async def test_self_rag_answers_alone_when_it_judges_retrieval_unneeded(monkeypatch):
    backend = install_backend(monkeypatch, {})
    llm = ScriptedLLM(["[No] It is common knowledge.", "Paris"])
    result = await _loop("self_rag").run(pipeline_input("What is the capital of France?", llm))
    assert backend.calls == [] and result.agent_summary == "Paris" and result.chunks == []
    assert llm.prompts[0].endswith("Instruction:What is the capital of France?\nNeed retrieval?: ")


async def test_self_rag_keeps_the_answer_its_critiques_score_highest(monkeypatch):
    backend = install_backend(
        monkeypatch,
        {Q_BIDEN: [hit("d1", "Biden was a senator."), hit("d2", "Biden: Syracuse Law, 1968.")]},
    )
    llm = ScriptedLLM(
        [
            "[Yes]",
            # passage d1: irrelevant, unsupported, low utility
            "[Irrelevant]",
            "He went to Delaware.",
            "[No support / Contradictory]",
            "Perceived utility: 2",
            # passage d2: relevant, fully supported, high utility
            "[Relevant]",
            "[Relevant] Syracuse University College of Law. [Utility:5]",
            "[Fully supported]",
            "Perceived utility: 5",
        ]
    )
    result = await _loop("self_rag").run(pipeline_input(Q_BIDEN, llm))

    assert [c["query"] for c in backend.calls] == [Q_BIDEN]
    assert result.agent_summary == "Syracuse University College of Law."
    # The answer comes back with the passage it was written from.
    assert [c["chunk_id"] for c in result.chunks] == ["d2"]
    assert "[Retrieval]<paragraph>d2.txt\nBiden: Syracuse Law, 1968.</paragraph>" in llm.prompts[6]
    assert [c["issup"] for c in result.metrics.loop_trace["critiques"]] == [0.0, 1.0]


async def test_self_rag_always_mode_skips_the_retrieval_judgement(monkeypatch):
    backend = install_backend(monkeypatch, {Q_BIDEN: [hit("d1", "Biden: Syracuse Law.")]})
    llm = ScriptedLLM(["[Relevant]", "Syracuse.", "[Fully supported]", "4"])
    config = ArmConfig(self_rag_retrieval="always")
    result = await _loop("self_rag").run(pipeline_input(Q_BIDEN, llm, config))
    assert len(backend.calls) == 1 and result.agent_summary == "Syracuse."


LOOP_SETTINGS["self_rag"] = [
    "self_rag_passages",
    "self_rag_weight_rel",
    "self_rag_weight_sup",
    "self_rag_weight_use",
    "self_rag_retrieval",
]
SHORT_RUNS["self_rag"] = ["[Yes]", "[Relevant]", "Ada.", "[Fully supported]", "5"]


# ---------------------------------------------------------------------------
# Search-o1
# ---------------------------------------------------------------------------

BQ, EQ = "<|begin_search_query|>", "<|end_search_query|>"
BR, ER = "<|begin_search_result|>", "<|end_search_result|>"


async def test_search_o1_searches_each_query_and_inserts_what_was_read(monkeypatch):
    backend = install_backend(
        monkeypatch,
        {
            "founder of craigslist": [hit("c1", "Craigslist was founded by Craig Newmark.")],
            "Craig Newmark birth date": [hit("c2", "Craig Newmark, born December 6, 1952.")],
        },
    )
    llm = ScriptedLLM(
        [
            # A provider that ignores ``stop`` runs on past the query: cut there.
            f"I need the founder first.\n{BQ}founder of craigslist{EQ}\n\nThe founder is",
            "The page names him.\n\n**Final Information**\n\nCraig Newmark founded craigslist.",
            f"Now his birth date.\n{BQ}Craig Newmark birth date{EQ}",
            "**Final Information**\n\nCraig Newmark was born on December 6, 1952.",
            "So the answer is \\boxed{December 6, 1952}.",
        ]
    )
    result = await _loop("search_o1").run(pipeline_input(Q_CRAIG, llm))

    assert [c["query"] for c in backend.calls] == [
        "founder of craigslist",
        "Craig Newmark birth date",
    ]
    assert llm.stops[0] == [EQ]
    # The authors' instruction and question prompt, with the search limit.
    assert llm.prompts[0].startswith("You are a reasoning assistant with the ability")
    assert "The maximum number of search attempts is limited to 10." in llm.prompts[0]
    assert llm.prompts[0].endswith(f"\\boxed{{YOUR_ANSWER}}.\n\nQuestion:\n{Q_CRAIG}\n\n")
    # Reason-in-Documents reads the reasoning so far, the query and the passages.
    reader = llm.prompts[1]
    assert reader.startswith("**Task Instruction:**")
    assert "Step 1: I need the founder first." in reader
    assert "- **Current Search Query:**  \nfounder of craigslist\n" in reader
    assert '**Web Page 1:**\n{\n  "id": 1,\n  "title": "c1.txt",' in reader
    assert '"context": "Craigslist was founded by Craig Newmark."' in reader
    # Its account goes back into the reasoning, and generation resumes after it.
    assert llm.prompts[2].endswith(
        f"{BQ}founder of craigslist{EQ}\n\n{BR}Craig Newmark founded craigslist.{ER}\n\n"
    )
    assert "The founder is" not in llm.prompts[2]
    assert result.agent_summary == "December 6, 1952"
    assert [c["content"] for c in result.chunks] == [
        "Craigslist was founded by Craig Newmark.",
        "Craig Newmark, born December 6, 1952.",
    ]
    assert result.metrics.stop_reason == "answer"
    assert result.metrics.loop_trace["queries"] == [
        "founder of craigslist",
        "Craig Newmark birth date",
    ]


async def test_search_o1_answers_without_searching_when_no_query_is_written(monkeypatch):
    backend = install_backend(monkeypatch, {})
    llm = ScriptedLLM(["Paris is the capital. \\boxed{\\text{Paris}}"])
    result = await _loop("search_o1").run(pipeline_input("What is the capital of France?", llm))
    assert backend.calls == [] and result.agent_summary == "Paris" and result.chunks == []


async def test_search_o1_refuses_a_search_past_its_limit(monkeypatch):
    from lasagna.agentic.loops.search_o1 import LIMIT_MESSAGE

    backend = install_backend(monkeypatch, {"A": [hit("a", "alpha")]})
    llm = ScriptedLLM(
        [
            f"{BQ}A{EQ}",
            "**Final Information**\n\nalpha",
            f"{BQ}B{EQ}",  # over the limit of one: never searched
            "\\boxed{alpha}",
        ]
    )
    config = ArmConfig(search_o1_max_searches=1)
    result = await _loop("search_o1").run(pipeline_input("A and B?", llm, config))

    assert [c["query"] for c in backend.calls] == ["A"]
    assert "limited to 1." in llm.prompts[0]
    assert llm.prompts[3].endswith(f"{BQ}B{EQ}" + LIMIT_MESSAGE)
    assert result.agent_summary == "alpha"
    assert result.metrics.loop_trace["refused_searches"] == 1


async def test_search_o1_does_not_search_a_query_twice(monkeypatch):
    from lasagna.agentic.loops.search_o1 import REPEAT_MESSAGE

    backend = install_backend(monkeypatch, {"A": [hit("a", "alpha")]})
    llm = ScriptedLLM(
        [f"{BQ}A{EQ}", "**Final Information**\n\nalpha", f"{BQ}A{EQ}", "\\boxed{alpha}"]
    )
    result = await _loop("search_o1").run(pipeline_input("A?", llm))
    assert [c["query"] for c in backend.calls] == ["A"]
    assert llm.prompts[3].endswith(REPEAT_MESSAGE)
    assert result.agent_summary == "alpha"


async def test_search_o1_stops_at_its_turn_limit_with_no_answer(monkeypatch):
    backend = install_backend(monkeypatch, {}, default=[hit("a", "alpha")])
    llm = ScriptedLLM(
        [f"{BQ}A{EQ}", "nothing useful", f"{BQ}B{EQ}", "**Final Information**\n\nbeta"]
    )
    config = ArmConfig(search_o1_max_turns=2)
    result = await _loop("search_o1").run(pipeline_input("A and B?", llm, config))

    assert [c["query"] for c in backend.calls] == ["A", "B"]
    # A reader reply without the marker counts as finding nothing, as in the authors' code.
    assert llm.prompts[2].endswith(f"{BR}No helpful information found.{ER}\n\n")
    assert result.agent_summary == ""
    assert result.metrics.stop_reason == "max_turns"


async def test_search_o1_says_when_it_ends_without_a_boxed_answer(monkeypatch):
    install_backend(monkeypatch, {})
    llm = ScriptedLLM(["The answer is Paris."])
    result = await _loop("search_o1").run(pipeline_input("Capital of France?", llm))
    assert result.agent_summary == "" and result.metrics.stop_reason == "no_boxed_answer"


def test_search_o1_shortens_long_reasoning_as_the_authors_do():
    from lasagna.agentic.loops.search_o1 import _previous_reasoning

    assert _previous_reasoning("a\n\nb") == "Step 1: a\n\nStep 2: b"
    output = f"a\nb\n{BQ}q{EQ}\nd\ne\nf\ng\nh"
    assert _previous_reasoning(output) == (
        f"Step 1: a\n\n...\n\nStep 3: {BQ}q{EQ}\n\n...\n\nStep 6: f\n\nStep 7: g\n\nStep 8: h"
    )


async def test_search_o1_reads_a_query_written_in_thinking_the_provider_returns_apart(
    monkeypatch,
):
    from langchain_core.messages import AIMessage

    backend = install_backend(monkeypatch, {"Jaws director": [hit("j", "Spielberg.")]})
    llm = ScriptedLLM(
        [
            # Ollama with reasoning=True: the thinking, cut at the stop, comes
            # back in its own field and the reply text is empty.
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": f"First the director.\n{BQ}Jaws director"},
            ),
            "**Final Information**\n\nSteven Spielberg.",
            "\\boxed{Steven Spielberg}",
        ]
    )
    result = await _loop("search_o1").run(pipeline_input("Who directed Jaws?", llm))
    assert [c["query"] for c in backend.calls] == ["Jaws director"]
    assert f"First the director.\n{BQ}Jaws director{EQ}" in llm.prompts[2]
    assert result.agent_summary == "Steven Spielberg"


def test_search_o1_asks_ollama_for_its_thinking():
    from lasagna.agentic.loops.search_o1 import _thinking_llm

    class ChatOllama:  # the name the loop recognises; the real class is pydantic
        def __init__(self, reasoning):
            self.reasoning = reasoning

        def model_copy(self, update):
            return ChatOllama(**update)

    assert _thinking_llm(ChatOllama(None)).reasoning is True
    assert _thinking_llm(ChatOllama(False)).reasoning is False  # an explicit choice is kept
    other = ScriptedLLM([])
    assert _thinking_llm(other) is other


LOOP_SETTINGS["search_o1"] = [
    "search_o1_max_searches",
    "search_o1_max_turns",
    "search_o1_docs_per_search",
]
SHORT_RUNS["search_o1"] = [
    f"{BQ}Who wrote the first program?{EQ}",
    "**Final Information**\n\nAda Lovelace wrote it.",
    "\\boxed{Ada Lovelace}",
]


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


@pytest.mark.parametrize("name", LOOPS)
async def test_each_loops_answer_and_passages_reach_the_harness(name, monkeypatch):
    # The orchestrator turns a pipeline's result into SearchResults; the
    # evaluation reads the contexts off them and the loop's own answer off the
    # first one's metadata, and judges that answer against those contexts.
    from lasagna.api._search_orchestrator import SearchOrchestrator

    install_backend(monkeypatch, {}, default=[hit("a", "Ada wrote the first program.")])
    inp = pipeline_input("Who wrote the first program?", ScriptedLLM(SHORT_RUNS[name]))
    result = await _loop(name).run(inp)

    results = SearchOrchestrator._convert_agentic_results(None, result, "arm-session")
    assert [r.chunk.content for r in results] == ["Ada wrote the first program."]
    assert results[0].chunk.metadata["_agentic_answer"] == result.agent_summary
    assert results[0].chunk.metadata["_routing_tier_used"] == "comprehensive"
    assert results[0].document.document_id == "doc-a" and results[0].document.filename == "a.txt"
