"""Answer metrics that need no judge: ROUGE-L, answer containment, semantic
similarity, and ALCE citation quality with a stub entailment judge.

ROUGE-L values were checked once against ``rouge_score`` 0.1.2
(``RougeScorer(["rougeL"])``); it is not a dependency, so the arithmetic is
written out.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.answer_metrics import (
    answer_similarity,
    citation_precision_recall,
    contains_answer,
    has_citations,
    rouge_l,
)


class TestRougeL:
    def test_the_longest_common_subsequence_sets_precision_and_recall(self):
        # LCS(the cat lay on a mat, the cat sat on the mat) = the cat on mat = 4.
        # P = R = 4/6, F = 2/3. rouge_score: 0.6667.
        assert rouge_l("the cat lay on a mat", "the cat sat on the mat") == pytest.approx(2 / 3)

    def test_unequal_lengths(self):
        # pred tokens: curie won 2 nobel prizes in 1903 and 1911 (9);
        # ref tokens: marie curie won two nobel prizes (6); LCS = 4.
        # P = 4/9, R = 4/6, F = 16/30. rouge_score: 0.5333.
        pred = "Curie won 2 Nobel Prizes, in 1903 and 1911."
        assert rouge_l(pred, "Marie Curie won two Nobel prizes") == pytest.approx(16 / 30)

    def test_word_order_matters_where_token_f1_does_not(self):
        from lasagna.knowledge_evaluation_sdk.answer_metrics import token_f1

        assert token_f1("dog bites man", "man bites dog") == 1.0
        # LCS is one token: P = R = 1/3.
        assert rouge_l("dog bites man", "man bites dog") == pytest.approx(1 / 3)

    def test_the_best_alias_counts(self):
        assert rouge_l("Tour Eiffel", "Eiffel Tower", aliases=["Tour Eiffel"]) == 1.0

    def test_no_answer_policy_matches_token_f1(self):
        assert rouge_l("I don't know", "") == 1.0
        assert rouge_l("I don't know", "Paris") == 0.0
        assert rouge_l("Paris", "") == 0.0


class TestContainsAnswer:
    def test_a_sentence_that_states_the_answer_counts(self):
        # Exact match is 0 here; the answer is right.
        assert contains_answer("The Eiffel Tower is in Paris.", "Paris") == 1.0

    def test_normalization_is_the_squad_one(self):
        assert contains_answer("It was built by the Romans!", "Romans") == 1.0
        assert contains_answer("It is the Eiffel Tower.", "Eiffel Tower") == 1.0

    def test_the_reference_must_be_whole_tokens(self):
        assert contains_answer("The answer is not obvious.", "no") == 0.0
        assert contains_answer("It opened in 1990.", "1") == 0.0

    def test_any_alias_counts(self):
        assert (
            contains_answer("They call it La Tour Eiffel.", "Eiffel Tower", ["Tour Eiffel"]) == 1.0
        )

    def test_an_abstention_does_not_contain_the_answer(self):
        assert contains_answer("I don't know.", "Paris") == 0.0
        assert contains_answer("I don't know.", "") == 1.0


def _encoder(table):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return [table[t] for t in texts]

    return encode, calls


class TestAnswerSimilarity:
    def test_cosine_against_the_closest_reference(self):
        encode, calls = _encoder(
            {"a": [1.0, 0.0], "ref1": [0.0, 1.0], "ref2": [1.0, 1.0], "b": [3.0, 4.0]}
        )
        out = answer_similarity(["a", "b"], [["ref1", "ref2"], ["ref1"]], encode)
        # a vs ref2: 1 / sqrt(2); b vs ref1: 4 / 5.
        assert out == pytest.approx([2**-0.5, 0.8])
        assert len(calls) == 1, "one batch for the run"

    def test_a_question_without_a_reference_has_no_score_not_zero(self):
        encode, _ = _encoder({"a": [1.0, 0.0]})
        assert answer_similarity(["a"], [[""]], encode) == [None]

    def test_an_empty_answer_shares_nothing(self):
        encode, calls = _encoder({"ref": [1.0, 0.0]})
        assert answer_similarity([""], [["ref"]], encode) == [0.0]
        assert calls == []


PASSAGES = [
    "Paris is the capital of France.",
    "France is in Europe.",
    "Bananas are yellow.",
]


def _entails(premise, hypothesis):
    """Stub NLI: the premise states the hypothesis word for word."""
    return hypothesis.rstrip(".") in premise


class TestCitationQuality:
    def test_alce_recall_and_precision_by_hand(self):
        # s1 cites [1][3]: jointly entailed; [1] alone entails, [3] does not and
        #    [1] still does without it, so [3] is irrelevant.
        # s2 cites [2]: entailed.
        # s3 cites nothing: unsupported.
        # recall = 2/3 sentences; precision = 2 of 3 citations.
        answer = "Paris is the capital of France [1][3]. France is in Europe [2]. It has a river."
        out = citation_precision_recall(answer, PASSAGES, _entails)
        assert out == pytest.approx({"citation_recall": 2 / 3, "citation_precision": 2 / 3})

    def test_a_citation_to_a_passage_never_shown_supports_nothing(self):
        out = citation_precision_recall("France is in Europe [7].", PASSAGES, _entails)
        assert out == {"citation_recall": 0.0, "citation_precision": 0.0}

    def test_a_marker_opening_a_sentence_belongs_to_the_one_before(self):
        answer = "France is in Europe. [2] Bananas are yellow [3]."
        out = citation_precision_recall(answer, PASSAGES, _entails)
        assert out == {"citation_recall": 1.0, "citation_precision": 1.0}

    def test_comma_lists_are_citations(self):
        out = citation_precision_recall(
            "Paris is the capital of France [1, 2].", PASSAGES, _entails
        )
        # Jointly entailed; [2] alone does not entail and [1] does without it.
        assert out == {"citation_recall": 1.0, "citation_precision": 0.5}

    def test_an_answer_without_citations_is_not_scored(self):
        assert not has_citations("Paris is the capital of France.")
        assert citation_precision_recall("Paris is the capital.", PASSAGES, _entails) is None

    def test_an_undecided_judgement_leaves_the_answer_unscored(self):
        out = citation_precision_recall("France is in Europe [2].", PASSAGES, lambda p, h: None)
        assert out is None
