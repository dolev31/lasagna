"""A correct answer that also hedges must be scored, not zeroed as an abstention.

`is_abstention` searched the whole answer, so any answer *containing* "there is
no mention" or "the context does not" counted as a refusal:
`token_f1("Paris. There is no mention of the year.", "Paris")` was 0.0, and so
was its exact match. LLM answers hedge like this routinely, and generators
hedge at different rates, so token-F1 was biased down by a different amount for
each system being compared.

Normalization also ran in a different order from the official SQuAD script
(articles were removed before punctuation), so `the-end` normalized to `end`
where the official script gives `theend`, while the docstring said
"SQuAD-style".
"""

from __future__ import annotations

import re
import string
from collections import Counter

import pytest

from lasagna.knowledge_evaluation_sdk.answer_metrics import (
    abstention_rate,
    exact_match,
    is_abstention,
    normalize_answer,
    token_f1,
)


def _official_normalize(s):
    """`normalize_answer` from the official SQuAD v2.0 evaluation script."""

    def remove_articles(text):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _official_f1(prediction, gold):
    pred, ref = _official_normalize(prediction).split(), _official_normalize(gold).split()
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if not pred or not ref:
        return float(pred == ref)
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


class TestHedgedAnswersAreAnswers:
    @pytest.mark.parametrize(
        "answer",
        [
            "Paris. There is no mention of the year.",
            "Paris, although the context does not give the year",
            "The capital is Paris; the provided context does not say when it was founded.",
            "The context does not state the year, but it says the treaty was signed in Paris.",
            "Paris. I don't know the exact year.",
        ],
    )
    def test_a_hedged_answer_is_not_an_abstention(self, answer):
        assert not is_abstention(answer)
        assert token_f1(answer, "Paris") == pytest.approx(_official_f1(answer, "Paris"))
        assert token_f1(answer, "Paris") > 0

    def test_the_example_from_the_review_scores_like_the_official_script(self):
        answer = "Paris. There is no mention of the year."
        assert token_f1(answer, "Paris") == pytest.approx(0.25)


class TestRealAbstentionsAreStillAbstentions:
    @pytest.mark.parametrize(
        "answer",
        [
            "",
            "I don't know.",
            "I dont know",
            "I don\u2019t know.",
            "The provided context does not contain the answer.",
            "Unable to determine from the given passages",
            "There is no information about this in the context.",
            "I'm sorry, but I cannot answer this question based on the context.",
            "Unfortunately, the context does not mention who won.",
            "Based on the provided context, there is no mention of the winner.",
            "The question cannot be answered from the context.",
            "Not enough information to answer.",
            "No relevant information was found.",
            "The answer cannot be determined from the passages. They discuss other topics.",
        ],
    )
    def test_an_abstention_is_detected(self, answer):
        assert is_abstention(answer)

    def test_an_abstention_on_an_unanswerable_question_is_correct(self):
        assert exact_match("I'm sorry, but the context does not say.", "") == 1.0
        assert token_f1("I'm sorry, but the context does not say.", "") == 1.0

    def test_an_abstention_on_an_answerable_question_scores_zero(self):
        assert exact_match("I don't know.", "Paris") == 0.0
        assert token_f1("I don't know.", "Paris") == 0.0


class TestTheAbstentionRateIsReported:
    def test_it_is_the_share_of_answers_that_abstain(self):
        answers = ["Paris", "I don't know.", "Paris. There is no mention of the year.", ""]
        assert abstention_rate(answers) == pytest.approx(0.5)

    def test_no_answers_have_no_rate(self):
        assert abstention_rate([]) is None


class TestNormalizationMatchesTheOfficialScript:
    @pytest.mark.parametrize(
        "text",
        [
            "the-end",
            "Winnie-the-Pooh",
            "An apple a day",
            "  The Eiffel   Tower! ",
            "a.k.a. the 'Big Apple'",
            "U.S.A. (the)",
        ],
    )
    def test_same_output_as_the_official_normalizer(self, text):
        assert normalize_answer(text) == _official_normalize(text)

    def test_hyphenated_articles_are_kept_as_one_token(self):
        assert normalize_answer("the-end") == "theend"
        assert exact_match("theend", "the-end") == 1.0
