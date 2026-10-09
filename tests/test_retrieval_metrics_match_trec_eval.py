"""Retrieval metrics agree with trec_eval on hand-computed cases.

Every expected value below was worked out by hand and then checked against
``pytrec_eval`` (trec_eval 9.0.8) once, outside the suite; the reference is not
a dependency, so the arithmetic is written out instead.
"""

from __future__ import annotations

import math

import pytest

from lasagna.knowledge_evaluation_sdk.ir_metrics import (
    complete_recall_at_k,
    corpus_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    r_precision,
    recall_at_k,
)

GRADED = {"doc1": 2, "doc2": 1, "doc3": 1}
RANKED = ["doc4", "doc1", "doc3", "doc5", "doc2"]


def test_a_negative_grade_gains_nothing_in_ndcg():
    # {a: 2, b: -1, c: 1} ranked [b, a, c]. trec_eval: 0.6697.
    # DCG = 0 + 2/log2(3) + 1/log2(4); IDCG = 2/log2(2) + 1/log2(3).
    rel = {"a": 2, "b": -1, "c": 1}
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 + 1 / math.log2(3)
    assert ndcg_at_k(rel, ["b", "a", "c"], k=5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(rel, ["b", "a", "c"], k=5) == pytest.approx(0.6697, abs=1e-4)


def test_r_precision_is_precision_at_the_number_of_relevant_documents():
    # R = 3; the top 3 are doc4 (0), doc1 (2), doc3 (1): 2 of 3. trec_eval Rprec 0.6667.
    assert r_precision(GRADED, RANKED) == pytest.approx(2 / 3)


def test_r_precision_counts_ranks_past_the_list_as_misses():
    # 8 relevant, a 7-document list holding 5 of them: 5 / 8. trec_eval Rprec 0.625.
    rel = {f"r{i}": 1 for i in range(8)}
    ranked = ["r0", "x", "r1", "r2", "y", "r3", "r4"]
    assert r_precision(rel, ranked) == pytest.approx(5 / 8)


def test_r_precision_ignores_judged_non_relevant_documents_in_r():
    # A zero grade is a judgement, not a relevant document: R = 2, top 2 = [b, c].
    assert r_precision({"a": 1, "b": 0, "c": 1}, ["b", "c", "z", "a"]) == pytest.approx(0.5)


def test_r_precision_of_a_query_with_nothing_relevant_is_zero():
    assert r_precision({}, ["a"]) == 0.0
    assert r_precision({"a": 0}, ["a"]) == 0.0


def test_complete_recall_needs_every_supporting_document():
    # Two-hop question: both paragraphs are needed. Recall@3 is 0.5 and Hit@3 is
    # 1.0, but the question cannot be answered until rank 4.
    rel = {"p1": 1, "p2": 1}
    ranked = ["p1", "x", "y", "p2"]
    assert recall_at_k(rel, ranked, k=3) == 0.5
    assert hit_rate_at_k(rel, ranked, k=3) == 1.0
    assert complete_recall_at_k(rel, ranked, k=3) == 0.0
    assert complete_recall_at_k(rel, ranked, k=4) == 1.0


def test_complete_recall_with_one_relevant_document_is_hit_rate():
    for k in (1, 2, 3):
        assert complete_recall_at_k({"doc1": 1}, RANKED, k=k) == hit_rate_at_k(
            {"doc1": 1}, RANKED, k=k
        )


def test_complete_recall_is_unreachable_with_more_relevant_documents_than_k():
    rel = {f"r{i}": 1 for i in range(4)}
    assert complete_recall_at_k(rel, ["r0", "r1", "r2"], k=3) == 0.0


def test_complete_recall_counts_a_repeated_document_once():
    assert complete_recall_at_k({"p1": 1, "p2": 1}, ["p1", "p1", "p1"], k=3) == 0.0


def test_corpus_metrics_averages_the_new_metrics_over_queries():
    rels = [GRADED, {"p1": 1, "p2": 1}]
    lists = [RANKED, ["p1", "x", "p2"]]
    out = corpus_metrics(rels, lists, k=3)
    # Query 1: R-Prec 2/3, complete recall 0 (doc2 is at rank 5).
    # Query 2: R-Prec 1/2 (top 2 = p1, x), complete recall 1.
    assert out["r_precision@3"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert out["complete_recall@3"] == pytest.approx(0.5)


def test_corpus_r_precision_is_over_the_list_cut_at_k():
    # R = 3 but k = 2: only the top 2 are scored, so the value is 1/3, not 2/3.
    out = corpus_metrics([GRADED], [RANKED], k=2)
    assert out["r_precision@2"] == pytest.approx(1 / 3)
