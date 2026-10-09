"""Retrieval metrics agree with trec_eval on hand-computed cases.

Every expected value below was worked out by hand and then checked against
``pytrec_eval`` (trec_eval 9.0.8) once, outside the suite; the reference is not
a dependency, so the arithmetic is written out instead.
"""

from __future__ import annotations

import math

import pytest

from lasagna.knowledge_evaluation_sdk.ir_metrics import ndcg_at_k


def test_a_negative_grade_gains_nothing_in_ndcg():
    # {a: 2, b: -1, c: 1} ranked [b, a, c]. trec_eval: 0.6697.
    # DCG = 0 + 2/log2(3) + 1/log2(4); IDCG = 2/log2(2) + 1/log2(3).
    rel = {"a": 2, "b": -1, "c": 1}
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 + 1 / math.log2(3)
    assert ndcg_at_k(rel, ["b", "a", "c"], k=5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(rel, ["b", "a", "c"], k=5) == pytest.approx(0.6697, abs=1e-4)
