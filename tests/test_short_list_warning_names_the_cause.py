"""A short result list is explained by its cause, not always by chunk merging.

The warning advised raising retrieval_chunk_overfetch for every query that
scored fewer than k documents. BM25 returns only documents that share a term
with the query, so 75 of NFCorpus's 323 test queries (one rare word, such as
"amnesia") match fewer than ten documents in the whole corpus. Over-fetching
cannot change that, and the advice made a correct BEIR number look deflated.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.experiment import _short_list_warning


def test_a_retriever_that_ran_out_is_not_told_to_over_fetch():
    msg = _short_list_warning(short=75, exhausted=75, n_queries=323, k=10, overfetch=10)
    assert msg.startswith("75 of 323 queries had fewer than 10 distinct documents")
    assert "ran out of candidates" in msg
    assert "retrieval_chunk_overfetch" not in msg


def test_merged_chunks_still_get_the_over_fetch_advice():
    msg = _short_list_warning(short=5, exhausted=0, n_queries=50, k=10, overfetch=4)
    assert "raise retrieval_chunk_overfetch (now 4)" in msg
    assert "ran out of candidates" not in msg


def test_both_causes_are_counted_apart():
    msg = _short_list_warning(short=7, exhausted=4, n_queries=50, k=10, overfetch=4)
    assert "For 4 of them the retriever returned fewer results" in msg
    assert "For 3 of them the full request came back" in msg
