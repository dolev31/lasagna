"""The shipped BEIR reproduction sweep still measures what its README claims.

examples/reproduction/README.md sets Lasagna's numbers against BEIR's and
MTEB's. Those comparisons hold only while the sweep scores every test query
against the whole corpus, one uncut chunk per document, and reranks the BM25
top 100. A later edit that broke any of these would still run and still print
plausible numbers, so the file is checked here rather than trusted.
"""

from __future__ import annotations

import pytest

from tests._paths import PACKAGE_DIR

SWEEP = PACKAGE_DIR.parent.parent / "examples" / "reproduction" / "beir_tier1.yaml"

#: The longest document in either corpus, in characters (SciFact).
LONGEST_DOCUMENT = 10_128


@pytest.fixture(scope="module")
def tasks():
    if not SWEEP.is_file():
        pytest.skip("needs a source checkout")
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    return Experiment.from_file(str(SWEEP)).generate_tasks()


def test_every_arm_scores_the_full_test_set(tasks):
    assert {t.benchmark_name for t in tasks} == {"beir/scifact", "beir/nfcorpus"}
    assert len(tasks) == 28
    for t in tasks:
        assert t.split == "test" and t.n_samples is None and t.max_corpus_docs == 0


def test_each_document_is_one_uncut_chunk(tasks):
    for t in tasks:
        o = t.config_overrides
        assert o["chunking_strategy"] == "fixed" and o["chunk_size"] > LONGEST_DOCUMENT
        # Each of these takes precedence over chunking_strategy when on.
        assert not (o["preserve_headers"] or o["preserve_tables"] or o["enable_hierarchical"])
        assert o["max_content_length"] > LONGEST_DOCUMENT


def test_each_dataset_is_ingested_once_per_embedder(tasks):
    from lasagna.knowledge_evaluation_sdk.experiment import _corpus_cache_key

    assert len({(t.benchmark_name, _corpus_cache_key(t)) for t in tasks}) == 4


def test_the_cross_encoder_reranks_the_bm25_top_100(tasks):
    rerankers = [t for t in tasks if t.config_overrides.get("reranker")]
    assert len(rerankers) == 4
    for t in rerankers:
        o = t.config_overrides
        assert o["retriever"] == "bm25"
        assert o["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert o["metric_k"] * o["retrieval_chunk_overfetch"] == 100
