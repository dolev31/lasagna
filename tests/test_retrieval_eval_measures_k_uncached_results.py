"""Retrieval evaluation must measure k fresh results per query.

Three answer-time policies changed what a sweep measured, silently:
the search cache keyed on a dozen fields (not the reranker model), so one arm
was served another arm's lists; the default similarity threshold, applied to
min-max-normalized scores, left about 4 documents per query; adaptive-k cut
reranked lists to about 7. NDCG@10 was then computed on far fewer than 10.
"""

from __future__ import annotations

from lasagna.core.config import RAGConfig
from lasagna.knowledge_evaluation_sdk.experiment import _fix_retrieval_depth


def test_the_cache_key_changes_with_any_setting():
    base = RAGConfig()
    a = base.with_query_overrides({"reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    b = base.with_query_overrides({"reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2"})
    c = base.with_query_overrides({"reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    assert a.settings_fingerprint() != b.settings_fingerprint()
    assert a.settings_fingerprint() == c.settings_fingerprint()


def test_retrieval_eval_turns_off_truncation_and_caching():
    task_config = {
        "enable_search_cache": True,
        "similarity_threshold": 0.3,
        "chunk_cite_retrieval": {"strategy": "adaptive", "max_chunks": 20},
        "default_top_k": 20,
    }
    _fix_retrieval_depth(task_config, depth=40)
    assert task_config["enable_search_cache"] is False
    assert task_config["similarity_threshold"] == 0.0
    assert task_config["chunk_cite_retrieval"] == {"strategy": "fixed", "max_chunks": 20}
    assert task_config["default_top_k"] == 40


def test_the_threshold_override_reaches_the_config():
    cfg = RAGConfig().with_query_overrides(
        {"similarity_threshold": 0.0, "enable_search_cache": False}
    )
    assert cfg.similarity_threshold == 0.0
    assert cfg.enable_search_cache is False


def test_repeated_document_ids_count_once():
    # Three chunks of one relevant document used to give NDCG 2.13, Recall 3.0.
    from lasagna.knowledge_evaluation_sdk.ir_metrics import map_score, ndcg_at_k, recall_at_k

    rel = {"d1": 1}
    assert ndcg_at_k(rel, ["d1", "d1", "d1"], 10) == 1.0
    assert recall_at_k(rel, ["d1", "d1", "d1"], 10) == 1.0
    assert map_score(rel, ["d1", "d1", "d1"], 10) == 1.0
