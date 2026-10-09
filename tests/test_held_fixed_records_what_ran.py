"""The comparability key must describe what ran, not what the profile declared.

A sweep on bge-small-en-v1.5 / 384 with hierarchical chunks was recorded as
all-MiniLM-L6-v2 / semantic (the base profile's declaration), and a run on a
different embedder received the identical key.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.bundle import comparability_key, held_fixed_from


def _row(system):
    return {"benchmark_name": "beir/scifact", "split": "test", "n_samples": 60, "system": system}


def test_the_task_record_wins_over_the_declaration():
    declared = {
        "system": {"embedder_model": "all-MiniLM-L6-v2", "embedder_dim": 384, "chunker": "semantic"}
    }
    ran = {
        "embedder_model": "BAAI/bge-small-en-v1.5",
        "embedder_dim": 384,
        "chunker": "hierarchical",
        "bm25_stemming": True,
    }
    held = held_fixed_from(_row(ran), declared)
    assert held["embedder_model"] == "BAAI/bge-small-en-v1.5"
    assert held["chunker"] == "hierarchical"
    assert held["bm25_stemming"] is True


def test_different_embedders_get_different_keys():
    declared = {"system": {"embedder_model": "all-MiniLM-L6-v2", "embedder_dim": 384}}
    a = held_fixed_from(
        _row({"embedder_model": "BAAI/bge-small-en-v1.5", "embedder_dim": 384}), declared
    )
    b = held_fixed_from(
        _row({"embedder_model": "intfloat/e5-small-v2", "embedder_dim": 384}), declared
    )
    assert comparability_key(a) != comparability_key(b)


def test_stemming_changes_the_key():
    declared = {"system": {}}
    a = held_fixed_from(_row({"bm25_stemming": True}), declared)
    b = held_fixed_from(_row({"bm25_stemming": False}), declared)
    assert comparability_key(a) != comparability_key(b)


def test_structural_settings_are_recorded_as_the_chunker_that_ran():
    # The speed profile asks for "semantic", but preserve_headers sends every
    # document through structure-aware chunking; the record said "semantic".
    from lasagna.knowledge_evaluation_sdk.experiment import _ran_system

    ingestion = {
        "chunking_strategy": "semantic",
        "enable_hierarchical": False,
        "preserve_headers": True,
        "preserve_tables": False,
    }
    assert _ran_system("m", 384, ingestion, {})["chunker"] == "structure_aware"

    plain = dict(ingestion, preserve_headers=False)
    assert _ran_system("m", 384, plain, {})["chunker"] == "semantic"


def test_a_custom_reranker_is_recorded_instead_of_the_cross_encoder():
    from lasagna.knowledge_evaluation_sdk.experiment import _ran_system

    cfg = {"enable_reranking": True, "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    assert _ran_system("m", 384, {}, cfg, "template_length")["reranker_model"] == (
        "template_length"
    )
    assert _ran_system("m", 384, {}, cfg, "cross_encoder_adapted")["reranker_model"] == (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
