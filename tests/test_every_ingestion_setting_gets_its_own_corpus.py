"""An arm that changes any ingestion setting is ingested on its own.

The list of ingestion-time settings was written by hand and went stale:
settings added with new chunkers (semantic_breakpoint_method, chunk_unit,
recursive_separators, ...) were missing, so an arm sweeping only one of them
reused the control's corpus and measured nothing. It is now derived from the
stage map: whatever belongs to chunking, embedding or document processing.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.experiment import _corpus_cache_key, _ingestion_time_keys
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from lasagna.knowledge_evaluation_sdk.stages import stage_of


def _task(**overrides):
    return BenchmarkTask(
        benchmark_name="smoke",
        config_name="a",
        config_overrides=overrides,
        run_index=0,
        profile="speed",
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("chunk_size", 400),
        ("semantic_breakpoint_method", "percentile"),
        ("chunk_unit", "tokens"),
        ("recursive_separators", ["\n\n", "\n"]),
        ("embedding_model", "BAAI/bge-small-en-v1.5"),
        # Cuts every chunk before it is embedded and indexed. Filed as
        # operational, it never reached the ingest: a whole-document arm
        # (chunk_size 100000) still had its documents cut at 4,000 characters.
        ("max_content_length", 100_000),
    ],
)
def test_a_changed_ingestion_setting_is_a_different_corpus(key, value):
    assert key in _ingestion_time_keys()
    assert _corpus_cache_key(_task(**{key: value})) != _corpus_cache_key(_task())


def test_every_chunking_and_embedding_setting_counts():
    from lasagna.core.config import rag_config

    for key in rag_config._config:
        if stage_of(key) in ("chunker", "embedder", "document_processor"):
            assert key in _ingestion_time_keys(), key


def test_query_time_settings_share_the_corpus():
    assert _corpus_cache_key(_task(hybrid_alpha=0.3)) == _corpus_cache_key(_task())
