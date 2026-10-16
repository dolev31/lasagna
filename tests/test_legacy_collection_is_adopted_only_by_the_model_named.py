"""The pre-0.1.0 collection is adopted only by the model the user names as its maker.

Before 0.1.0 every model wrote to ``rag_documents_v2``, and nothing recorded
which one: no collection stamp, a generic collection description, no model on
the rows (the session table names a profile per session, and sessions at one
size shared and overwrote the collection). The first model to open it at its
size adopted it and stamped itself as its maker, so bge-large, opening bge-m3's
1024-dimension documents, searched them with the wrong vectors from then on.
"""

from __future__ import annotations

import pytest
from loguru import logger

import lasagna.api._collection_manager as cm
from tests.test_collection_identity import BGE_LARGE, BGE_M3, MINILM, _config
from tests.test_collection_identity import store  # noqa: F401 - the fixture `legacy` reads

LEGACY = "rag_documents_v2"


@pytest.fixture(autouse=True)
def no_confirmation(monkeypatch):
    monkeypatch.delenv("LASAGNA_LEGACY_COLLECTION_MODEL", raising=False)


@pytest.fixture
def legacy(request):
    """test_collection_identity's manager over an in-memory store."""
    return request.getfixturevalue("store")


def _open(legacy, settings):
    seen = []
    sink = logger.add(lambda m: seen.append(m.record), level="INFO")
    try:
        config = _config(**settings)
        legacy.manager.maybe_reload_embedding_manager(config)
    finally:
        logger.remove(sink)
    return config, [r for r in seen if r["level"].no >= 30]


def test_another_model_of_the_same_size_does_not_adopt_it_and_is_told_how(legacy):
    legacy.vectors.add(LEGACY, 1024, rows=50)  # written by bge-m3 under 0.1.0a1

    large, warnings = _open(legacy, BGE_LARGE)

    assert large.collection_name == "rag_documents_v2__baai_bge_large_en_v1_5__1024"
    assert legacy.stamps.get_collection_meta(LEGACY) is None
    assert cm.legacy_collections(legacy.vectors) == [
        {"name": LEGACY, "rows": 50, "embedding_dim": 1024}
    ]
    [warning] = warnings
    message = warning["message"]
    assert "did not record" in message
    assert "LASAGNA_LEGACY_COLLECTION_MODEL=BAAI/bge-large-en-v1.5" in message
    assert "50 chunks" in message


def test_the_model_named_as_its_maker_adopts_it(legacy, monkeypatch):
    legacy.vectors.add(LEGACY, 1024, rows=50)
    monkeypatch.setenv("LASAGNA_LEGACY_COLLECTION_MODEL", "BAAI/bge-m3")

    large, large_warnings = _open(legacy, BGE_LARGE)
    m3, m3_warnings = _open(legacy, BGE_M3)

    assert large.collection_name != LEGACY and not large_warnings
    assert m3.collection_name == LEGACY and not m3_warnings
    stamp = legacy.stamps.get_collection_meta(LEGACY)
    assert (stamp["embedding_model"], stamp["embedding_dim"]) == ("BAAI/bge-m3", 1024)


def test_once_stamped_the_setting_is_no_longer_needed(legacy, monkeypatch):
    legacy.vectors.add(LEGACY, 384, rows=50)
    monkeypatch.setenv("LASAGNA_LEGACY_COLLECTION_MODEL", "all-MiniLM-L6-v2")
    _open(legacy, MINILM)
    monkeypatch.delenv("LASAGNA_LEGACY_COLLECTION_MODEL")
    legacy.manager._legacy_decided = set()  # a new process decides again

    speed, warnings = _open(legacy, MINILM)
    assert speed.collection_name == LEGACY and not warnings


def test_a_legacy_collection_at_another_size_still_says_nothing(legacy):
    legacy.vectors.add(LEGACY, 384, rows=50)
    default, warnings = _open(legacy, BGE_M3)
    assert default.collection_name != LEGACY
    assert not warnings
