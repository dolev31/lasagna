"""`lasagna doctor` checks that the models the session searches with are on disk.

It counted the folders in the Hugging Face cache and said "ok, 4 model(s)
cached" while the profile's own models (BGE-M3 and the mxbai reranker) were
not among them, so the first search still downloaded 3.1 GB.
"""

from __future__ import annotations

import pytest

from lasagna.core import diagnostics, model_downloads
from lasagna.core.session_config import get_knowledge_config_manager

#: What a cache holding only the small models answers.
CACHED = {
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
}


@pytest.fixture(autouse=True)
def small_models_only(monkeypatch):
    monkeypatch.setattr(model_downloads, "is_cached", lambda model: model in CACHED)


def test_four_cached_models_are_not_the_default_profiles():
    check = diagnostics._check_model_cache("default")
    assert check.status == "warning"
    assert "BAAI/bge-m3" in check.detail and "not downloaded" in check.detail
    assert "downloads" in check.advice and "--profile speed" in check.advice


def test_a_speed_session_is_ready():
    session = "doctor-speed-probe"
    get_knowledge_config_manager(session_id=session).set_active_profile("speed", session_id=session)
    check = diagnostics._check_model_cache(session)
    assert check.status == "ok", check
    assert "all-MiniLM-L6-v2" in check.detail and session in check.detail
