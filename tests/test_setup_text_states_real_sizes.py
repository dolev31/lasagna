"""What `lasagna init` and `lasagna doctor` say is computed from the profiles.

`init` said a local install "needs ~2GB disk" whatever profile was chosen
(bge-m3 with its reranker is 3.2 GB, speed is 92 MB); `doctor` did not say which profile, session or settings file
was in force. The text now comes from `lasagna.core.setup_text`.
"""

from __future__ import annotations

import os

from lasagna.core.setup_text import (
    MODEL_DOWNLOAD_BYTES,
    doctor_context,
    profile_download_text,
    profile_models,
)


def test_speed_downloads_one_small_model():
    assert profile_download_text("speed") == "downloads 92 MB: all-MiniLM-L6-v2 (92 MB)"


def test_standard_names_its_embedder_and_reranker_with_their_sizes():
    text = profile_download_text("standard")
    assert text.startswith("downloads 3.2 GB:")
    assert "BAAI/bge-m3 (2.3 GB)" in text and "mixedbread-ai/mxbai-rerank-large-v1 (881 MB)" in text


def test_the_settings_file_downloads_what_its_embedder_and_reranker_are():
    """No profile (the settings file) is what a fresh session runs, and the gate measures."""
    from lasagna.core.config import rag_config

    assert profile_models(None) == [rag_config.embedding_model, rag_config.reranker_model]
    assert profile_download_text(None) == profile_download_text("standard")


def test_the_fallback_embedder_is_not_counted():
    """It is fetched only when the embedder cannot load."""
    from lasagna.core.config import rag_config

    assert rag_config.embedding_model_fallback not in profile_models(None)


def test_every_model_a_built_in_profile_downloads_has_a_recorded_size():
    from lasagna.core.session_config._profiles import ProfileMixin

    missing = {
        model
        for profile in ProfileMixin._PROFILE_SPECS
        for model in profile_models(profile)
        if model not in MODEL_DOWNLOAD_BYTES
    }
    assert not missing, f"record the size of {sorted(missing)} in setup_text.MODEL_DOWNLOAD_BYTES"


def test_doctor_says_which_session_profile_and_files_are_in_force():
    rows = dict(doctor_context("default"))
    assert set(rows) == {"session", "profile", "settings file", "data directory"}
    assert rows["session"] == "default" and rows["profile"]
    assert rows["settings file"].endswith(".toml")
    assert os.path.expanduser("~") not in " ".join(rows.values())
