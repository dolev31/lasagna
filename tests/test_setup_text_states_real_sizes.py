"""What `lasagna init` and `lasagna doctor` say is computed from the profiles.

`init` said a local install "needs ~2GB disk" whatever profile was chosen
(bge-m3 with its reranker is 3.1 GB, speed is 91 MB) and saved an empty API key
without a word; `doctor` did not say which profile, session or settings file
was in force. The text now comes from `lasagna.core.setup_text`.
"""

from __future__ import annotations

import os

from lasagna.core.setup_text import (
    MODEL_WEIGHT_BYTES,
    api_key_problem,
    doctor_context,
    profile_download_text,
    profile_models,
)


def test_speed_downloads_one_small_model():
    assert profile_download_text("speed") == "downloads 91 MB: all-MiniLM-L6-v2 (91 MB)"


def test_standard_names_its_embedder_and_reranker_with_their_sizes():
    text = profile_download_text("standard")
    assert text.startswith("downloads 3.1 GB:")
    assert "BAAI/bge-m3 (2.3 GB)" in text and "mixedbread-ai/mxbai-rerank-large-v1 (870 MB)" in text


def test_every_model_a_built_in_profile_downloads_has_a_recorded_size():
    from lasagna.core.session_config._profiles import ProfileMixin

    missing = {
        model
        for profile in ProfileMixin._PROFILE_SPECS
        for model in profile_models(profile)
        if model not in MODEL_WEIGHT_BYTES
    }
    assert not missing, f"record the size of {sorted(missing)} in setup_text.MODEL_WEIGHT_BYTES"


def test_an_empty_or_broken_key_is_refused_with_what_to_do():
    assert "cannot be empty" in api_key_problem("OPENAI_API_KEY", "")
    assert "cannot be empty" in api_key_problem("OPENAI_API_KEY", "   ")
    assert "space or a line break" in api_key_problem("OPENAI_API_KEY", "sk-abc def")
    assert api_key_problem("OPENAI_API_KEY", "sk-abc") is None


def test_doctor_says_which_session_profile_and_files_are_in_force():
    rows = dict(doctor_context("default"))
    assert set(rows) == {"session", "profile", "settings file", "data directory"}
    assert rows["session"] == "default" and rows["profile"]
    assert rows["settings file"].endswith(".toml")
    assert os.path.expanduser("~") not in " ".join(rows.values())
