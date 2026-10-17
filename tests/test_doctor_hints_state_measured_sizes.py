"""``lasagna doctor`` states the measured download sizes and which profiles need a key.

With no Hugging Face cache it said the first search downloads "~5 GB on the
default profile" and that the speed profile is "an 80 MB model"; the
downloads measured in ``core/setup_text.py`` are 3.2 GB (bge-m3 and the mxbai
reranker) and 92 MB. With no API key it said "Retrieval works without one", but the
default configuration sends complex queries through agentic retrieval, which
calls an LLM; only the speed profile retrieves without a key. Both hints are
now computed from the profiles and the measured sizes.
"""

from __future__ import annotations

import pytest

from lasagna.core import diagnostics, setup_text
from lasagna.core.config import rag_config
from lasagna.core.session_config._profiles import ProfileMixin
from lasagna.core.setup_text import MODEL_DOWNLOAD_BYTES, human_size

STALE = ("~5 GB", "80 MB", "80MB", "4.5 GB", "3.1 GB", "91 MB")


def _size(model: str) -> str:
    return human_size(MODEL_DOWNLOAD_BYTES[model])


def test_the_cache_hint_states_the_measured_sizes(tmp_path, monkeypatch):
    # huggingface_hub reads HF_HOME once, at import: setting it here left the
    # check reading the real cache, so the test failed wherever mxbai was cached.
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty" / "hub"))

    check = diagnostics._check_model_cache()

    default = [rag_config.embedding_model, rag_config.reranker_model]
    assert all(_size(m) in check.advice and m in check.advice for m in default), check.advice
    total = human_size(sum(MODEL_DOWNLOAD_BYTES[m] for m in default))
    assert f"downloads {total}" in check.advice
    speed = ProfileMixin._PROFILE_SPECS["speed"]["embedding_model"]
    assert f"--profile speed`) downloads {_size(speed)}" in check.advice
    assert not any(s in check.advice for s in STALE), check.advice


def test_only_profiles_without_an_llm_step_are_called_keyless():
    keyless = setup_text.keyless_profiles()

    assert keyless == ["speed"]
    for name, spec in ProfileMixin._PROFILE_SPECS.items():
        has_llm_step = any(
            spec.get(k, getattr(rag_config, k, False)) for k in setup_text.LLM_RETRIEVAL_SETTINGS
        )
        assert (name in keyless) is not has_llm_step, name
    # The settings file's own configuration routes complex queries to agentic retrieval.
    assert rag_config.use_agentic_rag and setup_text.needs_llm_key(None)


def test_the_key_hint_names_which_profiles_need_one():
    check = diagnostics._check_llm_credentials({"llm_credentials_present": False})

    assert check.advice.startswith("Retrieval works without one only on the speed profile")
    assert "The default configuration and the standard, balanced and max_quality profiles" in (
        check.advice
    )
    assert "Retrieval works without one." not in check.advice


@pytest.mark.parametrize(
    "path",
    [
        "core/diagnostics.py",
        "core/setup_text.py",
        "mcp/server.py",
        "configurator/schema.py",
        "settings.lasagna.toml",
        "skills/lasagna-support/SKILL.md",
    ],
)
def test_no_stale_size_is_left_where_users_read_it(path):
    from tests._paths import PACKAGE_DIR

    text = (PACKAGE_DIR / path).read_text()
    assert not any(s in text for s in STALE), [s for s in STALE if s in text]
