"""Every setting can be set as LASAGNA_<NAME>, with RAG_<NAME> still read.

The package is Lasagna, and its settings were only reachable through ``RAG_``
variables. ``LASAGNA_`` now names the same settings and wins when both are set,
so existing deployments keep working while new ones use the project's name.
Session tokens are issued as "lasagna" and tokens issued as "rag-knowledge"
before this release stay valid.
"""

from __future__ import annotations

import pytest

from lasagna.core import config as config_module
from lasagna.core.config import RAGConfig


def test_a_lasagna_variable_sets_the_setting(monkeypatch):
    monkeypatch.setenv("LASAGNA_CHUNK_SIZE", "500")
    assert RAGConfig().chunk_size == 500


def test_the_lasagna_name_wins_over_the_rag_name(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "400")
    monkeypatch.setenv("LASAGNA_CHUNK_SIZE", "500")
    assert RAGConfig().chunk_size == 500


def test_the_rag_name_alone_still_works(monkeypatch):
    monkeypatch.delenv("LASAGNA_CHUNK_SIZE", raising=False)
    monkeypatch.setenv("RAG_CHUNK_SIZE", "400")
    assert RAGConfig().chunk_size == 400


def test_a_bad_lasagna_value_is_refused_under_its_own_name(monkeypatch):
    monkeypatch.setenv("LASAGNA_HYBRID_ALPHA", "2")
    error = getattr(config_module, "ConfigError", ValueError)
    with pytest.raises(error, match="LASAGNA_HYBRID_ALPHA='2'"):
        RAGConfig()


def test_lasagna_variables_that_are_not_settings_are_left_alone(monkeypatch):
    from loguru import logger

    monkeypatch.setattr(config_module, "_warned_unknown_env", set(), raising=False)
    monkeypatch.setenv("LASAGNA_LOG_FORMAT", "json")
    monkeypatch.setenv("LASAGNA_DATASET_REVISIONS", "{}")
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        RAGConfig()
    finally:
        logger.remove(sink)
    assert not [m for m in seen if "LASAGNA_" in m], seen


def test_the_settings_file_can_be_named_with_either_prefix(monkeypatch, tmp_path):
    settings = tmp_path / "mine.toml"
    settings.write_text("[rag]\nchunk_size = 640\n")
    monkeypatch.setenv("LASAGNA_SETTINGS_FILE", str(settings))
    assert RAGConfig().chunk_size == 640


def test_env_get_prefers_lasagna_and_treats_empty_as_unset(monkeypatch):
    from lasagna.core.paths import env_get

    monkeypatch.setenv("RAG_SOMETHING", "rag")
    assert env_get("SOMETHING") == "rag"
    monkeypatch.setenv("LASAGNA_SOMETHING", "")
    assert env_get("SOMETHING") == "rag"
    monkeypatch.setenv("LASAGNA_SOMETHING", "lasagna")
    assert env_get("SOMETHING") == "lasagna"


class TestSessionTokens:
    @pytest.fixture(autouse=True)
    def secret(self, monkeypatch):
        monkeypatch.setenv("RAG_JWT_SECRET", "x" * 32)

    def _token(self, issuer):
        import time

        import jwt

        from lasagna.api.rest import session_token

        payload = {
            "sub": "u",
            "sid": "s",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            "iss": issuer,
            "typ": "session",
        }
        return jwt.encode(payload, session_token._get_secret(), algorithm="HS256")

    def test_new_tokens_are_issued_as_lasagna(self):
        import jwt

        from lasagna.api.rest.session_token import create_session_token

        token, _ = create_session_token("s", "u")
        assert jwt.decode(token, options={"verify_signature": False})["iss"] == "lasagna"

    def test_a_token_issued_as_rag_knowledge_is_still_accepted(self):
        from lasagna.api.rest.session_token import decode_session_token

        assert decode_session_token(self._token("rag-knowledge")) == ("s", "u")

    def test_a_token_from_another_issuer_is_refused(self):
        from lasagna.api.rest.session_token import decode_session_token

        assert decode_session_token(self._token("someone-else")) is None


def test_a_typo_in_lasagna_log_level_leaves_the_default(monkeypatch):
    """`LASAGNA_LOG_LEVEL` is the `log_level` setting's name, and the CLI's
    console level, so a typo in it is ignored rather than fatal."""
    monkeypatch.setenv("LASAGNA_LOG_LEVEL", "NOPE")
    monkeypatch.delenv("RAG_LOG_LEVEL", raising=False)
    assert RAGConfig().log_level == "INFO"
