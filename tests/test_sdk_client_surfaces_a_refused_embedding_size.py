"""An SDK client whose embedding size the store refuses is told so.

With the persistent session store, the client's settings go through the
guarded config write. A refusal there was swallowed with every other bridge
error at debug level, so the client would have searched on the session's
other embedding model, silently, rather than the one it was configured with.
"""

from __future__ import annotations

import pytest

from lasagna.api._profile_guard import EmbeddingSizeConflict


class _RefusingProvider:
    def set_config_overrides(self, session_id, overrides):
        raise EmbeddingSizeConflict("stored documents were embedded at 384")

    def remove_config_overrides(self, session_id, keys):
        pass


def test_the_refusal_reaches_the_caller(monkeypatch):
    from lasagna.sdk import KnowledgeClient, KnowledgeConfig

    monkeypatch.setattr("lasagna.providers.get_session_provider", lambda: _RefusingProvider())
    client = KnowledgeClient(
        config=KnowledgeConfig(embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768),
        session_id="sdk-refused",
    )
    with pytest.raises(EmbeddingSizeConflict, match="384"):
        client._apply_runtime_config()
