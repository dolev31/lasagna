"""``embedding_model_fallback`` is a server setting, and its load follows the download policy.

It slipped past both checks a session's model names go through. The key-name
check looked for ``_model`` and ``_models`` endings, so a session could name a
relative folder on the server (``models/x``) as its fallback, and the fallback
load, after the primary failed, skipped ``model_downloads.check``: a server
that refuses downloads inside a request downloaded the fallback there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.core import model_downloads
from lasagna.core.model_downloads import ModelNotDownloaded, refuse_downloads_in_requests
from lasagna.core.session_config._session_keys import SessionSettingRefused, check_session_settings


@pytest.mark.parametrize("value", ["models/x", "BAAI/bge-base-en-v1.5"])
def test_a_session_cannot_set_the_fallback(value):
    with pytest.raises(SessionSettingRefused, match="embedding_model_fallback is a server setting"):
        check_session_settings({"embedding_model_fallback": value})


def test_the_fallback_load_is_refused_when_it_would_download(monkeypatch):
    import lasagna.infrastructure.embedding.manager as manager

    loaded = []

    def transformer(path, device=None):
        loaded.append(path)
        raise OSError("the primary's weights are corrupt")

    monkeypatch.setattr(manager, "SentenceTransformer", transformer, raising=False)
    monkeypatch.setattr(manager, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(manager, "strict_stages", lambda: False)
    monkeypatch.setattr(model_downloads, "_allowed", lambda: False)
    monkeypatch.setattr(
        model_downloads,
        "refusal",
        lambda model, *a, **k: (
            None if model == "primary/cached" else ModelNotDownloaded(model, offline=False)
        ),
    )
    embedder = object.__new__(manager.EmbeddingManager)
    embedder._config = SimpleNamespace(
        embedding_model="primary/cached",
        embedding_model_fallback="fallback/not-downloaded",
        use_gpu=False,
    )
    embedder.embedding_dim = 384
    monkeypatch.setattr(manager.EmbeddingManager, "_resolve_local_snapshot", staticmethod(str))

    with refuse_downloads_in_requests():
        with pytest.raises(ModelNotDownloaded, match="fallback/not-downloaded"):
            embedder._load_sentence_transformer()
    assert loaded == ["primary/cached"]  # the fallback was never handed to the loader
