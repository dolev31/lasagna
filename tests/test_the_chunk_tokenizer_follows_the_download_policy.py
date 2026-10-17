"""A server does not download a chunk tokenizer inside a request.

Models are refused when a server would have to download them
(``allow_model_download_in_requests``), but ``chunk_tokenizer`` (a session
setting, with ``chunk_unit: tokens``) went straight to
``AutoTokenizer.from_pretrained``, which downloaded whatever repository it
named while the upload waited.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core import model_downloads
from lasagna.processing.chunker import splitters


def test_an_uncached_tokenizer_is_refused_not_downloaded(monkeypatch):
    import transformers

    def download(*a, **k):
        raise AssertionError("downloaded the tokenizer")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", download)
    name = f"acme/tokenizer-{uuid.uuid4().hex[:8]}"
    with model_downloads.refuse_downloads_in_requests():
        with pytest.raises(model_downloads.ModelNotDownloaded, match="chunk tokenizer"):
            splitters.load_tokenizer(name)


def test_a_cached_tokenizer_is_looked_up_by_its_tokenizer_file(monkeypatch):
    asked = []

    def cached(repo, filename):
        asked.append(filename)
        return "/cache/tokenizer_config.json"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    with model_downloads.refuse_downloads_in_requests():
        model_downloads.check("acme/tok", kind="chunk tokenizer", filename="tokenizer_config.json")
    assert asked == ["tokenizer_config.json"]
