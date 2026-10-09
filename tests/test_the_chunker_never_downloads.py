"""Building a chunker never reaches the network.

``SemanticChunker.__init__`` used to call ``nltk.download("punkt")`` and
``nltk.download("punkt_tab")``. Every upload and every benchmark ingest built
one, so a library call contacted the network, wrote 64 MB under the user's
home directory, and split sentences one way online and another way offline.
Without the tokenizer data the chunker now splits with a regex, says so once
with the command that installs the data, and the run records the fallback.
"""

from __future__ import annotations

import pytest

from lasagna.processing.chunker import semantic


@pytest.fixture
def no_tokenizer_data(monkeypatch):
    """nltk installed, its punkt_tab data absent, and any download an error."""
    if not semantic.NLTK_AVAILABLE:
        pytest.skip("nltk is not installed")
    import nltk

    def _refuse(*args, **kwargs):
        raise AssertionError(f"nltk.download was called with {args} {kwargs}")

    def _absent(resource, *args, **kwargs):
        raise LookupError(resource)

    monkeypatch.setattr(nltk, "download", _refuse)
    monkeypatch.setattr(nltk.data, "find", _absent)
    semantic.punkt_available.cache_clear()
    monkeypatch.setattr(semantic, "_regex_split_noted", False)
    yield
    semantic.punkt_available.cache_clear()


def test_building_a_chunker_does_not_download_tokenizer_data(no_tokenizer_data):
    semantic.SemanticChunker()
    semantic.create_chunker("fixed")


def test_without_tokenizer_data_sentences_split_on_punctuation(no_tokenizer_data):
    chunker = semantic.SemanticChunker()
    assert chunker._split_sentences("One here. Two there! Three?") == [
        "One here.",
        "Two there!",
        "Three?",
    ]


def test_the_regex_fallback_is_announced_once_with_the_fix(no_tokenizer_data):
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        chunker = semantic.SemanticChunker()
        chunker._split_sentences("First. Second.")
        chunker._split_sentences("Third. Fourth.")
    finally:
        logger.remove(sink)

    notes = [m for m in messages if "regular expression" in m]
    assert len(notes) == 1, messages
    assert semantic.PUNKT_DOWNLOAD_COMMAND in notes[0]


def test_tokenizer_data_is_detected_without_downloading(monkeypatch):
    if not semantic.NLTK_AVAILABLE:
        pytest.skip("nltk is not installed")
    import nltk

    monkeypatch.setattr(
        nltk, "download", lambda *a, **k: pytest.fail("punkt_available() downloaded")
    )
    semantic.punkt_available.cache_clear()
    try:
        assert isinstance(semantic.punkt_available(), bool)
    finally:
        semantic.punkt_available.cache_clear()
