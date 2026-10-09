"""The fixed and recursive chunkers: exact boundaries, overlap, offsets, units.

Both are baselines a chunking result is measured against, so the boundaries
are checked against hand-computed expectations, and the recursive chunker
against LangChain's own implementation when it is installed.
"""

from __future__ import annotations

import random

import pytest

from lasagna.core.base_components import Chunk
from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.processing.chunker import splitters
from lasagna.processing.chunker.semantic import chunk_document
from lasagna.processing.chunker.splitters import FixedSizeChunker, RecursiveChunker


def _spans(chunks):
    return [(c.content, c.start_char, c.end_char) for c in chunks]


@pytest.mark.parametrize("name", ["fixed", "recursive"])
def test_the_chunker_passes_the_stage_contract(name):
    report = ComponentTestHarness().validate(registry.get("chunker", name))
    assert report.passed, report.summary()


@pytest.mark.parametrize("name", ["fixed", "recursive"])
def test_every_chunk_is_the_slice_its_offsets_name(name):
    text = ("Para one.\n\nPara two has  odd   spacing.\nAnd a line. " * 30).strip()
    chunks = registry.get("chunker", name, chunk_size=50, chunk_overlap=10).chunk(text)
    assert len(chunks) > 5
    for i, c in enumerate(chunks):
        assert text[c.start_char : c.end_char] == c.content
        assert c.chunk_index == i


class TestFixedWindows:
    def test_character_windows_overlap_by_the_overlap(self):
        chunks = FixedSizeChunker(chunk_size=4, chunk_overlap=1, unit="chars").chunk("abcdefghij")
        assert _spans(chunks) == [("abcd", 0, 4), ("defg", 3, 7), ("ghij", 6, 10)]

    def test_the_last_window_is_the_first_to_reach_the_end(self):
        chunks = FixedSizeChunker(chunk_size=4, chunk_overlap=2, unit="chars").chunk("abcdef")
        assert [c.content for c in chunks] == ["abcd", "cdef"]

    def test_word_windows_count_words_and_skip_the_whitespace_between(self):
        chunks = FixedSizeChunker(chunk_size=2, chunk_overlap=1, unit="words").chunk(
            "one two three four five"
        )
        assert _spans(chunks) == [
            ("one two", 0, 7),
            ("two three", 4, 13),
            ("three four", 8, 18),
            ("four five", 14, 23),
        ]

    def test_token_windows_follow_the_tokenizer_offsets(self, monkeypatch):
        class EveryThreeChars:
            def spans(self, text):
                return [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]

            def count(self, text):
                return len(self.spans(text))

        monkeypatch.setattr(splitters, "load_tokenizer", lambda name: EveryThreeChars())
        chunks = FixedSizeChunker(
            chunk_size=2, chunk_overlap=0, unit="tokens", tokenizer="stub"
        ).chunk("abcdefghijklm")
        assert _spans(chunks) == [("abcdef", 0, 6), ("ghijkl", 6, 12), ("m", 12, 13)]

    def test_tokens_without_a_tokenizer_are_refused(self):
        with pytest.raises(ValueError, match="chunk_tokenizer"):
            FixedSizeChunker(unit="tokens").chunk("some text")

    def test_an_overlap_as_large_as_the_window_is_refused(self):
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            FixedSizeChunker(chunk_size=4, chunk_overlap=4).chunk("abcdefgh")

    def test_it_reads_the_arms_settings_through_chunk_document(self):
        arm = rag_config.with_query_overrides({"chunk_size": 3, "chunk_overlap": 0})
        out = chunk_document("abcdefgh", "d", "s", "f", config=arm, chunker=FixedSizeChunker())
        assert [c for c, _ in out] == ["abc", "def", "gh"]
        assert [(m["start_char"], m["end_char"]) for _, m in out] == [(0, 3), (3, 6), (6, 8)]


class TestRecursiveSplitting:
    TEXT = "One two.\n\nThree four five six.\nSeven."

    def test_it_splits_on_the_coarsest_separator_first(self):
        chunks = RecursiveChunker(chunk_size=12, chunk_overlap=0, unit="chars").chunk(self.TEXT)
        # The paragraph fits; the second paragraph splits at its line break, and
        # its 21-character first line splits further at spaces.
        assert _spans(chunks) == [
            ("One two.", 0, 8),
            ("Three four", 10, 20),
            ("five six.", 21, 30),
            ("Seven.", 31, 37),
        ]

    def test_overlap_carries_the_tail_of_one_chunk_into_the_next(self):
        chunks = RecursiveChunker(
            chunk_size=5, chunk_overlap=2, separators=[" ", ""], unit="chars"
        ).chunk("a b c d e f")
        assert [c.content for c in chunks] == ["a b", "b c", "c d", "d e f"]

    def test_a_sentence_keeps_its_full_stop(self):
        chunks = RecursiveChunker(chunk_size=25, chunk_overlap=0, unit="chars").chunk(
            "First sentence here. Second sentence here."
        )
        assert [c.content for c in chunks] == ["First sentence here.", "Second sentence here."]

    def test_word_units_size_chunks_in_words(self):
        chunks = RecursiveChunker(chunk_size=3, chunk_overlap=0, unit="words").chunk(
            "a b c d e f g"
        )
        assert [c.content for c in chunks] == ["a b c", "d e f", "g"]

    def test_chunking_strategy_recursive_runs_it(self):
        arm = rag_config.with_query_overrides(
            {
                "chunking_strategy": "recursive",
                "chunk_size": 12,
                "chunk_overlap": 0,
                "chunk_unit": "chars",
                "enable_hierarchical": False,
                "preserve_headers": False,
                "preserve_tables": False,
            }
        )
        out = chunk_document(self.TEXT, "d", "s", "f", config=arm)
        assert [c for c, _ in out] == ["One two.", "Three four", "five six.", "Seven."]


class TestRecursiveMatchesLangChain:
    """Same algorithm, same output: checked on random text and settings."""

    @pytest.fixture(autouse=True)
    def _langchain(self):
        pytest.importorskip("langchain_text_splitters")

    @pytest.mark.parametrize(
        "separators,keep",
        [
            (["\n\n", "\n", " ", ""], "start"),  # LangChain's defaults
            (["\n\n", "\n", ". ", "? ", "! ", " ", ""], "end"),  # this package's defaults
            (["\n", ". "], "end"),  # no character fallback: oversized pieces pass through
        ],
    )
    def test_random_documents_split_identically(self, separators, keep):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        rng = random.Random(7)
        vocab = ["alpha", "beta.", "gamma?", "delta!", "eps\n", "zeta\n\n", "eta ", "x" * 30]
        for _ in range(150):
            text = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 80)))
            size = rng.randint(5, 150)
            overlap = rng.randint(0, size)
            theirs = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap, separators=separators, keep_separator=keep
            ).split_text(text)
            ours = RecursiveChunker(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=separators,
                keep_separator=keep,
                unit="chars",
            ).chunk(text)
            assert [c.content for c in ours] == theirs, (text, size, overlap)


def _cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, filename), str)


MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.skipif(not _cached(MINILM, "tokenizer.json"), reason=f"{MINILM} is not cached")
def test_token_windows_are_exact_in_a_real_tokenizer():
    # One WordPiece token per word here, so the windows can be read off by hand.
    chunks = FixedSizeChunker(chunk_size=4, chunk_overlap=2, unit="tokens", tokenizer=MINILM).chunk(
        "the cat sat on the mat"
    )
    assert _spans(chunks) == [("the cat sat on", 0, 14), ("sat on the mat", 8, 22)]
    assert all(isinstance(c, Chunk) for c in chunks)
