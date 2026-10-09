"""The proposition and contextual chunkers, with the LLM stubbed.

Both must raise without an LLM: returning plain chunks would make an arm named
for the technique score as the control.
"""

from __future__ import annotations

import json
import re

import pytest

from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.processing.chunker.llm_chunkers import (
    ContextualChunker,
    PropositionChunker,
    parse_propositions,
)
from lasagna.processing.chunker.semantic import chunk_document
from lasagna.processing.chunker.splitters import FixedSizeChunker


class _Message:
    def __init__(self, content):
        self.content = content


class PropositionLLM:
    """Answers with one proposition per sentence of the passage, pronouns resolved."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = prompt.rsplit("Content: ", 1)[1].rsplit("\nOutput:", 1)[0]
        sentences = [s for s in re.split(r"(?<=\.)\s+", content) if s]
        return _Message(json.dumps([s.replace("She", "Ada") for s in sentences]))


class ContextLLM:
    """Context that names the chunk it was asked about, and the document it saw."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        chunk = prompt.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        return _Message(f"About [{chunk}].")


@pytest.fixture
def no_llm(monkeypatch):
    import lasagna.providers as providers

    def unconfigured():
        raise providers.LLMProviderNotConfigured("No LLMProvider configured")

    monkeypatch.setattr(providers, "get_llm_provider", unconfigured)


class TestPropositions:
    TEXT = "Ada wrote code. She liked math. Bob ran far."

    def test_the_chunker_passes_the_stage_contract(self):
        report = ComponentTestHarness().validate(PropositionChunker(llm=PropositionLLM()))
        assert report.passed, report.summary()

    def test_each_proposition_is_a_chunk_spanning_its_passage(self):
        chunks = PropositionChunker(llm=PropositionLLM(), passage_size=5).chunk(self.TEXT)
        # Five-word passages: the recursive split puts each sentence in its own.
        assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
            ("Ada wrote code.", 0, 15),
            ("Ada liked math.", 16, 31),
            ("Bob ran far.", 32, 44),
        ]
        assert [c.metadata["passage_index"] for c in chunks] == [0, 1, 2]
        assert all(c.chunk_type == "proposition" for c in chunks)

    def test_the_prompt_is_the_papers_with_the_passage_as_content(self):
        llm = PropositionLLM()
        PropositionChunker(llm=llm, passage_size=100).chunk(self.TEXT, title="Ada", section="Life")
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        assert prompt.startswith('Decompose the "Content" into clear and simple propositions')
        assert "Decontextualize the proposition" in prompt
        assert prompt.endswith(f"Input: Title: Ada. Section: Life. Content: {self.TEXT}\nOutput:")

    def test_a_passage_is_sent_once_however_often_it_is_chunked(self):
        llm = PropositionLLM()
        chunker = PropositionChunker(llm=llm, passage_size=100)
        assert chunker.chunk(self.TEXT) == chunker.chunk(self.TEXT)
        assert len(llm.prompts) == 1

    def test_the_answer_may_be_fenced(self):
        assert parse_propositions('```json\n["A is B.", "C is D."]\n```') == ["A is B.", "C is D."]

    def test_an_unreadable_answer_raises_instead_of_indexing_the_passage(self):
        with pytest.raises(ValueError, match="list of propositions"):
            parse_propositions("I cannot help with that.")

    def test_without_an_llm_it_says_so(self, no_llm):
        with pytest.raises(RuntimeError, match="proposition chunker needs an LLM"):
            registry.get("chunker", "proposition").chunk(self.TEXT)


class TestContextualChunks:
    TEXT = "one two three four five six"

    def _chunker(self, llm=None):
        base = FixedSizeChunker(chunk_size=3, chunk_overlap=0, unit="words")
        return ContextualChunker(llm=llm or ContextLLM(), base_chunker=base)

    def test_the_chunker_passes_the_stage_contract(self):
        report = ComponentTestHarness().validate(self._chunker())
        assert report.passed, report.summary()

    def test_each_chunk_is_its_context_then_the_chunk_at_the_chunks_span(self):
        chunks = self._chunker().chunk(self.TEXT)
        assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
            ("About [one two three].\n\none two three", 0, 13),
            ("About [four five six].\n\nfour five six", 14, 27),
        ]
        assert chunks[0].metadata["context"] == "About [one two three]."

    def test_the_prompt_is_anthropics_with_the_whole_document(self):
        llm = ContextLLM()
        self._chunker(llm).chunk(self.TEXT)
        assert llm.prompts[0] == (
            f"<document>\n{self.TEXT}\n</document>\n"
            "Here is the chunk we want to situate within the whole document\n"
            "<chunk>\none two three\n</chunk>\n"
            "Please give a short succinct context to situate this chunk within the "
            "overall document for the purposes of improving search retrieval of the "
            "chunk. Answer only with the succinct context and nothing else."
        )

    def test_the_document_cap_limits_what_the_llm_sees(self):
        llm = ContextLLM()
        arm = rag_config.with_query_overrides({"contextual_max_document_chars": 7})
        self._chunker(llm).chunk(self.TEXT, config=arm)
        assert llm.prompts[0].startswith("<document>\none two\n</document>")

    def test_the_base_chunker_comes_from_the_arms_settings(self):
        arm = rag_config.with_query_overrides(
            {"contextual_base_chunker": "fixed", "chunk_size": 9, "chunk_overlap": 0}
        )
        chunks = ContextualChunker(llm=ContextLLM()).chunk(self.TEXT, config=arm)
        assert [c.metadata["base_chunker"] for c in chunks] == ["fixed"] * 3
        assert chunks[0].content.endswith("\n\none two t")

    def test_combining_it_with_the_pipeline_flag_is_refused(self):
        arm = rag_config.with_query_overrides({"enable_contextual_retrieval": True})
        with pytest.raises(ValueError, match="twice"):
            self._chunker().chunk(self.TEXT, config=arm)

    def test_a_failed_llm_call_raises_instead_of_leaving_the_chunk_bare(self):
        class Broken:
            def invoke(self, prompt):
                raise ConnectionError("rate limited")

        with pytest.raises(ConnectionError):
            self._chunker(Broken()).chunk(self.TEXT)

    def test_without_an_llm_it_says_so(self, no_llm):
        with pytest.raises(RuntimeError, match="contextual chunker needs an LLM"):
            registry.get("chunker", "contextual").chunk(self.TEXT)

    def test_ingest_keeps_the_base_chunks_span(self):
        out = chunk_document(self.TEXT, "d", "s", "f", chunker=self._chunker())
        assert [(m["start_char"], m["end_char"]) for _, m in out] == [(0, 13), (14, 27)]
