"""Semantic Chunking for RAG System.

This module provides intelligent text chunking strategies:
- Semantic chunking based on embedding similarity
- Sentence-window chunking for context preservation
- Hierarchical chunking (parent-child) for better retrieval
- Structure-aware chunking that preserves headers and tables
"""

import re
import hashlib
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

import numpy as np
from loguru import logger

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    # Said once, when a sentence is first split, not at import: every command
    # imports this module, including ones that never chunk anything.
    NLTK_AVAILABLE = False

from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.base_components import Chunk as _BaseChunk

#: The one-time command that installs nltk's sentence tokenizer data. Nothing
#: in Lasagna downloads it for you.
PUNKT_DOWNLOAD_COMMAND = "python -m nltk.downloader punkt_tab"


@lru_cache(maxsize=1)
def punkt_available() -> bool:
    """Whether nltk and its sentence tokenizer data (punkt_tab) are installed.

    Only looks; never downloads. Constructing a chunker used to call
    ``nltk.download``, so a library call reached the network, wrote 64 MB under
    the user's home directory, and split sentences differently depending on
    whether the machine happened to be online.
    """
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        return False
    return True


_regex_split_noted = False


def _note_regex_split() -> None:
    """Say once per process that sentences are split with a regex, and how to fix it."""
    global _regex_split_noted
    if _regex_split_noted:
        return
    _regex_split_noted = True
    if NLTK_AVAILABLE:
        reason = "nltk's sentence tokenizer data (punkt_tab) is not installed"
        fix = PUNKT_DOWNLOAD_COMMAND
    else:
        reason = "nltk is not installed"
        fix = f"pip install nltk, then {PUNKT_DOWNLOAD_COMMAND}"
    logger.warning(
        f"Sentences are split with a regular expression because {reason}. Chunk "
        f"boundaries can differ from a machine that has it, and run bundles record "
        f"this as an active fallback. To use nltk's splitter, run once: {fix}"
    )


#: Whether this process has looked for (and tried to fetch) NLTK's Punkt data.

#: Breakpoint methods for semantic chunking. "absolute" is this package's
#: original rule (split where adjacent sentences fall below semantic_threshold,
#: within min/max sizes); the others are Kamradt's, with LangChain's
#: SemanticChunker defaults for semantic_breakpoint_amount.
BREAKPOINT_DEFAULTS: Dict[str, Optional[float]] = {
    "absolute": None,
    "percentile": 95.0,
    "standard_deviation": 3.0,
    "interquartile": 1.5,
    "gradient": 95.0,
}


def breakpoint_threshold(
    distances: List[float], method: str, amount: float
) -> Tuple[float, np.ndarray]:
    """The distance above which a gap between sentences is a breakpoint.

    As LangChain's ``SemanticChunker`` computes it. Returns the threshold and
    the series it applies to: the distances, or for ``gradient`` their
    gradient. ``interquartile`` is the mean plus ``amount`` interquartile ranges
    (LangChain's form, not Tukey's Q3 + 1.5 IQR).
    """
    d = np.asarray(distances, dtype=float)
    if method == "percentile":
        return float(np.percentile(d, amount)), d
    if method == "standard_deviation":
        return float(np.mean(d) + amount * np.std(d)), d
    if method == "interquartile":
        q1, q3 = np.percentile(d, [25, 75])
        return float(np.mean(d) + amount * (q3 - q1)), d
    if method == "gradient":
        g = np.gradient(d, range(len(d)))
        return float(np.percentile(g, amount)), g
    raise ValueError(
        f"semantic_breakpoint_method must be one of {sorted(BREAKPOINT_DEFAULTS)}, got {method!r}"
    )


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    HIERARCHICAL = "hierarchical"
    #: Sections split at headers, tables kept whole, each section chunked
    #: semantically. Runs when preserve_headers or preserve_tables is on.
    STRUCTURE_AWARE = "structure_aware"


#: Strategies named after a structural setting. With that setting off, what is
#: left is the strategy their sections are chunked with: semantic. The preset
#: profiles declared "semantic" until they named the chunker that runs, so an
#: arm that turns the structure off chunks exactly as it did then.
_STRUCTURAL_STRATEGIES = frozenset(
    {ChunkingStrategy.STRUCTURE_AWARE.value, ChunkingStrategy.HIERARCHICAL.value}
)


def chunker_needs_embeddings(config) -> bool:
    """Whether the built-in chunking for ``config`` may chunk semantically.

    Benchmark ingest hands the chunker the embedding model only when this is
    true. It was ``chunking_strategy == "semantic"``; a structural strategy
    counts too, because its sections (or, with the structure off, the whole
    text) are chunked semantically.
    """
    strategy = str(getattr(config, "chunking_strategy", "") or "")
    return strategy == ChunkingStrategy.SEMANTIC.value or strategy in _STRUCTURAL_STRATEGIES


class Chunk(_BaseChunk):
    """A text chunk with stable ID computation and serialization.

    Extends the canonical ``base_components.Chunk`` with:
    - ``stable_id``: deterministic content-based hash for citation tracking
    - ``to_dict()``: serialization for storage backends
    """

    @property
    def stable_id(self) -> str:
        """Generate a stable ID based on content hash."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return content_hash[: rag_config.citation_id_length]

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_type": self.chunk_type,
            "stable_id": self.stable_id,
            **self.metadata,
        }


class SemanticChunker:
    """Semantic chunking based on embedding similarity.

    Splits text at points where semantic similarity drops below threshold,
    creating coherent chunks that preserve meaning.
    """

    def __init__(self, embedding_manager=None, config: Optional[RAGConfig] = None):
        """Initialize semantic chunker.

        Args:
            embedding_manager: Optional embedding manager for semantic splitting
            config: The settings to chunk with. Defaults to the process-wide
                config; ``chunk_document`` passes the arm's, so a swept
                ``chunk_size`` reaches the chunker instead of the global value.
        """
        self.embedding_manager = embedding_manager
        self._config = config if config is not None else rag_config

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences: nltk's punkt when installed, else a regex."""
        if punkt_available():
            try:
                return sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        else:
            _note_regex_split()

        # Fallback: simple regex-based splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _sentence_starts(text: str, sentences: List[str]) -> List[int]:
        """Where each sentence starts in ``text``, searching forward in order."""
        starts = []
        pos = 0
        for sentence in sentences:
            start = text.find(sentence, pos)
            if start == -1:
                start = pos  # the tokenizer changed the sentence; keep the order
            starts.append(start)
            pos = start + len(sentence)
        return starts

    @staticmethod
    def _source_text(text: str, start: int, end: int, joined: str) -> Tuple[str, int, int]:
        """A chunk's content as the source has it, and its span.

        Sentences joined with single spaces rewrote the source: "**Is the
        widget free?** Yes" became "**Is the widget free? ** Yes", and line
        breaks became spaces. The slice from the first sentence's start to the
        last one's end is the source text; it is used when it holds the same
        characters as the joined sentences, whitespace aside (a tokenizer that
        changed a sentence keeps the joined text).
        """
        sliced = text[start:end]
        if "".join(sliced.split()) == "".join(joined.split()):
            return sliced, start, end
        return joined.strip(), start, end

    def chunk_semantic(
        self,
        text: str,
        threshold: Optional[float] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[Chunk]:
        """Split text using semantic similarity.

        Groups sentences together until semantic similarity drops.

        Args:
            text: Text to chunk
            threshold: Similarity threshold for splitting (default from config)
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters

        Returns:
            List of Chunk objects
        """
        if not self.embedding_manager:
            logger.warning("No embedding manager - falling back to fixed chunking")
            return self.chunk_fixed(text)

        # `is not None`, not `or`: a threshold of 0.0 is a value, not a gap.
        threshold = threshold if threshold is not None else self._config.semantic_threshold
        min_size = min_size if min_size is not None else self._config.min_chunk_size
        max_size = max_size if max_size is not None else self._config.max_chunk_size

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            start = max(text.find(sentences[0]), 0)
            content, start, end = self._source_text(
                text, start, start + len(sentences[0]), sentences[0]
            )
            return [Chunk(content=content, chunk_index=0, start_char=start, end_char=end)]

        # Get embeddings for all sentences
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.embedding_manager.encode_documents_batch(sentences, show_progress=False)

        chunks = []
        current_chunk = []
        current_start = 0

        # Pre-compute sentence start positions for accurate char tracking
        sentence_starts = self._sentence_starts(text, sentences)

        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            current_chunk.append(sentence)
            current_text = " ".join(current_chunk)

            # Check if we should split
            should_split = False

            # Check max size
            if len(current_text) >= max_size:
                should_split = True

            # Check semantic similarity with next sentence
            elif i < len(sentences) - 1:
                next_embedding = embeddings[i + 1]
                similarity = self.embedding_manager.similarity(embedding, next_embedding)

                if similarity < threshold and len(current_text) >= min_size:
                    should_split = True

            # Create chunk if splitting or at end
            if should_split or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():
                    content, start, end = self._source_text(
                        text, current_start, sentence_starts[i] + len(sentences[i]), chunk_text
                    )
                    chunks.append(
                        Chunk(
                            content=content,
                            chunk_index=len(chunks),
                            start_char=start,
                            end_char=end,
                        )
                    )
                # Next chunk starts after this sentence's end in the original text
                if i + 1 < len(sentences):
                    current_start = sentence_starts[i + 1]
                current_chunk = []

        logger.info(f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks

    def chunk_semantic_breakpoints(
        self,
        text: str,
        method: Optional[str] = None,
        amount: Optional[float] = None,
        buffer_size: Optional[int] = None,
    ) -> List[Chunk]:
        """Kamradt's semantic chunking: split where meaning shifts most.

        Each sentence is embedded together with ``buffer_size`` neighbours on
        either side; a gap whose cosine distance to the next exceeds the
        method's threshold (95th percentile by default) is a breakpoint. No size
        limits apply, as in the original. Chunk text is the source slice from the
        first sentence of a group to the end of its last.

        Reference: G. Kamradt, "5 Levels of Text Splitting" (2024), as
        implemented by LangChain's ``SemanticChunker``.
        """
        cfg = self._config
        method = method or getattr(cfg, "semantic_breakpoint_method", "percentile")
        if method not in BREAKPOINT_DEFAULTS or method == "absolute":
            raise ValueError(
                f"breakpoint method must be one of "
                f"{sorted(m for m in BREAKPOINT_DEFAULTS if m != 'absolute')}, got {method!r}"
            )
        if amount is None:
            amount = getattr(cfg, "semantic_breakpoint_amount", None)
        if amount is None:
            amount = BREAKPOINT_DEFAULTS[method]
        if buffer_size is None:
            buffer_size = getattr(cfg, "semantic_buffer_size", 1)
        if self.embedding_manager is None:
            raise RuntimeError("semantic chunking needs an embedding model, and none was given")

        sentences = self._split_sentences(text)
        if not sentences:
            return []
        starts = self._sentence_starts(text, sentences)
        ends = [st + len(sentence) for st, sentence in zip(starts, sentences)]

        def group(first: int, last: int) -> Chunk:
            return Chunk(
                content=text[starts[first] : ends[last]],
                chunk_index=0,
                start_char=starts[first],
                end_char=ends[last],
                metadata={"sentence_start": first, "sentence_end": last + 1},
            )

        n = len(sentences)
        # Too few gaps to rank: one sentence is one chunk, and the gradient of a
        # single distance is undefined (LangChain returns the two sentences).
        if n == 1 or (method == "gradient" and n == 2):
            out = [group(i, i) for i in range(n)]
        else:
            b = int(buffer_size)
            windows = [" ".join(sentences[max(0, i - b) : i + b + 1]) for i in range(n)]
            vectors = np.asarray(
                self.embedding_manager.encode_documents_batch(windows, show_progress=False),
                dtype=float,
            )
            norms = np.linalg.norm(vectors, axis=1)
            norms[norms == 0] = 1.0
            unit = vectors / norms[:, None]
            distances = [1.0 - float(np.dot(unit[i], unit[i + 1])) for i in range(n - 1)]
            threshold, series = breakpoint_threshold(distances, method, float(amount))
            breaks = [i for i, value in enumerate(series) if value > threshold]
            out = []
            first = 0
            for i in breaks:
                out.append(group(first, i))
                first = i + 1
            if first < n:
                out.append(group(first, n - 1))

        for i, chunk in enumerate(out):
            chunk.chunk_index = i
        logger.info(f"Semantic chunking ({method}): {n} sentences -> {len(out)} chunks")
        return out

    def chunk_semantic_configured(self, text: str) -> List[Chunk]:
        """Semantic chunking by the configured ``semantic_breakpoint_method``."""
        method = getattr(self._config, "semantic_breakpoint_method", "absolute")
        if method == "absolute" or self.embedding_manager is None:
            return self.chunk_semantic(text)
        return self.chunk_semantic_breakpoints(text)

    def chunk_fixed(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[Chunk]:
        """Split text into fixed-size overlapping chunks.

        Falls back method when semantic chunking isn't available.

        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks

        Returns:
            List of Chunk objects
        """
        chunk_size = chunk_size if chunk_size is not None else self._config.chunk_size
        overlap = overlap if overlap is not None else self._config.chunk_overlap

        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)
        # Where the last chunk's window began: the loop guard below reads it,
        # and a chunk's start_char is the stripped text's start, which can be later.
        last_window = None

        while start < text_length:
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < text_length:
                search_start = max(start, end - int(chunk_size * 0.2))
                chunk_text = text[search_start:end]

                for delimiter in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                    last_break = chunk_text.rfind(delimiter)
                    if last_break != -1:
                        end = search_start + last_break + len(delimiter)
                        break

            raw = text[start:end]
            chunk_content = raw.strip()
            if chunk_content:
                # The span of the stripped text, inside the source: end could
                # pass the end of the text (a 233-character text gave 0-500).
                first = start + len(raw) - len(raw.lstrip())
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_index=len(chunks),
                        start_char=first,
                        end_char=first + len(chunk_content),
                    )
                )
                last_window = start

            # Move with overlap
            start = end - overlap
            # Prevent backward movement (infinite loop guard)
            if last_window is not None and start <= last_window:
                start = end

        logger.info(f"Fixed chunking: {text_length} chars -> {len(chunks)} chunks")
        return chunks

    def chunk_sentence_window(
        self, text: str, window_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[Chunk]:
        """Split text using sentence windows.

        Creates chunks of N consecutive sentences with overlap.

        Args:
            text: Text to chunk
            window_size: Number of sentences per chunk
            overlap: Number of sentences to overlap

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If overlap >= window_size (would cause infinite loop)
        """
        # `or` turned a requested overlap of 0 into the configured default of 1.
        window_size = window_size if window_size is not None else self._config.window_size
        overlap = overlap if overlap is not None else self._config.sentence_overlap

        # Validate to prevent infinite loop
        if overlap >= window_size:
            logger.warning(
                f"sentence_overlap ({overlap}) >= window_size ({window_size}), "
                f"adjusting overlap to {window_size - 1}"
            )
            overlap = max(0, window_size - 1)

        sentences = self._split_sentences(text)
        if not sentences:
            return []
        starts = self._sentence_starts(text, sentences)

        chunks = []
        i = 0
        step = window_size - overlap

        # Ensure we always make progress
        if step <= 0:
            step = 1

        while i < len(sentences):
            window = sentences[i : i + window_size]
            chunk_content = " ".join(window)
            last = i + len(window) - 1

            if chunk_content.strip():
                chunks.append(
                    Chunk(
                        content=chunk_content.strip(),
                        chunk_index=len(chunks),
                        start_char=starts[i],
                        end_char=starts[last] + len(sentences[last]),
                        metadata={"sentence_start": i, "sentence_end": i + len(window)},
                    )
                )

            # The window that reaches the last sentence is the last one. Going
            # on emitted trailing windows wholly inside the one before, each a
            # duplicate retrieval hit.
            if i + window_size >= len(sentences):
                break
            i += step

        logger.info(f"Sentence window chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks


class HierarchicalChunker:
    """Hierarchical (parent-child) chunking for better retrieval.

    Creates large parent chunks for context and small child chunks for search.
    Search uses child chunks, but retrieval returns parent context.
    """

    def __init__(self, semantic_chunker: SemanticChunker):
        """Initialize with a semantic chunker.

        Args:
            semantic_chunker: Base chunker for creating chunks
        """
        self.semantic_chunker = semantic_chunker

    def chunk_hierarchical(
        self, text: str, parent_size: Optional[int] = None, child_size: Optional[int] = None
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """Create hierarchical parent-child chunks.

        Args:
            text: Text to chunk
            parent_size: Size of parent chunks
            child_size: Size of child chunks

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        cfg = self.semantic_chunker._config
        parent_size = parent_size if parent_size is not None else cfg.parent_chunk_size
        child_size = child_size if child_size is not None else cfg.child_chunk_size

        # Create parent chunks
        parent_chunks = self.semantic_chunker.chunk_fixed(text, chunk_size=parent_size, overlap=0)

        # Create child chunks within each parent
        child_chunks = []
        for parent in parent_chunks:
            parent_id = parent.stable_id

            children = self.semantic_chunker.chunk_fixed(
                parent.content, chunk_size=child_size, overlap=50
            )

            # Track total siblings per parent for auto-merge threshold calculation
            total_siblings = len(children)
            for child in children:
                # Offset child positions to be document-relative
                # (chunk_fixed returns positions relative to its input,
                #  which here is parent.content, not the full document)
                child.start_char += parent.start_char
                child.end_char += parent.start_char
                child.parent_chunk_id = parent_id
                child.chunk_index = len(child_chunks)
                child.metadata["total_siblings"] = total_siblings
                child_chunks.append(child)

        logger.info(
            f"Hierarchical chunking: {len(parent_chunks)} parents, {len(child_chunks)} children"
        )
        return parent_chunks, child_chunks


class StructureAwareChunker:
    """Structure-aware chunking that preserves document structure.

    Handles headers, tables, code blocks, and maintains context.
    """

    # Regex patterns for structure detection
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    TABLE_PATTERN = re.compile(r"(\|[^\n]+\|\n)+", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    PAGE_MARKER_PATTERN = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
    #: A chunk made only of these is a horizontal rule (``---``, ``***``) or a
    #: table border (``|---|---|``): it carries no text, and alone it matched
    #: queries about the text around it, ranking first at score 1.0.
    RULE_ONLY_PATTERN = re.compile(r"[\s\-*_=|]*")

    def __init__(self, semantic_chunker: SemanticChunker):
        """Initialize with a base chunker.

        Args:
            semantic_chunker: Base chunker for text content
        """
        self.semantic_chunker = semantic_chunker

    def chunk_with_structure(self, text: str) -> List[Chunk]:
        """Chunk text while preserving structure.

        Args:
            text: Text with possible structure (headers, tables, code)

        Returns:
            List of chunks with structure metadata
        """
        if not text or not text.strip():
            return []

        chunks = []
        current_header = None
        page_sections = self._split_by_pages(text)

        for page_section in page_sections:
            page_text = page_section["content"]
            page_number = page_section.get("page_number")
            page_offset = page_section.get("start", 0)

            # Process text by sections within the page
            sections = self._split_by_headers(page_text)

            for section in sections:
                section_text = section["content"]
                section_header = section.get("header")
                section_offset = page_offset + section.get("start", 0)

                if section_header:
                    current_header = section_header

                # Handle special content types
                if self.semantic_chunker._config.preserve_tables:
                    section_text, table_chunks = self._extract_tables(
                        section_text, len(chunks), current_header, page_number
                    )
                    for table_chunk in table_chunks:
                        if self.RULE_ONLY_PATTERN.fullmatch(table_chunk.content):
                            continue
                        table_chunk.chunk_index = len(chunks)
                        table_chunk.start_char += section_offset
                        table_chunk.end_char += section_offset
                        chunks.append(table_chunk)

                # Chunk remaining text
                text_chunks = self.semantic_chunker.chunk_semantic_configured(section_text)

                # Spans are found in the section as written: with its tables
                # taken out, the chunked copy's positions no longer matched it.
                cursor = 0
                for chunk in text_chunks:
                    if self.RULE_ONLY_PATTERN.fullmatch(chunk.content):
                        continue
                    chunk.chunk_index = len(chunks)
                    chunk.section_header = current_header
                    chunk.page_number = page_number
                    found = section["content"].find(chunk.content, cursor)
                    if found >= 0:
                        chunk.start_char = section_offset + found
                        chunk.end_char = chunk.start_char + len(chunk.content)
                        cursor = found + len(chunk.content)
                    else:
                        chunk.start_char += section_offset
                        chunk.end_char += section_offset
                    chunks.append(chunk)

        logger.info(f"Structure-aware chunking: {len(chunks)} chunks")
        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text by header markers outside fenced code.

        A ``#`` line inside a fenced block is a shell or Python comment, and
        reading it as a header cut the block in two. Headers are looked for in
        a copy with each block blanked to spaces (newlines kept, so every offset
        still points into ``text``); sections are cut from ``text`` itself.
        """
        sections = []
        current_pos = 0
        current_header = None
        outside_code = self.CODE_BLOCK_PATTERN.sub(
            lambda block: re.sub(r"[^\n]", " ", block.group(0)), text
        )

        for match in self.HEADER_PATTERN.finditer(outside_code):
            # Add content before this header
            if match.start() > current_pos:
                content, start = _stripped_at(text, current_pos, match.start())
                if content:
                    sections.append({"content": content, "header": current_header, "start": start})

            current_header = match.group(2).strip()
            current_pos = match.end()

        # Add remaining content
        if current_pos < len(text):
            content, start = _stripped_at(text, current_pos, len(text))
            if content:
                sections.append({"content": content, "header": current_header, "start": start})

        return sections if sections else [{"content": text, "header": None, "start": 0}]

    def _split_by_pages(self, text: str) -> List[Dict[str, Any]]:
        """Split text into page sections based on [Page N] markers."""
        matches = list(self.PAGE_MARKER_PATTERN.finditer(text))
        if not matches:
            return [{"content": text, "page_number": None, "start": 0}]

        sections = []

        # Capture any leading text before the first marker
        if matches[0].start() > 0:
            preface, first = _stripped_at(text, 0, matches[0].start())
            if preface:
                sections.append({"content": preface, "page_number": None, "start": first})

        for i, match in enumerate(matches):
            page_number = int(match.group(1))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content, start = _stripped_at(text, match.end(), end)
            if content:
                sections.append(
                    {
                        "content": content,
                        "page_number": page_number,
                        "start": start,
                    }
                )

        return sections

    def _extract_tables(
        self, text: str, chunk_start_index: int, header: Optional[str], page: Optional[int]
    ) -> Tuple[str, List[Chunk]]:
        """Extract tables and return remaining text with table chunks."""
        table_chunks = []
        # Collect all table matches first, then remove in reverse order
        # to avoid offset shifts and duplicate-content replacement issues
        matches = list(self.TABLE_PATTERN.finditer(text))

        for match in matches:
            table_content = match.group(0)

            table_chunks.append(
                Chunk(
                    content=table_content.strip(),
                    chunk_index=chunk_start_index + len(table_chunks),
                    chunk_type="table",
                    section_header=header,
                    page_number=page,
                    start_char=match.start(),
                    end_char=match.start() + len(table_content.strip()),
                )
            )

        # Remove tables in reverse order to preserve earlier match offsets
        remaining_text = text
        for match in reversed(matches):
            remaining_text = (
                remaining_text[: match.start()] + "\n\n" + remaining_text[match.end() :]
            )

        return remaining_text.strip(), table_chunks


def _stripped_at(text: str, start: int, end: int) -> Tuple[str, int]:
    """``text[start:end]`` stripped, and where the stripped text starts in ``text``."""
    raw = text[start:end]
    return raw.strip(), start + len(raw) - len(raw.lstrip())


#: (requested, effective) chunking pairs already warned about in this process.
_WARNED_OVERRIDES: set = set()


def resolve_effective_strategy(config, requested=None) -> str:
    """The strategy that will actually run, given the structural flags.

    ``chunking_strategy`` is not the top of the dispatch: ``enable_hierarchical``
    wins, then ``preserve_headers``/``preserve_tables``, and only then the
    strategy. All three of those default to True, so out of the box the
    requested strategy never reaches the dispatch at all.

    That is invisible and it is worse than a crash. A chunking ablation sweeping
    ``fixed`` against ``semantic`` produces byte-identical arms, a difference of
    exactly zero, and a confident conclusion that chunking does not matter.
    """
    requested = requested or config.chunking_strategy

    structure_off = False
    if config.enable_hierarchical:
        effective = ChunkingStrategy.HIERARCHICAL.value
    elif config.preserve_headers or config.preserve_tables:
        effective = ChunkingStrategy.STRUCTURE_AWARE.value
    elif str(requested) in _STRUCTURAL_STRATEGIES:
        effective, structure_off = ChunkingStrategy.SEMANTIC.value, True
    else:
        effective = str(requested)

    if effective != str(requested) and (str(requested), effective) not in _WARNED_OVERRIDES:
        # Once per (requested, effective) pair: it is called per document, and
        # a corpus ingest printed this thousands of times.
        _WARNED_OVERRIDES.add((str(requested), effective))
        # loguru formats with braces, not %-style: a "%r" here renders
        # literally and the names never appear.
        if structure_off:
            setting = (
                "enable_hierarchical"
                if str(requested) == ChunkingStrategy.HIERARCHICAL.value
                else "preserve_headers or preserve_tables"
            )
            logger.warning(
                f"chunking_strategy={str(requested)!r} needs {setting}, which is off, "
                f"so 'semantic' runs. Name the strategy you want, or turn the setting on."
            )
        else:
            logger.warning(
                f"chunking_strategy={str(requested)!r} is being overridden by "
                f"structural settings; {effective!r} will run instead. Set "
                f"enable_hierarchical=False and preserve_headers=preserve_tables=False "
                f"for the requested strategy to take effect. Two configs differing "
                f"only in chunking_strategy produce identical chunks under these "
                f"settings."
            )
    return effective


def create_chunker(
    strategy: Optional[str] = None, embedding_manager=None, config: Optional[RAGConfig] = None
) -> SemanticChunker:
    """Factory function to create appropriate chunker.

    Args:
        strategy: Chunking strategy to use (from config if not specified)
        embedding_manager: Embedding manager for semantic chunking

    Returns:
        Configured chunker instance
    """
    config = config or rag_config
    # Resolves and warns when a structural flag shadows the request.
    effective = resolve_effective_strategy(config, strategy)

    chunker = SemanticChunker(embedding_manager=embedding_manager, config=config)
    # Carried on the instance so a caller can see what it actually got. The
    # previous version logged the requested strategy and returned a chunker
    # that ignored it, which made the log actively misleading.
    chunker.effective_strategy = effective

    logger.info("Created chunker; effective strategy: %s", effective)
    return chunker


def adapt_component_chunks(chunks: Any, text: str) -> List["Chunk"]:
    """Convert a ``BaseChunker``'s output into the internal chunk shape.

    The public contract returns ``base_components.Chunk``; the ingest path needs
    the subclass in this module, whose only addition is ``stable_id`` -- a
    content hash used for citation tracking and de-duplication. Recomputing it
    here rather than requiring component authors to produce one keeps the public
    ABC as small as it looks.

    ``start_char``/``end_char`` are filled in by locating the chunk in the source
    when a component leaves them at zero. Citations carry character spans, so a
    component that only returns text would otherwise produce chunks that cannot
    be highlighted -- silently, and only in the UI.
    """
    adapted: List[Chunk] = []
    cursor = 0
    for i, c in enumerate(chunks):
        content = getattr(c, "content", None)
        if content is None:
            raise TypeError(
                f"Chunker returned {type(c).__name__} at index {i}; a BaseChunker "
                f"must return objects with a `content` attribute (see "
                f"lasagna.core.base_components.Chunk)."
            )

        start = getattr(c, "start_char", 0) or 0
        end = getattr(c, "end_char", 0) or 0
        if not end:
            # Locate from the cursor so repeated text maps to successive
            # occurrences rather than all to the first.
            found = text.find(content, cursor)
            if found < 0:
                # Fall back to searching from the start: a component may return
                # chunks out of document order.
                found = text.find(content)
            if found >= 0:
                start, end = found, found + len(content)
                cursor = end
            else:
                # The chunk is not a substring of the source -- a summarising or
                # rewriting chunker. Leave the span empty rather than inventing
                # one: (0, len(content)) would point a citation at unrelated
                # text near the top of the document, and highlight it
                # confidently.
                start, end = 0, 0

        adapted.append(
            Chunk(
                content=content,
                chunk_index=getattr(c, "chunk_index", i) or i,
                metadata=dict(getattr(c, "metadata", {}) or {}),
                page_number=getattr(c, "page_number", None),
                start_char=start,
                end_char=end,
                section_header=getattr(c, "section_header", None),
                parent_chunk_id=getattr(c, "parent_chunk_id", None),
                chunk_type=getattr(c, "chunk_type", "text") or "text",
            )
        )
    return adapted


def chunk_document(
    text: str,
    document_id: str,
    session_id: str,
    filename: str,
    embedding_manager=None,
    config: Optional[RAGConfig] = None,
    return_parents: bool = False,
    chunker: Any = None,
) -> Union[
    List[Tuple[str, Dict[str, Any]]], Tuple[List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]]]
]:
    """Chunk a document with full metadata.

    Convenience function that creates chunks ready for storage.

    Args:
        text: Document text
        document_id: Document ID
        session_id: Session ID
        filename: Original filename
        embedding_manager: Optional embedding manager
        config: Optional RAG config override
        return_parents: If True and hierarchical chunking is enabled,
            returns (child_tuples, parent_dicts) instead of just child_tuples.

    Returns:
        If return_parents=False: List of (chunk_text, metadata) tuples (children only).
        If return_parents=True and hierarchical enabled:
            Tuple of (child_tuples, parent_dicts) where parent_dicts contain
            {stable_id, document_id, session_id, content, chunk_index, start_char, end_char, metadata}.
    """
    config = config or rag_config
    strategy = config.chunking_strategy

    # A registered component replaces the built-in strategies outright. It is
    # not merged with them: `enable_hierarchical` and `preserve_*` describe how
    # the built-in chunker should behave, and silently applying them on top of
    # someone else's algorithm would mean the thing under test is not the thing
    # that ran -- the exact failure that made chunking ablations meaningless.
    component = chunker
    if component is not None:
        from lasagna.core.base_components import BaseChunker

        if not isinstance(component, BaseChunker):
            raise TypeError(
                f"chunker must implement BaseChunker, got {type(component).__name__}. "
                f"Run `lasagna component validate <module:Class>` to check."
            )
        logger.info(f"Chunking with injected component: {component.name}")
    else:
        resolve_effective_strategy(config, strategy)
    # The arm's config, not the process-wide one: the strategies below read
    # chunk_size and friends from it, and with the global config two arms
    # sweeping chunk_size produced the same chunks. Only the built-in
    # strategies need it (and its NLTK check).
    chunker = (
        SemanticChunker(embedding_manager=embedding_manager, config=config)
        if component is None
        else None
    )

    parent_chunks_data: Optional[List[Dict[str, Any]]] = None

    def _line_number_at(idx: int) -> int:
        return text.count("\n", 0, idx) + 1

    # Component branch first: it is the whole point of injecting one.
    if component is not None:
        # Built-ins read the arm's settings (chunk_size, the breakpoint method);
        # a user's chunker is shown only the public contract.
        extra = {"config": config} if getattr(component, "_forwards_session_config", False) else {}
        chunks = adapt_component_chunks(component.chunk(text, **extra), text)
    # Hierarchical chunking: parent-child structure for auto-merge retrieval
    elif config.enable_hierarchical:
        hierarchical_chunker = HierarchicalChunker(chunker)
        parent_chunks, chunks = hierarchical_chunker.chunk_hierarchical(
            text,
            parent_size=config.parent_chunk_size,
            child_size=config.child_chunk_size,
        )

        if return_parents:
            # Build parent records for SQLite storage
            # Pre-count children per parent (O(C) instead of O(P*C))
            from collections import Counter

            children_per_parent = Counter(c.parent_chunk_id for c in chunks)

            parent_chunks_data = []
            for parent in parent_chunks:
                parent_chunks_data.append(
                    {
                        "stable_id": parent.stable_id,
                        "document_id": document_id,
                        "session_id": session_id,
                        "content": parent.content,
                        "chunk_index": parent.chunk_index,
                        "start_char": parent.start_char,
                        "end_char": parent.end_char,
                        "metadata": {
                            "filename": filename,
                            "chunk_type": "parent",
                            "child_count": children_per_parent.get(parent.stable_id, 0),
                            "line_start": _line_number_at(parent.start_char),
                            "line_end": _line_number_at(parent.end_char),
                        },
                    }
                )
    # Standard chunking strategies (non-hierarchical)
    elif config.preserve_headers or config.preserve_tables:
        structure_chunker = StructureAwareChunker(chunker)
        chunks = structure_chunker.chunk_with_structure(text)
    elif (
        strategy == ChunkingStrategy.SEMANTIC or str(strategy) in _STRUCTURAL_STRATEGIES
    ) and embedding_manager:
        # A structural strategy with its structure off; see _STRUCTURAL_STRATEGIES.
        chunks = chunker.chunk_semantic_configured(text)
    elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
        chunks = chunker.chunk_sentence_window(text)
    elif strategy == ChunkingStrategy.RECURSIVE:
        # It used to fall through to fixed-size chunking under this name, and
        # was refused until there was a real implementation to run.
        from .splitters import RecursiveChunker

        chunks = adapt_component_chunks(RecursiveChunker(config=config).chunk(text), text)
    else:
        chunks = chunker.chunk_fixed(text)

    # Convert to tuples with metadata
    result = []
    for chunk in chunks:
        start_char = chunk.start_char
        end_char = chunk.end_char
        line_start = _line_number_at(start_char)
        line_end = _line_number_at(end_char)

        metadata = {
            "document_id": document_id,
            "session_id": session_id,
            "filename": filename,
            "chunk_index": chunk.chunk_index,
            "total_chunks": len(chunks),
            "stable_id": chunk.stable_id,
            "chunk_type": chunk.chunk_type,
            "start_char": start_char,
            "end_char": end_char,
            "line_start": line_start,
            "line_end": line_end,
        }

        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        if chunk.section_header:
            metadata["section_header"] = chunk.section_header
        if chunk.parent_chunk_id:
            metadata["parent_chunk_id"] = chunk.parent_chunk_id
        if chunk.metadata.get("total_siblings"):
            metadata["total_siblings"] = chunk.metadata["total_siblings"]

        result.append((chunk.content, metadata))

    if return_parents and parent_chunks_data is not None:
        return result, parent_chunks_data

    return result


# ---------------------------------------------------------------------------
# ABC-compliant adapters
#
# These adapters bridge between the public ``BaseChunker`` ABC (which uses
# ``chunk(text) -> List[Chunk]``) and the internal chunker classes (which use
# method names like ``chunk_semantic()``, ``chunk_hierarchical()``, etc.).
# ---------------------------------------------------------------------------

from lasagna.core.base_components import BaseChunker, Chunk as BaseChunk  # noqa: E402


class SemanticChunkerAdapter(BaseChunker):
    """Semantic chunking: split where adjacent sentences' embeddings diverge.

    ``semantic_breakpoint_method`` picks the rule. ``percentile``,
    ``standard_deviation``, ``interquartile`` and ``gradient`` are Kamradt's
    (G. Kamradt, "5 Levels of Text Splitting", 2024; LangChain's
    ``SemanticChunker``): embed each sentence with ``semantic_buffer_size``
    neighbours, and break where the cosine distance to the next exceeds the
    method's threshold over the document, set by ``semantic_breakpoint_amount``
    (95th percentile, 3 standard deviations, 1.5 IQR, 95th percentile of the
    gradient). ``absolute``, the default for continuity with earlier results,
    breaks where similarity falls below ``semantic_threshold`` while the chunk is
    within ``min_chunk_size``..``max_chunk_size`` characters.

    Embeds with the arm's ``embedding_model`` unless an ``embedding_manager`` is
    given, loading it once per process. Without a model to load it raises; it
    no longer falls back to fixed-size chunking under this name.
    """

    # Reads the arm's breakpoint method, thresholds and embedding model.
    _forwards_session_config = True

    def __init__(self, embedding_manager=None, config=None, **kwargs):
        self._embedding_manager = embedding_manager
        self._config = config

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        if not text or not text.strip():
            return []
        config = kwargs.get("config") or self._config or rag_config
        manager = self._embedding_manager
        if manager is None:
            from lasagna.infrastructure.embedding.manager import shared_embedding_manager

            manager = shared_embedding_manager(config)
        inner = SemanticChunker(embedding_manager=manager, config=config)
        method = kwargs.get("breakpoint_method") or config.semantic_breakpoint_method
        if method != "absolute":
            return inner.chunk_semantic_breakpoints(
                text,
                method=method,
                amount=kwargs.get("breakpoint_amount"),
                buffer_size=kwargs.get("buffer_size"),
            )
        # Only forward kwargs accepted by chunk_semantic, under its own names:
        # passing min_chunk_size through as-is raised TypeError.
        accepted = {}
        if "threshold" in kwargs:
            accepted["threshold"] = kwargs["threshold"]
        if "min_chunk_size" in kwargs:
            accepted["min_size"] = kwargs["min_chunk_size"]
        if "max_chunk_size" in kwargs:
            accepted["max_size"] = kwargs["max_chunk_size"]
        return inner.chunk_semantic(text, **accepted)


class HierarchicalChunkerAdapter(BaseChunker):
    """Adapts ``HierarchicalChunker`` to the public ``BaseChunker`` ABC.

    Returns child chunks (the searchable units) from the hierarchical
    parent-child structure.  Use :meth:`chunk_with_parents` if you also
    need the parent chunks for context merging.
    """

    _ACCEPTED_KWARGS = frozenset({"parent_size", "child_size"})

    # Reads the arm's parent_chunk_size and child_chunk_size.
    _forwards_session_config = True

    def __init__(self, semantic_chunker=None, config=None, **kwargs):
        self._fixed_chunker = semantic_chunker is not None
        sc = semantic_chunker or SemanticChunker(config=config)
        self._inner = HierarchicalChunker(sc)

    def _filter_kwargs(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k in self._ACCEPTED_KWARGS}

    def _for(self, kwargs: dict) -> HierarchicalChunker:
        config = kwargs.get("config")
        if config is None or self._fixed_chunker:
            return self._inner
        return HierarchicalChunker(SemanticChunker(config=config))

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        _parents, children = self._for(kwargs).chunk_hierarchical(
            text, **self._filter_kwargs(kwargs)
        )
        return children

    def chunk_with_parents(self, text: str, **kwargs) -> Tuple[List[BaseChunk], List[BaseChunk]]:
        """Chunk text and return both parent and child chunks.

        Returns:
            Tuple of (parent_chunks, child_chunks).
        """
        return self._for(kwargs).chunk_hierarchical(text, **self._filter_kwargs(kwargs))


class SentenceWindowChunkerAdapter(BaseChunker):
    """Sliding windows of whole sentences: the ``sentence_window`` strategy.

    Each chunk is ``window_size`` consecutive sentences, and consecutive chunks
    share ``sentence_overlap`` of them. Sentences come from NLTK's Punkt
    tokenizer when its data is installed, else a regex on terminal punctuation.
    The chunk text joins its sentences with single spaces; ``start_char`` and
    ``end_char`` span them in the source.

    Not LlamaIndex's ``SentenceWindowNodeParser``, which indexes single
    sentences and swaps in their neighbours after retrieval.

    Settings: ``window_size``, ``sentence_overlap``.
    """

    # Reads the arm's window_size and sentence_overlap.
    _forwards_session_config = True

    def __init__(
        self,
        config=None,
        window_size: Optional[int] = None,
        sentence_overlap: Optional[int] = None,
        **kwargs,
    ):
        self._config = config
        self._window_size = window_size
        self._overlap = sentence_overlap

    @property
    def name(self) -> str:
        return "sentence_window"

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        inner = SemanticChunker(config=kwargs.get("config") or self._config)
        return inner.chunk_sentence_window(
            text, window_size=self._window_size, overlap=self._overlap
        )


class StructureAwareChunkerAdapter(BaseChunker):
    """Adapts ``StructureAwareChunker`` to the public ``BaseChunker`` ABC.

    Tables become chunks of their own and headers mark sections; the text in
    between is chunked semantically, as on the built-in path, with the arm's
    embedding model. Without an ``embedding_manager`` it loads that model
    (once per process) rather than chunking the text by size, which is what it
    did unannounced when built with no arguments.
    """

    # Reads the arm's table, semantic and embedding settings.
    _forwards_session_config = True

    def __init__(self, semantic_chunker=None, config=None, embedding_manager=None, **kwargs):
        self._semantic_chunker = semantic_chunker
        self._config = config
        self._embedding_manager = embedding_manager

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        if not text or not text.strip():
            return []
        sc = self._semantic_chunker
        if sc is None:
            config = kwargs.get("config") or self._config or rag_config
            manager = self._embedding_manager
            if manager is None:
                from lasagna.infrastructure.embedding.manager import shared_embedding_manager

                manager = shared_embedding_manager(config)
            sc = SemanticChunker(embedding_manager=manager, config=config)
        return StructureAwareChunker(sc).chunk_with_structure(text)
