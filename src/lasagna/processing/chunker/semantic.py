"""Semantic Chunking for RAG System - SOTA Implementation.

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


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    HIERARCHICAL = "hierarchical"


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

    def __init__(self, embedding_manager=None):
        """Initialize semantic chunker.

        Args:
            embedding_manager: Optional embedding manager for semantic splitting
        """
        self.embedding_manager = embedding_manager

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

        threshold = threshold or rag_config.semantic_threshold
        min_size = min_size or rag_config.min_chunk_size
        max_size = max_size or rag_config.max_chunk_size

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [Chunk(content=sentences[0], chunk_index=0)]

        # Get embeddings for all sentences
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.embedding_manager.encode_documents_batch(sentences, show_progress=False)

        chunks = []
        current_chunk = []
        current_start = 0

        # Pre-compute sentence start positions for accurate char tracking
        sentence_starts = []
        pos = 0
        for sentence in sentences:
            start = text.find(sentence, pos)
            if start == -1:
                start = pos  # fallback if sentence was modified by tokenizer
            sentence_starts.append(start)
            pos = start + len(sentence)

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
                    chunk_end_char = sentence_starts[i] + len(sentences[i])
                    chunks.append(
                        Chunk(
                            content=chunk_text.strip(),
                            chunk_index=len(chunks),
                            start_char=current_start,
                            end_char=chunk_end_char,
                        )
                    )
                # Next chunk starts after this sentence's end in the original text
                if i + 1 < len(sentences):
                    current_start = sentence_starts[i + 1]
                current_chunk = []

        logger.info(f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks

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
        chunk_size = chunk_size if chunk_size is not None else rag_config.chunk_size
        overlap = overlap if overlap is not None else rag_config.chunk_overlap

        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

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

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_index=len(chunks),
                        start_char=start,
                        end_char=end,
                    )
                )

            # Move with overlap
            start = end - overlap
            # Prevent backward movement (infinite loop guard)
            if chunks and start <= chunks[-1].start_char:
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
        window_size = window_size or rag_config.window_size
        overlap = overlap or rag_config.sentence_overlap

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

        chunks = []
        i = 0
        step = window_size - overlap

        # Ensure we always make progress
        if step <= 0:
            step = 1

        while i < len(sentences):
            window = sentences[i : i + window_size]
            chunk_content = " ".join(window)

            if chunk_content.strip():
                chunks.append(
                    Chunk(
                        content=chunk_content.strip(),
                        chunk_index=len(chunks),
                        metadata={"sentence_start": i, "sentence_end": i + len(window)},
                    )
                )

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
        parent_size = parent_size if parent_size is not None else rag_config.parent_chunk_size
        child_size = child_size if child_size is not None else rag_config.child_chunk_size

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
                if rag_config.preserve_tables:
                    section_text, table_chunks = self._extract_tables(
                        section_text, len(chunks), current_header, page_number
                    )
                    for table_chunk in table_chunks:
                        table_chunk.start_char += section_offset
                        table_chunk.end_char += section_offset
                    chunks.extend(table_chunks)

                # Chunk remaining text
                text_chunks = self.semantic_chunker.chunk_semantic(section_text)

                for chunk in text_chunks:
                    chunk.chunk_index = len(chunks)
                    chunk.section_header = current_header
                    chunk.page_number = page_number
                    chunk.start_char += section_offset
                    chunk.end_char += section_offset
                    chunks.append(chunk)

        logger.info(f"Structure-aware chunking: {len(chunks)} chunks")
        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text by header markers."""
        sections = []
        current_pos = 0
        current_header = None

        for match in self.HEADER_PATTERN.finditer(text):
            # Add content before this header
            if match.start() > current_pos:
                content = text[current_pos : match.start()].strip()
                if content:
                    sections.append(
                        {"content": content, "header": current_header, "start": current_pos}
                    )

            current_header = match.group(2).strip()
            current_pos = match.end()

        # Add remaining content
        if current_pos < len(text):
            content = text[current_pos:].strip()
            if content:
                sections.append(
                    {"content": content, "header": current_header, "start": current_pos}
                )

        return sections if sections else [{"content": text, "header": None, "start": 0}]

    def _split_by_pages(self, text: str) -> List[Dict[str, Any]]:
        """Split text into page sections based on [Page N] markers."""
        matches = list(self.PAGE_MARKER_PATTERN.finditer(text))
        if not matches:
            return [{"content": text, "page_number": None, "start": 0}]

        sections = []

        # Capture any leading text before the first marker
        if matches[0].start() > 0:
            preface = text[: matches[0].start()].strip()
            if preface:
                sections.append({"content": preface, "page_number": None, "start": 0})

        for i, match in enumerate(matches):
            page_number = int(match.group(1))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
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
                    end_char=match.end(),
                )
            )

        # Remove tables in reverse order to preserve earlier match offsets
        remaining_text = text
        for match in reversed(matches):
            remaining_text = (
                remaining_text[: match.start()] + "\n\n" + remaining_text[match.end() :]
            )

        return remaining_text.strip(), table_chunks


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

    Raises on ``recursive``: the enum accepts it, the configurator documents it
    as "splits by separators", and the dispatch falls through to fixed-size
    chunking. Running one algorithm under another's name is the one option worse
    than not offering it.
    """
    requested = requested or config.chunking_strategy
    if str(requested) == ChunkingStrategy.RECURSIVE.value:
        raise ValueError(
            "chunking_strategy='recursive' is declared but not implemented -- it "
            "would silently run fixed-size chunking. Use 'fixed', 'semantic' or "
            "'sentence_window'."
        )

    if config.enable_hierarchical:
        effective = ChunkingStrategy.HIERARCHICAL.value
    elif config.preserve_headers or config.preserve_tables:
        effective = "structure_aware"
    else:
        effective = str(requested)

    if effective != str(requested) and (str(requested), effective) not in _WARNED_OVERRIDES:
        # Once per (requested, effective) pair: it is called per document, and
        # a corpus ingest printed this thousands of times.
        _WARNED_OVERRIDES.add((str(requested), effective))
        # loguru formats with braces, not %-style: a "%r" here renders
        # literally and the names never appear.
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
    # Resolves and warns; also rejects 'recursive' rather than quietly running
    # fixed-size chunking under that name.
    effective = resolve_effective_strategy(config, strategy)

    chunker = SemanticChunker(embedding_manager=embedding_manager)
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
    chunker = SemanticChunker(embedding_manager=embedding_manager)

    parent_chunks_data: Optional[List[Dict[str, Any]]] = None

    def _line_number_at(idx: int) -> int:
        return text.count("\n", 0, idx) + 1

    # Component branch first: it is the whole point of injecting one.
    if component is not None:
        chunks = adapt_component_chunks(component.chunk(text), text)
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
    elif strategy == ChunkingStrategy.SEMANTIC and embedding_manager:
        chunks = chunker.chunk_semantic(text)
    elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
        chunks = chunker.chunk_sentence_window(text)
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
    """Adapts ``SemanticChunker`` to the public ``BaseChunker`` ABC."""

    def __init__(self, embedding_manager=None, **kwargs):
        self._inner = SemanticChunker(embedding_manager=embedding_manager)

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        # Only forward kwargs accepted by chunk_semantic
        accepted = {}
        if "threshold" in kwargs:
            accepted["threshold"] = kwargs["threshold"]
        if "min_chunk_size" in kwargs:
            accepted["min_chunk_size"] = kwargs["min_chunk_size"]
        if "max_chunk_size" in kwargs:
            accepted["max_chunk_size"] = kwargs["max_chunk_size"]
        return self._inner.chunk_semantic(text, **accepted)


class HierarchicalChunkerAdapter(BaseChunker):
    """Adapts ``HierarchicalChunker`` to the public ``BaseChunker`` ABC.

    Returns child chunks (the searchable units) from the hierarchical
    parent-child structure.  Use :meth:`chunk_with_parents` if you also
    need the parent chunks for context merging.
    """

    _ACCEPTED_KWARGS = frozenset({"parent_size", "child_size"})

    def __init__(self, semantic_chunker=None, **kwargs):
        sc = semantic_chunker or SemanticChunker()
        self._inner = HierarchicalChunker(sc)

    def _filter_kwargs(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k in self._ACCEPTED_KWARGS}

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        _parents, children = self._inner.chunk_hierarchical(text, **self._filter_kwargs(kwargs))
        return children

    def chunk_with_parents(self, text: str, **kwargs) -> Tuple[List[BaseChunk], List[BaseChunk]]:
        """Chunk text and return both parent and child chunks.

        Returns:
            Tuple of (parent_chunks, child_chunks).
        """
        return self._inner.chunk_hierarchical(text, **self._filter_kwargs(kwargs))


class StructureAwareChunkerAdapter(BaseChunker):
    """Adapts ``StructureAwareChunker`` to the public ``BaseChunker`` ABC."""

    def __init__(self, semantic_chunker=None, **kwargs):
        sc = semantic_chunker or SemanticChunker()
        self._inner = StructureAwareChunker(sc)

    def chunk(self, text: str, **kwargs) -> List[BaseChunk]:
        # chunk_with_structure() accepts no extra parameters
        return self._inner.chunk_with_structure(text)
