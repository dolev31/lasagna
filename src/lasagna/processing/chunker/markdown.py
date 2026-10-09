"""Markdown header splitting: one chunk per section, carrying its header path.

Sections are what LangChain's ``MarkdownHeaderTextSplitter`` produces: a
header opens a section, a header of the same or a higher level closes it, and
lines inside fenced code blocks are never headers. Each chunk records the
headers above it (``metadata["header_path"]``), which is the point of the
method: a passage under "Installation > Windows" can be retrieved as such.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from lasagna.core.base_components import BaseChunker, Chunk

from .splitters import RecursiveChunker, _setting, _strip

_HEADER = re.compile(r"(#{1,6})(?:[ \t]+(.*))?$")
_FENCES = ("```", "~~~")


def _lines(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` of every line, line break included."""
    out = []
    pos = 0
    for line in text.splitlines(keepends=True):
        out.append((pos, pos + len(line)))
        pos += len(line)
    return out


class MarkdownHeaderChunker(BaseChunker):
    """Split Markdown on its headers, keeping each chunk's header path.

    LangChain's ``MarkdownHeaderTextSplitter`` semantics: ATX headers (``#`` to
    ``######``, up to ``markdown_header_levels`` deep) open sections; a deeper
    header stays in its section's text; fenced code is never split on. With
    ``markdown_strip_headers`` (LangChain's default) the header line is left out
    of the chunk text, since it is in the metadata; otherwise it opens the chunk,
    and a header with no text of its own joins its first subsection.

    LangChain leaves sections at whatever length they are and has a second
    splitter run after it. With ``markdown_split_long_sections`` (the default) a
    section longer than ``chunk_size`` is split here by the recursive chunker,
    each piece keeping the section's headers; turn it off to get LangChain's
    output, one chunk per section.

    Chunk text is the section's slice of the source with its outer whitespace
    trimmed, so ``start_char``/``end_char`` locate it exactly. (LangChain joins
    the section's stripped lines instead, dropping blank lines and indentation.)

    Metadata: ``header_path`` (the header texts, outermost first) and
    ``headers`` (``{"h1": ..., "h2": ...}``); ``section_header`` is the
    innermost one.

    Settings: ``markdown_header_levels``, ``markdown_strip_headers``,
    ``markdown_split_long_sections``, and for that split ``chunk_size``,
    ``chunk_overlap``, ``chunk_unit``, ``recursive_separators``.
    """

    # Reads the arm's header levels and section-splitting settings.
    _forwards_session_config = True

    def __init__(
        self,
        config: Any = None,
        header_levels: Optional[int] = None,
        strip_headers: Optional[bool] = None,
        split_long_sections: Optional[bool] = None,
        **kwargs: Any,
    ):
        self._config = config
        self._levels = header_levels
        self._strip_headers = strip_headers
        self._split_long = split_long_sections

    @property
    def name(self) -> str:
        return "markdown_header"

    def _resolve(self, config: Any) -> Tuple[int, bool, bool]:
        levels = int(
            self._levels if self._levels is not None else _setting(config, "markdown_header_levels")
        )
        if not 1 <= levels <= 6:
            raise ValueError(f"markdown_header_levels must be 1 to 6, got {levels}")
        strip = (
            self._strip_headers
            if self._strip_headers is not None
            else _setting(config, "markdown_strip_headers")
        )
        split = (
            self._split_long
            if self._split_long is not None
            else _setting(config, "markdown_split_long_sections")
        )
        return levels, bool(strip), bool(split)

    def sections(self, text: str, levels: int, strip_headers: bool) -> List[Tuple[int, int, list]]:
        """``(start, end, [(level, title), ...])`` for every section, in order."""
        out: List[Tuple[int, int, list]] = []
        stack: List[Tuple[int, str]] = []
        body_start = 0
        fence: Optional[str] = None
        for start, end in _lines(text):
            stripped = text[start:end].strip()
            if fence is not None:
                if stripped.startswith(fence):
                    fence = None
                continue
            opener = next((f for f in _FENCES if stripped.startswith(f)), None)
            if opener is not None:
                fence = opener
                continue
            match = _HEADER.match(stripped)
            if match is None or len(match.group(1)) > levels:
                continue
            out.append((body_start, start, list(stack)))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, (match.group(2) or "").strip()))
            body_start = end if strip_headers else start
        out.append((body_start, len(text), list(stack)))

        kept = []
        for start, end, path in out:
            span = _strip(text, (start, end))
            if span is not None:
                kept.append((span[0], span[1], path))
        if not strip_headers:
            kept = self._fold_bare_headers(text, kept)
        return kept

    @staticmethod
    def _fold_bare_headers(text: str, sections: List[Tuple[int, int, list]]) -> list:
        """Join a section that is only its header line to the subsection after it.

        Kept as its own chunk, a bare "# Title" retrieves as a near-empty passage.
        LangChain merges it into the first child section the same way.
        """
        out: list = []
        for start, end, path in sections:
            if out:
                p_start, p_end, p_path = out[-1]
                last_line = text[p_start:p_end].rsplit("\n", 1)[-1].strip()
                if _HEADER.match(last_line) and len(path) > len(p_path):
                    # Chains: "# A" joins "## B", and that joins "### C" below it.
                    out[-1] = (p_start, end, path)
                    continue
            out.append((start, end, path))
        return out

    def chunk(self, text: str, **kwargs: Any) -> List[Chunk]:
        config = kwargs.get("config") or self._config
        levels, strip_headers, split_long = self._resolve(config)
        if not text or not text.strip():
            return []

        splitter = RecursiveChunker(config=config) if split_long else None
        chunks: List[Chunk] = []
        for start, end, path in self.sections(text, levels, strip_headers):
            pieces = [(start, end)]
            if splitter is not None:
                inner = splitter.spans(text[start:end], config)
                pieces = [(start + a, start + b) for a, b in inner] or pieces
            titles = [title for _, title in path]
            for a, b in pieces:
                chunks.append(
                    Chunk(
                        content=text[a:b],
                        chunk_index=len(chunks),
                        start_char=a,
                        end_char=b,
                        section_header=titles[-1] if titles else None,
                        metadata={
                            "header_path": list(titles),
                            "headers": {f"h{level}": title for level, title in path},
                        },
                    )
                )
        return chunks
