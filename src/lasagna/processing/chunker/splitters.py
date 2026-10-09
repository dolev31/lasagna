"""Size-based chunkers: fixed windows and recursive separator splitting.

The two baselines nearly every chunking paper reports against. Both measure
size in one unit, set by ``chunk_unit``:

* ``chars``: characters, Python ``len``;
* ``words``: runs of non-whitespace;
* ``tokens``: tokens of ``chunk_tokenizer``, a Hugging Face tokenizer name or
  ``tiktoken:<encoding>``. Without a tokenizer, ``tokens`` is refused rather
  than approximated by words: 512 words is not 512 tokens.

Every chunk is a slice of the source text, so ``start_char``/``end_char``
locate it exactly and ``text[start_char:end_char] == chunk.content``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

from lasagna.core.base_components import BaseChunker, Chunk
from lasagna.core.config import rag_config

Span = Tuple[int, int]

#: Accepted values of ``chunk_unit``.
UNITS = ("chars", "words", "tokens")

_WORD = re.compile(r"\S+")


def _setting(config: Any, name: str) -> Any:
    """``name`` from the arm's config, or from the process-wide one."""
    return getattr(config if config is not None else rag_config, name)


class _Tokenizer:
    """Token spans and counts from a fast Hugging Face tokenizer or tiktoken."""

    def __init__(self, name: str):
        self.name = name
        self._hf = None
        self._tiktoken = None
        if name.startswith("tiktoken:"):
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    f"chunk_tokenizer={name!r} needs tiktoken: pip install tiktoken"
                ) from exc
            encoding = name.split(":", 1)[1]
            try:
                self._tiktoken = tiktoken.get_encoding(encoding)
            except ValueError:
                self._tiktoken = tiktoken.encoding_for_model(encoding)
        else:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(name)
            if not getattr(tok, "is_fast", False):
                # Only fast tokenizers report character offsets, and without
                # them a token window cannot be mapped back onto the text.
                raise ValueError(
                    f"chunk_tokenizer={name!r} has no fast tokenizer, so token "
                    f"offsets are unavailable"
                )
            self._hf = tok

    def spans(self, text: str) -> List[Span]:
        """Each token's character span in ``text``, in order."""
        if self._hf is not None:
            enc = self._hf(
                text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
            )
            return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]
        ids = self._tiktoken.encode(text, disallowed_special=())
        decoded, starts = self._tiktoken.decode_with_offsets(ids)
        if decoded != text:
            raise ValueError(f"{self.name} does not round-trip this text; offsets would drift")
        ends = list(starts[1:]) + [len(text)]
        return [(s, e) for s, e in zip(starts, ends) if e > s]

    def count(self, text: str) -> int:
        """Number of tokens in ``text`` (no special tokens)."""
        if self._hf is not None:
            return len(self._hf(text, add_special_tokens=False, verbose=False)["input_ids"])
        return len(self._tiktoken.encode(text, disallowed_special=()))


@lru_cache(maxsize=8)
def load_tokenizer(name: str) -> _Tokenizer:
    """The tokenizer named by ``chunk_tokenizer``, loaded once per process."""
    return _Tokenizer(name)


class _Measure:
    """Length and unit boundaries in one of :data:`UNITS`."""

    def __init__(self, unit: str, tokenizer: Optional[str] = None):
        if unit not in UNITS:
            raise ValueError(f"chunk_unit must be one of {UNITS}, got {unit!r}")
        if unit == "tokens" and not tokenizer:
            raise ValueError(
                "chunk_unit='tokens' needs chunk_tokenizer (a Hugging Face tokenizer "
                "name, or 'tiktoken:<encoding>'); use chunk_unit='words' to count words"
            )
        self.unit = unit
        self._tok = load_tokenizer(tokenizer) if unit == "tokens" else None

    def length(self, text: str) -> int:
        if self.unit == "chars":
            return len(text)
        if self.unit == "words":
            return len(_WORD.findall(text))
        return self._tok.count(text)

    def spans(self, text: str) -> List[Span]:
        """The span of every unit in ``text``."""
        if self.unit == "chars":
            return [(i, i + 1) for i in range(len(text))]
        if self.unit == "words":
            return [m.span() for m in _WORD.finditer(text)]
        return self._tok.spans(text)


def _measure(config: Any, unit: Optional[str], tokenizer: Optional[str]) -> _Measure:
    return _Measure(
        unit or _setting(config, "chunk_unit"),
        tokenizer or _setting(config, "chunk_tokenizer"),
    )


def _to_chunks(text: str, spans: Sequence[Span], **metadata: Any) -> List[Chunk]:
    """Chunks for non-blank spans, numbered in order, each with its own metadata."""
    out: List[Chunk] = []
    for start, end in spans:
        content = text[start:end]
        if not content.strip():
            continue
        out.append(
            Chunk(
                content=content,
                chunk_index=len(out),
                start_char=start,
                end_char=end,
                metadata=dict(metadata),
            )
        )
    return out


def _check_sizes(size: int, overlap: int, *, strict_overlap: bool) -> None:
    if size <= 0:
        raise ValueError(f"chunk_size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap must not be negative, got {overlap}")
    if overlap > size or (strict_overlap and overlap == size):
        raise ValueError(f"chunk_overlap ({overlap}) must be smaller than chunk_size ({size})")


# ---------------------------------------------------------------------------
# Fixed-size windows
# ---------------------------------------------------------------------------


def fixed_windows(n_units: int, size: int, overlap: int) -> List[Span]:
    """Unit-index windows of ``size``, starting every ``size - overlap`` units.

    The last window is the first one to reach the end, as in LangChain's
    ``TokenTextSplitter``: no trailing window lies wholly inside the previous.
    """
    windows: List[Span] = []
    start = 0
    while start < n_units:
        end = min(start + size, n_units)
        windows.append((start, end))
        if end == n_units:
            break
        start += size - overlap
    return windows


class FixedSizeChunker(BaseChunker):
    """Fixed-size windows with overlap, in characters, words or tokens.

    The baseline every chunking study compares against: windows of
    ``chunk_size`` units start every ``chunk_size - chunk_overlap`` units, and no
    boundary moves to a sentence or word end. A character window can cut a word
    in half; that is the method, not a defect. With ``chunk_unit='tokens'`` the
    windows are exact in the configured tokenizer: each holds ``chunk_size``
    tokens of the whole document's tokenization, mapped back to the text through
    the tokenizer's character offsets.

    Settings: ``chunk_size``, ``chunk_overlap``, ``chunk_unit``,
    ``chunk_tokenizer``.
    """

    # Reads the arm's chunk_size, chunk_overlap, chunk_unit and chunk_tokenizer.
    _forwards_session_config = True

    def __init__(
        self,
        config: Any = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        unit: Optional[str] = None,
        tokenizer: Optional[str] = None,
        **kwargs: Any,
    ):
        self._config = config
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._unit = unit
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(self, text: str, **kwargs: Any) -> List[Chunk]:
        config = kwargs.get("config") or self._config
        size = int(self._size if self._size is not None else _setting(config, "chunk_size"))
        overlap = int(
            self._overlap if self._overlap is not None else _setting(config, "chunk_overlap")
        )
        _check_sizes(size, overlap, strict_overlap=True)
        measure = _measure(config, self._unit, self._tokenizer)
        if not text or not text.strip():
            return []

        if measure.unit == "chars":
            spans = fixed_windows(len(text), size, overlap)
        else:
            units = measure.spans(text)
            spans = [
                (units[a][0], units[b - 1][1]) for a, b in fixed_windows(len(units), size, overlap)
            ]
        return _to_chunks(text, spans, unit=measure.unit)


# ---------------------------------------------------------------------------
# Recursive separator splitting
# ---------------------------------------------------------------------------


def _split_on(text: str, start: int, end: int, separator: str, keep: str) -> List[Span]:
    """Split ``text[start:end]`` at every occurrence of ``separator``.

    The separator stays with the piece before it (``keep='end'``) or the piece
    after it (``'start'``), so the pieces tile the span with nothing dropped.
    An empty separator splits into characters.
    """
    if not separator:
        return [(i, i + 1) for i in range(start, end)]
    cuts = [start]
    pos = text.find(separator, start, end)
    while pos != -1:
        cuts.append(pos + len(separator) if keep == "end" else pos)
        pos = text.find(separator, pos + len(separator), end)
    cuts.append(end)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _strip(text: str, span: Span) -> Optional[Span]:
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _merge(text: str, splits: List[Span], measure: _Measure, size: int, overlap: int) -> List[Span]:
    """Pack consecutive pieces into chunks of at most ``size``, carrying ``overlap``.

    LangChain's ``_merge_splits`` with an empty join separator (the separators
    are kept inside the pieces): sizes are summed piece by piece, and after each
    chunk pieces are dropped from the front until at most ``overlap`` remains.
    """
    chunks: List[Span] = []
    current: List[Span] = []
    lengths: List[int] = []
    total = 0
    for piece in splits:
        n = measure.length(text[piece[0] : piece[1]])
        if total + n > size and current:
            stripped = _strip(text, (current[0][0], current[-1][1]))
            if stripped is not None:
                chunks.append(stripped)
            while total > overlap or (total + n > size and total > 0):
                total -= lengths.pop(0)
                current.pop(0)
        current.append(piece)
        lengths.append(n)
        total += n
    if current:
        stripped = _strip(text, (current[0][0], current[-1][1]))
        if stripped is not None:
            chunks.append(stripped)
    return chunks


def recursive_split(
    text: str,
    span: Span,
    separators: Sequence[str],
    measure: _Measure,
    size: int,
    overlap: int,
    keep: str,
) -> List[Span]:
    """LangChain's recursive splitting, on character spans of ``text``.

    Split on the first separator that occurs in the span; pack the pieces
    shorter than ``size`` into chunks; split any piece that is not with the
    remaining, finer separators.
    """
    start, end = span
    separator, finer = separators[-1], []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            break
        if text.find(sep, start, end) != -1:
            separator, finer = sep, list(separators[i + 1 :])
            break

    chunks: List[Span] = []
    small: List[Span] = []
    for piece in _split_on(text, start, end, separator, keep):
        if measure.length(text[piece[0] : piece[1]]) < size:
            small.append(piece)
            continue
        if small:
            chunks.extend(_merge(text, small, measure, size, overlap))
            small = []
        if finer:
            chunks.extend(recursive_split(text, piece, finer, measure, size, overlap, keep))
        else:
            # Nothing finer to split on: the piece goes out as it is, oversized
            # and unstripped, as LangChain emits it.
            chunks.append(piece)
    if small:
        chunks.extend(_merge(text, small, measure, size, overlap))
    return chunks


class RecursiveChunker(BaseChunker):
    """Recursive separator splitting: paragraph, then line, sentence, word, character.

    The algorithm of LangChain's ``RecursiveCharacterTextSplitter``, the most
    widely used chunking baseline: split on the coarsest separator present, pack
    the pieces into chunks of at most ``chunk_size`` with ``chunk_overlap``
    carried between them, and re-split any piece still too large with the next
    separator. With ``recursive_separators=["\\n\\n", "\\n", " ", ""]`` and
    ``recursive_keep_separator='start'`` it reproduces LangChain's defaults
    exactly; the default here adds a sentence level, and keeps each separator at
    the end of its piece so a sentence keeps its full stop.

    Settings: ``chunk_size``, ``chunk_overlap``, ``chunk_unit``,
    ``chunk_tokenizer``, ``recursive_separators``, ``recursive_keep_separator``.
    """

    # Reads the arm's sizes, unit and separators.
    _forwards_session_config = True

    def __init__(
        self,
        config: Any = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[Sequence[str]] = None,
        keep_separator: Optional[str] = None,
        unit: Optional[str] = None,
        tokenizer: Optional[str] = None,
        **kwargs: Any,
    ):
        self._config = config
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._separators = list(separators) if separators is not None else None
        self._keep = keep_separator
        self._unit = unit
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "recursive"

    def spans(self, text: str, config: Any = None) -> List[Span]:
        """The chunk spans of ``text``, for callers that sub-split a section."""
        config = config or self._config
        size = int(self._size if self._size is not None else _setting(config, "chunk_size"))
        overlap = int(
            self._overlap if self._overlap is not None else _setting(config, "chunk_overlap")
        )
        _check_sizes(size, overlap, strict_overlap=False)
        separators = list(
            self._separators
            if self._separators is not None
            else _setting(config, "recursive_separators")
        )
        keep = self._keep or _setting(config, "recursive_keep_separator")
        if keep not in ("start", "end"):
            raise ValueError(f"recursive_keep_separator must be 'start' or 'end', got {keep!r}")
        if not separators:
            raise ValueError("recursive_separators must name at least one separator")
        measure = _measure(config, self._unit, self._tokenizer)
        if not text or not text.strip():
            return []
        return recursive_split(text, (0, len(text)), separators, measure, size, overlap, keep)

    def chunk(self, text: str, **kwargs: Any) -> List[Chunk]:
        return _to_chunks(text, self.spans(text, kwargs.get("config")))
