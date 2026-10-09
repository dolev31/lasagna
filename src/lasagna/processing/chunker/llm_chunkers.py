"""Chunkers that call an LLM at ingestion: propositions, and contextual chunks.

Both need an LLM and raise without one. Neither falls back to plain chunks:
an arm named for a technique that quietly indexed something else would score
as the control under the technique's name.
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.core.base_components import BaseChunker, Chunk

from .splitters import RecursiveChunker, _setting


def _chat_model(component: str, model: Optional[str], max_tokens: int) -> Any:
    """The configured LLM at temperature 0, or RuntimeError naming the component."""
    from lasagna.providers import get_llm_provider

    try:
        llm = get_llm_provider().get_llm(model_name=model, temperature=0.0, max_tokens=max_tokens)
    except RuntimeError as exc:
        raise RuntimeError(f"{component} needs an LLM, and none is configured ({exc})") from exc
    if llm is None:
        raise RuntimeError(f"{component} needs an LLM, and none is configured")
    return llm


def _complete(llm: Any, prompt: str) -> str:
    """The text of one completion, from a LangChain chat model or a ``generate`` callable."""
    if hasattr(llm, "invoke"):
        out = llm.invoke(prompt)
    elif hasattr(llm, "generate"):
        out = llm.generate(prompt)
    else:
        raise TypeError(f"{type(llm).__name__} has neither invoke() nor generate()")
    return str(getattr(out, "content", out)).strip()


# ---------------------------------------------------------------------------
# Propositions (Dense X Retrieval)
# ---------------------------------------------------------------------------

#: The prompt Chen et al. used to have GPT-4 write the propositions their
#: Propositionizer was trained on (arXiv:2312.06648, appendix), with its
#: one-shot example.
PROPOSITION_PROMPT = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Input: Title: Ēostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare's scratch or form and a lapwing's nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare's scratch or form and a lapwing's nest look very similar.", "Both hares and lapwing's nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America." ]

Input: {input}
Output:"""


def parse_propositions(output: str) -> List[str]:
    """The list of strings in an LLM's answer, which may be fenced or prefixed.

    Raises ValueError when there is no list of strings to read: an unreadable
    answer is a failure to report, not a passage to index in its place.
    """
    start, end = output.find("["), output.rfind("]")
    if start != -1 and end > start:
        blob = output[start : end + 1]
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(blob)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return [v.strip() for v in value if v.strip()]
    raise ValueError(f"expected a JSON list of propositions, got: {output[:200]!r}")


class PropositionChunker(BaseChunker):
    """Propositions as retrieval units: Dense X Retrieval (Chen et al., 2023).

    The document is split into passages of ``proposition_passage_size`` words
    (100, the passage length of the paper's Wikipedia corpus), and an LLM
    rewrites each passage as a list of propositions: atomic, self-contained
    statements with pronouns resolved, each one chunk. The prompt is the one
    the authors gave GPT-4 to produce the Propositionizer's training data; the
    paper then distilled it into Flan-T5-large, which this does not use.

    A proposition is rewritten text, so it is not a slice of the source: its
    ``start_char``/``end_char`` span the passage it came from, and
    ``metadata["passage_index"]`` names it. Title and section, when passed as
    ``chunk(text, title=..., section=...)``, are put in the prompt as the paper
    formats its input.

    Settings: ``proposition_model`` (None for the configured default LLM),
    ``proposition_passage_size``. Reference: arXiv:2312.06648.
    """

    # Reads the arm's model and passage size.
    _forwards_session_config = True

    def __init__(
        self,
        config: Any = None,
        llm: Any = None,
        passage_size: Optional[int] = None,
        **kwargs: Any,
    ):
        self._config = config
        self._llm = llm
        self._llms: Dict[Optional[str], Any] = {}
        self._passage_size = passage_size
        # One answer per prompt per instance, so re-chunking is deterministic
        # and an ingest retried after a failure does not pay twice.
        self._cache: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "proposition"

    def _resolve_llm(self, config: Any) -> Any:
        if self._llm is not None:
            return self._llm
        model = _setting(config, "proposition_model")
        with self._lock:
            if model not in self._llms:
                self._llms[model] = _chat_model("proposition chunker", model, max_tokens=2048)
            return self._llms[model]

    def propositions(self, passage: str, title: str = "", section: str = "", llm: Any = None):
        """The propositions an LLM writes for one passage."""
        fields = []
        if title:
            fields.append(f"Title: {title}.")
        if section:
            fields.append(f"Section: {section}.")
        fields.append(f"Content: {passage}")
        prompt = PROPOSITION_PROMPT.format(input=" ".join(fields))
        llm = llm or self._resolve_llm(self._config)
        key = hashlib.sha256(f"{id(llm)}\n{prompt}".encode()).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        found = parse_propositions(_complete(llm, prompt))
        with self._lock:
            self._cache[key] = found
        return list(found)

    def chunk(self, text: str, **kwargs: Any) -> List[Chunk]:
        if not text or not text.strip():
            return []
        config = kwargs.get("config") or self._config
        llm = self._resolve_llm(config)
        size = int(
            self._passage_size
            if self._passage_size is not None
            else _setting(config, "proposition_passage_size")
        )
        passages = RecursiveChunker(
            config=config, chunk_size=size, chunk_overlap=0, unit="words"
        ).spans(text, config)

        chunks: List[Chunk] = []
        for p_index, (start, end) in enumerate(passages):
            found = self.propositions(
                text[start:end], kwargs.get("title", ""), kwargs.get("section", ""), llm
            )
            for prop_index, prop in enumerate(found):
                chunks.append(
                    Chunk(
                        content=prop,
                        chunk_index=len(chunks),
                        start_char=start,
                        end_char=end,
                        chunk_type="proposition",
                        metadata={"passage_index": p_index, "proposition_index": prop_index},
                    )
                )
        logger.info(f"Propositions: {len(passages)} passages -> {len(chunks)} propositions")
        return chunks


# ---------------------------------------------------------------------------
# Contextual Retrieval (Anthropic)
# ---------------------------------------------------------------------------


class ContextualChunker(BaseChunker):
    """Contextual Retrieval (Anthropic, 2024): each chunk prefixed with LLM context.

    Chunks the document with ``contextual_base_chunker`` (any registered
    chunker; ``recursive`` by default), then asks an LLM, with Anthropic's
    published prompt, for a short context that situates each chunk in the
    whole document, and prepends it: the chunk text is ``context + "\\n\\n" +
    chunk``, so both the embedding and BM25 see it. ``start_char``/``end_char``
    are the base chunk's span in the source; ``metadata["context"]`` holds the
    generated text.

    Anthropic put the whole document in the prompt, with prompt caching to
    make that affordable. ``contextual_max_document_chars`` caps what is sent
    (50,000 characters by default; None sends everything). Do not combine with
    ``enable_contextual_retrieval``, the pipeline flag for the same technique:
    the context would be prepended twice, so that combination is refused.

    Settings: ``contextual_base_chunker``, ``contextual_model``,
    ``contextual_max_document_chars``. Reference:
    https://www.anthropic.com/news/contextual-retrieval
    """

    # Reads the arm's base chunker, model and document cap.
    _forwards_session_config = True

    def __init__(
        self,
        config: Any = None,
        llm: Any = None,
        base_chunker: Any = None,
        **kwargs: Any,
    ):
        self._config = config
        self._llm = llm
        self._base = base_chunker
        # By (model, document cap): the enricher caches each chunk's context.
        self._enrichers: Dict[tuple, Any] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "contextual"

    def _base_chunker(self, config: Any) -> BaseChunker:
        if self._base is not None:
            return self._base
        name = _setting(config, "contextual_base_chunker")
        if name == self.name:
            raise ValueError("contextual_base_chunker cannot be 'contextual' itself")
        from lasagna.core.registry import registry

        return registry.get("chunker", name, config=config)

    def _enricher(self, config: Any) -> Any:
        from lasagna.processing.contextual import ContextualEnricher

        cap = _setting(config, "contextual_max_document_chars")
        model = _setting(config, "contextual_model")
        key = (None if self._llm is not None else model, cap)
        with self._lock:
            if key not in self._enrichers:
                llm = self._llm or _chat_model("contextual chunker", model, max_tokens=256)
                # Strict: a failed call raises rather than leaving the chunk bare.
                self._enrichers[key] = ContextualEnricher(
                    llm_provider=llm, max_document_chars=cap, strict=True
                )
            return self._enrichers[key]

    def chunk(self, text: str, **kwargs: Any) -> List[Chunk]:
        if not text or not text.strip():
            return []
        config = kwargs.get("config") or self._config
        if _setting(config, "enable_contextual_retrieval"):
            raise ValueError(
                "the contextual chunker and enable_contextual_retrieval both prepend "
                "context; with both on every chunk would get it twice. Turn the flag off."
            )
        enricher = self._enricher(config)
        base = self._base_chunker(config)
        extra = {"config": config} if getattr(base, "_forwards_session_config", False) else {}

        chunks: List[Chunk] = []
        for piece in base.chunk(text, **extra):
            context = enricher.context_for(text, piece.content)
            if not context:
                raise RuntimeError(
                    f"contextual chunker: the LLM returned no context for chunk {len(chunks)}"
                )
            metadata = dict(piece.metadata or {})
            metadata.update(context=context, base_chunker=getattr(base, "name", ""))
            chunks.append(
                Chunk(
                    content=f"{context}\n\n{piece.content}",
                    chunk_index=len(chunks),
                    metadata=metadata,
                    page_number=piece.page_number,
                    start_char=piece.start_char,
                    end_char=piece.end_char,
                    section_header=piece.section_header,
                    parent_chunk_id=piece.parent_chunk_id,
                    chunk_type=piece.chunk_type,
                )
            )
        return chunks
