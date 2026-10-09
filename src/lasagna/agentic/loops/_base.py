"""What every published loop shares: its LLM, its search, and its result.

A loop gets three things from the pipeline that runs it:

* **The LLM**: ``AgenticPipelineInput.llm``, the model the orchestrator built
  for this query. ``agentic_models[<loop name>]`` pins another model for one
  loop. A loop constructed by hand with no LLM falls back to the configured
  provider, and raises when there is none: a loop that skipped its reasoning
  would score as plain retrieval under the loop's name.
* **The arm's settings**: ``AgenticPipelineInput.session_config``, the
  session's config with the arm's per-query overrides applied.
* **Search**: ``RAGBackend().search(..., _from_routing=True)`` with those
  per-query overrides, the same retrieval the standard tier runs (hybrid
  search, reranking, fusion, under the arm's settings). An agentic arm then
  measures its loop inside the complete pipeline, never against a private
  retriever. ``_from_routing`` stops the search re-entering the agentic tier.

A loop returns the answer it produced (``agent_summary``) and the passages it
read (``chunks``), so an evaluation judges the answer against what the loop
actually retrieved.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.agentic.agentic_rag import (
    AgenticRAGMetrics,
    AgenticRAGResult,
    HopData,
    RetrievedChunkData,
)
from lasagna.agentic.prompts import resolve_node_model
from lasagna.core.base_components import AgenticPipelineInput, BaseAgenticPipeline


def loop_llm(method: str, pipeline_input: AgenticPipelineInput, config: Any) -> Any:
    """The chat model a loop reasons with, or RuntimeError when there is none."""
    pinned = resolve_node_model(method, config)
    llm = None if pinned else pipeline_input.llm
    if llm is None:
        try:
            from lasagna.providers import get_llm_provider

            llm = get_llm_provider().get_llm(model_name=pinned)
        except Exception as exc:
            raise RuntimeError(f"{method} needs an LLM, and none is configured") from exc
    if llm is None:
        raise RuntimeError(f"{method} needs an LLM, and none is configured")
    return llm


def message_text(response: Any) -> str:
    """The text of a chat-model reply (an AIMessage, or anything printable)."""
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        # Some providers return content blocks; keep the text ones.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def cut_at(text: str, stops: Sequence[str]) -> str:
    """``text`` up to the first stop sequence, for providers that ignore ``stop``."""
    end = len(text)
    for stop in stops:
        i = text.find(stop)
        if i != -1:
            end = min(end, i)
    return text[:end]


#: A candidate sentence end: terminal punctuation, space, then a new sentence.
_BOUNDARY = re.compile(r"[.!?]\s+(?=[A-Z0-9\"'(\[])")
#: Words whose period does not end a sentence ("Dr. Caligari", "St. Louis").
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "no", "mt", "inc", "ltd", "co"}
)


def first_sentence(text: str) -> str:
    """The first sentence of ``text`` (all of it when it is one sentence).

    The papers split with NLTK's Punkt (FLARE) or spaCy (IRCoT); neither is a
    required dependency here, and a split that changed with what happens to be
    installed would change the queries a run sends. This rule is fixed: a
    sentence ends at ``.``, ``!`` or ``?`` before a capitalised word, except
    after an initial ("F.W. Murnau") or a common abbreviation ("Dr. No").
    """
    text = text.strip()
    for match in _BOUNDARY.finditer(text):
        if text[match.start()] == ".":
            before = text[: match.start()].rsplit(None, 1)
            word = before[-1].lower() if before else ""
            if word in _ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)*[a-z]", word):
                continue
        return text[: match.start() + 1].strip()
    return text


def _chunk_from_result(result: Any) -> RetrievedChunkData:
    """A search hit as the agentic layer's chunk record (as the retrieve node builds it)."""
    chunk = result.chunk
    metadata = dict(chunk.metadata or {})
    page = metadata.get("page_number")
    if page is None:
        page = getattr(chunk, "page_number", None)
    document = getattr(result, "document", None)
    return RetrievedChunkData(
        chunk_id=str(chunk.chunk_id),
        document_id=str(chunk.document_id or ""),
        document_name=str(getattr(document, "filename", "") or ""),
        content=chunk.content or "",
        score=float(result.score or 0.0),
        chunk_index=int(getattr(chunk, "chunk_index", 0) or 0),
        page_number=page,
        section_header=metadata.get("section_header"),
        metadata=metadata,
    )


def interleave(lists: Iterable[Sequence[RetrievedChunkData]]) -> List[RetrievedChunkData]:
    """Merge ranked lists round-robin, keeping each chunk's first appearance.

    A multi-step loop's answer draws on every step, so the passages it returns
    lead with each step's best one rather than all of the first step's. An
    evaluation that reads only the first few contexts then still sees the
    evidence for every hop.
    """
    lists = [list(x) for x in lists]
    out: List[RetrievedChunkData] = []
    seen: set = set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for ranked in lists:
            if rank < len(ranked) and ranked[rank].chunk_id not in seen:
                seen.add(ranked[rank].chunk_id)
                out.append(ranked[rank])
    return out


def passage_block(chunks: Sequence[RetrievedChunkData], header: str = "Title") -> str:
    """Passages as the papers lay them out: a title line, then the text."""
    return "\n\n".join(
        f"{header}: {c.document_name or c.document_id or 'untitled'}\n{c.content.strip()}"
        for c in chunks
    )


class LoopRun:
    """One question's run of a loop: its LLM, its searches and their results."""

    def __init__(
        self, method: str, pipeline_input: AgenticPipelineInput, config: Any, llm: Any
    ) -> None:
        self.method = method
        self.input = pipeline_input
        self.config = config
        self.llm = llm
        self.hops: List[HopData] = []
        self.llm_calls = 0
        self.llm_ms = 0.0
        self.trace: Dict[str, Any] = {"method": method, "queries": []}
        self._start = time.time()

    def search(self, query: str, top_k: int) -> List[RetrievedChunkData]:
        """Run the pipeline's own search for ``query``; one hop."""
        from lasagna.api.rag_backend import RAGBackend

        started = time.time()
        results = RAGBackend().search(
            query=query,
            session_id=self.input.session_id,
            top_k=top_k,
            document_ids=self.input.document_filter,
            trace_id=self.input.trace_id,
            _from_routing=True,
            # The arm's settings, as the standard tier would search with them.
            # Without these the loop's searches read the shared session store,
            # which concurrent arms overwrite.
            query_config_overrides=dict(getattr(self.config, "_query_overrides", None) or {})
            or None,
        )
        chunks = [_chunk_from_result(r) for r in results or []]
        self.hops.append(
            HopData(
                hop_index=len(self.hops),
                query_used=query,
                retrieved_chunks=chunks,
                search_latency_ms=(time.time() - started) * 1000,
            )
        )
        self.trace["queries"].append(query)
        logger.debug(f"[{self.method}] hop {len(self.hops) - 1}: {len(chunks)} passages")
        return chunks

    async def ask(self, prompt: str, stop: Optional[Sequence[str]] = None, llm: Any = None) -> Any:
        """One LLM call; returns the raw reply (see ``message_text``)."""
        model = llm if llm is not None else self.llm
        started = time.time()
        if stop:
            response = await model.ainvoke([HumanMessage(content=prompt)], stop=list(stop))
        else:
            response = await model.ainvoke([HumanMessage(content=prompt)])
        self.llm_calls += 1
        self.llm_ms += (time.time() - started) * 1000
        return response

    async def complete(self, prompt: str, stop: Optional[Sequence[str]] = None) -> str:
        """One LLM call; returns its text, cut at the first stop sequence."""
        text = message_text(await self.ask(prompt, stop=stop))
        return cut_at(text, stop) if stop else text

    def result(
        self,
        answer: str,
        evidence: Sequence[RetrievedChunkData],
        stop_reason: str,
        steps: int,
        max_steps: int,
    ) -> AgenticRAGResult:
        """The loop's answer and the passages it read, in the shapes the pipeline reads."""
        chunks: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        sections: List[str] = []
        hop_of = {c.chunk_id: h.hop_index for h in self.hops for c in h.retrieved_chunks}
        for i, c in enumerate(evidence, start=1):
            chunks.append(
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "filename": c.document_name,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "score": c.score,
                    "metadata": dict(c.metadata),
                }
            )
            citations.append(
                {
                    "index": i,
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "chunk_id": c.chunk_id,
                    "content_preview": c.content[:200],
                    "score": round(c.score, 3),
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "hop_index": hop_of.get(c.chunk_id),
                }
            )
            sections.append(f"[{i}]\n{c.content}")

        self.trace.update(stop_reason=stop_reason, steps=steps, llm_calls=self.llm_calls)
        if chunks:
            # Travels with the first result into the pipeline's SearchResult
            # metadata, where an evaluation can read what the loop did.
            chunks[0]["metadata"]["_agentic_loop"] = dict(self.trace)

        total_ms = (time.time() - self._start) * 1000
        config = self.config
        metrics = AgenticRAGMetrics(
            agentic_enabled=True,
            hop_count=len(self.hops),
            agent_iterations_used=steps,
            max_iterations_configured=max_steps,
            total_latency_ms=total_ms,
            hop_search_latencies_ms=[h.search_latency_ms for h in self.hops],
            agent_orchestration_ms=total_ms,
            agent_llm_calls_ms=self.llm_ms,
            accounted_latency_ms=self.llm_ms + sum(h.search_latency_ms for h in self.hops),
            unique_docs_across_hops=len({c.document_id for c in evidence}),
            citation_count=len(evidence),
            stop_reason=stop_reason,
            hybrid_search_enabled=bool(getattr(config, "enable_hybrid_search", False)),
            reranking_enabled=bool(getattr(config, "enable_reranking", False)),
            mmr_enabled=bool(getattr(config, "enable_mmr", False)),
            hyde_enabled=bool(getattr(config, "enable_hyde", False)),
            loop_trace=dict(self.trace),
        )
        context = "\n\n".join(sections)
        return AgenticRAGResult(
            formatted_context=f"<knowledge_base>\n{context}\n</knowledge_base>" if context else "",
            chunks=chunks,
            citations=citations,
            agent_summary=answer.strip(),
            search_count=len(self.hops),
            hops=list(self.hops),
            metrics=metrics,
        )


class LoopPipeline(BaseAgenticPipeline):
    """Base for the published loops: resolves config and LLM, then runs ``_loop``."""

    #: Registry name; also the key for ``agentic_models`` and error messages.
    method: str = ""

    def __init__(self, session_config: Any = None, **kwargs: Any) -> None:
        self._session_config = session_config

    async def run(self, input: AgenticPipelineInput) -> AgenticRAGResult:
        from lasagna.core.config import rag_config

        config = input.session_config or self._session_config or rag_config
        run = LoopRun(self.method, input, config, loop_llm(self.method, input, config))
        return await self._loop(run)

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


def setting(config: Any, key: str) -> Any:
    """A loop setting from the arm's config, else the process config (its defaults)."""
    value = getattr(config, key, None)
    if value is None:
        from lasagna.core.config import rag_config

        value = getattr(rag_config, key)
    return value
