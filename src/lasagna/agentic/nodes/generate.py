"""Generate Node — structured LLM answer generation with citation tracking.

This node replaces the ReAct agent's free-form ``AIMessage`` with a
deterministic generation step.  It formats graded (or refined) chunks as
numbered context, calls the LLM with a grounded-generation prompt, extracts
``[N]`` citation references from the response, and records timing.

Owned state fields (write):
    ``generation``, ``agent_summary``, ``generation_citations``,
    ``timing_spans``
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from ..prompts import resolve_prompt


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

_GENERATION_PROMPT = """\
You are a knowledge-retrieval assistant.  Answer the user's question using
ONLY the numbered documents provided below.  Follow these rules strictly:

1. Base your answer ONLY on the provided documents — do NOT use outside knowledge.
2. Cite sources using [N] notation matching the document numbers (e.g. [1], [2]).
3. Place each citation immediately after the sentence or phrase it supports.
4. Include specific facts, numbers, and direct quotes where available.
5. If the documents do NOT contain enough information to answer, say so explicitly
   and do NOT invent citations.

## User Question
{question}

## Documents
{context}

## Answer (with [N] citations):"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_chunks_as_context(chunks: List[Any]) -> str:
    """Format a list of ``RetrievedChunkData`` objects as numbered context.

    Each chunk is rendered as::

        [N]
        <content>

    separated by blank lines.
    """
    sections: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        content = getattr(chunk, "content", str(chunk))
        sections.append(f"[{idx}]\n{content}")
    return "\n\n".join(sections)


def _extract_citation_refs(text: str) -> List[str]:
    """Extract unique ``[N]`` citation references from *text*."""
    return list(dict.fromkeys(re.findall(r"\[(\d+)\]", text)))


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def generate_node(state: RAGGraphState) -> Dict[str, Any]:
    """Generate a grounded answer from retrieved chunks.

    Reads ``graded_chunks`` (preferred) or ``refined_chunks`` as the evidence
    pool, formats them for the LLM, invokes generation, and extracts citation
    references.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``generation``, ``agent_summary``,
        ``generation_citations``, ``timing_spans``.
    """
    llm = state.get("llm")
    question = state.get("original_question", "")

    # Prefer graded chunks (post-CRAG), fall back to refined, then retrieved.
    chunks = (
        state.get("graded_chunks")
        or state.get("refined_chunks")
        or state.get("retrieved_chunks")
        or []
    )

    if not chunks:
        logger.warning("[Generate] No chunks available for generation")
        return {
            "generation": "I could not find relevant information in the knowledge base to answer this question.",
            "agent_summary": "",
            "generation_citations": [],
        }

    context_text = _format_chunks_as_context(chunks)
    prompt = resolve_prompt("generate", _GENERATION_PROMPT, state.get("session_config")).format(
        question=question, context=context_text
    )

    gen_start = time.time()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        generation_text: str = response.content.strip()
    except Exception:
        logger.exception("[Generate] LLM invocation failed")
        generation_text = "Generation failed due to an internal error."

    gen_ms = (time.time() - gen_start) * 1000

    citation_refs = _extract_citation_refs(generation_text)

    logger.info(
        "[Generate] Generated answer with {n_cites} citation(s) in {ms:.1f} ms",
        n_cites=len(citation_refs),
        ms=gen_ms,
    )

    # Merge timing into existing spans.
    timing_spans: Dict[str, float] = dict(state.get("timing_spans", {}))
    timing_spans["generation_ms"] = gen_ms

    return {
        "generation": generation_text,
        "agent_summary": generation_text,
        "generation_citations": citation_refs,
        "timing_spans": timing_spans,
    }
