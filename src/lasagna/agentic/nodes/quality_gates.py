"""Post-generation quality gates — hallucination and answer quality checks.

Inspired by Self-RAG (ISSUP / ISUSE grading).  Both nodes are opt-in and
gated by config flags (``enable_hallucination_check`` / ``enable_answer_quality_check``).
When disabled, the graph builder simply skips wiring them.

Each node calls the LLM with a structured grading prompt and returns a
classification label consumed by the ``route_after_quality_check`` edge
function.

Owned state fields (write):
    ``hallucination_check``, ``answer_quality``, ``quality_check_attempts``,
    ``timing_spans``
"""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from ..prompts import resolve_prompt


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_HALLUCINATION_PROMPT = """\
You are a hallucination auditor.  Given a set of source documents and a
generated answer, determine whether every claim in the answer is directly
supported by the source documents.

## Source Documents
{context}

## Generated Answer
{generation}

## Instructions
Classify the answer as ONE of the following (return ONLY the label):

- **supported** — Every factual claim is directly supported by the sources.
- **partially_supported** — Most claims are supported, but some minor details
  are not explicitly found in the sources.
- **not_supported** — The answer contains significant claims that cannot be
  traced to the sources.

Classification:"""


_ANSWER_QUALITY_PROMPT = """\
You are an answer quality evaluator.  Given the user's original question
and the generated answer, determine whether the answer adequately addresses
the question.

## Original Question
{question}

## Generated Answer
{generation}

## Instructions
Classify the answer as ONE of the following (return ONLY the label):

- **addresses_question** — The answer directly and fully addresses the
  question with relevant, specific information.
- **partially_addresses** — The answer touches on the topic but misses key
  aspects or is too vague.
- **does_not_address** — The answer fails to address the question or is
  entirely off-topic.

Classification:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_HALLUCINATION_LABELS = {"supported", "partially_supported", "not_supported"}
_VALID_QUALITY_LABELS = {"addresses_question", "partially_addresses", "does_not_address"}


def _parse_label(raw: str, valid_labels: set, default: str) -> str:
    """Extract a classification label from the LLM response.

    Labels are checked longest-first to avoid substring false positives
    (e.g. ``"not_supported"`` should not match ``"supported"``).
    """
    cleaned = raw.strip().lower().replace("**", "").strip()
    # Sort by length descending so longer labels match before shorter substrings
    for label in sorted(valid_labels, key=len, reverse=True):
        if label in cleaned:
            return label
    return default


def _format_chunks_as_context(chunks: list) -> str:
    """Render chunks as numbered text for grading prompts."""
    sections = []
    for idx, chunk in enumerate(chunks, 1):
        content = getattr(chunk, "content", str(chunk))
        sections.append(f"[{idx}]\n{content}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def check_hallucination_node(state: RAGGraphState) -> Dict[str, Any]:
    """Grade whether the generation is grounded in the source documents (ISSUP).

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``hallucination_check``, ``quality_check_attempts``,
        ``timing_spans``.
    """
    llm = state.get("llm")
    generation = state.get("generation", "")
    chunks = (
        state.get("graded_chunks")
        or state.get("refined_chunks")
        or state.get("retrieved_chunks")
        or []
    )
    quality_check_attempts: int = state.get("quality_check_attempts", 0)

    if not llm or not generation or not chunks:
        logger.warning("[HallucinationCheck] Missing LLM, generation, or chunks: skipping")
        return {
            "hallucination_check": "supported",
            "quality_check_attempts": quality_check_attempts,
        }

    context_text = _format_chunks_as_context(chunks)
    prompt = resolve_prompt(
        "check_hallucination", _HALLUCINATION_PROMPT, state.get("session_config")
    ).format(context=context_text, generation=generation)

    start = time.time()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        label = _parse_label(response.content, _VALID_HALLUCINATION_LABELS, "supported")
    except Exception:
        logger.exception("[HallucinationCheck] LLM call failed: defaulting to 'supported'")
        label = "supported"

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "[HallucinationCheck] Result: {label} ({ms:.1f} ms)",
        label=label,
        ms=elapsed_ms,
    )

    timing_spans: Dict[str, float] = dict(state.get("timing_spans", {}))
    timing_spans["quality_check_ms"] = timing_spans.get("quality_check_ms", 0) + elapsed_ms

    return {
        "hallucination_check": label,
        "quality_check_attempts": quality_check_attempts + 1,
        "timing_spans": timing_spans,
    }


async def check_answer_quality_node(state: RAGGraphState) -> Dict[str, Any]:
    """Grade whether the generation adequately answers the question (ISUSE).

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``answer_quality``, ``quality_check_attempts``,
        ``timing_spans``.
    """
    llm = state.get("llm")
    generation = state.get("generation", "")
    question = state.get("original_question", "")
    quality_check_attempts: int = state.get("quality_check_attempts", 0)

    if not llm or not generation:
        logger.warning("[AnswerQualityCheck] Missing LLM or generation: skipping")
        return {
            "answer_quality": "addresses_question",
            "quality_check_attempts": quality_check_attempts,
        }

    prompt = resolve_prompt(
        "check_answer_quality", _ANSWER_QUALITY_PROMPT, state.get("session_config")
    ).format(question=question, generation=generation)

    start = time.time()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        label = _parse_label(response.content, _VALID_QUALITY_LABELS, "addresses_question")
    except Exception:
        logger.exception("[AnswerQualityCheck] LLM call failed: defaulting to 'addresses_question'")
        label = "addresses_question"

    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "[AnswerQualityCheck] Result: {label} ({ms:.1f} ms)",
        label=label,
        ms=elapsed_ms,
    )

    timing_spans: Dict[str, float] = dict(state.get("timing_spans", {}))
    timing_spans["quality_check_ms"] = timing_spans.get("quality_check_ms", 0) + elapsed_ms

    return {
        "answer_quality": label,
        "quality_check_attempts": quality_check_attempts + 1,
        "timing_spans": timing_spans,
    }
