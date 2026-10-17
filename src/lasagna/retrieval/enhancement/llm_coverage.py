"""LLM-based coverage scorer for RAG retrieval completeness assessment.

This module provides an LLM-powered alternative to the keyword-based
``CoverageScorer`` in ``coverage.py``. Instead of relying on simple keyword
overlap, ``LLMCoverageScorer`` prompts a language model to perform
aspect-level analysis of the query, determining which sub-topics are
addressed by the retrieved chunks and which remain uncovered.

Advantages over the keyword-based ``CoverageScorer``:
- Understands semantic equivalence (e.g. "ML" and "machine learning")
- Identifies implicit aspects that keyword splitting cannot detect
- Produces more reliable coverage ratios for complex, multi-faceted queries
- Generates actionable follow-up suggestions grounded in actual gaps
"""

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.retrieval.enhancement.coverage import (
    CoverageScore,
    QueryAspect,
)

_COVERAGE_PROMPT = """\
You are an information-retrieval analyst. Your task is to evaluate how well \
a set of retrieved text chunks covers the aspects of a user query.

## User Query
{query}

## Retrieved Chunks
{chunks_text}

## Instructions
1. Identify every distinct aspect or sub-topic the query asks about.
2. For each aspect, decide whether it is **covered** (the chunks contain \
relevant information) or **missing** (no chunk addresses it).
3. Compute a coverage_ratio between 0.0 and 1.0 representing the fraction \
of aspects that are adequately covered.
4. If any aspects are missing, suggest a single concise follow-up query that \
would help retrieve the missing information. If all aspects are covered, \
set suggested_followup to null.

Return ONLY a JSON object (no markdown fences, no explanation) with exactly \
these keys:
{{
  "aspects_found": ["<covered aspect 1>", ...],
  "aspects_missing": ["<missing aspect 1>", ...],
  "coverage_ratio": <float between 0.0 and 1.0>,
  "suggested_followup": "<follow-up query string or null>"
}}
"""


def _build_default_score() -> CoverageScore:
    """Return a safe fallback ``CoverageScore`` used when the LLM call fails."""
    return CoverageScore(
        aspects=[],
        coverage_ratio=0.5,
        weighted_coverage=0.5,
        uncovered_aspects=[],
        confidence=0.0,
        suggested_followup=None,
    )


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Render chunks into a numbered text block for the prompt."""
    parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
        content = chunk.get("content", "")
        parts.append(f"[{chunk_id}]\n{content}")
    return "\n\n".join(parts)


class LLMCoverageScorer:
    """Evaluate retrieval coverage using an LLM for aspect-level analysis.

    Unlike the keyword-based ``CoverageScorer``, this class delegates the
    aspect extraction and matching to a language model, which can understand
    synonyms, paraphrases, and implicit information needs.

    Usage::

        scorer = LLMCoverageScorer()
        score = await scorer.score_coverage(query, chunks, llm)
    """

    async def score_coverage(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        llm: Any,
    ) -> CoverageScore:
        """Score how completely the retrieved chunks cover the query.

        Args:
            query: The user's search query.
            chunks: Retrieved chunks, each a dict with at least ``"content"``
                and ``"chunk_id"`` keys.
            llm: A LangChain-compatible chat model that supports
                ``ainvoke(messages)``.

        Returns:
            A ``CoverageScore`` populated from the LLM's analysis. On any
            failure the method falls back to a neutral default score with
            a ``coverage_ratio`` of 0.5.
        """
        if not query or not query.strip():
            logger.warning("LLMCoverageScorer: empty query, returning default score")
            return _build_default_score()

        if not chunks:
            logger.debug("LLMCoverageScorer: no chunks provided, nothing is covered")
            return CoverageScore(
                aspects=[],
                coverage_ratio=0.0,
                weighted_coverage=0.0,
                uncovered_aspects=[],
                confidence=0.0,
                suggested_followup=f"Find information about: {query}",
            )

        chunks_text = _format_chunks(chunks)
        prompt_text = _COVERAGE_PROMPT.format(
            query=query,
            chunks_text=chunks_text,
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt_text)])
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw, query)
        except Exception:
            logger.exception("LLMCoverageScorer: LLM invocation failed, returning default score")
            return _build_default_score()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, query: str) -> CoverageScore:
        """Parse the LLM's JSON response into a ``CoverageScore``.

        Args:
            raw: The raw string returned by the LLM.
            query: The original user query (used for context in logging).

        Returns:
            A populated ``CoverageScore``, or the default fallback on parse
            failure.
        """
        try:
            # Strip potential markdown code fences the LLM may add despite
            # the prompt asking it not to.
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (optionally with language tag)
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data: Dict[str, Any] = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError) as exc:
            logger.warning(f"LLMCoverageScorer: failed to parse the LLM's response: {exc}")
            return _build_default_score()

        aspects_found: List[str] = data.get("aspects_found", [])
        aspects_missing: List[str] = data.get("aspects_missing", [])
        raw_ratio = data.get("coverage_ratio", 0.5)
        suggested_followup = data.get("suggested_followup")

        # Clamp coverage_ratio to [0.0, 1.0]
        try:
            coverage_ratio = max(0.0, min(1.0, float(raw_ratio)))
        except (TypeError, ValueError):
            coverage_ratio = 0.5

        # Build QueryAspect objects so the result is fully compatible with
        # downstream consumers that inspect .aspects.
        all_aspects: List[QueryAspect] = []
        for text in aspects_found:
            all_aspects.append(
                QueryAspect(
                    text=text,
                    keywords=set(),
                    is_covered=True,
                    coverage_strength=1.0,
                )
            )
        for text in aspects_missing:
            all_aspects.append(
                QueryAspect(
                    text=text,
                    keywords=set(),
                    is_covered=False,
                    coverage_strength=0.0,
                )
            )

        total_aspects = len(all_aspects)
        covered_count = len(aspects_found)

        # Derive confidence from the consistency between the LLM-reported
        # ratio and the discrete aspect counts.
        if total_aspects > 0:
            discrete_ratio = covered_count / total_aspects
            # High agreement between reported ratio and discrete ratio
            # increases confidence.
            agreement = 1.0 - abs(coverage_ratio - discrete_ratio)
            confidence = min(1.0, coverage_ratio * 0.6 + agreement * 0.4)
        else:
            confidence = 0.0

        return CoverageScore(
            aspects=all_aspects,
            coverage_ratio=coverage_ratio,
            weighted_coverage=coverage_ratio,
            uncovered_aspects=aspects_missing,
            confidence=confidence,
            suggested_followup=suggested_followup if suggested_followup else None,
        )
