"""Speculative RAG: parallel draft generation with verification.

Generates multiple draft answers from different context subsets in parallel,
then uses a verifier to select the most faithful and relevant answer.

Reference: Wang et al., "Speculative RAG: Enhancing Retrieval Augmented
Generation through Drafting", 2024.

This module provides a post-retrieval generation enhancement that can be
used alongside or instead of standard single-pass generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from loguru import logger


@dataclass
class DraftAnswer:
    """A candidate answer generated from a subset of contexts."""

    answer: str
    context_indices: List[int]  # which chunks were used
    confidence: float = 0.0
    verification_score: float = 0.0


class SpeculativeRAG:
    """Generate and verify multiple draft answers in parallel.

    Strategy:
    1. Partition retrieved contexts into N overlapping subsets.
    2. Generate a draft answer from each subset (in parallel).
    3. Verify each draft against all contexts using a verifier LLM.
    4. Return the highest-scoring draft.

    Args:
        draft_llm: LLM for generating draft answers (can be smaller/faster).
        verifier_llm: LLM for verifying drafts (typically stronger model).
        n_drafts: Number of parallel drafts to generate.
        subset_size: Number of context chunks per draft.
    """

    def __init__(
        self,
        draft_llm,
        verifier_llm=None,
        n_drafts: int = 3,
        subset_size: int = 3,
    ):
        self._draft_llm = draft_llm
        self._verifier_llm = verifier_llm or draft_llm
        self._n_drafts = n_drafts
        self._subset_size = subset_size

    async def generate(
        self,
        query: str,
        contexts: List[str],
    ) -> Tuple[str, List[DraftAnswer]]:
        """Generate the best answer via speculative parallel drafting.

        Args:
            query: The user's question.
            contexts: Retrieved context chunks (ordered by relevance).

        Returns:
            Tuple of (best_answer, all_drafts).
        """
        if not contexts:
            return "", []

        # Create context subsets — overlapping windows
        subsets = self._create_subsets(contexts)
        if not subsets:
            subsets = [list(range(min(self._subset_size, len(contexts))))]

        # Generate drafts sequentially to avoid rate-limit exhaustion on
        # proxied LLM endpoints.  The per-draft latency
        # is typically <3s, so total overhead is modest (n_drafts * 3s).
        drafts: list = []
        for subset_indices in subsets:
            try:
                draft = await self._generate_draft(query, contexts, subset_indices)
                drafts.append(draft)
            except Exception as e:
                drafts.append(e)

        # Filter out failures
        valid_drafts = [d for d in drafts if isinstance(d, DraftAnswer) and d.answer.strip()]

        if not valid_drafts:
            logger.warning("Speculative RAG: all drafts failed, falling back to single generation")
            fallback = await self._generate_draft(
                query, contexts, list(range(min(5, len(contexts))))
            )
            if isinstance(fallback, DraftAnswer):
                return fallback.answer, [fallback]
            return "", []

        # Verify drafts against all contexts
        if len(valid_drafts) > 1:
            valid_drafts = await self._verify_drafts(query, contexts, valid_drafts)

        # Sort by verification score
        valid_drafts.sort(key=lambda d: d.verification_score, reverse=True)

        best = valid_drafts[0]
        logger.info(
            f"Speculative RAG: selected draft with score {best.verification_score:.2f} "
            f"from {len(valid_drafts)} candidates"
        )
        return best.answer, valid_drafts

    def _create_subsets(self, contexts: List[str]) -> List[List[int]]:
        """Create overlapping context subsets for diverse draft generation."""
        n = len(contexts)
        k = min(self._subset_size, n)

        if n <= k:
            # Not enough contexts for distinct subsets.
            # Generate multiple drafts with the same full context — the LLM's
            # sampling temperature still provides diversity across drafts.
            return [list(range(n))] * self._n_drafts

        # Enough contexts for sliding windows
        n_drafts = min(self._n_drafts, max(2, n - k + 1))
        stride = max(1, (n - k) // max(1, n_drafts - 1))

        subsets = []
        seen = set()
        for i in range(n_drafts):
            start = min(i * stride, n - k)
            key = start  # dedup identical windows
            if key in seen:
                continue
            seen.add(key)
            subsets.append(list(range(start, start + k)))

        return subsets

    async def _generate_draft(
        self,
        query: str,
        contexts: List[str],
        subset_indices: List[int],
    ) -> DraftAnswer:
        """Generate a single draft answer from a context subset."""
        subset_texts = [contexts[i] for i in subset_indices if i < len(contexts)]
        context_block = "\n\n---\n\n".join(
            f"[Context {i + 1}]: {t}" for i, t in enumerate(subset_texts)
        )

        prompt = (
            f"Answer the following question based ONLY on the provided context. "
            f"Be concise and factual.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        try:
            response = await self._draft_llm.ainvoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            return DraftAnswer(
                answer=answer.strip(),
                context_indices=subset_indices,
                confidence=1.0,
            )
        except Exception as e:
            logger.warning(f"Speculative RAG draft generation failed: {e}")
            return DraftAnswer(answer="", context_indices=subset_indices, confidence=0.0)

    async def _verify_drafts(
        self,
        query: str,
        contexts: List[str],
        drafts: List[DraftAnswer],
    ) -> List[DraftAnswer]:
        """Verify each draft against all contexts using the verifier LLM."""
        all_context = "\n\n".join(f"[{i + 1}]: {c}" for i, c in enumerate(contexts[:10]))

        async def _verify_one(draft: DraftAnswer) -> DraftAnswer:
            prompt = (
                f"You are a fact-checking assistant. Given the contexts and a candidate answer, "
                f"rate the answer's faithfulness and relevance on a scale of 0.0 to 1.0.\n\n"
                f"Contexts:\n{all_context}\n\n"
                f"Question: {query}\n"
                f"Candidate Answer: {draft.answer}\n\n"
                f"Respond with ONLY a number between 0.0 and 1.0 representing the quality score."
            )
            try:
                response = await self._verifier_llm.ainvoke(prompt)
                score_text = response.content if hasattr(response, "content") else str(response)
                # Parse score from response
                score = float(score_text.strip().split()[0])
                draft.verification_score = max(0.0, min(1.0, score))
            except (ValueError, IndexError):
                draft.verification_score = 0.5  # neutral if parsing fails
            except Exception as e:
                logger.debug(f"Speculative RAG verification failed: {e}")
                draft.verification_score = 0.5
            return draft

        # Verify sequentially to respect LLM proxy rate limits.
        verified = []
        for d in drafts:
            try:
                result = await _verify_one(d)
                verified.append(result)
            except Exception:
                verified.append(d)  # keep the draft with default score

        return [v for v in verified if isinstance(v, DraftAnswer)]
