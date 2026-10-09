"""RAG Evaluation Module.

Provides evaluation framework for RAG quality testing.
Wraps the existing knowledge_evaluation_sdk.

Example:
    ```python
    from lasagna.sdk.evaluation import (
        KnowledgeEvaluator,
        EvalConfig,
        EvalResult,
    )

    # Create evaluator
    evaluator = KnowledgeEvaluator(client)

    # Run evaluation
    results = await evaluator.evaluate(test_cases)

    # Print metrics
    print(f"Precision: {results.precision:.2f}")
    print(f"Recall: {results.recall:.2f}")
    ```
"""

import os
import time as _time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from ..exceptions import EvaluationError, JudgeError

# Re-export from existing evaluation SDK
from ...knowledge_evaluation_sdk.evaluator import (
    KnowledgeEvaluator as BackendEvaluator,
)
from ...knowledge_evaluation_sdk.judge import LLMJudge
from ...knowledge_evaluation_sdk.models import (
    EvalResult as EvaluationResult,
    EvalMetrics as EvaluationMetrics,
    EvalCase as TestCase,
    JudgeScore as JudgeResult,
)

# Answering prompt — same as backend evaluator.py for consistency
_ANSWERING_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Use ONLY the information from the context below to answer the user's question.
If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}

Answer the question accurately and concisely based on the context above."""


@dataclass
class EvalConfig:
    """Configuration for RAG evaluation.

    Attributes:
        judge_model: Model name for LLM-as-judge evaluation.
        judge_model_platform: Platform for the judge model.
        answer_model: Model name for generating answers from context.
            Defaults to judge_model if not set.
        answer_model_platform: Platform for the answer model.
            Defaults to judge_model_platform if not set.
        answer_temperature: Temperature for answer generation.
        num_samples: Number of samples to evaluate.
        metrics: List of metrics to compute.
        verbose: Print progress during evaluation.
        temperature: Temperature for judge responses (0 for consistency).
    """

    judge_model: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    judge_model_platform: str = "openai"
    answer_model: Optional[str] = None
    answer_model_platform: Optional[str] = None
    answer_temperature: float = 0.0
    num_samples: int = 100
    metrics: List[str] = field(
        default_factory=lambda: [
            "precision",
            "recall",
            "f1",
            "mrr",
            "ndcg",
        ]
    )
    verbose: bool = True
    temperature: float = 0.0

    @property
    def effective_answer_model(self) -> str:
        """Answer model, defaulting to judge_model."""
        return self.answer_model or self.judge_model

    @property
    def effective_answer_platform(self) -> str:
        """Answer platform, defaulting to judge_model_platform."""
        return self.answer_model_platform or self.judge_model_platform


@dataclass
class EvalResult:
    """Results from RAG evaluation.

    Attributes:
        precision: Retrieval precision (relevant retrieved / total retrieved)
        recall: Retrieval recall (relevant retrieved / total relevant)
        f1: F1 score (harmonic mean of precision and recall)
        mrr: Mean Reciprocal Rank
        ndcg: Normalized Discounted Cumulative Gain
        latency_ms: Average search latency in milliseconds
        num_samples: Number of samples evaluated
        details: Per-sample evaluation details
    """

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    latency_ms: float = 0.0
    num_samples: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_backend(cls, result: EvaluationResult) -> "EvalResult":
        """Create from backend evaluation result.

        Maps backend EvalResult fields (case, judge_score, latency_ms, etc.)
        to this SDK wrapper's flat structure.
        """
        score = result.judge_score.score if result.judge_score else 0
        # Normalize 1-5 to 0-1: (score - 1) / 4 maps 1→0.0, 5→1.0
        # Clamped to [0, 1] for safety.
        normalized = max(0.0, min(1.0, (score - 1) / 4)) if score > 0 else 0.0
        return cls(
            precision=normalized,
            recall=0.0,
            f1=0.0,
            mrr=0.0,
            ndcg=0.0,
            latency_ms=result.latency_ms,
            num_samples=1,
            details=[result.to_dict()],
        )


class KnowledgeEvaluator:
    """Evaluator for RAG system quality.

    Provides comprehensive evaluation of retrieval quality including:
    - Precision/Recall metrics
    - MRR (Mean Reciprocal Rank)
    - NDCG (Normalized Discounted Cumulative Gain)
    - LLM-as-Judge evaluation

    Example:
        ```python
        from lasagna.sdk.evaluation import (
            KnowledgeEvaluator,
            EvalConfig,
        )

        evaluator = KnowledgeEvaluator(client)

        # Simple evaluation with ground truth
        results = await evaluator.evaluate(
            test_cases=[
                {
                    "query": "What is revenue?",
                    "expected_docs": ["doc_123"],
                },
            ]
        )

        # With LLM judge
        results = await evaluator.evaluate_with_judge(
            test_cases=[
                {
                    "query": "What is revenue?",
                    "expected_answer": "Revenue is $10M",
                },
            ]
        )
        ```
    """

    def __init__(
        self,
        client,
        config: Optional[EvalConfig] = None,
    ):
        """Initialize evaluator.

        Args:
            client: KnowledgeClient instance
            config: Evaluation configuration
        """
        self._client = client
        self._config = config or EvalConfig()
        self._answering_model = None

    async def _get_answering_model(self):
        """Lazily initialize the answering model."""
        if self._answering_model is None:
            from lasagna.providers import get_llm_provider

            cfg = self._config
            model_settings: Dict[str, Any] = {
                "platform": cfg.effective_answer_platform,
                "temperature": cfg.answer_temperature,
            }
            if cfg.effective_answer_model:
                model_settings["model_name"] = cfg.effective_answer_model

            self._answering_model = get_llm_provider().get_chat_model(model_settings)
        return self._answering_model

    async def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> EvalResult:
        """Evaluate retrieval quality against ground truth.

        Args:
            test_cases: List of test cases with 'query' and 'expected_docs'
            top_k: Number of results to retrieve per query

        Returns:
            EvalResult with computed metrics

        Example:
            ```python
            results = await evaluator.evaluate([
                {
                    "query": "quarterly revenue",
                    "expected_docs": ["doc_123", "doc_456"],
                },
                {
                    "query": "employee count",
                    "expected_docs": ["doc_789"],
                },
            ])
            print(f"Precision: {results.precision:.2%}")
            ```
        """
        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_latency = 0.0
        details = []

        for i, case in enumerate(test_cases):
            query = case["query"]
            expected_docs = set(case.get("expected_docs", []))

            start_time = _time.monotonic()
            results = await self._client.search_documents(query, top_k=top_k)
            latency = (_time.monotonic() - start_time) * 1000

            retrieved_docs = {r.document.document_id for r in results}

            # Compute metrics
            relevant_retrieved = len(retrieved_docs & expected_docs)
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            recall = relevant_retrieved / len(expected_docs) if expected_docs else 0

            # MRR
            mrr = 0.0
            for rank, result in enumerate(results, 1):
                if result.document.document_id in expected_docs:
                    mrr = 1.0 / rank
                    break

            total_precision += precision
            total_recall += recall
            total_mrr += mrr
            total_latency += latency

            details.append(
                {
                    "query": query,
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "latency_ms": latency,
                    "retrieved": list(retrieved_docs),
                    "expected": list(expected_docs),
                }
            )

            if self._config.verbose and (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{len(test_cases)} samples")

        n = len(test_cases)
        avg_precision = total_precision / n if n else 0
        avg_recall = total_recall / n if n else 0
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        return EvalResult(
            precision=avg_precision,
            recall=avg_recall,
            f1=f1,
            mrr=total_mrr / n if n else 0,
            ndcg=0.0,  # Would need relevance grades
            latency_ms=total_latency / n if n else 0,
            num_samples=n,
            details=details,
        )

    async def evaluate_with_judge(
        self,
        test_cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> EvalResult:
        """Evaluate using LLM-as-Judge.

        Retrieves RAG context, generates an answer from the context using
        a chat model, then uses an LLM judge to score the generated answer
        against the expected answer on a 1-5 scale.

        Args:
            test_cases: List with 'query' and 'expected_answer'
            top_k: Number of results to retrieve

        Returns:
            EvalResult with judge scores (precision = avg normalized score)

        Raises:
            JudgeError: If LLM judge evaluation fails
            EvaluationError: If evaluation process fails

        Example:
            ```python
            results = await evaluator.evaluate_with_judge([
                {
                    "query": "What is the revenue?",
                    "expected_answer": "Revenue was $10M in Q3 2024",
                },
            ])
            for detail in results.details:
                print(f"Query: {detail['query']}")
                print(f"Score: {detail['score']}/5")
                print(f"Reasoning: {detail['reasoning']}")
            ```
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        cfg = self._config

        try:
            judge = LLMJudge(
                platform=cfg.judge_model_platform,
                model_name=cfg.judge_model,
                temperature=cfg.temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM judge: {e}")
            raise JudgeError(f"Failed to initialize LLM judge: {e}") from e

        # Initialize the answering model for generating responses from context
        try:
            answering_model = await self._get_answering_model()
        except Exception as e:
            logger.error(f"Failed to initialize answering model: {e}")
            raise EvaluationError(f"Failed to initialize answering model: {e}") from e

        details = []
        total_score = 0.0
        total_latency = 0.0

        for i, case in enumerate(test_cases):
            query = case["query"]
            expected = case.get("expected_answer", "")

            try:
                start_time = _time.monotonic()

                # Step 1: Retrieve RAG context
                context = await self._client.get_context(query, max_chunks=top_k)
                retrieval_time = (_time.monotonic() - start_time) * 1000

                # Step 2: Generate an answer from the context using a chat model
                if context:
                    messages = [
                        SystemMessage(content=_ANSWERING_PROMPT.format(context=context)),
                        HumanMessage(content=query),
                    ]
                    answer_result = await answering_model.ainvoke(messages)
                    response_text = answer_result.content
                else:
                    response_text = "No relevant context found in the knowledge base."

                # Step 3: Judge the generated answer (not the raw context)
                judge_start = _time.monotonic()
                try:
                    judge_result = await judge.evaluate(
                        question=query,
                        expected=expected,
                        response=response_text,
                        context=context,
                    )
                    judge_time = (_time.monotonic() - judge_start) * 1000

                    score = judge_result.score
                    level = (
                        judge_result.level.value
                        if hasattr(judge_result.level, "value")
                        else str(judge_result.level)
                    )
                    reasoning = judge_result.reasoning

                except Exception as e:
                    logger.warning(f"Judge failed for query '{query}': {e}")
                    score = 0
                    level = "error"
                    reasoning = f"Judge evaluation failed: {e}"
                    judge_time = 0

                total_score += score
                total_latency += retrieval_time

                truncated_response = (
                    response_text[:500] + "..." if len(response_text) > 500 else response_text
                )
                details.append(
                    {
                        "query": query,
                        "expected_answer": expected,
                        "response": truncated_response,
                        "actual_response": truncated_response,  # backward-compat alias
                        "score": score,
                        "level": level,
                        "reasoning": reasoning,
                        "retrieval_time_ms": retrieval_time,
                        "judge_time_ms": judge_time,
                    }
                )

                if self._config.verbose and (i + 1) % 10 == 0:
                    print(
                        f"Evaluated {i + 1}/{len(test_cases)} samples (avg score: {total_score / (i + 1):.2f}/5)"
                    )

            except Exception as e:
                logger.error(f"Evaluation failed for query '{query}': {e}")
                details.append(
                    {
                        "query": query,
                        "expected_answer": expected,
                        "response": "",
                        "actual_response": "",  # backward-compat alias
                        "error": str(e),
                        "score": 0,
                    }
                )

        n = len(test_cases)
        if n == 0:
            raise EvaluationError("No test cases provided for evaluation")

        # Normalize score from 1-5 scale to 0-1: (score - 1) / 4 maps 1→0.0, 5→1.0
        # Clamp to [0, 1] — avg_score can be in (0, 1) when some cases fail
        # with score=0, which would produce a negative value without clamping.
        avg_score = total_score / n
        normalized_precision = max(0.0, min(1.0, (avg_score - 1) / 4)) if avg_score > 0 else 0.0

        return EvalResult(
            precision=normalized_precision,
            recall=0.0,  # Not applicable for judge evaluation
            f1=0.0,  # Not applicable for judge evaluation
            mrr=0.0,  # Not applicable for judge evaluation
            ndcg=0.0,  # Not applicable for judge evaluation
            latency_ms=total_latency / n,
            num_samples=n,
            details=details,
        )


__all__ = [
    "KnowledgeEvaluator",
    "EvalConfig",
    "EvalResult",
    # Backend exports
    "LLMJudge",
    "EvaluationResult",
    "EvaluationMetrics",
    "TestCase",
    "JudgeResult",
]
