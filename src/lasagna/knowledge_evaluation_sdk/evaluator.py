"""Knowledge Evaluator - Main evaluation orchestrator.

Runs the SDK's knowledge/RAG feature against benchmark data and
uses LLM-as-a-judge for scoring.

Two evaluation modes:
1. RAG-only mode (default): Tests RAG retrieval + LLM answering directly
   using KnowledgeClient (no Lasagna dependency)
2. Full agent mode: Tests complete SDK invoke flow (requires cuga.sdk).
   All agent-framework coupling is confined to ``lasagna.integrations.cuga_agent``.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from lasagna.providers import get_llm_provider, get_config_provider

if TYPE_CHECKING:
    from lasagna.sdk import KnowledgeClient

from .models import EvalCase, EvalConfig, EvalMetrics, EvalResult
from .judge import LLMJudge


RAG_ANSWERING_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Use ONLY the information from the context below to answer the user's question.
If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}

Answer the question accurately and concisely based on the context above."""


class KnowledgeEvaluator:
    """Orchestrates knowledge/RAG evaluation using the SDK.

    This evaluator:
    1. Loads benchmark data from JSONL ground truth files
    2. Uploads knowledge documents via KnowledgeClient
    3. Runs queries through RAG retrieval + LLM answering
    4. Scores responses using LLM-as-a-judge
    5. Produces detailed metrics and reports
    """

    def __init__(self, config: EvalConfig, use_full_agent: bool = False):
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            use_full_agent: If True, use full CugaAgent invoke flow
                           (requires cuga.sdk). If False (default), use
                           KnowledgeClient directly (standalone).
        """
        self.config = config
        self.use_full_agent = use_full_agent
        self._client: Optional["KnowledgeClient"] = None
        self._agent = None  # Only used when use_full_agent=True
        self._judge: Optional[LLMJudge] = None
        self._answering_model = None
        self._document_id: Optional[str] = None
        self._results: List[EvalResult] = []
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Initialize client and upload knowledge document (thread-safe)."""
        if self._client is not None or self._agent is not None:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._client is not None or self._agent is not None:
                return

            try:
                logger.info("Initializing Knowledge Evaluator...")

                if self.use_full_agent:
                    try:
                        from lasagna.integrations import cuga_agent as _adapter

                        self._agent = _adapter.create_agent()
                    except ImportError:
                        logger.warning(
                            "cuga.sdk not available, falling back to RAG-only mode. "
                            "Install the optional agent package for agent mode."
                        )
                        self.use_full_agent = False

                if not self.use_full_agent:
                    # RAG-only mode — standalone, no Lasagna dependency
                    from lasagna.sdk import KnowledgeClient

                    self._client = KnowledgeClient(
                        session_id=f"eval_{uuid.uuid4().hex[:8]}",
                    )

                # Upload the knowledge document
                doc_path = Path(self.config.document_path)
                if not doc_path.exists():
                    raise FileNotFoundError(f"Document not found: {doc_path}")

                logger.info(f"Uploading document: {doc_path.name}")

                if self.use_full_agent and self._agent:
                    from lasagna.integrations import cuga_agent as _adapter

                    self._document_id = await _adapter.upload_document(self._agent, doc_path)
                else:
                    result = await self._client.upload(str(doc_path))
                    logger.info(f"Upload result: {result.document_id}, {result.chunk_count} chunks")
                    self._document_id = result.document_id
                    if self._document_id:
                        self._client.set_focused_documents([self._document_id])
                        logger.info(f"Set focused document: {self._document_id}")

                # Initialize the judge
                self._judge = LLMJudge(
                    platform=self.config.judge_model_platform,
                    model_name=self.config.judge_model_name,
                )

                # Initialize answering model for RAG-only mode
                if not self.use_full_agent:
                    model_settings = (
                        get_config_provider().get_llm_settings("agent.code.model") or {}
                    )
                    self._answering_model = get_llm_provider().get_chat_model(model_settings)
                    logger.info("Using RAG-only mode (direct retrieval + LLM answering)")
                else:
                    logger.info("Using full agent mode")

                logger.info("Knowledge Evaluator initialized successfully")

            except Exception:
                # Rollback partially-initialized state so the next call
                # retries from scratch rather than operating in zombie state.
                logger.error("Evaluator initialization failed, rolling back state")
                if self._client:
                    try:
                        await self._client.close()
                    except Exception:
                        pass
                self._client = None
                self._agent = None
                self._judge = None
                self._answering_model = None
                self._document_id = None
                raise

    def load_cases(self) -> List[EvalCase]:
        """Load evaluation cases from ground truth JSONL file."""
        gt_path = Path(self.config.ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        cases = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cases.append(EvalCase.from_jsonl_line(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")

        logger.info(f"Loaded {len(cases)} evaluation cases from {gt_path.name}")
        return cases

    async def evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single case.

        Args:
            case: The evaluation case to run.

        Returns:
            EvalResult with the evaluation outcome.
        """
        await self._ensure_initialized()

        start_time = time.perf_counter()
        rag_context: Optional[str] = None
        rag_used = False
        agent_response = ""
        error: Optional[str] = None
        rag_latency = 0.0

        try:
            # Get RAG context
            rag_start = time.perf_counter()
            try:
                if self.use_full_agent and self._agent:
                    from lasagna.integrations import cuga_agent as _adapter

                    rag_context = await _adapter.search(
                        self._agent,
                        case.utterance,
                        document_ids=[self._document_id] if self._document_id else None,
                    )
                else:
                    rag_context = await self._client.get_context(
                        case.utterance,
                        document_ids=[self._document_id] if self._document_id else None,
                    )
                rag_used = bool(rag_context and len(rag_context) > 10)
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                rag_context = None

            rag_end = time.perf_counter()
            rag_latency = (rag_end - rag_start) * 1000

            if self.use_full_agent and self._agent:
                from lasagna.integrations import cuga_agent as _adapter

                agent_response = await _adapter.invoke(self._agent, case.utterance)
            else:
                # RAG-only mode: Direct RAG + LLM answering
                if rag_context and rag_used:
                    messages = [
                        SystemMessage(content=RAG_ANSWERING_PROMPT.format(context=rag_context)),
                        HumanMessage(content=case.utterance),
                    ]
                    response = await self._answering_model.ainvoke(messages)
                    agent_response = response.content
                else:
                    agent_response = "No relevant context found in the knowledge base."

        except Exception as e:
            logger.error(f"Evaluation failed for case: {case.utterance[:50]}... - {e}")
            error = str(e)

        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000

        # Score with LLM judge
        judge_score = None
        if error is None and self._judge:
            try:
                judge_score = await self._judge.evaluate(
                    question=case.utterance,
                    expected=case.expected_response,
                    response=agent_response,
                    context=rag_context,
                )
            except Exception as e:
                logger.warning(f"Judge scoring failed: {e}")

        return EvalResult(
            case=case,
            agent_response=agent_response,
            rag_context=rag_context,
            rag_used=rag_used,
            judge_score=judge_score,
            latency_ms=total_latency,
            rag_latency_ms=rag_latency,
            error=error,
        )

    async def run(self) -> EvalMetrics:
        """Run the full evaluation.

        Returns:
            EvalMetrics with aggregated results.
        """
        await self._ensure_initialized()

        cases = self.load_cases()
        if not cases:
            logger.warning("No evaluation cases found")
            return EvalMetrics.from_results([])

        logger.info(f"Starting evaluation of {len(cases)} cases...")

        # Run evaluations sequentially to avoid MCP connection conflicts
        # Each case gets its own clean state
        self._results = []
        for i, case in enumerate(cases, 1):
            logger.info(f"Evaluating case {i}/{len(cases)}: {case.utterance[:50]}...")
            result = await self.evaluate_case(case)
            self._results.append(result)

            if self.config.verbose:
                status = "PASS" if result.passed else "FAIL"
                score = result.score
                logger.info(f"{status} [{score}/5] {case.utterance[:50]}...")

        metrics = EvalMetrics.from_results(self._results)

        # Save results
        await self._save_results(metrics)

        return metrics

    async def run_streaming(self) -> AsyncIterator[EvalResult]:
        """Run evaluation with streaming results.

        Yields:
            EvalResult for each case as it completes.
        """
        await self._ensure_initialized()

        cases = self.load_cases()
        self._results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def eval_with_semaphore(case: EvalCase) -> EvalResult:
            async with semaphore:
                return await self.evaluate_case(case)

        # Create tasks
        tasks = [asyncio.create_task(eval_with_semaphore(case)) for case in cases]

        # Yield results as they complete; cancel outstanding tasks on error
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self._results.append(result)
                yield result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _save_results(self, metrics: EvalMetrics) -> None:
        """Save evaluation results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSONL
        results_path = output_dir / f"eval_results_{timestamp}.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for result in self._results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        # Save metrics summary as JSON
        metrics_path = output_dir / f"eval_metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        # Save human-readable report
        report_path = output_dir / f"eval_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(metrics.summary())
            f.write("\n\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")
            for i, result in enumerate(self._results, 1):
                f.write(f"Case {i}: {result.case.utterance}\n")
                f.write(f"Score: {result.score}/5")
                if result.judge_score:
                    f.write(f" ({result.judge_score.level.value})\n")
                    f.write(f"Reasoning: {result.judge_score.reasoning}\n")
                else:
                    f.write("\n")
                if result.error:
                    f.write(f"Error: {result.error}\n")
                f.write("-" * 40 + "\n\n")

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"  - Results: {results_path.name}")
        logger.info(f"  - Metrics: {metrics_path.name}")
        logger.info(f"  - Report:  {report_path.name}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._agent:
            from lasagna.integrations import cuga_agent as _adapter

            await _adapter.cleanup(self._agent)
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Client cleanup error: {e}")

    @property
    def results(self) -> List[EvalResult]:
        """Get the list of evaluation results."""
        return self._results

    @property
    def metrics(self) -> EvalMetrics:
        """Get aggregated metrics from current results."""
        return EvalMetrics.from_results(self._results)
