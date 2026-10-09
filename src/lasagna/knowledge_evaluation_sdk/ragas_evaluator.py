"""RAGAS evaluation integration for RAG pipeline benchmarking.

Wraps the ragas library to evaluate RAG pipelines on standard metrics
like Faithfulness, Context Precision, Context Recall, and Factual Correctness.

Requirements: pip install lasagna-rag[eval]

Example:
    from lasagna.knowledge_evaluation_sdk.ragas_evaluator import RAGASEvaluator

    evaluator = RAGASEvaluator()
    results = evaluator.evaluate_from_samples(
        questions=["What is RAG?"],
        answers=["RAG is Retrieval-Augmented Generation..."],
        contexts=[["RAG combines retrieval with generation..."]],
        ground_truths=["RAG is a technique that..."],
    )
    print(results)  # {'faithfulness': 0.92, 'context_precision': 0.85, ...}
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger


# Lazy imports to avoid hard dependency
def _import_ragas():
    """Import ragas components with helpful error messages."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextRecall,
            FactualCorrectness,
            LLMContextPrecisionWithReference,
        )
        from ragas import EvaluationDataset, SingleTurnSample

        return {
            "evaluate": ragas_evaluate,
            "Faithfulness": Faithfulness,
            "AnswerRelevancy": AnswerRelevancy,
            "LLMContextRecall": LLMContextRecall,
            "FactualCorrectness": FactualCorrectness,
            "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample": SingleTurnSample,
        }
    except ImportError:
        raise ImportError(
            "ragas is required for evaluation. Install with: pip install lasagna-rag[eval]"
        )


# Mapping of metric names to RAGAS metric classes
METRIC_MAP = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_recall": "LLMContextRecall",
    "factual_correctness": "FactualCorrectness",
    "context_precision": "LLMContextPrecisionWithReference",
}

# RAGAS v0.4 uses full class names as result keys, not our short aliases.
# This mapping converts RAGAS result keys back to our short names.
_RAGAS_KEY_MAP = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "answer_relevance"],
    "context_recall": ["context_recall", "llm_context_recall"],
    "factual_correctness": ["factual_correctness"],
    "context_precision": [
        "context_precision",
        "llm_context_precision_with_reference",
    ],
}


def _extract_score(result: Any, metric_name: str) -> float:
    """Extract a scalar score from a RAGAS result for a given metric.

    RAGAS v0.4 returns lists of per-sample scores from result[key],
    not aggregated scalars. This function averages them.
    """
    candidates = _RAGAS_KEY_MAP.get(metric_name, [metric_name])
    for key in candidates:
        try:
            val = result[key]
        except (KeyError, IndexError):
            continue
        if isinstance(val, list):
            nums = [v for v in val if isinstance(v, (int, float)) and v == v]  # skip NaN
            dropped = len(val) - len(nums)

            if not nums:
                # Every judgement failed. Returning 0.0 here is indistinguishable
                # from the model genuinely scoring zero, which is how a totally
                # failed evaluation has previously been mistaken for a real
                # result. Say so loudly; the caller still gets a number because
                # these scores are JSON-serialised and NaN is not valid JSON.
                logger.error(
                    f"Metric '{metric_name}': all {len(val)} judgements were "
                    f"unusable (NaN). Reporting 0.0, but this is NOT a real "
                    f"score -- treat this run as failed, not as zero."
                )
                return 0.0

            if dropped:
                # The mean is over survivors, so the denominator silently
                # changed. A score over 60 of 100 samples is a different claim
                # from a score over 100.
                logger.warning(
                    f"Metric '{metric_name}': {dropped} of {len(val)} judgements "
                    f"were unusable (NaN) and excluded. Score is the mean over "
                    f"the remaining {len(nums)} samples, not all {len(val)}."
                )
            return sum(nums) / len(nums)
        if isinstance(val, (int, float)):
            return float(val)
    raise KeyError(f"Metric '{metric_name}' not found in RAGAS result")


DEFAULT_METRICS = ["faithfulness", "context_recall", "context_precision"]


class RAGASEvaluator:
    """Evaluate RAG pipelines using RAGAS metrics.

    Supports both raw sample evaluation and pipeline-integrated evaluation.
    """

    def __init__(self, llm: Any = None, embeddings: Any = None):
        """Initialize the evaluator.

        Args:
            llm: Optional LangChain-compatible LLM for evaluation.
                 If None, ragas will use its default.
            embeddings: Optional embedding model for evaluation metrics.
        """
        self._llm = llm
        self._embeddings = embeddings

    def evaluate_from_samples(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Evaluate from raw question/answer/context samples.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved context lists (one list per question)
            ground_truths: Optional list of ground truth answers
            metrics: List of metric names (default: faithfulness, context_recall, context_precision)

        Returns:
            Dict mapping metric name to score
        """
        ragas = _import_ragas()
        metric_names = metrics or DEFAULT_METRICS

        # Build RAGAS samples
        samples = []
        for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
            sample_kwargs = {
                "user_input": q,
                "response": a,
                "retrieved_contexts": ctx,
            }
            if ground_truths and i < len(ground_truths):
                sample_kwargs["reference"] = ground_truths[i]
            samples.append(ragas["SingleTurnSample"](**sample_kwargs))

        dataset = ragas["EvaluationDataset"](samples=samples)

        # Build metric instances
        metric_instances = []
        for name in metric_names:
            ragas_class_name = METRIC_MAP.get(name)
            if ragas_class_name and ragas_class_name in ragas:
                metric_instances.append(ragas[ragas_class_name]())
            else:
                logger.warning(f"Unknown metric '{name}', skipping")

        if not metric_instances:
            logger.warning("No valid metrics specified")
            return {}

        # Run evaluation — batch_size limits concurrent LLM gateway calls to
        # avoid rate-limit exhaustion (hosted proxies often allow little concurrency).
        eval_kwargs = {"dataset": dataset, "metrics": metric_instances}
        if self._llm:
            eval_kwargs["llm"] = self._llm
        if self._embeddings:
            eval_kwargs["embeddings"] = self._embeddings
        if batch_size:
            eval_kwargs["batch_size"] = batch_size

        try:
            result = ragas["evaluate"](**eval_kwargs)
            scores = {}
            for name in metric_names:
                try:
                    scores[name] = _extract_score(result, name)
                except KeyError:
                    # A requested metric RAGAS never returned. Silently omitting
                    # it makes the metric look un-requested rather than failed.
                    logger.warning(
                        f"Metric '{name}' was requested but absent from the "
                        f"RAGAS result; it will be missing from this run's scores."
                    )
            return scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}

    def evaluate(
        self,
        dataset: Any = None,
        pipeline: Any = None,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate a pipeline on a dataset (for KnowledgePipeline integration).

        This is the method called by KnowledgePipeline.evaluate().

        Args:
            dataset: Evaluation dataset
            pipeline: KnowledgePipeline instance
            metrics: List of metric names

        Returns:
            Dict mapping metric name to score
        """
        if dataset is None:
            logger.warning("No dataset provided for evaluation")
            return {}

        # If dataset is already a list of dicts, convert
        if isinstance(dataset, list) and dataset and isinstance(dataset[0], dict):
            return self.evaluate_from_samples(
                questions=[d.get("question", d.get("user_input", "")) for d in dataset],
                answers=[d.get("answer", d.get("response", "")) for d in dataset],
                contexts=[d.get("contexts", d.get("retrieved_contexts", [])) for d in dataset],
                ground_truths=[d.get("ground_truth", d.get("reference")) for d in dataset],
                metrics=metrics,
            )

        # If it's already a RAGAS EvaluationDataset, use it directly
        ragas = _import_ragas()
        metric_names = metrics or DEFAULT_METRICS
        metric_instances = []
        for name in metric_names:
            ragas_class_name = METRIC_MAP.get(name)
            if ragas_class_name and ragas_class_name in ragas:
                metric_instances.append(ragas[ragas_class_name]())

        eval_kwargs = {"dataset": dataset, "metrics": metric_instances}
        if self._llm:
            eval_kwargs["llm"] = self._llm
        if self._embeddings:
            eval_kwargs["embeddings"] = self._embeddings

        try:
            result = ragas["evaluate"](**eval_kwargs)
            scores = {}
            for name in metric_names:
                try:
                    scores[name] = _extract_score(result, name)
                except KeyError:
                    # A requested metric RAGAS never returned. Silently omitting
                    # it makes the metric look un-requested rather than failed.
                    logger.warning(
                        f"Metric '{name}' was requested but absent from the "
                        f"RAGAS result; it will be missing from this run's scores."
                    )
            return scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}

    def evaluate_detailed(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> "DetailedEvalResult":
        """Evaluate with both aggregate and per-sample scores.

        Unlike ``evaluate_from_samples()`` which returns only aggregate scores,
        this method preserves per-sample scores — essential for paired
        significance testing via ``AblationResults.significance_test()``.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved context lists (one list per question)
            ground_truths: Optional list of ground truth answers
            metrics: List of metric names
            batch_size: Max concurrent RAGAS LLM calls (limits proxy load)

        Returns:
            DetailedEvalResult with ``.aggregate`` (Dict[str, float]) and
            ``.per_sample`` (List[Dict[str, float]]) attributes.
        """
        ragas = _import_ragas()
        metric_names = metrics or DEFAULT_METRICS

        # Build RAGAS samples
        samples = []
        for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
            sample_kwargs = {
                "user_input": q,
                "response": a,
                "retrieved_contexts": ctx,
            }
            if ground_truths and i < len(ground_truths):
                sample_kwargs["reference"] = ground_truths[i]
            samples.append(ragas["SingleTurnSample"](**sample_kwargs))

        dataset = ragas["EvaluationDataset"](samples=samples)

        # Build metric instances
        metric_instances = []
        active_metric_names = []
        for name in metric_names:
            ragas_class_name = METRIC_MAP.get(name)
            if ragas_class_name and ragas_class_name in ragas:
                metric_instances.append(ragas[ragas_class_name]())
                active_metric_names.append(name)
            else:
                logger.warning(f"Unknown metric '{name}', skipping")

        if not metric_instances:
            logger.warning("No valid metrics specified")
            return DetailedEvalResult(aggregate={}, per_sample=[])

        # Run evaluation — batch_size limits concurrent LLM calls to avoid
        # rate-limit exhaustion (hosted proxies often allow little concurrency).
        eval_kwargs = {"dataset": dataset, "metrics": metric_instances}
        if self._llm:
            eval_kwargs["llm"] = self._llm
        if self._embeddings:
            eval_kwargs["embeddings"] = self._embeddings
        if batch_size:
            eval_kwargs["batch_size"] = batch_size

        try:
            result = ragas["evaluate"](**eval_kwargs)

            # Extract aggregate scores (RAGAS v0.4 returns lists, not scalars)
            aggregate: Dict[str, float] = {}
            for name in active_metric_names:
                try:
                    aggregate[name] = _extract_score(result, name)
                except KeyError:
                    logger.warning(
                        f"Metric '{name}' was requested but absent from the "
                        f"RAGAS result; it will be missing from the aggregate."
                    )

            # Extract per-sample scores from RAGAS result DataFrame
            per_sample: List[Dict[str, float]] = []
            try:
                df = result.to_pandas()
                logger.debug(f"RAGAS DataFrame columns: {list(df.columns)}, shape: {df.shape}")
                # Build reverse map: DataFrame column → our metric name
                col_to_name: Dict[str, str] = {}
                for name in active_metric_names:
                    for candidate in _RAGAS_KEY_MAP.get(name, [name]):
                        if candidate in df.columns:
                            col_to_name[candidate] = name
                            break

                if not col_to_name:
                    logger.warning(
                        f"No metric columns matched in RAGAS DataFrame. "
                        f"Expected: {active_metric_names}, "
                        f"Available: {[c for c in df.columns if c not in ('question', 'answer', 'contexts', 'ground_truth')]}"
                    )

                for _, row in df.iterrows():
                    sample_scores: Dict[str, float] = {}
                    for col, name in col_to_name.items():
                        val = row[col]
                        if isinstance(val, (int, float)) and val == val:  # skip NaN
                            sample_scores[name] = float(val)
                    per_sample.append(sample_scores)

                if per_sample:
                    logger.info(
                        f"Extracted per-sample scores for {len(per_sample)} samples, "
                        f"metrics: {list(col_to_name.values())}"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not extract per-sample scores via to_pandas(): {e}. "
                    f"Result type: {type(result).__name__}. "
                    "Trying list-based fallback."
                )
                # Fallback: RAGAS result[metric] returns lists of per-sample scores
                try:
                    n_samples = len(questions) if questions else 0
                    per_sample = [{} for _ in range(n_samples)]
                    for name in active_metric_names:
                        for candidate in _RAGAS_KEY_MAP.get(name, [name]):
                            try:
                                vals = result[candidate]
                                if isinstance(vals, list) and len(vals) == n_samples:
                                    for i, v in enumerate(vals):
                                        if isinstance(v, (int, float)) and v == v:
                                            per_sample[i][name] = float(v)
                                    break
                            except (KeyError, IndexError):
                                continue
                    # Remove empty dicts
                    if all(not d for d in per_sample):
                        per_sample = []
                        logger.warning("List-based fallback also yielded no per-sample scores")
                    else:
                        logger.info(
                            f"Extracted per-sample scores via list fallback "
                            f"for {sum(1 for d in per_sample if d)} / {n_samples} samples"
                        )
                except Exception as e2:
                    logger.warning(f"List-based fallback also failed: {e2}")
                    per_sample = []

            return DetailedEvalResult(aggregate=aggregate, per_sample=per_sample)

        except Exception as e:
            logger.error(f"RAGAS detailed evaluation failed: {e}")
            return DetailedEvalResult(aggregate={}, per_sample=[])


@dataclass
class DetailedEvalResult:
    """Evaluation result containing both aggregate and per-sample scores.

    Attributes:
        aggregate: Dict mapping metric name to aggregate score.
        per_sample: List of dicts, one per sample, mapping metric name to score.
            Empty list if per-sample extraction failed.
    """

    aggregate: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict[str, float]] = field(default_factory=list)

    def scores_for_metric(self, metric: str) -> List[float]:
        """Extract per-sample scores for a single metric.

        Args:
            metric: Metric name (e.g., "faithfulness")

        Returns:
            List of per-sample scores.

        Raises:
            ValueError: If no per-sample scores are available.
        """
        if not self.per_sample:
            raise ValueError(
                f"No per-sample scores available for metric '{metric}'. "
                "Ensure evaluate_detailed() completed successfully."
            )
        return [s.get(metric, 0.0) for s in self.per_sample]
