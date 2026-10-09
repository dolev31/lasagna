"""KnowledgePipeline: composable RAG pipeline for researchers.

Allows researchers to compose, swap, evaluate, and serialize RAG pipelines
for reproducible experiments.

Example:
    from lasagna.core.pipeline import KnowledgePipeline

    # Build pipeline with defaults
    pipeline = KnowledgePipeline()

    # Swap a component
    pipeline.replace("reranker", "my_novel_reranker")

    # Evaluate (scores pre-supplied data, does NOT run retrieval)
    results = pipeline.evaluate(dataset=my_dataset, metrics=["faithfulness"])

    # Run + evaluate (retrieves through the pipeline, then scores)
    results = pipeline.run_and_evaluate(dataset=my_dataset, metrics=["faithfulness"])

    # Serialize for reproducibility
    config = pipeline.serialize()
"""

import json
import os
import sys
import time
import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger

from .registry import registry, COMPONENT_TYPES


def _capture_environment() -> Dict[str, Any]:
    """Capture runtime environment for reproducibility metadata.

    This is attached to PipelineResult.metadata (run artifacts),
    NOT to pipeline.serialize() (which must stay deterministic).
    """
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": sys.platform,
    }
    try:
        from lasagna.sdk._version import __version__ as _lasagna_version

        env["lasagna_version"] = _lasagna_version
    except ImportError:  # pragma: no cover - version module always ships
        pass
    try:
        import torch

        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import sentence_transformers

        env["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass
    return env


def _warn_if_not_interchangeable(component_type: str, name: str) -> None:
    """Say so when a selected component does not implement its stage contract.

    Comparing components is the reason this package exists, and a comparison is
    only meaningful between components that speak the same contract. Some
    built-ins predate the typed contracts and take or return different shapes,
    so swapping one for a contract-compliant component measures the interface
    difference as well as the algorithm.

    This warns rather than raises: those components still work through the
    pipeline they were written for. But a researcher choosing one for a
    comparison should be told, and told which alternative is comparable, rather
    than discovering it from a TypeError three stages downstream.
    """
    try:
        lister = {
            "reranker": registry.list_rerankers,
            "chunker": registry.list_chunkers,
            "embedder": registry.list_embedders,
            "query_enhancer": registry.list_enhancers,
        }.get(component_type)
        if lister is None:
            return

        entries = {e["name"]: e for e in lister()}
        chosen = entries.get(name)
        if chosen is None or chosen.get("implements_contract", True):
            return

        alternatives = sorted(n for n, e in entries.items() if e.get("implements_contract") is True)
        message = (
            f"{component_type} '{name}' does not implement the "
            f"{component_type} contract, so it is not interchangeable with "
            f"components that do. A comparison against one of "
            f"{alternatives} would also be measuring the interface difference. "
            f"Contract-compliant {component_type}s: {alternatives or 'none registered'}"
        )
        logger.warning(message)
        # The package's logs are off unless the host turns them on, and this
        # is the one warning a researcher writing a script must see.
        warnings.warn(message, UserWarning, stacklevel=3)
    except Exception:  # pragma: no cover - advisory only, must never block a swap
        pass


class PipelineResult:
    """Result from a pipeline evaluation run.

    Attributes:
        metrics: Aggregate metric scores (e.g., {"faithfulness": 0.87}).
        pipeline_config: Serialized pipeline config (deterministic).
        duration_seconds: Wall-clock time for the evaluation.
        metadata: Run-level metadata including environment info.
        per_sample: Optional per-sample scores for significance testing.
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        pipeline_config: Dict[str, str],
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
        per_sample: Optional[List[Dict[str, float]]] = None,
    ):
        self.metrics = metrics
        self.pipeline_config = pipeline_config
        self.duration_seconds = duration_seconds
        self.per_sample = per_sample or []
        self.metadata = metadata or {}
        # Auto-capture environment in run artifacts (not in serialize())
        self.metadata.setdefault("environment", _capture_environment())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "metrics": self.metrics,
            "pipeline_config": self.pipeline_config,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": self.metadata,
        }
        if self.per_sample:
            result["per_sample"] = self.per_sample
        return result

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"PipelineResult({metrics_str}, duration={self.duration_seconds:.1f}s)"


#: Stages a swapped component actually reaches at query time.
#:
#: The registry accepts seven component types, and ``KnowledgePipeline`` accepts
#: all seven as constructor arguments and serializes them. Only these four are
#: injected into the search path. ``chunker``, ``retriever`` and
#: ``document_processor`` act at *ingest* time, and ingest is configured through
#: a process-global backend rather than handed component instances -- so passing
#: one is recorded and has no effect on the pipeline that runs.
#:
#: Kept as a constant because the README's central claim is a count. A number in
#: prose drifts; a number derived from here cannot.
QUERY_TIME_STAGES = ("reranker", "query_enhancer", "embedder", "agentic_pipeline", "retriever")

#: Stages injected at ingest time, through `ingest()`.
INGEST_TIME_STAGES = ("chunker", "document_processor")

#: The stage that writes the answer from what retrieval returned. The pipeline
#: retrieves; whoever answers (the evaluation harness, an application) asks the
#: pipeline for its generator with `get_component("generator")`.
ANSWER_TIME_STAGES = ("generator",)

#: Empty: every registrable stage now reaches the pipeline. Kept as a named
#: constant rather than deleted so a future half-wired stage has an honest
#: place to be declared, instead of being silently absent from both lists.
INGEST_TIME_STAGES_NOT_YET_INJECTED: tuple = ()


class KnowledgePipeline:
    """Composable RAG pipeline that uses registered components.

    The pipeline resolves component names from the registry at execution
    time, allowing researchers to swap any component and re-run experiments.

    Config application:
        When ``config`` is provided (directly or via ``from_profile()``),
        the pipeline applies it as session-scoped overrides before each
        ``query()`` and ``ingest()`` call. This ensures profile settings
        (e.g., enable_reranking, chunk_size) actually affect execution.

    Component overrides:
        When components are registered/replaced in the pipeline, they are
        passed as overrides to the search orchestrator so that the
        researcher's custom implementations are actually used during search.
    """

    def __init__(
        self,
        embedder: Optional[str] = None,
        chunker: Optional[str] = None,
        reranker: Optional[str] = None,
        retriever: Optional[str] = None,
        document_processor: Optional[str] = None,
        query_enhancer: Optional[str] = None,
        agentic_pipeline: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize pipeline with component names.

        Args:
            embedder: Registered embedder name (or None for default)
            chunker: Registered chunker name (or None for default)
            reranker: Registered reranker name (or None for default)
            retriever: Registered retriever name (or None for default)
            document_processor: Registered processor name (or None for default)
            query_enhancer: Registered enhancer name (or None for default)
            agentic_pipeline: Registered agentic pipeline name (or None for
                config-driven resolution).  When set, queries are forced
                through the COMPREHENSIVE routing tier.
            config: Additional pipeline configuration (applied as session overrides)
        """
        self._component_names: Dict[str, Optional[str]] = {
            "embedder": embedder,
            "chunker": chunker,
            "reranker": reranker,
            "retriever": retriever,
            "document_processor": document_processor,
            "query_enhancer": query_enhancer,
            "agentic_pipeline": agentic_pipeline,
        }
        self._config = config or {}
        self._instances: Dict[str, Any] = {}
        self._hooks: Dict[str, Dict[str, List]] = {}
        self._node_overrides: Dict[str, str] = {}

    # =========================================================================
    # Component management
    # =========================================================================

    def replace(self, component_type: str, name: str) -> "KnowledgePipeline":
        """Replace a component in the pipeline.

        Args:
            component_type: Component type to replace
            name: Registered name of the new component

        Returns:
            self (for chaining)
        """
        if component_type not in COMPONENT_TYPES:
            raise ValueError(
                f"Unknown component type '{component_type}'. Valid types: {sorted(COMPONENT_TYPES)}"
            )
        if not registry.has(component_type, name):
            raise KeyError(
                f"No {component_type} named '{name}' in registry. "
                f"Available: {registry.list(component_type)}"
            )
        _warn_if_not_interchangeable(component_type, name)
        self._component_names[component_type] = name
        # Clear cached instance so it's re-created on next access
        self._instances.pop(component_type, None)
        return self

    def register(
        self,
        component_type: str,
        name: str,
        cls: Type,
        **kwargs,
    ) -> "KnowledgePipeline":
        """Register a component class and set it as the active component.

        Convenience method combining ``registry.register_class()`` and
        ``pipeline.replace()`` in one call.

        Args:
            component_type: Component type (e.g., "reranker")
            name: Unique name for this implementation
            cls: The component class
            **kwargs: Additional kwargs passed to registry

        Returns:
            self (for chaining)
        """
        registry.register_class(component_type, name, cls, **kwargs)
        return self.replace(component_type, name)

    def get_component(self, component_type: str, **kwargs) -> Any:
        """Get an instantiated component from the pipeline.

        Args:
            component_type: Component type to get
            **kwargs: Constructor arguments (if not already instantiated)

        Returns:
            Component instance, or None if no component is set for this type.
        """
        if component_type in self._instances:
            return self._instances[component_type]

        name = self._component_names.get(component_type)
        if name is None:
            return None

        instance = registry.get(component_type, name, **kwargs)
        self._instances[component_type] = instance
        return instance

    def generate(
        self, query: str, contexts: List[str], *, llm: Any = None, instructions: str = ""
    ) -> str:
        """Write the answer from ``contexts`` with the pipeline's generator.

        ``concise`` unless ``replace("generator", ...)`` named another; ``llm``
        is the model the generator calls (a strategy may bring its own).
        """
        generator = self.get_component("generator") or registry.get("generator", "concise")
        return generator.generate(query, list(contexts), llm=llm, instructions=instructions)

    # =========================================================================
    # Agentic pipeline / node overrides
    # =========================================================================

    def set_node_override(self, node_name: str, registered_name: str) -> "KnowledgePipeline":
        """Override an individual graph node in the built-in structured pipeline.

        Args:
            node_name: Name of the graph node to replace (e.g., ``"retrieve"``).
            registered_name: Name of the graph node callable registered via
                ``registry.register_graph_node()``.

        Returns:
            self (for chaining)
        """
        from lasagna.agentic.graph_builder import VALID_NODE_NAMES

        if node_name not in VALID_NODE_NAMES:
            raise ValueError(f"Unknown node '{node_name}'. Valid: {sorted(VALID_NODE_NAMES)}")
        if not registry.has_graph_node(registered_name):
            raise KeyError(
                f"No graph_node '{registered_name}' in registry. "
                f"Available: {[n['name'] for n in registry.list_graph_nodes()]}"
            )
        self._node_overrides[node_name] = registered_name
        # Clear cached agentic pipeline instance
        self._instances.pop("agentic_pipeline", None)
        return self

    def _resolve_agentic_pipeline(self) -> Any:
        """Resolve and instantiate the agentic pipeline override.

        If node overrides are set but no explicit pipeline is configured,
        defaults to ``"default"`` (the built-in structured graph) so the
        node overrides are actually applied.
        """
        name = self._component_names.get("agentic_pipeline")
        if name is None and self._node_overrides:
            # Node overrides require the structured graph pipeline
            name = "default"
        if name is None:
            return None

        if "agentic_pipeline" in self._instances:
            return self._instances["agentic_pipeline"]

        node_overrides = self._resolve_node_overrides()
        instance = registry.get("agentic_pipeline", name, node_overrides=node_overrides)
        self._instances["agentic_pipeline"] = instance
        return instance

    def _resolve_node_overrides(self) -> Optional[Dict[str, Any]]:
        """Resolve registered node override names to callables."""
        if not self._node_overrides:
            return None
        return {
            node_name: registry.get_graph_node(reg_name)
            for node_name, reg_name in self._node_overrides.items()
        }

    # =========================================================================
    # Middleware / Hooks
    # =========================================================================

    def hook(
        self,
        component_type: str,
        when: str = "after",
    ) -> Callable:
        """Register a hook that fires before/after a component executes.

        Hooks are only active when queries go through ``KnowledgePipeline``
        (the research path). The production singleton path is not affected.

        Args:
            component_type: Component type to hook (e.g., ``"reranker"``).
            when: ``"before"`` or ``"after"``.

        Returns:
            Decorator that registers the hook function.

        Example::

            @pipeline.hook("reranker", "after")
            def log_reranking(result, duration_ms, **kwargs):
                print(f"Reranked {len(result)} docs in {duration_ms:.1f}ms")

            @pipeline.hook("reranker", "before")
            def log_query(query, documents, **kwargs):
                print(f"Reranking {len(documents)} docs for: {query}")
        """
        if when not in ("before", "after"):
            raise ValueError(f"'when' must be 'before' or 'after', got '{when}'")

        def decorator(fn: Callable) -> Callable:
            self._hooks.setdefault(component_type, {}).setdefault(when, []).append(fn)
            return fn

        return decorator

    def _wrap_with_hooks(self, component_type: str, instance: Any) -> Any:
        """Wrap a component instance with registered hooks.

        Returns a ``_HookedComponent`` proxy if hooks are registered,
        otherwise returns the instance unchanged.
        """
        hooks = self._hooks.get(component_type)
        if not hooks:
            return instance
        return _HookedComponent(instance, hooks)

    # =========================================================================
    # Session config application
    # =========================================================================

    def _apply_session_config(self, session_id: str) -> None:
        """Apply pipeline config as session-scoped overrides.

        This ensures that profile settings (enable_reranking, chunk_size,
        hybrid_alpha, etc.) actually affect the SearchOrchestrator's behavior
        when ``query()`` or ``ingest()`` is called.
        """
        if not self._config:
            return
        try:
            from .session_config import get_knowledge_config_manager

            mgr = get_knowledge_config_manager()
            mgr.patch_config(self._config, session_id=session_id)
        except Exception as e:
            from .strict import strict_stages

            if strict_stages():
                raise ValueError(f"Could not apply the pipeline's settings: {e}") from e
            logger.warning(f"Could not apply pipeline config to session: {e}")

    # =========================================================================
    # Evaluate (scores pre-supplied data — backward compat)
    # =========================================================================

    def evaluate(
        self,
        dataset: Any = None,
        metrics: Optional[List[str]] = None,
        evaluator: Any = None,
        **kwargs,
    ) -> PipelineResult:
        """Score pre-supplied data with an evaluator (does NOT run retrieval).

        This is a pure scoring method — it passes the dataset directly to
        the evaluator without running any retrieval through the pipeline.
        Use ``run_and_evaluate()`` when you want the pipeline to retrieve
        and then score.

        Args:
            dataset: Evaluation dataset (RAGAS EvaluationDataset, list of dicts, etc.)
            metrics: List of metric names to compute
            evaluator: Optional evaluator instance (e.g., RAGASEvaluator)
            **kwargs: Additional arguments passed to the evaluator

        Returns:
            PipelineResult with computed metrics
        """
        start = time.time()
        metrics_list = metrics or ["ndcg@10"]

        if evaluator is not None:
            result_metrics = evaluator.evaluate(
                dataset=dataset,
                pipeline=self,
                metrics=metrics_list,
                **kwargs,
            )
        else:
            logger.warning("No evaluator provided. Install ragas: pip install lasagna-rag[eval]")
            result_metrics = {}

        duration = time.time() - start
        return PipelineResult(
            metrics=result_metrics,
            pipeline_config=self.serialize(),
            duration_seconds=duration,
            metadata=kwargs,
        )

    # =========================================================================
    # Run + Evaluate (retrieves through the pipeline, then scores)
    # =========================================================================

    def run_and_evaluate(
        self,
        dataset: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        evaluator: Any = None,
        top_k: int = 5,
        session_id: str = "__pipeline_eval__",
        answer_generator: Any = None,
        **kwargs,
    ) -> PipelineResult:
        """Retrieve through the pipeline, then score with an evaluator.

        Unlike ``evaluate()`` which only scores pre-supplied data, this
        method actually runs each query through the pipeline's search path
        (honoring component overrides and config), collects the retrieved
        contexts, and then scores the results. This is essential for valid
        ablation studies — swapping a component actually changes the results.

        Dataset format:
            Each dict should have at minimum:
            - ``"question"`` or ``"user_input"``: the query string
            - ``"ground_truth"`` or ``"reference"``: expected answer (optional)
            - ``"expected_docs"``: list of relevant doc IDs (optional, for IR metrics)

        Args:
            dataset: List of evaluation case dicts.
            metrics: Metric names for the evaluator (e.g., ["faithfulness"]).
            evaluator: Evaluator instance (e.g., RAGASEvaluator). If None,
                only IR metrics (precision, recall) are computed.
            top_k: Number of results to retrieve per query.
            session_id: Session namespace for isolation.
            answer_generator: Optional callable(query, context) -> str that
                generates an answer from retrieved context. If None, the
                concatenated chunk contents are used as the "answer".
            **kwargs: Extra arguments forwarded to the evaluator.

        Returns:
            PipelineResult with metrics and per-sample scores.
        """
        if not dataset:
            logger.warning("Empty dataset provided to run_and_evaluate()")
            return PipelineResult(
                metrics={},
                pipeline_config=self.serialize(),
                duration_seconds=0.0,
                metadata={"num_samples": 0, "top_k": top_k},
            )

        start = time.time()
        if evaluator is not None:
            metrics_list = metrics or ["faithfulness"]
        else:
            metrics_list = metrics  # IR metrics are auto-computed, no defaults needed

        # Apply pipeline config so search uses our settings
        self._apply_session_config(session_id)

        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[Optional[str]] = []
        per_sample_latencies: List[float] = []
        all_retrieved_ids: List[List[str]] = []

        for case in dataset:
            question = case.get("question") or case.get("user_input", "")
            ground_truth = case.get("ground_truth")
            if ground_truth is None:
                ground_truth = case.get("reference")

            # Run retrieval through the pipeline
            q_start = time.time()
            try:
                search_results = self.query(
                    query=question,
                    top_k=top_k,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{question[:50]}...': {e}")
                search_results = []
            q_latency = (time.time() - q_start) * 1000

            # Extract context chunks and doc IDs from search results
            chunk_texts = []
            retrieved_ids = []
            for r in search_results:
                content = ""
                doc_id = None
                if hasattr(r, "chunk") and hasattr(r.chunk, "content"):
                    content = r.chunk.content
                    if hasattr(r.chunk, "document_id"):
                        doc_id = str(r.chunk.document_id)
                elif hasattr(r, "content"):
                    content = r.content
                    # ScoredDocument and similar objects store doc_id at top level
                    if hasattr(r, "document_id") and r.document_id:
                        doc_id = str(r.document_id)
                elif isinstance(r, dict):
                    content = r.get("content", "")
                    doc_id = str(r.get("document_id", ""))
                if content:
                    chunk_texts.append(content)
                if doc_id:
                    retrieved_ids.append(doc_id)
            all_retrieved_ids.append(retrieved_ids)

            # Generate answer from context
            if answer_generator is not None:
                context_str = "\n\n".join(chunk_texts)
                try:
                    answer = answer_generator(question, context_str)
                    if answer is None:
                        answer = context_str
                except Exception as e:
                    logger.warning(f"Answer generation failed: {e}")
                    answer = context_str
            else:
                answer = "\n\n".join(chunk_texts)

            questions.append(question)
            answers.append(answer)
            contexts.append(chunk_texts)
            ground_truths.append(ground_truth)
            per_sample_latencies.append(q_latency)

        # Score with evaluator
        result_metrics: Dict[str, float] = {}
        per_sample_scores: List[Dict[str, float]] = []

        if evaluator is not None and answer_generator is None:
            logger.warning(
                "No answer_generator provided — answer-level metrics "
                "(faithfulness, relevancy) are evaluated on raw concatenated "
                "chunks, which may produce misleading scores. Provide "
                "answer_generator=... for reliable answer-quality metrics."
            )

        if evaluator is not None:
            # Ground truths must stay positionally aligned with questions.
            # Replace None entries with empty strings rather than filtering,
            # which would shift indices and corrupt per-sample score alignment.
            aligned_ground_truths = [gt if gt is not None else "" for gt in ground_truths]

            # Try detailed evaluation first (preserves per-sample scores)
            if hasattr(evaluator, "evaluate_detailed"):
                try:
                    detailed = evaluator.evaluate_detailed(
                        questions=questions,
                        answers=answers,
                        contexts=contexts,
                        ground_truths=aligned_ground_truths,
                        metrics=metrics_list,
                    )
                    result_metrics = detailed.aggregate
                    per_sample_scores = detailed.per_sample
                except Exception as e:
                    logger.warning(f"Detailed evaluation failed, falling back to aggregate: {e}")

            # Fall back to aggregate-only evaluation
            if not result_metrics and hasattr(evaluator, "evaluate_from_samples"):
                try:
                    result_metrics = evaluator.evaluate_from_samples(
                        questions=questions,
                        answers=answers,
                        contexts=contexts,
                        ground_truths=aligned_ground_truths,
                        metrics=metrics_list,
                    )
                except Exception as e:
                    logger.error(f"Evaluation scoring failed: {e}")
        else:
            # Compute IR metrics when evaluator is not provided
            has_ir_data = any(
                case.get("expected_docs")  # truthy: non-empty dict/list
                for case in dataset
            )
            if has_ir_data:
                from ..knowledge_evaluation_sdk.ir_metrics import corpus_metrics

                relevance_maps = []
                for case in dataset:
                    expected = case.get("expected_docs")
                    if isinstance(expected, dict):
                        relevance_maps.append(expected)
                    elif isinstance(expected, list):
                        relevance_maps.append({str(d): 1 for d in expected})
                    else:
                        relevance_maps.append({})

                result_metrics = corpus_metrics(relevance_maps, all_retrieved_ids, k=top_k)
                logger.info(
                    f"Computed IR metrics (no evaluator): "
                    f"{', '.join(f'{k}={v:.4f}' for k, v in result_metrics.items())}"
                )
            else:
                logger.info(
                    "No evaluator and no expected_docs in dataset. "
                    "Returning retrieval-only results without metrics."
                )

        duration = time.time() - start

        # User-provided kwargs go first; computed values override to prevent
        # accidental overwriting of reserved keys like num_samples.
        metadata = {**kwargs}
        metadata.update(
            {
                "num_samples": len(dataset),
                "top_k": top_k,
                "avg_retrieval_latency_ms": (
                    sum(per_sample_latencies) / len(per_sample_latencies)
                    if per_sample_latencies
                    else 0
                ),
            }
        )

        return PipelineResult(
            metrics=result_metrics,
            pipeline_config=self.serialize(),
            duration_seconds=duration,
            metadata=metadata,
            per_sample=per_sample_scores,
        )

    def serialize(self) -> Dict[str, Any]:
        """Serialize pipeline configuration for reproducibility.

        Returns only deterministic configuration — no timestamps, hardware
        info, or other non-deterministic data. Environment metadata is
        captured in ``PipelineResult.metadata`` instead.

        Returns:
            Dict with all component names and configuration
        """
        result: Dict[str, Any] = {
            "components": {
                ct: name for ct, name in self._component_names.items() if name is not None
            },
            "config": self._config,
        }
        if self._node_overrides:
            result["node_overrides"] = dict(self._node_overrides)
        return result

    # =========================================================================
    # Profile-based factory
    # =========================================================================

    @classmethod
    def from_profile(cls, profile_name: str) -> "KnowledgePipeline":
        """Create a pipeline pre-configured from a built-in profile.

        Loads the named profile (speed / standard / balanced / max_quality /
        default) and stores its configuration dict. The config is applied
        as session-scoped overrides when ``query()`` or ``ingest()`` is
        called, ensuring the profile settings actually affect execution.

        Args:
            profile_name: One of the built-in profile names.

        Returns:
            KnowledgePipeline with profile config stored.
        """
        from .session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager()
        profile = next(
            (p for p in mgr._default_profiles if p.name == profile_name),
            None,
        )
        if profile is None:
            available = [p.name for p in mgr._default_profiles]
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
        return cls(config=profile.config)

    @classmethod
    def with_defaults(cls) -> "KnowledgePipeline":
        """Create a pipeline with all built-in defaults pre-registered.

        Convenience factory for researchers who want a ready-to-use pipeline
        without needing to understand profiles or configuration.

        Returns:
            KnowledgePipeline ready for use with default components.
        """
        return cls.from_profile("default")

    # =========================================================================
    # Ingest & query
    # =========================================================================

    def ingest(
        self,
        path: str,
        session_id: str = "__pipeline__",
    ) -> Any:
        """Ingest a document (or directory) into the knowledge base.

        Uses the full RAG document pipeline:
        extraction -> chunking -> embedding -> indexing.

        Applies pipeline config as session overrides before processing.

        Args:
            path: File path or directory to ingest.
            session_id: Session namespace for isolation.

        Returns:
            Document metadata, or list of Document metadata when *path*
            is a directory.
        """
        from lasagna.api.rag_backend import get_rag_backend

        # Apply pipeline config so chunking/processing uses our settings
        self._apply_session_config(session_id)

        backend = get_rag_backend()

        # A registered chunker is injected here, the same way a reranker is
        # injected into the query path. Before this, `KnowledgePipeline(
        # chunker=...)` was accepted, serialized, and never reached ingestion --
        # so a chunking sweep produced identical arms and a difference of zero.
        chunker = self.get_component("chunker")
        if chunker is not None:
            chunker = self._wrap_with_hooks("chunker", chunker)
        processor = self.get_component("document_processor")
        if processor is not None:
            processor = self._wrap_with_hooks("document_processor", processor)

        if os.path.isdir(path):
            results = []
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    doc = backend.upload_document(
                        file_path=fpath,
                        session_id=session_id,
                        filename=fname,
                        _chunker_override=chunker,
                        _document_processor_override=processor,
                    )
                    results.append(doc)
            return results

        return backend.upload_document(
            file_path=path,
            session_id=session_id,
            filename=os.path.basename(path),
            _chunker_override=chunker,
            _document_processor_override=processor,
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
        session_id: str = "__pipeline__",
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Run the full SOTA search pipeline.

        Applies pipeline config as session overrides and passes any
        registered component overrides to the search orchestrator.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            session_id: Session namespace for isolation.
            config_overrides: Optional flat dict of config keys to override for
                this single query call only (e.g. ``{"enable_reranking": False}``).
                Takes highest priority; never written to the session store.
                Used by the eval SDK for lock-free concurrent ablation studies.

        Returns:
            List of SearchResult objects with ``.chunk``, ``.score``,
            and ``.document`` attributes.
        """
        from lasagna.api.rag_backend import get_rag_backend

        # Apply pipeline config as session-scoped overrides so the search
        # orchestrator sees our settings.  When config_overrides are supplied
        # (eval SDK / ablation path) the caller takes responsibility for
        # propagating all relevant settings via the per-query override mechanism,
        # so we skip the session store write entirely — this avoids concurrent
        # tasks racing to overwrite each other's config on a shared session ID.
        if not config_overrides:
            self._apply_session_config(session_id)

        backend = get_rag_backend()

        # Resolve component overrides from the registry, wrapped with hooks
        override_kwargs: Dict[str, Any] = {}
        reranker = self.get_component("reranker")
        if reranker is not None:
            override_kwargs["_reranker_override"] = self._wrap_with_hooks("reranker", reranker)
        enhancer = self.get_component("query_enhancer")
        if enhancer is not None:
            override_kwargs["_enhancer_override"] = self._wrap_with_hooks(
                "query_enhancer", enhancer
            )
        embedder = self.get_component("embedder")
        if embedder is not None:
            override_kwargs["_embedder_override"] = self._wrap_with_hooks("embedder", embedder)
        retriever = self.get_component("retriever")
        if retriever is not None:
            override_kwargs["_retriever_override"] = self._wrap_with_hooks("retriever", retriever)

        # Resolve agentic pipeline / node overrides
        pipeline_name = self._component_names.get("agentic_pipeline")
        if pipeline_name or self._node_overrides:
            agentic_pipeline = self._resolve_agentic_pipeline()
            if agentic_pipeline is not None:
                override_kwargs["_agentic_pipeline_override"] = agentic_pipeline
            node_overrides = self._resolve_node_overrides()
            if node_overrides:
                override_kwargs["_graph_node_overrides"] = node_overrides
            # Force comprehensive tier so the agentic path is taken
            override_kwargs["routing_tier"] = "comprehensive"

        # When config_overrides request agentic RAG, force routing to
        # comprehensive tier so the agentic pipeline is actually invoked.
        # Without this, the query router classifies queries independently
        # and routes to "standard" regardless of use_agentic_rag.
        if config_overrides and config_overrides.get("use_agentic_rag"):
            override_kwargs.setdefault("routing_tier", "comprehensive")

        if config_overrides:
            override_kwargs["query_config_overrides"] = config_overrides

        return backend.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            **override_kwargs,
        )

    # =========================================================================
    # Serialization helpers
    # =========================================================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePipeline":
        """Create a pipeline from a serialized dict.

        Args:
            data: Dict from serialize()

        Returns:
            KnowledgePipeline instance
        """
        components = data.get("components", {})
        config = data.get("config", {})
        pipeline = cls(
            embedder=components.get("embedder"),
            chunker=components.get("chunker"),
            reranker=components.get("reranker"),
            retriever=components.get("retriever"),
            document_processor=components.get("document_processor"),
            query_enhancer=components.get("query_enhancer"),
            agentic_pipeline=components.get("agentic_pipeline"),
            config=config,
        )
        # Restore node overrides
        for node_name, reg_name in data.get("node_overrides", {}).items():
            pipeline._node_overrides[node_name] = reg_name
        return pipeline

    @classmethod
    def from_json(cls, path: str) -> "KnowledgePipeline":
        """Load pipeline from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> "KnowledgePipeline":
        """Load pipeline from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_json(self, path: str) -> None:
        """Save pipeline to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    def to_yaml(self, path: str) -> None:
        """Save pipeline to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(self.serialize(), f, default_flow_style=False)

    def __repr__(self) -> str:
        active = {k: v for k, v in self._component_names.items() if v}
        return f"KnowledgePipeline({active})"


# =============================================================================
# Hook proxy
# =============================================================================


class _HookedComponent:
    """Transparent proxy that fires before/after hooks around method calls.

    Wraps any component instance so that when the search orchestrator
    calls its methods (e.g. ``reranker.rerank()``), the registered
    hooks execute. The proxy delegates attribute access to the underlying
    instance, intercepting callable attributes to inject timing and hooks.
    """

    def __init__(self, wrapped: Any, hooks: Dict[str, List[Callable]]):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_hooks", hooks)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr

        before_hooks = self._hooks.get("before", [])
        after_hooks = self._hooks.get("after", [])

        if not before_hooks and not after_hooks:
            return attr

        @functools.wraps(attr)
        def hooked(*args, **kwargs):
            # Fire "before" hooks
            for hook_fn in before_hooks:
                try:
                    hook_fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Before-hook failed: {e}")

            # Execute the real method with timing
            start = time.perf_counter()
            result = attr(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            # Fire "after" hooks
            for hook_fn in after_hooks:
                try:
                    hook_fn(result=result, duration_ms=duration_ms, **kwargs)
                except Exception as e:
                    logger.warning(f"After-hook failed: {e}")

            return result

        return hooked
