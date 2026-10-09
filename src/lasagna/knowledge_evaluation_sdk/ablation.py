"""Ablation study helper for RAG pipeline experiments.

Automates the process of sweeping component variants and comparing results.

Example:
    from lasagna.knowledge_evaluation_sdk.ablation import sweep
    from lasagna.core.pipeline import KnowledgePipeline

    base_pipeline = KnowledgePipeline(reranker="default")
    results = sweep(
        pipeline=base_pipeline,
        component="reranker",
        variants=["mxbai-rerank-large", "ms-marco-MiniLM", "bge-reranker-v2"],
        dataset=my_dataset,
        metrics=["faithfulness", "context_precision"],
        evaluator=my_evaluator,
    )
    print(results.to_dataframe())
    results.plot()

    # Significance testing (automatic with per-sample scores)
    sig = results.significance_test("mxbai-rerank-large", "bge-reranker-v2", "faithfulness")
    print(f"p={sig['p_value']:.4f}, significant={sig['significant']}")

    # Persistence
    results.save("reranker_sweep.json")
    loaded = AblationResults.load("reranker_sweep.json")
"""

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text.

    Uses a placeholder strategy so that command-based escapes (which
    contain ``{}``) are not double-escaped by the brace replacement step.
    """
    # Phase 1: Replace chars whose escapes use LaTeX commands (contain braces)
    # with null-byte-delimited placeholders that won't be touched by Phase 2.
    _PH = "\x00"
    placeholders = {
        "\\": f"{_PH}BACKSLASH{_PH}",
        "~": f"{_PH}TILDE{_PH}",
        "^": f"{_PH}CARET{_PH}",
    }
    for char, ph in placeholders.items():
        text = text.replace(char, ph)

    # Phase 2: Simple prefix escapes (these don't contain braces).
    for char in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, f"\\{char}")

    # Phase 3: Swap placeholders for the real LaTeX commands.
    text = text.replace(f"{_PH}BACKSLASH{_PH}", r"\textbackslash{}")
    text = text.replace(f"{_PH}TILDE{_PH}", r"\textasciitilde{}")
    text = text.replace(f"{_PH}CARET{_PH}", r"\textasciicircumflex{}")
    return text


class AblationResults:
    """Container for ablation study results with analysis utilities.

    Stores both aggregate metrics and per-sample scores (when available)
    for each variant. Per-sample scores are automatically used by
    ``significance_test()`` when available.
    """

    def __init__(
        self,
        results: List[Dict[str, Any]],
        per_sample_scores: Optional[Dict[str, List[Dict[str, float]]]] = None,
    ):
        """Initialize with a list of result dicts.

        Args:
            results: List of dicts with keys: variant, metrics,
                duration_seconds, pipeline_config.
            per_sample_scores: Optional dict mapping variant names to lists
                of per-sample score dicts. Each inner dict maps metric names
                to float scores for one sample.
        """
        self._results = results

        # Accept per-sample scores from either the explicit argument
        # or inline in each result dict under the "per_sample" key.
        pss: Dict[str, List[Dict[str, float]]] = dict(per_sample_scores or {})
        for r in results:
            inline = r.get("per_sample")
            if inline and r.get("variant") and r["variant"] not in pss:
                # Convert {metric: [scores]} → [{metric: score}, ...]
                if isinstance(inline, dict):
                    first_val = next(iter(inline.values()), None)
                    if isinstance(first_val, list):
                        n_samples = len(first_val)
                        pss[r["variant"]] = [
                            {m: vals[i] for m, vals in inline.items()} for i in range(n_samples)
                        ]
        self._per_sample_scores = pss

    @property
    def variants(self) -> List[str]:
        return [r["variant"] for r in self._results]

    @property
    def metrics(self) -> Dict[str, List[float]]:
        """Get metrics organized by metric name.

        Derives metric names from ALL results, not just the first.
        This handles cases where the first variant errored (empty metrics).
        """
        if not self._results:
            return {}
        all_metric_names: set = set()
        for r in self._results:
            all_metric_names.update(r.get("metrics", {}).keys())
        metric_names = sorted(all_metric_names)
        return {name: [r["metrics"].get(name, 0.0) for r in self._results] for name in metric_names}

    def to_dict(self) -> List[Dict[str, Any]]:
        return self._results

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis.

        Returns:
            pandas DataFrame with variants as rows and metrics as columns
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install lasagna-rag[eval]")

        rows = []
        for r in self._results:
            row = {"variant": r["variant"], "duration_s": r["duration_seconds"]}
            row.update(r.get("metrics", {}))
            rows.append(row)
        return pd.DataFrame(rows).set_index("variant")

    def plot(self, figsize=(10, 6), title: Optional[str] = None):
        """Plot comparison chart of metrics across variants.

        Args:
            figsize: Figure size tuple
            title: Optional plot title

        Returns:
            matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required: pip install lasagna-rag[eval]")

        df = self.to_dataframe()
        metric_cols = [c for c in df.columns if c != "duration_s"]

        fig, ax = plt.subplots(figsize=figsize)
        df[metric_cols].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title(title or "Ablation Study: Component Comparison")
        ax.legend(loc="best")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def best(self, metric: str) -> str:
        """Return the variant name with the highest score for a metric.

        Raises:
            ValueError: If there are no results or no variant has the metric.
        """
        if not self._results:
            raise ValueError("No results to determine best variant")
        best_idx = -1
        best_score = -float("inf")
        for i, r in enumerate(self._results):
            score = r.get("metrics", {}).get(metric, -float("inf"))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1 or best_score == -float("inf"):
            raise ValueError(
                f"No variant has metric '{metric}'. "
                f"Available metrics: {sorted(self.metrics.keys())}"
            )
        return self._results[best_idx]["variant"]

    def significance_test(
        self,
        variant_a: str,
        variant_b: str,
        metric: str,
        scores_a: Optional[List[float]] = None,
        scores_b: Optional[List[float]] = None,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Paired significance test between two variants.

        The same test as ``BenchmarkResults.significance_test``: a sign-flip
        randomization p-value (exact up to 30 non-zero differences), a t
        interval at ``1 - alpha``, and Cohen's ``d_z``. This used to be its own
        paired bootstrap, which rejected a true null about 9% of the time at
        12 queries, and its own effect size, which returned 0.0 for a perfectly
        consistent effect.

        When ``scores_a`` / ``scores_b`` are not provided, the method
        automatically looks up stored per-sample scores from the ablation
        run. This makes significance testing seamless after ``sweep()``
        or ``grid_search()``.

        Args:
            variant_a: First variant name (must be in ``self.variants``).
            variant_b: Second variant name (must be in ``self.variants``).
            metric: Metric name to compare.
            scores_a: Per-sample scores for variant A. If ``None``,
                auto-retrieved from stored per-sample scores.
            scores_b: Per-sample scores for variant B (same length).
                If ``None``, auto-retrieved from stored per-sample scores.
            n_bootstrap: Random sign patterns drawn when the sample is too
                large to enumerate. The name is kept for compatibility.
            alpha: Significance level (default 0.05).
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys: ``p_value``, ``significant``,
            ``confidence_interval_95``, ``effect_size``,
            ``mean_a``, ``mean_b``, ``difference``, ``n_samples``.

        Raises:
            ImportError: If numpy is not installed.
            ValueError: If score arrays are missing or mismatched.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required: pip install lasagna-rag[eval]")

        # Auto-retrieve per-sample scores from stored data when not explicitly
        # provided. Only positions both variants scored are kept: a missing
        # score used to be read as 0.0, which is a measurement nobody made.
        if (
            scores_a is None
            and scores_b is None
            and variant_a in self._per_sample_scores
            and variant_b in self._per_sample_scores
        ):
            pairs = [
                (sa[metric], sb[metric])
                for sa, sb in zip(
                    self._per_sample_scores[variant_a], self._per_sample_scores[variant_b]
                )
                if metric in sa and metric in sb
            ]
            scores_a = [x for x, _ in pairs]
            scores_b = [y for _, y in pairs]
        for side, variant in (("a", variant_a), ("b", variant_b)):
            stored = self._per_sample_scores.get(variant)
            if (scores_a if side == "a" else scores_b) is None and stored:
                if any(metric not in s for s in stored):
                    raise ValueError(
                        f"Variant {variant!r} did not score {metric!r} on every sample; "
                        "pass both score lists, or neither so they are paired."
                    )
                filled = [s[metric] for s in stored]
                scores_a, scores_b = (filled, scores_b) if side == "a" else (scores_a, filled)

        if scores_a is None or scores_b is None:
            raise ValueError(
                "Per-sample scores required for significance testing. "
                "Either pass scores_a/scores_b explicitly, or use "
                "sweep()/grid_search() which collect them automatically."
            )

        from .results import BenchmarkResults, _sign_flip_p_value, _t_interval

        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        if len(a) != len(b):
            raise ValueError(f"Score arrays must have equal length ({len(a)} vs {len(b)})")

        n = len(a)
        if n < 2:
            raise ValueError(f"Need at least 2 paired samples for significance testing, got {n}")

        differences = a - b
        p_value, method = _sign_flip_p_value(differences, n_resamples=n_bootstrap, seed=seed)
        interval = _t_interval(differences, alpha)

        return {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "difference": float(np.mean(differences)),
            "p_value": p_value,
            "significant": p_value < alpha,
            "confidence_interval": interval,
            "confidence_level": 1.0 - alpha,
            "confidence_interval_95": interval,
            "ci_method": "t",
            "test": f"paired sign-flip ({method.replace('_', ' ')})",
            "effect_size": BenchmarkResults._cohens_dz(differences),
            "n_samples": n,
            "n_bootstrap": n_bootstrap,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save results to a JSON file for later analysis.

        Args:
            path: File path (should end in .json)
        """
        data = {
            "results": self._results,
            "per_sample_scores": self._per_sample_scores,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"AblationResults saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AblationResults":
        """Load results from a JSON file.

        Args:
            path: File path to load from

        Returns:
            AblationResults instance
        """
        with open(path) as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with per_sample)
        if isinstance(data, list):
            return cls(results=data)

        return cls(
            results=data.get("results", []),
            per_sample_scores=data.get("per_sample_scores"),
        )

    # =========================================================================
    # Export
    # =========================================================================

    def to_markdown(self, caption: Optional[str] = None) -> str:
        """Export results as a Markdown table.

        Args:
            caption: Optional caption above the table.

        Returns:
            Markdown-formatted string
        """
        if not self._results:
            return "*No results*"

        all_metric_names: set = set()
        for r in self._results:
            all_metric_names.update(r.get("metrics", {}).keys())
        metric_names = sorted(all_metric_names)
        lines = []

        if caption:
            lines.append(f"**{caption}**\n")

        # Header
        header = "| Variant | " + " | ".join(metric_names) + " | Duration (s) |"
        sep = "|" + "---|" * (len(metric_names) + 2)
        lines.append(header)
        lines.append(sep)

        # Find best formatted value per metric (compare strings to avoid float issues)
        best_per_metric: Dict[str, str] = {}
        for name in metric_names:
            scores = [r.get("metrics", {}).get(name, -float("inf")) for r in self._results]
            best_per_metric[name] = f"{max(scores):.4f}"

        # Rows
        for r in self._results:
            cells = [r["variant"]]
            for name in metric_names:
                val = r.get("metrics", {}).get(name, 0.0)
                formatted = f"{val:.4f}"
                if formatted == best_per_metric.get(name):
                    formatted = f"**{formatted}**"
                cells.append(formatted)
            duration = r.get("duration_seconds")
            cells.append(f"{duration:.1f}" if duration is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_latex(
        self,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        bold_best: bool = True,
    ) -> str:
        """Export results as a LaTeX table suitable for academic papers.

        Args:
            caption: Table caption.
            label: LaTeX label for cross-referencing.
            bold_best: Whether to bold the best score per metric.

        Returns:
            LaTeX-formatted string
        """
        if not self._results:
            return "% No results"

        all_metric_names: set = set()
        for r in self._results:
            all_metric_names.update(r.get("metrics", {}).keys())
        metric_names = sorted(all_metric_names)

        # Find best formatted value per metric (compare strings to avoid float issues)
        best_per_metric: Dict[str, str] = {}
        if bold_best:
            for name in metric_names:
                scores = [r.get("metrics", {}).get(name, -float("inf")) for r in self._results]
                best_per_metric[name] = f"{max(scores):.4f}"

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]
        if caption:
            lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        col_spec = "l" + "c" * len(metric_names)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header
        header_cells = ["Variant"] + [_escape_latex(m) for m in metric_names]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        # Rows
        for r in self._results:
            cells = [_escape_latex(r["variant"])]
            for name in metric_names:
                val = r.get("metrics", {}).get(name, 0.0)
                formatted = f"{val:.4f}"
                if bold_best and formatted == best_per_metric.get(name):
                    formatted = f"\\textbf{{{formatted}}}"
                cells.append(formatted)
            lines.append(" & ".join(cells) + " \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        if not self._results:
            return "AblationResults(empty)"
        n = len(self._results)
        has_per_sample = bool(self._per_sample_scores)
        return (
            f"AblationResults({n} variants, metrics={sorted(self.metrics)}, "
            f"per_sample={'yes' if has_per_sample else 'no'})"
        )


def sweep(
    pipeline: Any,
    component: str,
    variants: List[str],
    dataset: Any = None,
    metrics: Optional[List[str]] = None,
    evaluator: Any = None,
    **kwargs,
) -> AblationResults:
    """Run an ablation study by sweeping component variants.

    For each variant, replaces the specified component in the pipeline
    and runs ``run_and_evaluate()`` to generate answers through the
    pipeline and score them. This ensures that each variant actually
    affects the results.

    Args:
        pipeline: KnowledgePipeline instance (will be modified in-place)
        component: Component type to sweep (e.g., "reranker")
        variants: List of registered component names to test
        dataset: Evaluation dataset (list of dicts with 'question',
            'ground_truth', and optionally 'contexts' keys)
        metrics: List of metric names
        evaluator: Evaluator instance (e.g., RAGASEvaluator)
        **kwargs: Additional arguments passed to run_and_evaluate()

    Returns:
        AblationResults with per-variant metrics and per-sample scores
    """
    _INGESTION_ONLY = {"chunker", "document_processor"}
    if component in _INGESTION_ONLY:
        raise ValueError(
            f"Cannot ablate '{component}' at query time. "
            f"{component} is an ingestion-time component — documents must be "
            f"re-ingested with each variant for results to differ. "
            f"Use pipeline.ingest() with each variant before evaluating."
        )

    results = []
    per_sample_scores: Dict[str, List[Dict[str, float]]] = {}

    # Save original component name so we can restore it after the sweep.
    # This prevents the pipeline from being left in a modified state.
    original_name = pipeline._component_names.get(component)

    try:
        for variant_name in variants:
            logger.info(f"Ablation: testing {component}='{variant_name}'")
            start = time.time()

            try:
                # Swap component
                pipeline.replace(component, variant_name)

                # Run through pipeline and evaluate
                eval_result = pipeline.run_and_evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    evaluator=evaluator,
                    **kwargs,
                )

                duration = time.time() - start
                results.append(
                    {
                        "variant": variant_name,
                        "metrics": eval_result.metrics,
                        "duration_seconds": round(duration, 3),
                        "pipeline_config": eval_result.pipeline_config,
                    }
                )

                # Store per-sample scores if available
                if eval_result.per_sample:
                    per_sample_scores[variant_name] = eval_result.per_sample

                logger.info(
                    f"Ablation: {component}='{variant_name}' -> "
                    f"{eval_result.metrics} ({duration:.1f}s)"
                )

            except Exception as e:
                logger.error(f"Ablation: {component}='{variant_name}' failed: {e}")
                results.append(
                    {
                        "variant": variant_name,
                        "metrics": {},
                        "duration_seconds": time.time() - start,
                        "pipeline_config": {},
                        "error": str(e),
                    }
                )
    finally:
        # Restore the pipeline to its original state.
        # Only set the key if it existed before; otherwise remove it
        # to avoid adding spurious None entries.
        if original_name is not None:
            pipeline._component_names[component] = original_name
        else:
            pipeline._component_names.pop(component, None)
        pipeline._instances.pop(component, None)

    return AblationResults(results, per_sample_scores=per_sample_scores)


def sweep_pipelines(
    pipeline: Any,
    variants: List[str],
    dataset: Any = None,
    metrics: Optional[List[str]] = None,
    evaluator: Any = None,
    **kwargs,
) -> AblationResults:
    """Sweep agentic pipeline variants.

    Convenience wrapper around :func:`sweep` for the ``agentic_pipeline``
    component type.  Ensures queries route through the COMPREHENSIVE tier
    so the agentic path is actually exercised.

    Args:
        pipeline: KnowledgePipeline instance (modified in-place).
        variants: Registered agentic pipeline names to test (e.g.,
            ``["default", "legacy_react", "my_custom"]``).
        dataset: Evaluation dataset.
        metrics: Metric names to compute.
        evaluator: Evaluator instance.
        **kwargs: Passed to ``pipeline.run_and_evaluate()``.

    Returns:
        AblationResults with per-variant metrics.
    """
    # Force agentic path — override (not setdefault) to guarantee the
    # comprehensive tier is active regardless of existing config values.
    _original_config = dict(pipeline._config)
    pipeline._config["use_agentic_rag"] = True
    pipeline._config["routing_default_tier"] = "comprehensive"

    try:
        return sweep(
            pipeline=pipeline,
            component="agentic_pipeline",
            variants=variants,
            dataset=dataset,
            metrics=metrics,
            evaluator=evaluator,
            **kwargs,
        )
    finally:
        pipeline._config = _original_config


def sweep_nodes(
    pipeline: Any,
    node_name: str,
    variants: List[str],
    dataset: Any = None,
    metrics: Optional[List[str]] = None,
    evaluator: Any = None,
    **kwargs,
) -> AblationResults:
    """Sweep individual graph node variants in the built-in structured pipeline.

    For each variant, calls ``pipeline.set_node_override(node_name, variant)``
    and evaluates. The pipeline is restored to its original state after sweep.

    Requires ``use_agentic_rag=True`` and ``use_structured_graph=True`` (or
    ``agentic_pipeline="default"``) in the pipeline config.

    Args:
        pipeline: KnowledgePipeline instance (modified in-place).
        node_name: Graph node to override (e.g., ``"retrieve"``).
        variants: Registered graph node names to test.
        dataset: Evaluation dataset.
        metrics: Metric names to compute.
        evaluator: Evaluator instance.
        **kwargs: Passed to ``pipeline.run_and_evaluate()``.

    Returns:
        AblationResults with per-variant metrics.
    """
    # Force structured graph path — node overrides only apply to the built-in
    # "default" pipeline, so we must explicitly set agentic_pipeline too.
    _original_config = dict(pipeline._config)
    _original_pipeline = pipeline._component_names.get("agentic_pipeline")
    pipeline._config["use_agentic_rag"] = True
    pipeline._config["use_structured_graph"] = True
    pipeline._config["routing_default_tier"] = "comprehensive"
    pipeline._component_names["agentic_pipeline"] = "default"

    original_overrides = dict(pipeline._node_overrides)
    results_list: List[Dict[str, Any]] = []
    per_sample_scores: Dict[str, List[Dict[str, float]]] = {}

    try:
        for variant_name in variants:
            logger.info(f"Node ablation: testing {node_name}='{variant_name}'")
            start = time.time()

            try:
                pipeline.set_node_override(node_name, variant_name)

                eval_result = pipeline.run_and_evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    evaluator=evaluator,
                    **kwargs,
                )

                duration = time.time() - start
                results_list.append(
                    {
                        "variant": f"{node_name}={variant_name}",
                        "metrics": eval_result.metrics,
                        "duration_seconds": round(duration, 3),
                        "pipeline_config": eval_result.pipeline_config,
                    }
                )

                if eval_result.per_sample:
                    per_sample_scores[f"{node_name}={variant_name}"] = eval_result.per_sample

                logger.info(
                    f"Node ablation: {node_name}='{variant_name}' -> "
                    f"{eval_result.metrics} ({duration:.1f}s)"
                )

            except Exception as e:
                logger.error(f"Node ablation: {node_name}='{variant_name}' failed: {e}")
                results_list.append(
                    {
                        "variant": f"{node_name}={variant_name}",
                        "metrics": {},
                        "duration_seconds": time.time() - start,
                        "pipeline_config": {},
                        "error": str(e),
                    }
                )
    finally:
        # Restore original state
        pipeline._node_overrides = original_overrides
        pipeline._instances.pop("agentic_pipeline", None)
        pipeline._config = _original_config
        # Restore original pipeline component name
        if _original_pipeline is not None:
            pipeline._component_names["agentic_pipeline"] = _original_pipeline
        else:
            pipeline._component_names.pop("agentic_pipeline", None)

    return AblationResults(results_list, per_sample_scores=per_sample_scores)


def grid_search(
    pipeline: Any,
    components: Dict[str, List[str]],
    dataset: Any = None,
    metrics: Optional[List[str]] = None,
    evaluator: Any = None,
    **kwargs,
) -> AblationResults:
    """Run a full factorial sweep over multiple component dimensions.

    Unlike :func:`sweep`, which varies a single component, ``grid_search``
    tests every combination of variants across multiple component slots.

    Uses ``run_and_evaluate()`` to generate answers through the pipeline
    and score them, ensuring that component changes are reflected in results.

    Example::

        results = grid_search(
            pipeline=KnowledgePipeline(),
            components={
                "reranker": ["mxbai-rerank-large", "bge-reranker-v2"],
                "chunker": ["fixed", "semantic"],
            },
            dataset=my_dataset,
            metrics=["faithfulness"],
            evaluator=my_evaluator,
        )
        # Tests 2 x 2 = 4 configurations

    Args:
        pipeline: KnowledgePipeline instance (modified in-place).
        components: Dict mapping component types to variant name lists.
        dataset: Evaluation dataset.
        metrics: Metric names to compute.
        evaluator: Evaluator instance (e.g., RAGASEvaluator).
        **kwargs: Passed to ``pipeline.run_and_evaluate()``.

    Returns:
        AblationResults with one entry per combination.
    """
    import itertools

    _INGESTION_ONLY = {"chunker", "document_processor"}
    blocked = set(components.keys()) & _INGESTION_ONLY
    if blocked:
        raise ValueError(
            f"Cannot ablate ingestion-time components at query time: "
            f"{sorted(blocked)}. Documents must be re-ingested with each "
            f"variant for results to differ."
        )

    for ct, vl in components.items():
        if not vl:
            raise ValueError(
                f"Empty variant list for component '{ct}'. "
                f"Each component must have at least one variant."
            )

    component_types = list(components.keys())
    variant_lists = [components[ct] for ct in component_types]

    total = 1
    for vl in variant_lists:
        total *= len(vl)

    results: List[Dict[str, Any]] = []
    per_sample_scores: Dict[str, List[Dict[str, float]]] = {}

    # Save original component names so we can restore after grid search
    original_names = {ct: pipeline._component_names.get(ct) for ct in component_types}

    try:
        for i, combo in enumerate(itertools.product(*variant_lists), 1):
            variant_name = "__".join(f"{ct}={v}" for ct, v in zip(component_types, combo))
            logger.info(f"Grid search: {i}/{total} — {variant_name}")

            start = time.time()
            try:
                for ct, v in zip(component_types, combo):
                    pipeline.replace(ct, v)

                eval_result = pipeline.run_and_evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    evaluator=evaluator,
                    **kwargs,
                )
                duration = time.time() - start
                results.append(
                    {
                        "variant": variant_name,
                        "metrics": eval_result.metrics,
                        "duration_seconds": round(duration, 3),
                        "pipeline_config": eval_result.pipeline_config,
                    }
                )

                # Store per-sample scores if available
                if eval_result.per_sample:
                    per_sample_scores[variant_name] = eval_result.per_sample

                logger.info(
                    f"Grid search: {variant_name} -> {eval_result.metrics} ({duration:.1f}s)"
                )

            except Exception as e:
                logger.error(f"Grid search: {variant_name} failed: {e}")
                results.append(
                    {
                        "variant": variant_name,
                        "metrics": {},
                        "duration_seconds": time.time() - start,
                        "pipeline_config": {},
                        "error": str(e),
                    }
                )
    finally:
        # Restore the pipeline to its original state.
        # Only set keys that existed before; remove others to avoid
        # adding spurious None entries.
        for ct, orig_name in original_names.items():
            if orig_name is not None:
                pipeline._component_names[ct] = orig_name
            else:
                pipeline._component_names.pop(ct, None)
            pipeline._instances.pop(ct, None)

    return AblationResults(results, per_sample_scores=per_sample_scores)
