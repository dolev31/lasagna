"""Knowledge Evaluation SDK for RAG Feature Testing.

This module provides evaluation tools for testing the SDK's knowledge/RAG feature
using LLM-as-a-judge scoring methodology.

Example:
    ```python
    from lasagna.knowledge_evaluation_sdk import (
        KnowledgeEvaluator,
        EvalConfig,
    )

    evaluator = KnowledgeEvaluator(
        config=EvalConfig(
            document_path="./data/knowledge_base.docx",
            ground_truth_path="./data/ground_truth.jsonl",
        )
    )

    results = await evaluator.run()
    print(results.summary())
    ```
"""

from .models import EvalCase, EvalResult, EvalMetrics, EvalConfig
from .evaluator import KnowledgeEvaluator
from .judge import JudgeUnavailableError, LLMJudge


# Lazy imports for optional dependencies (ragas, pandas, matplotlib, datasets)
def __getattr__(name):
    _lazy = {
        "RAGASEvaluator": (".ragas_evaluator", "RAGASEvaluator"),
        "DetailedEvalResult": (".ragas_evaluator", "DetailedEvalResult"),
        "AblationResults": (".ablation", "AblationResults"),
        "sweep": (".ablation", "sweep"),
        "sweep_pipelines": (".ablation", "sweep_pipelines"),
        "sweep_nodes": (".ablation", "sweep_nodes"),
        "grid_search": (".ablation", "grid_search"),
        # IR metrics (pure Python, no heavy deps)
        "ndcg_at_k": (".ir_metrics", "ndcg_at_k"),
        "precision_at_k": (".ir_metrics", "precision_at_k"),
        "recall_at_k": (".ir_metrics", "recall_at_k"),
        "mrr": (".ir_metrics", "mrr"),
        "map_score": (".ir_metrics", "map_score"),
        "f1_at_k": (".ir_metrics", "f1_at_k"),
        "hit_rate_at_k": (".ir_metrics", "hit_rate_at_k"),
        "corpus_metrics": (".ir_metrics", "corpus_metrics"),
        # Benchmark datasets
        "load_benchmark": (".datasets", "load_benchmark"),
        "list_benchmarks": (".datasets", "list_benchmarks"),
        # Benchmark framework
        "Experiment": (".experiment", "Experiment"),
        "ExperimentConfig": (".experiment", "ExperimentConfig"),
        "run_benchmark": (".experiment", "run_benchmark"),
        "BenchmarkResults": (".results", "BenchmarkResults"),
        "ReportGenerator": (".reporting", "ReportGenerator"),
        "BaseBenchmarkAdapter": (".benchmarks", "BaseBenchmarkAdapter"),
        "benchmark_registry": (".benchmarks", "benchmark_registry"),
        "BenchmarkInfo": (".benchmarks", "BenchmarkInfo"),
        "BenchmarkSample": (".benchmarks", "BenchmarkSample"),
        "BenchmarkCategory": (".benchmarks", "BenchmarkCategory"),
        "BenchmarkEvalMode": (".benchmarks", "BenchmarkEvalMode"),
        # Private evaluation set
        "DatasetBuilder": (".private_eval.builder", "DatasetBuilder"),
        "DatasetManifest": (".private_eval.manifest", "DatasetManifest"),
        "ObjectiveAggregator": (".private_eval.aggregator", "ObjectiveAggregator"),
        "TradeoffAnalyzer": (".private_eval.tradeoff_analyzer", "TradeoffAnalyzer"),
        "PrivateEvalAdapter": (".private_eval.adapter", "PrivateEvalAdapter"),
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalMetrics",
    "EvalConfig",
    "KnowledgeEvaluator",
    "LLMJudge",
    "JudgeUnavailableError",
    # RAGAS (lazy-loaded)
    "RAGASEvaluator",
    "DetailedEvalResult",
    # Ablation (lazy-loaded)
    "AblationResults",
    "sweep",
    "sweep_pipelines",
    "sweep_nodes",
    "grid_search",
    # IR metrics
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "map_score",
    "f1_at_k",
    "hit_rate_at_k",
    "corpus_metrics",
    # Benchmark datasets
    "load_benchmark",
    "list_benchmarks",
    # Benchmark framework
    "Experiment",
    "ExperimentConfig",
    "run_benchmark",
    "BenchmarkResults",
    "ReportGenerator",
    "BaseBenchmarkAdapter",
    "benchmark_registry",
    "BenchmarkInfo",
    "BenchmarkSample",
    "BenchmarkCategory",
    "BenchmarkEvalMode",
    # Private evaluation set
    "DatasetBuilder",
    "DatasetManifest",
    "ObjectiveAggregator",
    "TradeoffAnalyzer",
    "PrivateEvalAdapter",
]
