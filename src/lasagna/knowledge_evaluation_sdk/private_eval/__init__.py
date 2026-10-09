"""Private evaluation set for RAG configuration optimization.

Two-track evaluation dataset composed from public benchmarks:
- **Retrieval track**: BEIR-derived samples with qrels for IR metrics (live mode)
- **Generation track**: QA samples with ground truth for answer quality (offline mode)

Usage::

    from lasagna.knowledge_evaluation_sdk.private_eval import (
        DatasetBuilder,
        DatasetManifest,
        ObjectiveAggregator,
        TradeoffAnalyzer,
    )

    # Build the dataset (downloads from HuggingFace on first run)
    builder = DatasetBuilder(seed=42)
    manifest = builder.build()

    # Register adapters for experiment runner
    from lasagna.knowledge_evaluation_sdk.private_eval.adapter import (
        register_private_eval_adapters,
    )
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
    register_private_eval_adapters(benchmark_registry)
"""

from .adapter import PrivateEvalAdapter, register_private_eval_adapters
from .aggregator import ConfigResult, ObjectiveAggregator
from .builder import DatasetBuilder
from .manifest import DatasetManifest
from .tradeoff_analyzer import (
    ConfigEfficiency,
    MarginalAnalysis,
    TierRecommendation,
    TradeoffAnalyzer,
)


def build_dataset(seed: int = 42, output_dir: str | None = None) -> DatasetManifest:
    """Convenience function to build the private evaluation dataset."""
    builder = DatasetBuilder(seed=seed, output_dir=output_dir)
    return builder.build()


def load_split(track: str, split: str = "eval", dataset_dir: str | None = None):
    """Load a specific split from the private eval dataset.

    Args:
        track: "retrieval" or "generation"
        split: "train", "eval", or "test"
        dataset_dir: Dataset directory (default: built-in dataset/ folder)

    Returns:
        List of BenchmarkSample objects.
    """
    from pathlib import Path

    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent / "dataset")
    adapter = PrivateEvalAdapter(track=track, dataset_dir=dataset_dir)
    return adapter.load(split=split)


__all__ = [
    "DatasetBuilder",
    "DatasetManifest",
    "ObjectiveAggregator",
    "ConfigResult",
    "TradeoffAnalyzer",
    "ConfigEfficiency",
    "MarginalAnalysis",
    "TierRecommendation",
    "PrivateEvalAdapter",
    "register_private_eval_adapters",
    "build_dataset",
    "load_split",
]
