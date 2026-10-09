"""Benchmark adapter registry and base classes.

Public API::

    from lasagna.knowledge_evaluation_sdk.benchmarks import (
        benchmark_registry,
        BaseBenchmarkAdapter,
        BenchmarkInfo,
        BenchmarkSample,
        BenchmarkCategory,
        BenchmarkEvalMode,
    )

    # List all registered benchmarks
    for name, info in benchmark_registry.list_benchmarks().items():
        print(f"{name}: {info.description}")

    # Load samples from a benchmark
    adapter = benchmark_registry.get("beir/nfcorpus")
    samples = adapter.load(split="test", n_samples=100)
"""

from ._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)
from ._registry import BenchmarkRegistry, benchmark_registry

__all__ = [
    "BaseBenchmarkAdapter",
    "BenchmarkCategory",
    "BenchmarkEvalMode",
    "BenchmarkInfo",
    "BenchmarkRegistry",
    "BenchmarkSample",
    "benchmark_registry",
]
