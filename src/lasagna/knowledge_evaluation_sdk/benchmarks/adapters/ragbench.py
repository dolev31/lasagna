"""RAGBench benchmark adapter — TRACe-style RAG evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

if TYPE_CHECKING:
    from .._registry import BenchmarkRegistry

# RAGBench sub-datasets available via the config parameter.
_RAGBENCH_CONFIGS = [
    "hotpotqa",
    "squad",
    "msmarco",
    "pubmedqa",
    "techqa",
]


class RAGBenchAdapter(BaseBenchmarkAdapter):
    """RAGBench — TRACe-style RAG evaluation across multiple domains."""

    #: Each sample carries the documents it was answered from, so they pool
    #: into a corpus. Gold-only, like the other pooled benchmarks.
    pools_gold_context = True

    def __init__(self, config: str = "hotpotqa") -> None:
        self._config = config

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name=f"ragbench/{self._config}",
            description=f"RAGBench ({self._config}) — TRACe-style RAG evaluation",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            paper="arxiv:2407.11005",
            license="Apache 2.0",
            tags=["rag", "trace", self._config],
        )

    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        from ...datasets import load_benchmark as _load_legacy, _BENCHMARKS

        # Ensure the specific config variant is in the legacy catalogue
        bench_name = f"ragbench/{self._config}" if self._config != "hotpotqa" else "ragbench"
        if bench_name not in _BENCHMARKS:
            _BENCHMARKS[bench_name] = {
                "hf_dataset": "rungalileo/ragbench",
                "description": f"RAGBench ({self._config})",
                "format": "ragbench",
                "config": self._config,
            }

        rows = _load_legacy(
            name=bench_name,
            split=split,
            n_samples=n_samples,
            cache_dir=cache_dir,
            seed=seed,
        )
        return [
            BenchmarkSample(
                query_id=str(i),
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                context=row.get("context", ""),
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["test"]


def register(registry: "BenchmarkRegistry") -> None:
    # Register the default variant and per-config variants
    for config in _RAGBENCH_CONFIGS:
        registry.register_adapter(
            f"ragbench/{config}",
            RAGBenchAdapter(config=config),
        )
