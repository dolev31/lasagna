"""MultiHop-RAG benchmark adapter."""

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


class MultiHopRAGAdapter(BaseBenchmarkAdapter):
    """MultiHop-RAG — multi-hop RAG benchmark from news articles."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="multihop_rag",
            description="MultiHop-RAG — multi-hop RAG benchmark from news articles",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=2_556,
            paper="arxiv:2401.15391",
            license="MIT",
            tags=["multi_hop", "rag", "news"],
        )

    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        from ...datasets import load_benchmark as _load_legacy

        rows = _load_legacy(
            name="multihop_rag",
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
                hops=2,
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["train", "test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("multihop_rag", MultiHopRAGAdapter())
