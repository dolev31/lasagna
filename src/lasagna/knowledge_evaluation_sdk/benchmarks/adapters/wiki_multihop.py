"""2WikiMultiHopQA benchmark adapter."""

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


class WikiMultiHopAdapter(BaseBenchmarkAdapter):
    """2WikiMultiHopQA — multi-hop reasoning over two Wikipedia articles."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="wiki_multihop",
            description="2WikiMultihopQA: generation-only question/answer samples (no corpus)",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=192_606,
            paper="arxiv:2011.01060",
            license="Apache 2.0",
            tags=["multi_hop", "wikipedia"],
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
            name="wiki_multihop",
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
        return ["train", "validation", "test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("wiki_multihop", WikiMultiHopAdapter())
