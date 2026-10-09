"""Bamboogle benchmark adapter — 125 multi-hop questions."""

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


class BamboogleAdapter(BaseBenchmarkAdapter):
    """Bamboogle — 125 multi-hop questions that Google cannot answer directly."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="bamboogle",
            description="Bamboogle — 125 multi-hop questions that Google cannot answer directly",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=125,
            paper="arxiv:2304.09848",
            license="MIT",
            tags=["multi_hop", "small_benchmark"],
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
            name="bamboogle",
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
                hops=row.get("hops", 2),
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("bamboogle", BamboogleAdapter())
