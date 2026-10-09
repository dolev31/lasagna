"""MuSiQue benchmark adapter — multi-hop questions requiring 2-4 composition steps."""

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


class MuSiQueAdapter(BaseBenchmarkAdapter):
    """MuSiQue — multi-hop questions requiring 2-4 reasoning steps."""

    #: The supporting paragraphs travel with each question. MuSiQue's
    #: distractors are discarded at conversion, so a pooled corpus is
    #: gold-only and easier than the published task; the other questions'
    #: paragraphs are the only negatives.
    pools_gold_context = True

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="musique",
            description="MuSiQue — multi-hop questions requiring 2-4 composition steps",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=25_000,
            paper="arxiv:2108.00573",
            license="CC BY 4.0",
            tags=["multi_hop", "compositional"],
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
            name="musique",
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
                context=row.get("context", ""),  # gold supporting paragraphs
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["train", "validation", "test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("musique", MuSiQueAdapter())
