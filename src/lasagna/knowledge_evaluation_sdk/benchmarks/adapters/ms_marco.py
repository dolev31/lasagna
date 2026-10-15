"""MS MARCO benchmark adapter."""

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


class MSMARCOAdapter(BaseBenchmarkAdapter):
    """MS MARCO — passage ranking (8.8M passages)."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ms_marco",
            description="MS MARCO: passage ranking (8.8M passages)",
            category=BenchmarkCategory.PASSAGE_RANKING,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=808_731,
            num_documents=8_800_000,
            paper="arxiv:1611.09268",
            license="MIT",
            tags=["passage_ranking", "qa"],
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
            name="ms_marco",
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
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["train", "validation", "test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("ms_marco", MSMARCOAdapter())
