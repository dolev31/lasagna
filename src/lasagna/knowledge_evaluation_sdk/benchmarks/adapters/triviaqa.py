"""TriviaQA benchmark adapter."""

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


class TriviaQAAdapter(BaseBenchmarkAdapter):
    """TriviaQA — trivia question answering (95K QA pairs)."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="triviaqa",
            description="TriviaQA: trivia question answering (95K QA pairs)",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=95_000,
            paper="arxiv:1705.03551",
            license="Apache 2.0",
            tags=["qa", "trivia"],
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
            name="triviaqa",
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
                ground_truth_answers=row.get("ground_truth_aliases", []),
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["train", "validation", "test"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("triviaqa", TriviaQAAdapter())
