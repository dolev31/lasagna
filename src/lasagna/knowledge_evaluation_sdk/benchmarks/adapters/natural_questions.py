"""Natural Questions benchmark adapter."""

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


class NaturalQuestionsAdapter(BaseBenchmarkAdapter):
    """Natural Questions — Google search questions with Wikipedia answers."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="natural_questions",
            description="Natural Questions: Google search questions with Wikipedia answers",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=7_830,
            paper="arxiv:1901.08634",
            license="Apache 2.0",
            tags=["qa", "open_domain"],
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
            name="natural_questions",
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
        return ["train", "validation"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("natural_questions", NaturalQuestionsAdapter())
