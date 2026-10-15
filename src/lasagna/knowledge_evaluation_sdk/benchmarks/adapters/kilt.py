"""KILT benchmark adapter — Knowledge Intensive Language Tasks."""

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

_KILT_TASKS = {
    "nq": {
        "description": "KILT Natural Questions: knowledge-intensive QA",
        "paper": "arxiv:2009.02252",
    },
    "triviaqa": {
        "description": "KILT TriviaQA: trivia questions as knowledge-intensive task",
        "paper": "arxiv:2009.02252",
    },
}


class KILTAdapter(BaseBenchmarkAdapter):
    """KILT — Knowledge Intensive Language Tasks."""

    def __init__(self, task: str = "nq") -> None:
        self._task = task
        self._meta = _KILT_TASKS[task]

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name=f"kilt/{self._task}",
            description=self._meta["description"],
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            paper=self._meta.get("paper"),
            license="MIT",
            tags=["kilt", "knowledge_intensive", self._task],
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
            name=f"kilt/{self._task}",
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
    for task in _KILT_TASKS:
        registry.register_adapter(f"kilt/{task}", KILTAdapter(task=task))
