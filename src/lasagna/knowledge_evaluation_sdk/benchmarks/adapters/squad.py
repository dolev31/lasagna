"""SQuAD v2 benchmark adapter."""

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


class SQuADv2Adapter(BaseBenchmarkAdapter):
    """SQuAD v2 — reading comprehension with unanswerable questions."""

    #: Each question ships the paragraph it was written from, so pooling those
    #: paragraphs across the sample gives a real corpus to retrieve against.
    #: Easier than the real task -- the pool is all gold, no hard negatives --
    #: which is why `corpus_provenance` keeps these numbers out of any
    #: comparison with a native corpus.
    pools_gold_context = True

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="squad_v2",
            description="SQuAD v2 — reading comprehension with unanswerable questions",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=11_873,
            paper="arxiv:1806.03822",
            license="CC BY-SA 4.0",
            tags=["qa", "reading_comprehension"],
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
            name="squad_v2",
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
                ground_truth_answers=list(row.get("ground_truth_answers") or []),
                context=row.get("context", ""),
                # SQuAD v2 marks a question unanswerable by an empty answer set.
                # Said explicitly so the harness scores the abstention instead
                # of dropping the question as missing data.
                metadata={"unanswerable": not (row.get("ground_truth") or "").strip()},
            )
            for i, row in enumerate(rows)
        ]

    def available_splits(self) -> List[str]:
        return ["train", "validation"]


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("squad_v2", SQuADv2Adapter())
