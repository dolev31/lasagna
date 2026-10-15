"""FEVER benchmark adapter — Fact Extraction and VERification."""

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


class FEVERAdapter(BaseBenchmarkAdapter):
    """FEVER — Fact Extraction and VERification (185K claims)."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="fever",
            description="FEVER: Fact Extraction and VERification (185K claims)",
            category=BenchmarkCategory.FACT_VERIFICATION,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=185_445,
            paper="arxiv:1803.05355",
            license="CC BY-SA 3.0",
            tags=["fact_verification", "claims"],
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
            name="fever",
            split=split,
            n_samples=n_samples,
            cache_dir=cache_dir,
            seed=seed,
        )
        samples = []
        for i, row in enumerate(rows):
            sample = BenchmarkSample(
                query_id=str(i),
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
            )
            # Apply label filter if requested
            if filters and "label" in filters:
                if sample.ground_truth != filters["label"]:
                    continue
            samples.append(sample)
        return samples

    def available_splits(self) -> List[str]:
        return ["train", "labelled_dev", "paper_dev", "paper_test"]

    def available_filters(self) -> Dict[str, List[str]]:
        return {"label": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]}


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("fever", FEVERAdapter())
