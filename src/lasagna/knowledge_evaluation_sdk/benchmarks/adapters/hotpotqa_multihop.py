"""HotpotQA multi-hop benchmark adapter.

This adapter uses the full HotpotQA dataset (with supporting facts and
multi-hop annotations), distinct from the BEIR/hotpotqa retrieval-only
variant.
"""

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


class HotpotQAMultiHopAdapter(BaseBenchmarkAdapter):
    """HotpotQA — multi-hop QA with supporting facts."""

    #: The supporting passages travel with each question. HotpotQA's
    #: distractor paragraphs are discarded at conversion, so a pooled corpus is
    #: gold-only and easier than the published task -- the negatives are other
    #: questions' supporting passages, not the adversarial ones HotpotQA
    #: selected. `beir/hotpotqa` is the same questions against a real corpus.
    pools_gold_context = True

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="hotpotqa",
            description="HotpotQA — multi-hop question answering with supporting facts",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=112_779,
            paper="arxiv:1809.09600",
            license="CC BY-SA 4.0",
            tags=["multi_hop", "qa", "supporting_facts"],
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
            name="hotpotqa",
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
                hops=row.get("hops", 2),
                difficulty=row.get("difficulty", ""),
                context=row.get("context", ""),  # gold supporting passages
            )
            if row.get("supporting_titles"):
                sample.metadata["supporting_titles"] = row["supporting_titles"]
            if row.get("type"):
                sample.metadata["type"] = row["type"]
            # Apply difficulty filter if requested
            if filters and "difficulty" in filters:
                if sample.difficulty and sample.difficulty != filters["difficulty"]:
                    continue
            samples.append(sample)
        return samples

    def available_splits(self) -> List[str]:
        return ["train", "validation"]

    def available_filters(self) -> Dict[str, List[str]]:
        return {"difficulty": ["easy", "medium", "hard"]}


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("hotpotqa", HotpotQAMultiHopAdapter())
