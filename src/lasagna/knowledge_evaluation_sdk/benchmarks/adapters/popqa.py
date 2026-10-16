"""PopQA benchmark adapter — entity-centric QA across popularity tiers.

PopQA (Mallen et al., 2022) tests how well systems answer questions about
entities with varying levels of Wikipedia popularity.  It is the standard
benchmark for studying when retrieval augmentation helps vs. when LLMs can
rely on parametric knowledge.

Popularity tiers are computed as tertiles of ``s_pop`` (subject entity
Wikipedia monthly pageviews), following the original paper's primary analysis.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

if TYPE_CHECKING:
    from .._registry import BenchmarkRegistry

# The 16 relationship types in PopQA (Wikidata property labels).
_RELATIONSHIP_TYPES = [
    "occupation",
    "place of birth",
    "genre",
    "father",
    "mother",
    "country",
    "director",
    "capital of",
    "screenwriter",
    "composer",
    "color",
    "religion",
    "sport",
    "author",
    "capital",
    "producer",
]


def _assign_tier(s_pop: int, thresholds: Tuple[float, float]) -> str:
    """Assign a popularity tier based on tertile thresholds."""
    if s_pop <= thresholds[0]:
        return "tail"
    elif s_pop <= thresholds[1]:
        return "torso"
    return "head"


def _compute_tertile_thresholds(
    rows: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Compute 33rd and 66th percentile thresholds from s_pop values.

    Uses linear interpolation (matching numpy's default) to produce
    thresholds that partition the data into three roughly equal groups
    even when there are many ties.
    """
    pops = sorted(row.get("s_pop", 0) for row in rows)
    n = len(pops)
    if n == 0:
        return (0.0, 0.0)

    def _percentile(sorted_vals: List[int], p: float) -> float:
        """Percentile with linear interpolation (numpy-compatible)."""
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return float(sorted_vals[f])
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    t1 = _percentile(pops, 33.33)
    t2 = _percentile(pops, 66.67)
    return (t1, t2)


class PopQAAdapter(BaseBenchmarkAdapter):
    """PopQA — 14K entity-centric QA pairs across popularity tiers."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="popqa",
            description=(
                "PopQA: entity-centric QA testing knowledge across popularity tiers (14K pairs)"
            ),
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=14_267,
            paper="arxiv:2212.10511",
            license="CC BY-SA 4.0",
            tags=["qa", "entity_centric", "popularity_aware", "long_tail"],
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

        # Load the FULL dataset (no n_samples) so we can filter first.
        rows = _load_legacy(
            name="popqa",
            split=split,
            n_samples=None,
            cache_dir=cache_dir,
            seed=seed,
        )

        # Compute tertile thresholds from the full dataset.
        thresholds = _compute_tertile_thresholds(rows)

        # Assign tiers and build samples.
        samples: List[BenchmarkSample] = []
        for i, row in enumerate(rows):
            s_pop = row.get("s_pop", 0)
            o_pop = row.get("o_pop", 0)
            tier = _assign_tier(s_pop, thresholds)
            rel_type = row.get("prop", "")

            # Apply filters BEFORE sampling.
            if filters:
                if "popularity_tier" in filters and tier != filters["popularity_tier"]:
                    continue
                if "relationship_type" in filters and rel_type != filters["relationship_type"]:
                    continue

            metadata: Dict[str, Any] = {
                "s_pop": s_pop,
                "o_pop": o_pop,
                "popularity_tier": tier,
                "relationship_type": rel_type,
                "subject": row.get("subj", ""),
                "object": row.get("obj", ""),
            }
            # Preserve answer aliases for future alias-aware evaluation
            aliases = row.get("ground_truth_aliases")
            if aliases:
                metadata["ground_truth_aliases"] = aliases

            sample = BenchmarkSample(
                query_id=str(i),
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                metadata=metadata,
            )
            samples.append(sample)

        # Deterministic sampling AFTER filtering.
        if n_samples is not None and n_samples < len(samples):
            rng = random.Random(seed)
            samples = rng.sample(samples, n_samples)

        return samples

    def available_splits(self) -> List[str]:
        return ["test"]

    def available_filters(self) -> Dict[str, List[str]]:
        return {
            "popularity_tier": ["head", "torso", "tail"],
            "relationship_type": _RELATIONSHIP_TYPES,
        }


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("popqa", PopQAAdapter())
