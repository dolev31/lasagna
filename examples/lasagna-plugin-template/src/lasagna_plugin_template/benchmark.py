"""A benchmark, complete enough to copy.

Implementing ``info()`` and ``load()`` is the contract. Implementing
``load_corpus()`` as well is what makes a benchmark usable in *live* mode --
the only mode that measures retrieval rather than answer quality against gold
context. Without it, Lasagna resolves ``mode="auto"`` to generation-only and
says so.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from lasagna.knowledge_evaluation_sdk.benchmarks import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

_CORPUS: Dict[str, str] = {
    "doc_kiln": (
        "An anagama kiln is fired continuously for several days. Wood ash "
        "settles on the pots and melts into a natural glaze."
    ),
    "doc_glaze": (
        "Celadon glaze takes its colour from iron oxide reduced in a "
        "low-oxygen atmosphere during firing."
    ),
    "doc_wheel": (
        "Throwing on a potter's wheel centres the clay first; an off-centre "
        "lump will wobble and collapse as the walls rise."
    ),
    "doc_bisque": (
        "Bisque firing hardens greenware before glazing, driving off chemically "
        "bound water so the pot can be handled safely."
    ),
}

_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "tmpl_1",
        "question": "Where does the glaze on anagama-fired pottery come from?",
        "ground_truth": "Wood ash settling on the pots and melting during the long firing.",
        # doc id -> graded relevance. 2 = answers it, 1 = useful context.
        "expected_docs": {"doc_kiln": 2, "doc_glaze": 1},
    },
    {
        "query_id": "tmpl_2",
        "question": "What gives celadon its colour?",
        "ground_truth": "Iron oxide reduced in a low-oxygen atmosphere.",
        "expected_docs": {"doc_glaze": 2},
    },
    {
        "query_id": "tmpl_3",
        "question": "Why is bisque firing done before glazing?",
        "ground_truth": "It hardens greenware so the pot can be handled safely.",
        "expected_docs": {"doc_bisque": 2},
    },
]


class TinyBenchmark(BaseBenchmarkAdapter):
    """Four documents, three queries, no download."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="template_tiny",
            description="Worked example of a third-party benchmark (not a real measurement)",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=len(_QUERIES),
            num_documents=len(_CORPUS),
            license="Apache-2.0",
            tags=["template", "example"],
        )

    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        if split != "test":
            # Refuse rather than quietly serving the split you do have: a run
            # reporting `split: validation` over test data is not comparable to
            # anyone else's, and nothing downstream can detect it.
            raise ValueError(f"template_tiny has only a 'test' split, got {split!r}")

        rows = list(_QUERIES)
        if n_samples is not None and n_samples < len(rows):
            # Seeded, so the same request returns the same questions.
            rows = random.Random(seed).sample(rows, n_samples)

        return [
            BenchmarkSample(
                query_id=r["query_id"],
                question=r["question"],
                ground_truth=r["ground_truth"],
                ground_truth_answers=[r["ground_truth"]],
                expected_docs=dict(r["expected_docs"]),
                # Gold context lets this benchmark also work in
                # generation-only mode, where no corpus is ingested.
                context=" ".join(_CORPUS[d] for d in sorted(r["expected_docs"])),
            )
            for r in rows
        ]

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        """Returning a corpus here is what enables live mode."""
        return dict(_CORPUS)
