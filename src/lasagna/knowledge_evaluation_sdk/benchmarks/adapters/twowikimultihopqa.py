"""2WikiMultihopQA (Ho et al., COLING 2020; arXiv:2011.01060).

Reads the official Apache-2.0 mirror's pinned Parquet files directly, without
executing its dataset script. Unlike a pooled-gold corpus, every context
paragraph is retained, including distractors and distinct versions of a title.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from .._base import (
    BaseBenchmarkAdapter,
    BenchmarkCategory,
    BenchmarkEvalMode,
    BenchmarkInfo,
    BenchmarkSample,
)

if TYPE_CHECKING:
    from .._registry import BenchmarkRegistry


class TwoWikiMultihopQAAdapter(BaseBenchmarkAdapter):
    """Validation queries and their full context corpus, plus training queries.

    ``load_corpus`` has no split parameter: it always reads validation, the
    default of ``load``, even after a call that explicitly loads train. The
    train split is available for sample export; its retrieval evaluation needs
    a separate matching corpus. Titles are IDs. Distinct paragraph versions
    sharing a title are combined, in source order, rather than overwritten.
    """

    hf_revision = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"  # pragma: allowlist secret

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="2wikimultihopqa",
            description="2WikiMultihopQA: multi-hop QA with gold and distractor paragraphs",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=12_576,
            num_documents=54_957,
            paper="arxiv:2011.01060",
            license="Apache-2.0",
            tags=["multi_hop", "qa", "distractors"],
        )

    def available_splits(self) -> List[str]:
        return ["train", "validation"]

    def _rows(self, split: str, cache_dir: Optional[str]) -> Iterator[Dict[str, Any]]:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            filename="dev.parquet" if split == "validation" else "train.parquet",
            revision=self.hf_revision,
            cache_dir=cache_dir,
        )
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()

    @staticmethod
    def _paragraphs(row: Dict[str, Any]) -> List[Tuple[str, str]]:
        # The mirror stores these nested fields as JSON strings in Parquet.
        return [(title, " ".join(sentences)) for title, sentences in json.loads(row["context"])]

    def load(
        self,
        split: str = "validation",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        if split not in self.available_splits():
            raise ValueError("2wikimultihopqa supports train and validation, not " + repr(split))
        if n_samples is not None and n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if filters:
            raise ValueError("2wikimultihopqa does not expose filters")
        if n_samples == 0:
            return []

        samples: List[BenchmarkSample] = []
        rng = random.Random(seed)
        for index, row in enumerate(self._rows(split, cache_dir)):
            gold = dict.fromkeys(title for title, _ in json.loads(row["supporting_facts"]))
            missing = gold.keys() - {title for title, _ in self._paragraphs(row)}
            if missing:
                raise ValueError(
                    f"2wikimultihopqa {row['_id']}: missing gold paragraphs {sorted(missing)}"
                )
            sample = BenchmarkSample(
                query_id=row["_id"],
                question=row["question"],
                ground_truth=row["answer"],
                expected_docs={title: 1 for title in gold},
                hops=len(gold),
                metadata={"type": row["type"]},
            )
            # Reservoir sampling bounds memory for the larger training split.
            if n_samples is None or len(samples) < n_samples:
                samples.append(sample)
            else:
                chosen = rng.randrange(index + 1)
                if chosen < n_samples:
                    samples[chosen] = sample
        return samples

    def load_corpus(
        self, cache_dir: Optional[str] = None, max_docs: int = 100_000
    ) -> Optional[Dict[str, str]]:
        """The validation split's full contexts, deduplicated by title.

        A cap limits distinct titles, but later versions of retained titles
        are still included. Zero means unlimited; truncation is explicit.
        """
        if max_docs < 0:
            raise ValueError("max_docs must be non-negative")
        paragraphs: Dict[str, List[str]] = {}
        seen: Dict[str, set[str]] = {}
        truncated = False
        for row in self._rows("validation", cache_dir):
            for title, text in self._paragraphs(row):
                if title not in paragraphs:
                    if max_docs and len(paragraphs) >= max_docs:
                        truncated = True
                        continue
                    paragraphs[title] = []
                    seen[title] = set()
                if text not in seen[title]:
                    paragraphs[title].append(text)
                    seen[title].add(text)
        self.corpus_truncated = truncated
        if truncated:
            logger.warning(
                "2wikimultihopqa corpus truncated to {} titles; set max_corpus_docs=0 for all",
                len(paragraphs),
            )
        return {title: title + "\n\n" + "\n\n".join(parts) for title, parts in paragraphs.items()}


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("2wikimultihopqa", TwoWikiMultihopQAAdapter())
