"""MIRACL Swahili (Zhang et al., arXiv:2210.09984).

MIRACL judges Wikipedia passages against questions native speakers wrote, in 18
languages; this adapter loads Swahili, the first benchmark here that is not in
English. Queries and qrels come from ``miracl/miracl`` and passages from
``miracl/miracl-corpus``, each read at its own pinned commit with
``hf_hub_download``, without executing a dataset script.
"""

from __future__ import annotations

import gzip
import json
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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


class MiraclSwahiliAdapter(BaseBenchmarkAdapter):
    """Swahili dev or train queries, searched against the full Swahili corpus.

    ``dev`` is the default split: at the pinned commit test-a and test-b have
    queries but no qrels, so they cannot be scored, and the paper reports its
    baselines on dev. Only judgments above 0 become ``expected_docs``; a query
    left with no relevant passage is dropped, as BEIR's loader drops them. The
    qrels and the corpus share MIRACL's own passage IDs (``<article>#<n>``), so
    no ID is rewritten.

    The two repositories have separate commits. ``hf_revision`` names both, so
    every result records exactly which data it read.
    """

    TOPICS_REVISION = "5be20db9509754dadad47689368639fcec739c00"  # pragma: allowlist secret
    CORPUS_REVISION = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"  # pragma: allowlist secret
    hf_revision = f"miracl/miracl@{TOPICS_REVISION}+miracl/miracl-corpus@{CORPUS_REVISION}"

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="miracl/sw",
            description="MIRACL (Swahili): passage retrieval over Swahili Wikipedia",
            category=BenchmarkCategory.PASSAGE_RANKING,
            eval_mode=BenchmarkEvalMode.RETRIEVAL,
            num_queries=482,
            num_documents=131_924,
            paper="arxiv:2210.09984",
            license="Apache-2.0",
            tags=["multilingual", "swahili", "retrieval"],
        )

    def available_splits(self) -> List[str]:
        return ["dev", "train"]

    @staticmethod
    def _download(repo_id: str, filename: str, revision: str, cache_dir: Optional[str]) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )

    def _topics(self, split: str, cache_dir: Optional[str]) -> Dict[str, str]:
        path = self._download(
            "miracl/miracl",
            f"miracl-v1.0-sw/topics/topics.miracl-v1.0-sw-{split}.tsv",
            self.TOPICS_REVISION,
            cache_dir,
        )
        topics: Dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    query_id, text = line.rstrip("\n").split("\t", 1)
                    topics[query_id] = text
        return topics

    def _relevant(self, split: str, cache_dir: Optional[str]) -> Dict[str, Dict[str, int]]:
        """``{query_id: {docid: relevance}}`` for every judged query, relevant passages only."""
        path = self._download(
            "miracl/miracl",
            f"miracl-v1.0-sw/qrels/qrels.miracl-v1.0-sw-{split}.tsv",
            self.TOPICS_REVISION,
            cache_dir,
        )
        relevant: Dict[str, Dict[str, int]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    query_id, _, docid, grade = line.split()
                    docs = relevant.setdefault(query_id, {})
                    if int(grade) > 0:
                        docs[docid] = int(grade)
        return relevant

    def load(
        self,
        split: str = "dev",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        if split not in self.available_splits():
            raise ValueError(
                f"miracl/sw has qrels for dev and train only, not {split!r}: "
                "use --split dev, or split: dev in a sweep"
            )
        if n_samples is not None and n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if filters:
            raise ValueError("miracl/sw does not expose filters")
        if n_samples == 0:
            return []

        topics = self._topics(split, cache_dir)
        relevant = self._relevant(split, cache_dir)
        unknown = sorted(relevant.keys() - topics.keys())
        if unknown:
            raise ValueError(f"miracl/sw {split}: qrels name queries with no topic: {unknown[:5]}")
        samples = [
            BenchmarkSample(query_id=query_id, question=text, expected_docs=relevant[query_id])
            for query_id, text in topics.items()
            if relevant.get(query_id)
        ]
        if n_samples is not None and n_samples < len(samples):
            samples = random.Random(seed).sample(samples, n_samples)
        return samples

    def load_corpus(
        self, cache_dir: Optional[str] = None, max_docs: int = 100_000
    ) -> Optional[Dict[str, str]]:
        """Every Swahili passage as ``title + "\\n\\n" + text``. Zero means no cap."""
        if max_docs < 0:
            raise ValueError("max_docs must be non-negative")
        path = self._download(
            "miracl/miracl-corpus",
            "miracl-corpus-v1.0-sw/docs-0.jsonl.gz",
            self.CORPUS_REVISION,
            cache_dir,
        )
        corpus: Dict[str, str] = {}
        truncated = False
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if max_docs and len(corpus) >= max_docs:
                    truncated = True
                    break
                doc = json.loads(line)
                title, text = doc.get("title", ""), doc["text"]
                corpus[doc["docid"]] = f"{title}\n\n{text}" if title else text
        self.corpus_truncated = truncated
        if truncated:
            logger.warning(
                "miracl/sw corpus truncated to {} passages; set max_corpus_docs=0 for all",
                len(corpus),
            )
        return corpus


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("miracl/sw", MiraclSwahiliAdapter())
