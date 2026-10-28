"""2WikiMultihopQA (Ho et al., COLING 2020; arXiv:2011.01060).

Reads the official Apache-2.0 mirror's pinned Parquet files directly, without
executing its dataset script. Unlike a pooled-gold corpus, every context
paragraph is retained, including distractors and distinct versions of a title.
"""

from __future__ import annotations

import hashlib
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
    a separate matching corpus. The first paragraph version uses its title as
    ID; later versions append a stable text hash. This keeps paragraph lengths
    intact and lets relevance labels identify the question's exact version.
    """

    hf_revision = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"  # pragma: allowlist secret

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="2wikimultihopqa",
            description="2WikiMultihopQA: retrieval and generation over full validation contexts",
            category=BenchmarkCategory.MULTI_HOP,
            eval_mode=BenchmarkEvalMode.END_TO_END,
            num_queries=12_576,
            num_documents=56_687,
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

    def _identified_rows(
        self, split: str, cache_dir: Optional[str]
    ) -> Iterator[Tuple[Dict[str, Any], List[Tuple[str, str, str]]]]:
        """Assign IDs in full source order, independently of sample/corpus caps."""
        versions: Dict[str, Dict[str, str]] = {}
        used: Dict[str, Tuple[str, str]] = {}
        for row in self._rows(split, cache_dir):
            paragraphs = []
            for title, text in self._paragraphs(row):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                known = versions.setdefault(title, {})
                if digest not in known:
                    identifier = title if not known else f"{title}#{digest[:16]}"
                    if identifier in used and used[identifier] != (title, digest):
                        raise ValueError(f"2wikimultihopqa: paragraph ID collision for {title!r}")
                    known[digest] = identifier
                    used[identifier] = (title, digest)
                paragraphs.append((known[digest], title, text))
            yield row, paragraphs

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
        for index, (row, paragraphs) in enumerate(self._identified_rows(split, cache_dir)):
            gold = dict.fromkeys(title for title, _ in json.loads(row["supporting_facts"]))
            own: Dict[str, set[str]] = {}
            for identifier, title, _ in paragraphs:
                own.setdefault(title, set()).add(identifier)
            missing = gold.keys() - own.keys()
            if missing:
                raise ValueError(
                    f"2wikimultihopqa {row['_id']}: missing gold paragraphs {sorted(missing)}"
                )
            ambiguous = [title for title in gold if len(own[title]) != 1]
            if ambiguous:
                raise ValueError(f"2wikimultihopqa {row['_id']}: ambiguous gold titles {ambiguous}")
            sample = BenchmarkSample(
                query_id=row["_id"],
                question=row["question"],
                ground_truth=row["answer"],
                expected_docs={identifier: 1 for title in gold for identifier in own[title]},
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
        """The validation split's full contexts, deduplicated by paragraph version.

        A cap limits distinct paragraphs. Zero means unlimited; truncation is
        explicit. Identical paragraphs shared by questions reuse the same ID.
        """
        if max_docs < 0:
            raise ValueError("max_docs must be non-negative")
        corpus: Dict[str, str] = {}
        truncated = False
        for _, paragraphs in self._identified_rows("validation", cache_dir):
            for identifier, title, text in paragraphs:
                if identifier not in corpus:
                    if max_docs and len(corpus) >= max_docs:
                        truncated = True
                        continue
                    corpus[identifier] = title + "\n\n" + text
        self.corpus_truncated = truncated
        if truncated:
            logger.warning(
                "2wikimultihopqa corpus truncated to {} paragraphs; set max_corpus_docs=0 for all",
                len(corpus),
            )
        return corpus


def register(registry: "BenchmarkRegistry") -> None:
    registry.register_adapter("2wikimultihopqa", TwoWikiMultihopQAAdapter())
