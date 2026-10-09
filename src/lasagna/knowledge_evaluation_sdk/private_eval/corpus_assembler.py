"""Corpus assembler for BEIR document corpora.

Loads document corpora from BEIR benchmark adapters, filters to documents
referenced by evaluation samples, and adds distractor documents for
realistic retrieval noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from loguru import logger


@dataclass
class CorpusData:
    """Assembled corpus with metadata."""

    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {doc_id: {"text": str, "title": str, "source_benchmark": str, "domain": str}}
    total_chars: int = 0
    distractor_count: int = 0
    benchmark_doc_counts: Dict[str, int] = field(default_factory=dict)


class CorpusAssembler:
    """Assembles document corpus from BEIR benchmark adapters."""

    def __init__(self, max_docs_per_benchmark: int = 50_000, seed: int = 42):
        self._max_docs = max_docs_per_benchmark
        self._seed = seed

    def assemble(
        self,
        retrieval_samples: List[Dict[str, Any]],
        benchmark_names: List[str],
        domain_map: Dict[str, str],
        n_distractors_per_benchmark: int = 100,
    ) -> CorpusData:
        """Load BEIR corpus, keep referenced docs + add distractors.

        Args:
            retrieval_samples: Samples with 'expected_docs' and 'source_benchmark'.
            benchmark_names: BEIR benchmark names to load corpus from.
            domain_map: Mapping from benchmark name to domain label.
            n_distractors_per_benchmark: Number of non-relevant docs to add per benchmark.

        Returns:
            CorpusData with all documents.
        """
        from ..benchmarks import benchmark_registry

        # Collect all referenced doc IDs per benchmark.
        # Expected_docs keys may be prefixed as "benchmark:raw_id" (from
        # builder) or raw "raw_id".  Strip the prefix to get the
        # benchmark-local ID that matches raw_corpus keys.
        needed_docs: Dict[str, Set[str]] = {}
        for sample in retrieval_samples:
            bench = sample.get("source_benchmark", "")
            if bench not in needed_docs:
                needed_docs[bench] = set()
            for doc_id in sample.get("expected_docs", {}):
                prefix = f"{bench}:"
                if doc_id.startswith(prefix):
                    needed_docs[bench].add(doc_id[len(prefix) :])
                else:
                    needed_docs[bench].add(doc_id)

        corpus_data = CorpusData()
        rng = random.Random(self._seed)

        for bench_name in benchmark_names:
            adapter = benchmark_registry.get(bench_name)
            if adapter is None:
                logger.warning(f"Benchmark adapter '{bench_name}' not found, skipping")
                continue

            raw_corpus = adapter.load_corpus(max_docs=self._max_docs)
            if raw_corpus is None:
                logger.warning(
                    f"Benchmark '{bench_name}' has no corpus (load_corpus returned None)"
                )
                continue

            domain = domain_map.get(bench_name, "unknown")
            needed = needed_docs.get(bench_name, set())
            bench_doc_count = 0

            # Add all referenced documents.  Prefix doc IDs with the
            # benchmark name to avoid cross-benchmark collisions (multiple
            # BEIR corpora can share raw doc IDs like "0", "1", etc.).
            found_count = 0
            for doc_id, text in raw_corpus.items():
                if doc_id in needed:
                    prefixed_id = f"{bench_name}:{doc_id}"
                    corpus_data.documents[prefixed_id] = {
                        "text": text,
                        "title": "",
                        "source_benchmark": bench_name,
                        "domain": domain,
                    }
                    bench_doc_count += 1
                    found_count += 1

            # Check coverage (compare against benchmark-local IDs)
            found_local = {did for did in raw_corpus if did in needed}
            missing = needed - found_local
            if missing:
                logger.warning(
                    f"{bench_name}: {len(missing)}/{len(needed)} referenced docs "
                    f"not found in corpus (corpus may be truncated)"
                )

            # Add distractor documents (non-relevant docs for noise)
            available_distractors = [did for did in raw_corpus if did not in needed]
            if available_distractors and n_distractors_per_benchmark > 0:
                n_to_add = min(n_distractors_per_benchmark, len(available_distractors))
                selected = rng.sample(available_distractors, n_to_add)
                for doc_id in selected:
                    distractor_id = f"{bench_name}:distractor_{doc_id}"
                    corpus_data.documents[distractor_id] = {
                        "text": raw_corpus[doc_id],
                        "title": "",
                        "source_benchmark": bench_name,
                        "domain": domain,
                    }
                    bench_doc_count += 1
                    corpus_data.distractor_count += 1

            corpus_data.benchmark_doc_counts[bench_name] = bench_doc_count
            logger.info(
                f"{bench_name}: {found_count} relevant + "
                f"{min(n_distractors_per_benchmark, len(available_distractors))} "
                f"distractor docs loaded"
            )

        # Compute total chars
        corpus_data.total_chars = sum(len(doc["text"]) for doc in corpus_data.documents.values())

        logger.info(
            f"Corpus assembled: {len(corpus_data.documents)} docs, "
            f"{corpus_data.total_chars:,} chars, "
            f"{corpus_data.distractor_count} distractors"
        )
        return corpus_data

    def add_gold_passages(
        self,
        corpus_data: CorpusData,
        gold_passages: Dict[str, Dict[str, Any]],
    ) -> None:
        """Add gold QA passages to an existing corpus.

        Used to create the e2e generation corpus which includes both BEIR
        retrieval documents and the gold supporting passages from QA
        benchmarks (SQuAD, HotpotQA, MuSiQue).

        Args:
            corpus_data: Existing CorpusData to augment (modified in place).
            gold_passages: ``{doc_id: {"text", "title", "source_benchmark",
                "domain", "source_sample_id"}}``
        """
        count = 0
        for doc_id, doc in gold_passages.items():
            if doc_id not in corpus_data.documents:
                corpus_data.documents[doc_id] = {
                    "text": doc["text"],
                    "title": doc.get("title", ""),
                    "source_benchmark": doc.get("source_benchmark", ""),
                    "domain": doc.get("domain", "general"),
                }
                count += 1

        corpus_data.benchmark_doc_counts["gold_passages"] = count
        corpus_data.total_chars = sum(len(d["text"]) for d in corpus_data.documents.values())
        logger.info(
            f"Added {count} gold passages to corpus (total: {len(corpus_data.documents)} docs)"
        )
