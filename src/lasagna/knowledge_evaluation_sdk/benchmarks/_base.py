"""Benchmark adapter ABC and shared data models.

All benchmark-related models are defined here and ONLY here.
Not duplicated in models.py or anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BenchmarkCategory(str, Enum):
    """Broad category for grouping benchmarks."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    FACT_VERIFICATION = "fact_verification"
    PASSAGE_RANKING = "passage_ranking"


class BenchmarkEvalMode(str, Enum):
    """What the benchmark measures.

    Distinct from the evaluator's ``rag_only`` / ``full_agent`` modes.
    """

    RETRIEVAL = "retrieval"  # IR metrics only (BEIR-style)
    GENERATION = "generation"  # Answer quality (SQuAD-style)
    END_TO_END = "end_to_end"  # Both retrieval + generation


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkInfo:
    """Static metadata describing a benchmark dataset."""

    name: str
    description: str
    category: BenchmarkCategory
    eval_mode: BenchmarkEvalMode
    num_queries: Optional[int] = None
    num_documents: Optional[int] = None
    paper: Optional[str] = None  # e.g. "arxiv:2104.08663"
    license: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSample:
    """Universal evaluation sample format.

    Every benchmark adapter converts its native data into this format.
    Fields are intentionally generous to cover single-hop QA, multi-hop
    reasoning, passage ranking, and fact verification benchmarks.
    """

    query_id: str
    question: str
    ground_truth: str = ""
    ground_truth_answers: List[str] = field(default_factory=list)
    expected_docs: Dict[str, int] = field(default_factory=dict)
    context: str = ""
    difficulty: str = ""
    domain: str = ""
    hops: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_eval_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dict format used by ``datasets.py``.

        This enables drop-in compatibility with
        ``KnowledgePipeline.run_and_evaluate()``.
        """
        d: Dict[str, Any] = {
            "question": self.question,
            "ground_truth": self.ground_truth,
        }
        if self.expected_docs:
            d["expected_docs"] = self.expected_docs
        if self.context:
            d["context"] = self.context
        return d


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class BaseBenchmarkAdapter(ABC):
    """Interface that every benchmark must implement.

    Subclasses are discovered and registered by :class:`BenchmarkRegistry`.
    """

    #: Set by an adapter when it loaded only part of the corpus. Declared here
    #: so every adapter answers the question, and so a caller never has to
    #: getattr-with-a-default and silently read "not truncated" from an adapter
    #: that simply never set it.
    corpus_truncated: bool = False

    #: Upstream dataset revision, when the adapter pinned one. Empty means the
    #: corpus came from a moving reference.
    hf_revision: str = ""

    #: Whether this benchmark's samples carry inline gold context that can be
    #: pooled into a searchable corpus. Opt-in, per adapter, because only the
    #: person who knows the dataset knows whether ``context`` holds passages or
    #: a paraphrase.
    #:
    #: Turning it on makes a benchmark that could only score generation able to
    #: measure retrieval as well -- see :meth:`build_pooled_gold_corpus` for
    #: what that measurement is and is not.
    pools_gold_context: bool = False

    @property
    def corpus_provenance(self) -> str:
        """Where the searchable corpus came from: ``native`` or ``pooled_gold``.

        Recorded with every result and hashed into the comparability key. A
        pooled-gold number and a real-corpus number answer different questions
        and must never land in the same leaderboard row.
        """
        return "pooled_gold" if self.pools_gold_context else "native"

    def build_pooled_gold_corpus(
        self,
        samples: List[BenchmarkSample],
    ) -> Optional[Dict[str, str]]:
        """A corpus assembled from the gold context of *all* the samples.

        Eight of the thirty-one public benchmarks ship questions with
        their gold context attached and no corpus to search: SQuAD v2,
        HotpotQA, MuSiQue and the five RAGBench domains. Retrieval could not be
        measured on any of them, whatever a researcher wanted to test.

        Pooling across samples is what makes the result a retrieval task rather
        than a formality: each query's own gold passages sit among every other
        query's, so the retriever has to find one relevant passage among
        hundreds. Pooling per-query would score 1.0 by construction.

        **It is easier than the real task, and it must be labelled as such.**
        Several of these datasets ship only the gold passages — HotpotQA and
        MuSiQue discard their distractors at conversion — so the pool contains
        no hard negatives, only other questions' answers. Recall against it
        will read higher than against a real corpus of the same size.
        ``corpus_provenance`` is hashed into the comparability key precisely so
        these numbers can never be compared against BEIR's.

        Returns ``None`` when no sample carries context, which is the honest
        answer for Natural Questions and TriviaQA: they ship the question and
        the answer string and nothing to retrieve. An empty corpus would score
        zero everywhere and look like a retrieval failure.
        """
        corpus: Dict[str, str] = {}
        for sample in samples:
            context = (sample.context or "").strip()
            if not context:
                continue
            # Split the way the offline builder does, so a pooled corpus built
            # here and one built there contain the same passages.
            passages = [p.strip() for p in context.split("\n\n") if p.strip()]
            for index, passage in enumerate(passages):
                corpus[f"gold:{sample.query_id}:{index}"] = passage

        if not corpus:
            return None
        return corpus

    def pooled_gold_expected_docs(
        self,
        samples: List[BenchmarkSample],
    ) -> Dict[str, Dict[str, int]]:
        """``{query_id: {doc_id: 1}}`` for a pooled corpus.

        Without this the samples have no ``expected_docs`` and every IR metric
        scores zero — which reads as a broken retriever rather than as missing
        labels.
        """
        expected: Dict[str, Dict[str, int]] = {}
        for sample in samples:
            context = (sample.context or "").strip()
            if not context:
                continue
            passages = [p for p in context.split("\n\n") if p.strip()]
            expected[sample.query_id] = {
                f"gold:{sample.query_id}:{index}": 1 for index in range(len(passages))
            }
        return expected

    @abstractmethod
    def info(self) -> BenchmarkInfo:
        """Return static metadata for this benchmark."""

    @abstractmethod
    def load(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BenchmarkSample]:
        """Load benchmark samples.

        Args:
            split: Dataset split (``"test"``, ``"train"``, ``"validation"``).
            n_samples: Limit the number of samples (deterministic).
            seed: Random seed for reproducible sampling.
            cache_dir: Override default cache directory.
            filters: Adapter-specific filtering (e.g. ``{"difficulty": "hard"}``).

        Returns:
            List of :class:`BenchmarkSample`.
        """

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        """Return ``{doc_id: text}`` for live-mode ingestion.

        Args:
            cache_dir: Override default cache directory.
            max_docs: Safety limit on number of documents loaded.
                Set to 0 to load all (may cause OOM on large corpora).

        Returns ``None`` if the benchmark does not ship a corpus
        (i.e. it is offline-only or provides context inline).
        """
        return None

    def available_splits(self) -> List[str]:
        """Splits this benchmark supports."""
        return ["test"]

    def available_filters(self) -> Dict[str, List[str]]:
        """Filterable dimensions and their legal values.

        Example: ``{"difficulty": ["easy", "medium", "hard"]}``
        """
        return {}
