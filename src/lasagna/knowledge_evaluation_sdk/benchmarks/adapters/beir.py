"""BEIR benchmark suite adapters.

Wraps the existing ``datasets.py`` loader for all BEIR datasets that ship
with qrels (nfcorpus, scifact, fiqa, hotpotqa, arguana) and adds the
remaining standard BEIR benchmarks.
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

# All BEIR sub-benchmarks shipped by BeIR on HuggingFace.
_BEIR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "beir/nfcorpus": {
        "hf_dataset": "BeIR/nfcorpus",
        "hf_qrels": "BeIR/nfcorpus-qrels",
        # Counts verified against the dataset itself, not estimated. The
        # previous values (3,600 queries / 170,000 docs) had the two
        # transposed and inflated the corpus 47x.
        "description": "NF Corpus — biomedical IR benchmark (3.2K queries, 3.6K docs)",
        "num_queries": 3237,
        "num_documents": 3633,
        "paper": "arxiv:1610.02806",
    },
    "beir/scifact": {
        "hf_dataset": "mteb/scifact",
        "hf_qrels": "mteb/scifact",
        "description": "SciFact — scientific claim verification (1.1K claims, 5.2K docs)",
        "num_queries": 1109,
        "num_documents": 5183,
        "paper": "arxiv:2004.14974",
    },
    "beir/fiqa": {
        "hf_dataset": "mteb/fiqa",
        "hf_qrels": "mteb/fiqa",
        "description": "FiQA — financial opinion QA (6.6K queries, 57.6K docs)",
        "num_queries": 6648,
        "num_documents": 57638,
    },
    "beir/hotpotqa": {
        "hf_dataset": "mteb/hotpotqa",
        "hf_qrels": "mteb/hotpotqa",
        "description": "HotpotQA (BEIR) — multi-hop reasoning (7.4K queries, 5.2M docs)",
        "num_queries": 7400,
        "num_documents": 5_200_000,
    },
    "beir/arguana": {
        "hf_dataset": "mteb/arguana",
        "hf_qrels": "mteb/arguana",
        "description": "ArguAna — argument retrieval (1.4K queries, 8.7K docs)",
        "num_queries": 1400,
        "num_documents": 8700,
    },
    "beir/msmarco": {
        "hf_dataset": "BeIR/msmarco",
        "hf_qrels": "BeIR/msmarco-qrels",
        "description": "MS MARCO (BEIR split) — passage retrieval (6.9K dev queries)",
        "num_queries": 6900,
        "num_documents": 8_800_000,
    },
    "beir/trec-covid": {
        "hf_dataset": "BeIR/trec-covid",
        "hf_qrels": "BeIR/trec-covid-qrels",
        "description": "TREC-COVID — COVID-19 related retrieval (50 queries, 171K docs)",
        "num_queries": 50,
        "num_documents": 171_000,
    },
    "beir/nq": {
        "hf_dataset": "BeIR/nq",
        "hf_qrels": "BeIR/nq-qrels",
        "description": "Natural Questions (BEIR) — Google search questions (3.5K queries)",
        "num_queries": 3500,
        "num_documents": 2_680_000,
    },
    "beir/quora": {
        "hf_dataset": "BeIR/quora",
        "hf_qrels": "BeIR/quora-qrels",
        "description": "Quora Duplicate Questions — duplicate detection (10K queries)",
        "num_queries": 10000,
        "num_documents": 523_000,
    },
    "beir/dbpedia-entity": {
        "hf_dataset": "BeIR/dbpedia-entity",
        "hf_qrels": "BeIR/dbpedia-entity-qrels",
        "description": "DBPedia-Entity — entity retrieval (400 queries, 4.6M docs)",
        "num_queries": 400,
        "num_documents": 4_600_000,
    },
    "beir/fever": {
        "hf_dataset": "BeIR/fever",
        "hf_qrels": "BeIR/fever-qrels",
        "description": "FEVER (BEIR) — fact verification retrieval (6.7K queries)",
        "num_queries": 6700,
        "num_documents": 5_400_000,
    },
    "beir/climate-fever": {
        "hf_dataset": "BeIR/climate-fever",
        "hf_qrels": "BeIR/climate-fever-qrels",
        "description": "Climate-FEVER — climate claim verification (1.5K queries)",
        "num_queries": 1500,
        "num_documents": 5_400_000,
    },
    "beir/scidocs": {
        "hf_dataset": "BeIR/scidocs",
        "hf_qrels": "BeIR/scidocs-qrels",
        "description": "SciDocs — scientific document recommendations (1K queries)",
        "num_queries": 1000,
        "num_documents": 25_000,
    },
}


class BEIRBenchmarkAdapter(BaseBenchmarkAdapter):
    """Adapter for a single BEIR sub-benchmark."""

    def __init__(self, name: str, meta: Dict[str, Any]) -> None:
        self._name = name
        self._meta = meta

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name=self._name,
            description=self._meta["description"],
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.RETRIEVAL,
            num_queries=self._meta.get("num_queries"),
            num_documents=self._meta.get("num_documents"),
            paper=self._meta.get("paper"),
            license="Varies by dataset",
            tags=["beir", "retrieval"],
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

        # Ensure the legacy catalogue knows about this benchmark.
        # The original datasets.py only has 5 BEIR entries; we add the rest
        # dynamically when first needed.
        _ensure_legacy_entry(self._name, self._meta)

        rows = _load_legacy(
            name=self._name,
            split=split,
            n_samples=n_samples,
            cache_dir=cache_dir,
            seed=seed,
        )
        return [
            BenchmarkSample(
                # The upstream id when the loader carried one; the position
                # only as a fallback. A positional id makes `sample_id_digest`
                # a function of n_samples alone, so two runs over completely
                # different questions certify as covering the same queries.
                query_id=str(row.get("query_id") or i),
                question=row.get("question", ""),
                ground_truth=row.get("ground_truth", ""),
                expected_docs=row.get("expected_docs", {}),
            )
            for i, row in enumerate(rows)
        ]

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        """Load BEIR corpus documents for live-mode ingestion.

        Args:
            cache_dir: Unused (kept for interface compat).
            max_docs: Safety limit to prevent OOM on million-doc corpora
                like HotpotQA (5.2M docs) or MS MARCO (8.8M docs).
                Set to 0 to load all (at your own risk).

        Returns:
            Dict mapping doc_id to text, or None if corpus unavailable.
        """
        from loguru import logger as _logger

        try:
            from datasets import load_dataset
        except ImportError:
            return None

        # A warning up front, from declared metadata, so a long ingest can be
        # cancelled early. The FLAG is set below from what actually loaded --
        # declared counts are hand-maintained and have been wrong by 47x
        # (nfcorpus said 170,000 against an actual 3,633), which set
        # `corpus_truncated` on runs where nothing was truncated and put a false
        # integrity flag in the bundle.
        num_documents = self._meta.get("num_documents", 0)
        if max_docs and num_documents and num_documents > max_docs:
            _logger.warning(
                f"BEIR corpus '{self._name}' is declared as {num_documents:,} docs "
                f"(exceeds max_docs={max_docs:,}); it will be cut to the first "
                f"{max_docs:,}. Set max_corpus_docs=0 to load all."
            )

        hf_name = self._meta["hf_dataset"]
        # Pinned the same way the queries and qrels are. This was an unpinned
        # `load_dataset`, so a pinned run fetched its QUERIES from the pin and
        # its CORPUS from HEAD -- and still stamped `hf_revision` on the result,
        # producing a bundle that claims a reproducible corpus it did not load.
        # A partial pin is worse than none: it looks like provenance.
        _rev = self._resolve_revision()
        _kw = {"revision": _rev} if _rev else {}
        try:
            corpus_ds = load_dataset(hf_name, "corpus", split="corpus", **_kw)
        except (ValueError, KeyError, RuntimeError):
            try:
                corpus_ds = load_dataset(hf_name, split="corpus", **_kw)
            except (ValueError, KeyError, RuntimeError):
                return None

        corpus: Dict[str, str] = {}
        truncated = False
        for row in corpus_ds:
            doc_id = str(row.get("_id", row.get("id", "")))
            title = row.get("title", "")
            text = row.get("text", "")
            full = f"{title}\n\n{text}".strip() if title else text
            if doc_id and full:
                corpus[doc_id] = full
            if max_docs and len(corpus) >= max_docs:
                # Set from the break, not from metadata. This is the only
                # statement that knows a document was actually left behind.
                truncated = True
                break

        self.corpus_truncated = truncated
        if truncated:
            _logger.warning(
                f"BEIR corpus '{self._name}' truncated to {len(corpus):,} docs. "
                f"A metric computed against part of a corpus is not that "
                f"benchmark's metric -- it simply comes out lower, and only the "
                f"`corpus_truncated` flag in the bundle records it."
            )
        return corpus if corpus else None

    def _resolve_revision(self) -> str:
        """The revision this benchmark is pinned to, or "".

        Reads the shared registry so a `pin_dataset_revision("beir/scifact", ...)`
        or `LASAGNA_DATASET_REVISIONS` entry covers the corpus as well as the
        queries -- BEIR keeps its own catalogue, and before this the two halves
        of the same benchmark could come from different revisions.
        """
        declared = getattr(self, "hf_revision", "") or ""
        if declared:
            return str(declared)
        try:
            from ...datasets import resolve_dataset_revision

            return resolve_dataset_revision(self._name)
        except Exception:  # noqa: BLE001 - provenance must never fail a load
            return ""

    def available_splits(self) -> List[str]:
        return ["test"]


def _ensure_legacy_entry(name: str, meta: Dict[str, Any]) -> None:
    """Add a BEIR benchmark to ``datasets._BENCHMARKS`` if missing."""
    from ...datasets import _BENCHMARKS

    if name not in _BENCHMARKS:
        _BENCHMARKS[name] = {
            "hf_dataset": meta["hf_dataset"],
            "hf_qrels": meta["hf_qrels"],
            "description": meta["description"],
            "format": "beir",
            # Carried through rather than dropped: this hand-copies keys, and
            # omitting `hf_revision` meant the catalogue tier of
            # `resolve_dataset_revision` could never be satisfied for any of the
            # dynamically registered BEIR datasets.
            "hf_revision": meta.get("hf_revision", ""),
        }


def register(registry: "BenchmarkRegistry") -> None:
    """Register all BEIR benchmarks."""
    for name, meta in _BEIR_BENCHMARKS.items():
        registry.register_adapter(name, BEIRBenchmarkAdapter(name, meta))
