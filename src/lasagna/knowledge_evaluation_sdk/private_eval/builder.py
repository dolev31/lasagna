"""DatasetBuilder — samples from multiple benchmarks and creates stratified splits.

Orchestrates the full dataset build pipeline:
  1. Load retrieval-track samples from BEIR benchmarks (with qrels)
  2. Load generation-track samples from QA benchmarks (with ground truth)
  3. Tag each sample with domain, difficulty, query_type
  4. Assemble BEIR corpus (referenced docs + distractors)
  5. Extract gold passages from generation samples → e2e corpus
  6. Stratified split into train/eval/test (60/20/20)
  7. Produce 3 track split sets: retrieval, e2e_generation, hallucination
  8. Save to JSONL + manifest

Usage::

    from lasagna.knowledge_evaluation_sdk.private_eval.builder import (
        DatasetBuilder,
    )

    builder = DatasetBuilder(seed=42, n_distractors_per_benchmark=500)
    manifest = builder.build()
    print(manifest.summary())
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .corpus_assembler import CorpusAssembler
from .manifest import (
    BenchmarkSourceSpec,
    CorpusStats,
    DatasetManifest,
    SplitStats,
    compute_jsonl_hash,
)
from .splits import StratifiedSplitter


# ---------------------------------------------------------------------------
# Benchmark sampling specs
# ---------------------------------------------------------------------------


@dataclass
class _BenchSpec:
    """Internal spec for a benchmark to sample from."""

    name: str
    n_samples: int
    split: str
    track: str  # "retrieval" or "generation"
    domain: str
    default_difficulty: str = "medium"


# Retrieval track: BEIR benchmarks with qrels + corpus
_RETRIEVAL_BENCHMARKS: List[_BenchSpec] = [
    _BenchSpec("beir/nfcorpus", 100, "test", "retrieval", "biomedical"),
    _BenchSpec("beir/fiqa", 100, "test", "retrieval", "finance"),
    _BenchSpec("beir/scifact", 80, "test", "retrieval", "science"),
    _BenchSpec("beir/arguana", 80, "test", "retrieval", "argumentation"),
    _BenchSpec("beir/hotpotqa", 100, "test", "retrieval", "general"),
]

# Generation track: QA benchmarks with ground_truth + context
_GENERATION_BENCHMARKS: List[_BenchSpec] = [
    _BenchSpec("squad_v2", 120, "validation", "generation", "general"),
    _BenchSpec("hotpotqa", 100, "validation", "generation", "general", "medium"),
    _BenchSpec("musique", 80, "validation", "generation", "general", "hard"),
    _BenchSpec("bamboogle", 125, "test", "generation", "general", "hard"),
]

# Domain map for corpus assembler
_DOMAIN_MAP: Dict[str, str] = {
    "beir/nfcorpus": "biomedical",
    "beir/fiqa": "finance",
    "beir/scifact": "science",
    "beir/arguana": "argumentation",
    "beir/hotpotqa": "general",
}


class DatasetBuilder:
    """Builds the private evaluation dataset from public benchmarks."""

    def __init__(
        self,
        seed: int = 42,
        output_dir: Optional[str] = None,
        n_distractors_per_benchmark: int = 500,
    ):
        self._seed = seed
        from .adapter import _default_dataset_dir

        self._output_dir = output_dir or _default_dataset_dir()
        self._n_distractors = n_distractors_per_benchmark
        self._splitter = StratifiedSplitter()

    def build(self) -> DatasetManifest:
        """Full build pipeline: load -> tag -> corpus -> split -> save.

        Produces three evaluation tracks:
          - **retrieval**: BEIR queries with qrels, searched against BEIR corpus
          - **e2e_generation**: QA queries with gold context ingested into corpus
          - **hallucination**: same QA queries, searched against BEIR corpus (no gold passages)

        Two corpus files are produced:
          - ``corpus_retrieval.jsonl``: BEIR docs + distractors
          - ``corpus_e2e.jsonl``: retrieval corpus + gold QA passages
        """
        logger.info("Starting private eval dataset build (v2.0, 3 tracks)...")

        # ── Load samples ─────────────────────────────────────────────────
        retrieval_samples = self._load_retrieval_track()
        generation_samples = self._load_generation_track()

        logger.info(
            f"Loaded {len(retrieval_samples)} retrieval + "
            f"{len(generation_samples)} generation samples"
        )

        # ── Assemble retrieval corpus (BEIR docs + distractors) ──────────
        corpus_retrieval = self._assemble_corpus(retrieval_samples)

        # ── Extract gold passages and build e2e corpus ───────────────────
        gold_passages = self._extract_gold_passages(generation_samples)
        logger.info(f"Extracted {len(gold_passages)} gold passages from generation samples")

        # E2E corpus = retrieval corpus + gold passages
        corpus_e2e = copy.deepcopy(corpus_retrieval)
        assembler = CorpusAssembler(seed=self._seed)
        assembler.add_gold_passages(corpus_e2e, gold_passages)

        # ── Tag generation samples ───────────────────────────────────────
        # Add has_gold_context flag and expected_docs for e2e samples
        for sample in generation_samples:
            ctx = sample.get("context", "")
            sample["has_gold_context"] = bool(ctx and ctx.strip())

        # Add expected_docs to samples that have gold context
        self._add_gold_expected_docs(generation_samples, gold_passages)

        # ── Split each track ─────────────────────────────────────────────
        r_train, r_eval, r_test = self._splitter.split(retrieval_samples, seed=self._seed)
        g_train, g_eval, g_test = self._splitter.split(generation_samples, seed=self._seed)

        # Derive e2e splits (only samples with gold context)
        e2e_train = [s for s in g_train if s.get("has_gold_context")]
        e2e_eval = [s for s in g_eval if s.get("has_gold_context")]
        e2e_test = [s for s in g_test if s.get("has_gold_context")]

        # Hallucination splits = ALL generation samples (no expected_docs)
        h_train = [self._strip_expected_docs(s) for s in g_train]
        h_eval = [self._strip_expected_docs(s) for s in g_eval]
        h_test = [self._strip_expected_docs(s) for s in g_test]

        logger.info(
            f"Splits (train/eval/test): Retrieval {len(r_train)}/{len(r_eval)}/{len(r_test)}, "
            f"E2E Gen {len(e2e_train)}/{len(e2e_eval)}/{len(e2e_test)}, "
            f"Hallucination {len(h_train)}/{len(h_eval)}/{len(h_test)}"
        )

        # ── Save everything ──────────────────────────────────────────────
        os.makedirs(self._output_dir, exist_ok=True)
        corpus_dir = os.path.join(self._output_dir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        files_written: Dict[str, str] = {}

        # Retrieval splits (unchanged sample IDs)
        for name, data in [
            ("retrieval_train", r_train),
            ("retrieval_eval", r_eval),
            ("retrieval_test", r_test),
        ]:
            files_written[f"{name}.jsonl"] = self._save_jsonl(
                data, os.path.join(self._output_dir, f"{name}.jsonl")
            )

        # Legacy generation splits (kept for backward compat)
        for name, data in [
            ("generation_train", g_train),
            ("generation_eval", g_eval),
            ("generation_test", g_test),
        ]:
            files_written[f"{name}.jsonl"] = self._save_jsonl(
                data, os.path.join(self._output_dir, f"{name}.jsonl")
            )

        # E2E generation splits (new)
        for name, data in [
            ("e2e_generation_train", e2e_train),
            ("e2e_generation_eval", e2e_eval),
            ("e2e_generation_test", e2e_test),
        ]:
            files_written[f"{name}.jsonl"] = self._save_jsonl(
                data, os.path.join(self._output_dir, f"{name}.jsonl")
            )

        # Hallucination splits (new)
        for name, data in [
            ("hallucination_train", h_train),
            ("hallucination_eval", h_eval),
            ("hallucination_test", h_test),
        ]:
            files_written[f"{name}.jsonl"] = self._save_jsonl(
                data, os.path.join(self._output_dir, f"{name}.jsonl")
            )

        # Save both corpora
        ret_corpus_path = os.path.join(corpus_dir, "corpus_retrieval.jsonl")
        self._save_corpus(corpus_retrieval, ret_corpus_path)
        files_written["corpus/corpus_retrieval.jsonl"] = ret_corpus_path

        e2e_corpus_path = os.path.join(corpus_dir, "corpus_e2e.jsonl")
        self._save_corpus(corpus_e2e, e2e_corpus_path)
        files_written["corpus/corpus_e2e.jsonl"] = e2e_corpus_path

        # Keep legacy corpus.jsonl as symlink/copy of retrieval corpus for compat
        legacy_corpus_path = os.path.join(corpus_dir, "corpus.jsonl")
        self._save_corpus(corpus_retrieval, legacy_corpus_path)
        files_written["corpus/corpus.jsonl"] = legacy_corpus_path

        # ── Build manifest ───────────────────────────────────────────────
        manifest = self._build_manifest_v2(
            retrieval_samples,
            generation_samples,
            r_train,
            r_eval,
            r_test,
            e2e_train,
            e2e_eval,
            e2e_test,
            h_train,
            h_eval,
            h_test,
            corpus_retrieval,
            corpus_e2e,
            files_written,
        )
        manifest.save(os.path.join(self._output_dir, "manifest.json"))

        # Save split stats
        self._save_split_stats(manifest)
        split_stats_path = os.path.join(self._output_dir, "split_stats.json")
        if os.path.isfile(split_stats_path):
            manifest.checksums["split_stats.json"] = compute_jsonl_hash(split_stats_path)
            manifest.save(os.path.join(self._output_dir, "manifest.json"))

        logger.info(f"Dataset built at {self._output_dir}")
        logger.info(manifest.summary())
        return manifest

    def _load_retrieval_track(self) -> List[Dict[str, Any]]:
        """Load BEIR benchmarks via benchmark_registry."""
        from ..benchmarks import benchmark_registry

        samples = []
        for spec in _RETRIEVAL_BENCHMARKS:
            adapter = benchmark_registry.get(spec.name)
            if adapter is None:
                logger.warning(f"Adapter '{spec.name}' not found, skipping")
                continue

            raw_samples = adapter.load(
                split=spec.split,
                n_samples=spec.n_samples,
                seed=self._seed,
            )
            for i, s in enumerate(raw_samples):
                # Prefix expected_docs keys with benchmark name to match
                # the prefixed doc IDs in corpus_assembler (avoids
                # cross-benchmark ID collisions).
                raw_expected = s.expected_docs or {}
                prefixed_expected = {
                    f"{spec.name}:{doc_id}": score for doc_id, score in raw_expected.items()
                }
                sample = {
                    "sample_id": f"{spec.name.replace('/', '_')}_{spec.split}_{i:04d}",
                    "question": s.question,
                    "expected_docs": prefixed_expected,
                    "domain": spec.domain,
                    "difficulty": self._assign_difficulty(s, spec),
                    "query_type": "factual",
                    "source_benchmark": spec.name,
                    "original_query_id": s.query_id,
                }
                samples.append(sample)

            logger.info(f"  {spec.name}: loaded {len(raw_samples)} samples")

        return samples

    def _load_generation_track(self) -> List[Dict[str, Any]]:
        """Load QA benchmarks via benchmark_registry."""
        from ..benchmarks import benchmark_registry

        samples = []
        for spec in _GENERATION_BENCHMARKS:
            adapter = benchmark_registry.get(spec.name)
            if adapter is None:
                logger.warning(f"Adapter '{spec.name}' not found, skipping")
                continue

            raw_samples = adapter.load(
                split=spec.split,
                n_samples=spec.n_samples,
                seed=self._seed,
            )
            for i, s in enumerate(raw_samples):
                hops = getattr(s, "hops", 1) or 1
                sample = {
                    "sample_id": f"{spec.name.replace('/', '_')}_{spec.split}_{i:04d}",
                    "question": s.question,
                    "ground_truth": s.ground_truth or "",
                    "ground_truth_answers": s.ground_truth_answers
                    or ([s.ground_truth] if s.ground_truth else []),
                    "context": s.context or "",
                    "domain": spec.domain,
                    "difficulty": self._assign_difficulty(s, spec),
                    "query_type": "multi_hop" if hops > 1 else "factual",
                    "hops": hops,
                    "source_benchmark": spec.name,
                }
                samples.append(sample)

            logger.info(f"  {spec.name}: loaded {len(raw_samples)} samples")

        return samples

    def _assign_difficulty(self, sample: Any, spec: _BenchSpec) -> str:
        """Assign difficulty based on sample properties and benchmark."""
        # Use explicit difficulty from sample if available
        if hasattr(sample, "difficulty") and sample.difficulty:
            return sample.difficulty

        # Heuristic assignment
        hops = getattr(sample, "hops", 1) or 1
        if hops >= 3:
            return "hard"
        if hops == 2:
            return "medium"

        # Unanswerable questions (SQuAD v2) are hard
        gt = getattr(sample, "ground_truth", "") or ""
        if not gt.strip():
            return "hard"

        return spec.default_difficulty

    def _assemble_corpus(self, retrieval_samples: List[Dict]) -> Any:
        """Assemble document corpus from BEIR adapters."""
        assembler = CorpusAssembler(seed=self._seed)
        benchmark_names = [s.name for s in _RETRIEVAL_BENCHMARKS]
        return assembler.assemble(
            retrieval_samples,
            benchmark_names,
            _DOMAIN_MAP,
            n_distractors_per_benchmark=self._n_distractors,
        )

    @staticmethod
    def _extract_gold_passages(
        generation_samples: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract gold passages from generation samples for corpus ingestion.

        For each sample with non-empty ``context``, splits on ``\\n\\n`` to
        get individual passages.  Each passage gets a unique doc_id:
        ``gold:{benchmark}:{sample_id}:{passage_idx}``.

        Returns:
            Dict mapping doc_id to passage metadata.
        """
        gold_docs: Dict[str, Dict[str, Any]] = {}
        for sample in generation_samples:
            context = sample.get("context", "")
            if not context or not context.strip():
                continue

            sample_id = sample["sample_id"]
            benchmark = sample.get("source_benchmark", "")
            domain = sample.get("domain", "general")

            passages = [p.strip() for p in context.split("\n\n") if p.strip()]
            for pidx, passage in enumerate(passages):
                doc_id = f"gold:{benchmark}:{sample_id}:{pidx}"
                gold_docs[doc_id] = {
                    "text": passage,
                    "title": f"Gold passage for {sample_id}",
                    "source_benchmark": benchmark,
                    "domain": domain,
                    "source_sample_id": sample_id,
                }
        return gold_docs

    @staticmethod
    def _add_gold_expected_docs(
        generation_samples: List[Dict[str, Any]],
        gold_passages: Dict[str, Dict[str, Any]],
    ) -> None:
        """Add ``expected_docs`` to generation samples that have gold context.

        Maps each sample to its gold passage doc_ids (relevance=1).
        This enables computing IR metrics on the e2e generation track.
        """
        # Build reverse index: sample_id → list of gold doc_ids
        sample_to_gold: Dict[str, Dict[str, int]] = {}
        for doc_id, doc in gold_passages.items():
            sid = doc.get("source_sample_id", "")
            if sid:
                sample_to_gold.setdefault(sid, {})[doc_id] = 1

        for sample in generation_samples:
            if sample.get("has_gold_context"):
                sample["expected_docs"] = sample_to_gold.get(sample["sample_id"], {})

    @staticmethod
    def _strip_expected_docs(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep copy of the sample with expected_docs removed.

        Used for hallucination track where samples should not have qrels.
        Deep copy prevents mutation leaking between e2e and hallucination tracks.
        """
        out = copy.deepcopy(sample)
        out.pop("expected_docs", None)
        return out

    def _save_jsonl(self, samples: List[Dict], path: str) -> str:
        """Save samples to JSONL file. Returns the path."""
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return path

    def _save_corpus(self, corpus_data: Any, path: str) -> None:
        """Save corpus to JSONL."""
        with open(path, "w") as f:
            for doc_id, doc in corpus_data.documents.items():
                record = {
                    "doc_id": doc_id,
                    "text": doc["text"],
                    "title": doc.get("title", ""),
                    "source_benchmark": doc.get("source_benchmark", ""),
                    "domain": doc.get("domain", ""),
                    "char_count": len(doc["text"]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save corpus metadata
        meta_path = os.path.join(os.path.dirname(path), "corpus_metadata.json")
        meta = {
            "doc_count": len(corpus_data.documents),
            "total_chars": corpus_data.total_chars,
            "distractor_count": corpus_data.distractor_count,
            "benchmark_doc_counts": corpus_data.benchmark_doc_counts,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _build_manifest_v2(
        self,
        retrieval_all,
        generation_all,
        r_train,
        r_eval,
        r_test,
        e2e_train,
        e2e_eval,
        e2e_test,
        h_train,
        h_eval,
        h_test,
        corpus_retrieval,
        corpus_e2e,
        files_written,
    ) -> DatasetManifest:
        """Build the v2.0 dataset manifest with 3-track support."""
        manifest = DatasetManifest(seed=self._seed, version="2.0.0")

        # Record benchmark sources
        for spec in _RETRIEVAL_BENCHMARKS:
            manifest.benchmarks.append(
                BenchmarkSourceSpec(
                    name=spec.name,
                    split=spec.split,
                    n_samples=spec.n_samples,
                    track="retrieval",
                    domain=spec.domain,
                )
            )
        for spec in _GENERATION_BENCHMARKS:
            manifest.benchmarks.append(
                BenchmarkSourceSpec(
                    name=spec.name,
                    split=spec.split,
                    n_samples=spec.n_samples,
                    track="e2e_generation",
                    domain=spec.domain,
                )
            )

        # Record split stats for all 3 tracks
        for name, samples in [
            ("retrieval_train", r_train),
            ("retrieval_eval", r_eval),
            ("retrieval_test", r_test),
            ("e2e_generation_train", e2e_train),
            ("e2e_generation_eval", e2e_eval),
            ("e2e_generation_test", e2e_test),
            ("hallucination_train", h_train),
            ("hallucination_eval", h_eval),
            ("hallucination_test", h_test),
        ]:
            manifest.splits[name] = self._compute_split_stats(samples)

        # Corpus stats (report e2e corpus as primary — superset)
        manifest.corpus = CorpusStats(
            doc_count=len(corpus_e2e.documents),
            total_chars=corpus_e2e.total_chars,
            benchmarks=corpus_e2e.benchmark_doc_counts,
            distractor_count=corpus_e2e.distractor_count,
        )

        # Checksums
        for filename, filepath in files_written.items():
            if os.path.isfile(filepath):
                manifest.checksums[filename] = compute_jsonl_hash(filepath)

        return manifest

    def _compute_split_stats(self, samples: List[Dict]) -> SplitStats:
        """Compute statistics for a split."""
        domains: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        query_types: Dict[str, int] = {}
        sample_ids = []

        for s in samples:
            d = s.get("domain", "unknown")
            domains[d] = domains.get(d, 0) + 1
            diff = s.get("difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
            qt = s.get("query_type", "unknown")
            query_types[qt] = query_types.get(qt, 0) + 1
            sample_ids.append(s["sample_id"])

        return SplitStats(
            count=len(samples),
            domains=domains,
            difficulties=difficulties,
            query_types=query_types,
            sample_ids=sample_ids,
        )

    def _save_split_stats(self, manifest: DatasetManifest) -> None:
        """Save split statistics to JSON."""
        stats = {}
        for name, split in manifest.splits.items():
            stats[name] = {
                "count": split.count,
                "domains": split.domains,
                "difficulties": split.difficulties,
                "query_types": split.query_types,
            }
        path = os.path.join(self._output_dir, "split_stats.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
