"""PrivateEvalAdapter — reads pre-built local JSONL splits.

Registers as ``private_eval/retrieval``, ``private_eval/e2e_generation``,
and ``private_eval/hallucination`` in the benchmark registry, allowing the
experiment runner to use the private evaluation set via standard YAML configs.

The legacy ``private_eval/generation`` name is deprecated: it emits a warning
and redirects to ``private_eval/e2e_generation`` (phase 1 deprecation).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def _default_dataset_dir() -> str:
    """Where the built evaluation set lives: never inside the installed package.

    Builds used to land in site-packages, which a reinstall wipes and which
    may not be writable. ``LASAGNA_EVAL_DATA_DIR`` overrides the default.
    """
    import os

    return os.environ.get("LASAGNA_EVAL_DATA_DIR") or str(
        Path.home() / ".cache" / "lasagna" / "private_eval"
    )


# Map track names to their corpus files.
_CORPUS_FILE_MAP: Dict[str, str] = {
    "retrieval": "corpus_retrieval.jsonl",
    "e2e_generation": "corpus_e2e.jsonl",
    "hallucination": "corpus_retrieval.jsonl",  # shares with retrieval
}


class PrivateEvalAdapter:
    """Reads pre-built private eval JSONL splits.

    Conforms to the BaseBenchmarkAdapter interface without inheriting
    from it (duck-typing), to avoid circular imports during registration.
    """

    def __init__(self, track: str, dataset_dir: Optional[str] = None):
        """
        Args:
            track: "retrieval", "e2e_generation", or "hallucination".
            dataset_dir: Path to the dataset directory. Defaults to ./dataset/.
        """
        self._track = track
        self._dataset_dir = dataset_dir or _default_dataset_dir()

    def info(self) -> Any:
        """Return benchmark metadata."""
        from ..benchmarks._base import BenchmarkCategory, BenchmarkEvalMode, BenchmarkInfo

        if self._track == "retrieval":
            return BenchmarkInfo(
                name="private_eval/retrieval",
                description="Private evaluation set: retrieval track (BEIR-derived, qrels)",
                category=BenchmarkCategory.SINGLE_HOP,
                eval_mode=BenchmarkEvalMode.RETRIEVAL,
                num_queries=460,
            )
        if self._track == "e2e_generation":
            return BenchmarkInfo(
                name="private_eval/e2e_generation",
                description="Private evaluation set: e2e generation (gold passages in corpus, IR+gen metrics)",
                category=BenchmarkCategory.SINGLE_HOP,
                eval_mode=BenchmarkEvalMode.GENERATION,
                num_queries=283,
            )
        if self._track == "hallucination":
            return BenchmarkInfo(
                name="private_eval/hallucination",
                description="Private evaluation set: hallucination resistance (no gold passages in corpus)",
                category=BenchmarkCategory.SINGLE_HOP,
                eval_mode=BenchmarkEvalMode.GENERATION,
                num_queries=425,
            )
        # Legacy fallback
        return BenchmarkInfo(
            name=f"private_eval/{self._track}",
            description=f"Private evaluation set: {self._track} track",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=0,
        )

    def load(
        self,
        split: str = "eval",
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Load samples from a pre-built JSONL split.

        Args:
            split: One of "train", "eval", "test".
            n_samples: Optional limit on number of samples.
            seed: Ignored (deterministic splits already built).

        Returns:
            List of BenchmarkSample objects.
        """
        from ..benchmarks._base import BenchmarkSample

        filename = f"{self._track}_{split}.jsonl"
        filepath = os.path.join(self._dataset_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Private eval data not found: {filepath}. "
                f"Run 'python -m lasagna.knowledge_evaluation_sdk"
                f".private_eval build' first."
            )

        samples = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sample = BenchmarkSample(
                    query_id=data.get("sample_id", ""),
                    question=data.get("question", ""),
                    ground_truth=data.get("ground_truth", ""),
                    ground_truth_answers=data.get("ground_truth_answers", []),
                    expected_docs=data.get("expected_docs", {}),
                    context=data.get("context", ""),
                    difficulty=data.get("difficulty", ""),
                    domain=data.get("domain", ""),
                    hops=data.get("hops", 1),
                    metadata={
                        "source_benchmark": data.get("source_benchmark", ""),
                        "query_type": data.get("query_type", ""),
                    },
                )
                samples.append(sample)

        if n_samples and len(samples) > n_samples:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:n_samples]

        logger.info(f"Loaded {len(samples)} samples from private_eval/{self._track} split={split}")
        return samples

    def load_corpus(
        self,
        cache_dir: Optional[str] = None,
        max_docs: int = 100_000,
    ) -> Optional[Dict[str, str]]:
        """Load corpus for this track.

        Each track loads from its designated corpus file:
          - retrieval / hallucination → corpus_retrieval.jsonl
          - e2e_generation → corpus_e2e.jsonl (includes gold passages)
        Falls back to legacy corpus.jsonl if track-specific file not found.
        """
        corpus_file = _CORPUS_FILE_MAP.get(self._track, "corpus.jsonl")
        corpus_path = os.path.join(self._dataset_dir, "corpus", corpus_file)

        # Fall back to legacy corpus.jsonl for v1 datasets
        if not os.path.exists(corpus_path):
            legacy_path = os.path.join(self._dataset_dir, "corpus", "corpus.jsonl")
            if os.path.exists(legacy_path):
                logger.warning(
                    f"Track-specific corpus '{corpus_file}' not found, "
                    f"falling back to legacy corpus.jsonl"
                )
                corpus_path = legacy_path
            else:
                logger.warning(f"Corpus file not found: {corpus_path}")
                return None

        corpus: Dict[str, str] = {}
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data["doc_id"]
                corpus[doc_id] = data["text"]
                # `max_docs` guards the comparison, because 0 means unlimited.
                # Without it `len(corpus) >= 0` is true on the first document
                # and the corpus comes back with exactly one entry -- an empty
                # retrieval target that scores zero and reads as a broken
                # retriever. `beir.py:234` has the same guard.
                if max_docs and len(corpus) >= max_docs:
                    break

        logger.info(f"Loaded corpus: {len(corpus)} documents from {corpus_file}")
        return corpus

    def available_splits(self) -> List[str]:
        """Return available splits."""
        return ["train", "eval", "test"]

    def available_filters(self) -> Dict[str, List[str]]:
        """Return available filters."""
        return {
            "domain": ["biomedical", "finance", "science", "argumentation", "general"],
            "difficulty": ["easy", "medium", "hard"],
        }


def register_private_eval_adapters(
    registry: Any,
    dataset_dir: Optional[str] = None,
) -> None:
    """Register all private eval adapters in the benchmark registry.

    Registers three tracks: retrieval, e2e_generation, hallucination.
    Also registers a deprecated ``private_eval/generation`` that warns
    and redirects to ``private_eval/e2e_generation``.
    """
    retrieval_adapter = PrivateEvalAdapter("retrieval", dataset_dir)
    e2e_adapter = PrivateEvalAdapter("e2e_generation", dataset_dir)
    hallucination_adapter = PrivateEvalAdapter("hallucination", dataset_dir)

    registry.register_adapter("private_eval/retrieval", retrieval_adapter)
    registry.register_adapter("private_eval/e2e_generation", e2e_adapter)
    registry.register_adapter("private_eval/hallucination", hallucination_adapter)

    # Deprecated redirect: private_eval/generation → e2e_generation
    registry.register_adapter(
        "private_eval/generation",
        _DeprecatedGenerationAdapter(e2e_adapter),
    )

    logger.debug(
        "Registered private_eval adapters: retrieval, e2e_generation, "
        "hallucination (+ deprecated generation redirect)"
    )


class _DeprecatedGenerationAdapter:
    """Wrapper that warns on use and delegates to e2e_generation adapter."""

    def __init__(self, delegate: PrivateEvalAdapter):
        self._delegate = delegate
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(
                "private_eval/generation is DEPRECATED. Redirecting to "
                "private_eval/e2e_generation. Update your YAML config. "
                "This redirect will be removed in the next release."
            )
            self._warned = True

    def info(self) -> Any:
        self._warn_once()
        return self._delegate.info()

    def load(self, *args: Any, **kwargs: Any) -> List[Any]:
        self._warn_once()
        return self._delegate.load(*args, **kwargs)

    def load_corpus(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, str]]:
        self._warn_once()
        return self._delegate.load_corpus(*args, **kwargs)

    def available_splits(self) -> List[str]:
        return self._delegate.available_splits()

    def available_filters(self) -> Dict[str, List[str]]:
        return self._delegate.available_filters()
