"""Standard benchmark dataset loader for RAG evaluation.

One-liner loading of BEIR, SQuAD, and other standard IR/QA benchmarks,
converted to the format expected by ``KnowledgePipeline.run_and_evaluate()``.

No competitor offers this — researchers can go from zero to evaluation
in a single call.

Example:
    from lasagna.knowledge_evaluation_sdk.datasets import (
        load_benchmark,
        list_benchmarks,
    )

    # Load a standard benchmark
    dataset = load_benchmark("beir/nfcorpus", split="test", n_samples=100)
    # Returns: [{"question": ..., "ground_truth": ..., "doc_id": ...}, ...]

    # Use directly with pipeline
    result = pipeline.run_and_evaluate(dataset, metrics=["faithfulness"])

    # See available benchmarks
    print(list_benchmarks())
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Bump this when converter logic changes to invalidate cached data.
_CACHE_VERSION = 3  # Bumped: BEIR output now includes expected_docs from qrels


# ---------------------------------------------------------------------------
# Dataset revisions
# ---------------------------------------------------------------------------
#
# A HuggingFace dataset name is a moving reference. `mteb/scifact` today and
# `mteb/scifact` in six months can be different data, and nothing about a
# results file would say so -- two people reporting numbers on "the same
# benchmark" may have measured different corpora.
#
# No revisions are shipped pinned. A SHA written here without being resolved
# against the Hub would be a fabricated provenance record, which is worse than
# an honest blank: it would satisfy `--strict` while proving nothing. What ships
# is the *mechanism*, so a run that cares can pin, and `hf_revision` on the
# results then records what was actually loaded.

#: benchmark name -> git revision (a commit SHA, tag or branch on the Hub).
_PINNED_REVISIONS: Dict[str, str] = {}

#: Environment variable holding a JSON object of the same shape, so a pin can
#: be set for a CI run without editing code.
_REVISION_ENV = "LASAGNA_DATASET_REVISIONS"


class DatasetNotCached(ValueError):
    """Offline, a benchmark's Hugging Face dataset is not in the local cache."""


def _offline() -> bool:
    from lasagna.core.config import env_flag
    from lasagna.core.model_downloads import offline

    return offline() or env_flag("HF_DATASETS_OFFLINE")


def dataset_cached(hf_name: str) -> bool:
    """Whether ``hf_name`` (``org/name``) loads from the local cache.

    Either the ``datasets`` cache holds it (``<org>___<name>``), or the hub
    cache has a snapshot whose files resolve.
    """
    org, _, name = hf_name.partition("/")
    try:
        import datasets.config as datasets_config

        if (Path(datasets_config.HF_DATASETS_CACHE) / f"{org}___{name}").is_dir():
            return True
    except Exception:  # noqa: BLE001 - no datasets library: only the hub cache counts
        pass
    try:
        from huggingface_hub import constants

        snapshots = Path(constants.HF_HUB_CACHE) / f"datasets--{org}--{name}" / "snapshots"
    except Exception:  # noqa: BLE001
        return False
    return snapshots.is_dir() and any(f.is_file() for f in snapshots.rglob("*"))


def hf_repos(benchmark: str, adapter: Any = None) -> List[str]:
    """The Hugging Face datasets ``benchmark`` loads (data and qrels), from its catalogue entry."""
    info = _BENCHMARKS.get(benchmark) or _BENCHMARKS.get(benchmark.split("/")[0]) or {}
    if not info:
        info = getattr(adapter, "_meta", None) or {}
    return list(dict.fromkeys(r for r in (info.get("hf_dataset"), info.get("hf_qrels")) if r))


def missing_offline_datasets(
    repos_by_benchmark: Dict[str, List[str]],
) -> Optional[DatasetNotCached]:
    """Offline, the benchmarks whose Hugging Face data is not cached, as one error.

    None when online or when everything is cached. Checked once before a run's
    tasks start: each task used to fail on its own, printing the same
    "dataset_info.json not found" for every arm.
    """
    if not _offline():
        return None
    missing = [
        benchmark
        for benchmark, repos in repos_by_benchmark.items()
        if any(not dataset_cached(repo) for repo in repos)
    ]
    if not missing:
        return None
    named = ", ".join(missing)
    verb = "is" if len(missing) == 1 else "are"
    return DatasetNotCached(
        f"{named} {verb} not in the Hugging Face cache and offline mode is on "
        f"(HF_HUB_OFFLINE or HF_DATASETS_OFFLINE): run once online to download it, or "
        f"unset HF_HUB_OFFLINE and HF_DATASETS_OFFLINE."
    )


def pin_dataset_revision(benchmark: str, revision: str) -> None:
    """Load ``benchmark`` from a fixed upstream revision from now on.

    The pin is recorded on every result produced afterwards, which is what
    lets ``lasagna results validate --strict`` insist on one.
    """
    _PINNED_REVISIONS[benchmark] = revision


def resolve_dataset_revision(benchmark: str) -> str:
    """The revision to load ``benchmark`` at, or ``""`` for whatever is current.

    Resolution order: an explicit :func:`pin_dataset_revision` call, then the
    ``LASAGNA_DATASET_REVISIONS`` environment variable, then the catalogue
    entry. Programmatic first, so a test or a script can override a CI-wide pin
    without unsetting it.
    """
    if benchmark in _PINNED_REVISIONS:
        return _PINNED_REVISIONS[benchmark]

    raw = os.environ.get(_REVISION_ENV, "").strip()
    if raw:
        try:
            from_env = json.loads(raw)
            if isinstance(from_env, dict) and from_env.get(benchmark):
                return str(from_env[benchmark])
        except (ValueError, TypeError):
            # A malformed pin must not silently load whatever is current --
            # that is the failure the pin exists to prevent -- but it also must
            # not take down a run that would otherwise be fine, so it is loud.
            logger.warning(
                f"{_REVISION_ENV} is not valid JSON, so no revision was pinned. "
                f"Results will record an empty hf_revision and `--strict` "
                f"validation will reject them."
            )

    return str((_BENCHMARKS.get(benchmark) or {}).get("hf_revision", "") or "")


# ---------------------------------------------------------------------------
# Benchmark catalogue
# ---------------------------------------------------------------------------

_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # BEIR benchmarks (widely used in IR research)
    "beir/nfcorpus": {
        "hf_dataset": "BeIR/nfcorpus",
        "hf_qrels": "BeIR/nfcorpus-qrels",
        "description": "NF Corpus: biomedical IR benchmark (3.6K queries, 170K docs)",
        "format": "beir",
    },
    "beir/scifact": {
        "hf_dataset": "mteb/scifact",
        "hf_qrels": "mteb/scifact",
        "description": "SciFact: scientific claim verification (1.1K claims, 5K docs)",
        "format": "beir",
    },
    "beir/fiqa": {
        "hf_dataset": "mteb/fiqa",
        "hf_qrels": "mteb/fiqa",
        "description": "FiQA: financial opinion QA (6.6K queries, 57K docs)",
        "format": "beir",
    },
    "beir/hotpotqa": {
        "hf_dataset": "mteb/hotpotqa",
        "hf_qrels": "mteb/hotpotqa",
        "description": "HotpotQA: multi-hop reasoning (7.4K queries, 5.2M docs)",
        "format": "beir",
    },
    "beir/arguana": {
        "hf_dataset": "mteb/arguana",
        "hf_qrels": "mteb/arguana",
        "description": "ArguAna: argument retrieval (1.4K queries, 8.7K docs)",
        "format": "beir",
    },
    # QA benchmarks
    "squad_v2": {
        "hf_dataset": "rajpurkar/squad_v2",
        "description": "SQuAD v2: reading comprehension with unanswerable questions",
        "format": "squad",
    },
    "natural_questions": {
        "hf_dataset": "google-research-datasets/natural_questions",
        "description": "Natural Questions: Google search questions with Wikipedia answers",
        "format": "nq",
        "config": "default",
    },
    # MS MARCO
    "ms_marco": {
        "hf_dataset": "microsoft/ms_marco",
        "description": "MS MARCO: passage ranking (8.8M passages)",
        "format": "msmarco",
        "config": "v2.1",
    },
    # -----------------------------------------------------------------------
    # Additional benchmarks (used by the benchmark adapter framework)
    # -----------------------------------------------------------------------
    "triviaqa": {
        "hf_dataset": "trivia_qa",
        "description": "TriviaQA: trivia question answering (95K QA pairs)",
        "format": "triviaqa",
        "config": "rc",
    },
    "hotpotqa": {
        "hf_dataset": "hotpot_qa",
        "description": "HotpotQA: multi-hop question answering with supporting facts",
        "format": "hotpotqa",
        "config": "fullwiki",
    },
    "musique": {
        "hf_dataset": "dgslibisey/MuSiQue",
        "description": "MuSiQue: multi-hop questions requiring 2-4 composition steps",
        "format": "musique",
    },
    "wiki_multihop": {
        "hf_dataset": "varun-v-rao/2WikiMultiHopQA",
        "description": "2WikiMultiHopQA: multi-hop reasoning over two Wikipedia articles",
        "format": "wiki_multihop",
    },
    "multihop_rag": {
        "hf_dataset": "yixuantt/MultiHopRAG",
        "description": "MultiHop-RAG: multi-hop RAG benchmark from news articles",
        "format": "multihop_rag",
    },
    "bamboogle": {
        "hf_dataset": "chiayewken/bamboogle",
        "description": "Bamboogle: 125 multi-hop questions that Google cannot answer directly",
        "format": "bamboogle",
    },
    "popqa": {
        "hf_dataset": "akariasai/PopQA",
        "description": "PopQA: entity-centric QA testing knowledge across popularity tiers (14K pairs)",
        "format": "popqa",
    },
    "ragbench": {
        "hf_dataset": "rungalileo/ragbench",
        "description": "RAGBench: TRACe-style RAG evaluation across 5 domains",
        "format": "ragbench",
        "config": "hotpotqa",
    },
    "kilt/nq": {
        "hf_dataset": "kilt_tasks",
        "description": "KILT Natural Questions: knowledge-intensive language tasks",
        "format": "kilt",
        "config": "nq",
    },
    "kilt/triviaqa": {
        "hf_dataset": "kilt_tasks",
        "description": "KILT TriviaQA: trivia questions as knowledge-intensive task",
        "format": "kilt",
        "config": "triviaqa",
    },
    "fever": {
        "hf_dataset": "fever",
        "description": "FEVER: Fact Extraction and VERification (185K claims)",
        "format": "fever",
        "config": "v1.0",
    },
}


def list_benchmarks() -> Dict[str, str]:
    """List available benchmark datasets with descriptions.

    Returns:
        Dict mapping benchmark name to description.
    """
    return {name: info["description"] for name, info in _BENCHMARKS.items()}


def load_benchmark(
    name: str,
    split: str = "test",
    n_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    allow_split_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """Load a standard benchmark dataset for RAG evaluation.

    Downloads and converts the dataset to the format expected by
    ``KnowledgePipeline.run_and_evaluate()``:

        [{"question": str, "ground_truth": str, "doc_id": str}, ...]

    Results are cached locally for fast reuse.

    Args:
        name: Benchmark name (e.g., ``"beir/nfcorpus"``, ``"squad_v2"``).
            Use ``list_benchmarks()`` to see available options.
        split: Dataset split — ``"test"``, ``"train"``, or ``"validation"``.
        n_samples: Optional limit on number of samples. When set, a
            deterministic random sample is drawn (seeded by ``seed``).
        cache_dir: Directory for cached datasets. Defaults to
            ``~/.cache/lasagna/benchmarks``.
        seed: Random seed for reproducible sampling.

    Returns:
        List of evaluation case dicts.

    Raises:
        ValueError: If the benchmark name is not recognised.
        ImportError: If the ``datasets`` library is not installed.
    """
    if name not in _BENCHMARKS:
        available = ", ".join(sorted(_BENCHMARKS.keys()))
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")

    # Check cache first
    cache_path = _cache_path(name, split, n_samples, seed, cache_dir)
    if cache_path.exists():
        logger.debug(f"Loading cached benchmark from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # Load from HuggingFace
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for benchmark loading. "
            "Install it with: pip install datasets"
        )

    info = _BENCHMARKS[name]
    hf_name = info["hf_dataset"]
    fmt = info["format"]

    # A dataset name is a moving reference; a revision is not. Empty means the
    # run cannot claim to have loaded any particular version of the data, which
    # `UNPINNED_DATASET` reports and `--strict` refuses to publish.
    revision = resolve_dataset_revision(name)
    logger.info(
        f"Downloading benchmark '{name}' from HuggingFace"
        + (f" at revision {revision}..." if revision else " (unpinned revision)...")
    )

    # The split actually loaded, which the cache is keyed on below.
    #
    # Both branches now assign it. The BEIR branch used to resolve splits inside
    # `_load_with_fallback` and never update this, so the re-key below could not
    # fire for BEIR: a substituted split was cached under the *requested* key and
    # every later run read train judgements believing they were test.
    actual_split = split

    # BEIR datasets with qrels: load queries + relevance judgments separately
    if fmt == "beir" and "hf_qrels" in info:
        import random

        # Queries are not split-sensitive for BEIR -- the `queries` config holds
        # every query and the qrels decide which are in play -- so a fallback
        # there is harmless and stays enabled.
        queries_ds, _ = _load_with_fallback(
            hf_name,
            config="queries",
            splits=["queries", "test", "validation", "train"],
            revision=revision,
        )
        # The qrels ARE the split. Substituting here is what turns a test-set
        # request into train-set judgements, so it obeys the caller's flag and
        # the split actually used is carried back out.
        qrels_ds, actual_split = _load_with_fallback(
            info["hf_qrels"],
            # Named, not None. The mteb repos hold `default` (the qrels),
            # `corpus` and `queries`; with no name, an offline load fails once
            # more than one of them is cached, which running the benchmark
            # always does.
            config=info.get("hf_qrels_config", "default"),
            splits=[split, "test", "validation", "train"],
            revision=revision,
            allow_split_fallback=allow_split_fallback,
        )

        # Join queries with qrels FIRST, then sample from the joined result.
        # Sampling before joining wastes most samples since many queries
        # lack qrels (e.g., nfcorpus has 3237 queries but only ~323 with
        # test qrels).
        samples = _convert_beir_with_qrels(queries_ds, qrels_ds)

        if n_samples is not None and len(samples) > n_samples:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:n_samples]
    else:
        # Standard single-dataset path
        load_kwargs: Dict[str, Any] = {}
        if "config" in info:
            load_kwargs["name"] = info["config"]
        if revision:
            load_kwargs["revision"] = revision

        try:
            ds = load_dataset(hf_name, split=split, **load_kwargs)
            actual_split = split
        except (ValueError, KeyError) as exc:
            # Substituting a split changes what the number means -- `train`
            # numbers reported as `test` are not comparable to anyone else's,
            # and the cache stores the result under the *requested* split's key,
            # so the substitution would persist invisibly across runs.
            #
            # Opt in explicitly, and the substitution is recorded rather than
            # merely logged.
            if not allow_split_fallback:
                available = _available_splits(hf_name, load_kwargs.get("name"))
                raise ValueError(
                    f"Split '{split}' is not available for benchmark '{name}'"
                    + (f" (available: {', '.join(available)})" if available else "")
                    + ". Choose one with --split (or `split:` on the benchmark in a "
                    "sweep). From Python, allow_split_fallback=True accepts a substitute "
                    "and records the split actually used."
                ) from exc

            fallbacks = [s for s in ["test", "validation", "train"] if s != split]
            ds = None
            actual_split = ""
            for fallback in fallbacks:
                try:
                    logger.warning(
                        f"Split '{split}' not found for '{name}', substituting '{fallback}'. "
                        f"These numbers are NOT comparable to '{split}' results."
                    )
                    ds = load_dataset(hf_name, split=fallback, **load_kwargs)
                    actual_split = fallback
                    break
                except (ValueError, KeyError):
                    continue
            if ds is None:
                raise ValueError(
                    f"No usable split found for '{name}'. Tried: {split}, {', '.join(fallbacks)}"
                )

        # Convert to our evaluation format
        converter = _FORMAT_CONVERTERS.get(fmt)
        if converter is None:
            raise ValueError(f"No converter for format '{fmt}'")

        # Sample from HF dataset BEFORE converting to avoid memory bloat
        # on large datasets (e.g., MS MARCO with 8.8M passages).
        if n_samples is not None and n_samples < len(ds):
            import random

            rng = random.Random(seed)
            indices = rng.sample(range(len(ds)), n_samples)
            ds = ds.select(indices)

        samples = converter(ds)

    # Cache under the split that was actually loaded, not the one requested.
    # Keying on the request is how a substitution outlives the run that made
    # it: every later call for `test` would be served `train` data from disk,
    # with no warning, because the warning only fires on a cache miss.
    if actual_split != split:
        cache_path = _cache_path(name, actual_split, n_samples, seed, cache_dir)
        logger.warning(
            f"Caching '{name}' under split '{actual_split}' (requested '{split}'), "
            f"so a later request for '{split}' does not silently reuse it."
        )

    # Atomic cache write — prevents corrupt JSON from interrupted writes.
    # Write to a temp file first, then rename (atomic on same filesystem).
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=cache_path.parent,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(samples, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.rename(cache_path)
    logger.info(f"Cached {len(samples)} samples to {cache_path}")

    return samples


# ---------------------------------------------------------------------------
# BEIR helpers
# ---------------------------------------------------------------------------


def _load_with_fallback(
    hf_name: str,
    config: Optional[str] = None,
    splits: Optional[List[str]] = None,
    revision: str = "",
    allow_split_fallback: bool = True,
):
    """Load a HuggingFace dataset, returning ``(dataset, split_actually_used)``.

    The split is returned rather than assumed. This function walks a list of
    candidates, so the caller cannot know which one it landed on — and the BEIR
    path used it for qrels with ``[requested, "test", "validation", "train"]``,
    so a benchmark that does not publish the requested split silently returned
    **train** judgements. Those were then cached under the requested split's key
    and stamped ``split: "test"`` on the result, with no warning and nothing in
    provenance. Training-set retrieval numbers reported as test-set numbers is
    the single worst thing this package could do quietly.

    ``allow_split_fallback=False`` refuses to substitute, matching the
    discipline the non-BEIR path has had all along (see ``load_benchmark``).
    """
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {}
    if config:
        load_kwargs["name"] = config
    if revision:
        load_kwargs["revision"] = revision

    candidates = list(splits or ["test"])
    if not allow_split_fallback:
        candidates = candidates[:1]

    errors: List[str] = []
    for index, s in enumerate(candidates):
        try:
            dataset = load_dataset(hf_name, split=s, **load_kwargs)
        except (ValueError, KeyError, RuntimeError) as exc:
            # The message, not only the type: "multiple configurations in the
            # cache" is not a missing split, and read as one it misleads.
            errors.append(f"{s}: {type(exc).__name__}: {str(exc)[:200]}")
            continue

        if index > 0:
            # Loud, because the number this produces is not the number the
            # caller asked for. `actual_split` travels back to the caller so
            # the cache is keyed on it and the results record it.
            logger.warning(
                f"Split {candidates[0]!r} is not available for {hf_name!r}; "
                f"substituting {s!r}. These numbers are NOT comparable to "
                f"{candidates[0]!r} results. Pass allow_split_fallback=False to "
                f"refuse the substitution."
            )
        return dataset, s

    raise ValueError(f"No usable split for '{hf_name}' (tried: {candidates}; {'; '.join(errors)})")


def _convert_beir_with_qrels(queries_ds, qrels_ds) -> List[Dict[str, Any]]:
    """Join BEIR queries with qrels for IR evaluation.

    Produces samples with ``expected_docs`` — a dict mapping doc IDs
    to graded relevance scores — compatible with ``corpus_metrics()``.
    """
    # Build query lookup: {query_id: query_text}
    queries: Dict[str, str] = {}
    for row in queries_ds:
        qid = str(row.get("_id", row.get("id", "")))
        text = row.get("text", row.get("title", ""))
        if qid and text:
            queries[qid] = text

    # Group qrels by query: {query_id: {corpus_id: score}}
    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = int(row.get("score", 0))
        if qid in queries and score > 0:
            qrels.setdefault(qid, {})[cid] = score

    # Build evaluation samples
    samples = []
    for qid, query_text in queries.items():
        expected = qrels.get(qid, {})
        if expected:  # Only include queries that have relevance judgments
            samples.append(
                {
                    # The upstream query id, carried rather than discarded.
                    #
                    # Adapters were assigning `query_id=str(i)` -- a positional
                    # index -- so `sample_id_digest`, which the bundle documents
                    # as proof that two runs scored the same questions, was a
                    # pure function of n_samples: beir/nfcorpus and squad_v2 at
                    # n=50 both hashed to 770b707572088fe8.
                    "query_id": qid,
                    "question": query_text,
                    "ground_truth": "",
                    "expected_docs": expected,
                }
            )

    if not samples:
        logger.warning(
            f"BEIR converter produced 0 samples with qrels. "
            f"Queries: {len(queries)}, qrels entries: {sum(len(v) for v in qrels.values())}"
        )
    return samples


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _convert_beir(ds) -> List[Dict[str, Any]]:
    """Convert a BEIR-format HuggingFace dataset.

    BEIR datasets on HuggingFace can have varying schemas (multi-config
    with corpus/queries/qrels, or flat rows). This converter probes the
    available columns and extracts what it can find.
    """
    samples = []
    columns = set(ds.column_names) if hasattr(ds, "column_names") else set()
    for row in ds:
        question = row.get("query") or row.get("question") or row.get("text", "")
        answer = row.get("answer", "")
        doc_id = str(row.get("_id", row.get("id", "")))
        if question:
            samples.append(
                {
                    "question": question,
                    "ground_truth": answer,
                    "doc_id": doc_id,
                }
            )
    if not samples:
        logger.warning(f"BEIR converter produced 0 samples. Columns in dataset: {columns}")
    return samples


def _convert_squad(ds) -> List[Dict[str, Any]]:
    """Convert SQuAD v2 format."""
    samples = []
    for row in ds:
        question = row.get("question", "")
        answers = row.get("answers", {})
        # SQuAD v2 answers: {"text": [...], "answer_start": [...]}
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        ground_truth = answer_texts[0] if answer_texts else ""
        context = row.get("context", "")

        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": ground_truth,
                # Every annotator's answer: the official script takes the best
                # match over all of them, and scoring against the first alone
                # understated EM and F1.
                "ground_truth_answers": list(dict.fromkeys(answer_texts)),
            }
            if context:
                sample["context"] = context
            samples.append(sample)
    return samples


def _convert_nq(ds) -> List[Dict[str, Any]]:
    """Convert Natural Questions format.

    NQ annotations can be a list of annotator dicts or a single dict,
    depending on the HuggingFace variant. This handles both.
    """
    samples = []
    for row in ds:
        question = row.get("question", {})
        if isinstance(question, dict):
            question = question.get("text", "")

        annotations = row.get("annotations", {})
        short_answers = []

        # Annotations may be a list of annotator dicts or a single dict
        if isinstance(annotations, list) and annotations:
            ann = annotations[0]
            sa = ann.get("short_answers", [])
            if sa and isinstance(sa[0], dict):
                short_answers = [a.get("text", "") for a in sa if a.get("text")]
            elif sa and isinstance(sa[0], str):
                short_answers = [a for a in sa if a]
        elif isinstance(annotations, dict):
            sa = annotations.get("short_answers", [])
            if sa and isinstance(sa[0], dict):
                short_answers = [a.get("text", "") for a in sa if a.get("text")]
            elif sa and isinstance(sa[0], str):
                short_answers = [a for a in sa if a]

        ground_truth = short_answers[0] if short_answers else ""

        if question:
            samples.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )
    return samples


def _convert_msmarco(ds) -> List[Dict[str, Any]]:
    """Convert MS MARCO format."""
    samples = []
    for row in ds:
        query = row.get("query", "")
        # MS MARCO has "answers" as a list and "passages" as passage dicts
        answers = row.get("answers", [])
        ground_truth = answers[0] if answers else ""

        if query and ground_truth:
            samples.append(
                {
                    "question": query,
                    "ground_truth": ground_truth,
                }
            )
    return samples


def _convert_triviaqa(ds) -> List[Dict[str, Any]]:
    """Convert TriviaQA format."""
    samples = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("answer", {})
        if isinstance(answer, dict):
            ground_truth = answer.get("value", "")
            aliases = answer.get("aliases", [])
        else:
            ground_truth = str(answer) if answer else ""
            aliases = []
        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": ground_truth,
            }
            if aliases:
                sample["ground_truth_aliases"] = aliases
            samples.append(sample)
    return samples


def _convert_hotpotqa(ds) -> List[Dict[str, Any]]:
    """Convert HotpotQA format (fullwiki/distractor setting).

    Extracts the gold supporting passages as ``context`` so offline
    evaluation can use them for RAGAS faithfulness/recall scoring.

    The ``context`` field in HotpotQA rows is a dict:
        {"title": [str, ...], "sentences": [[str, ...], ...]}
    The ``supporting_facts`` field identifies which of those passages
    are the gold evidence:
        {"title": [str, ...], "sent_id": [int, ...]}
    """
    samples = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("answer", "")
        level = row.get("level", "")
        q_type = row.get("type", "")
        supporting_facts = row.get("supporting_facts", {})
        ctx_field = row.get("context", {})
        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": answer,
                "hops": 2,
            }
            if level:
                sample["difficulty"] = level
            if q_type:
                sample["type"] = q_type
            # Extract supporting fact titles for multi-hop tracking
            sf_titles = set()
            if isinstance(supporting_facts, dict):
                sf_titles = set(supporting_facts.get("title", []))
            if sf_titles:
                sample["supporting_titles"] = list(sf_titles)
            # Build gold context from supporting passages.
            # ctx_field has parallel lists: title[i] → sentences[i] (list of strings).
            if isinstance(ctx_field, dict):
                titles = ctx_field.get("title", [])
                sentences = ctx_field.get("sentences", [])
                gold_passages = []
                for i, title in enumerate(titles):
                    if title in sf_titles and i < len(sentences):
                        passage = " ".join(sentences[i])
                        if passage.strip():
                            gold_passages.append(f"{title}: {passage}")
                if gold_passages:
                    sample["context"] = "\n\n".join(gold_passages)
            samples.append(sample)
    return samples


def _convert_musique(ds) -> List[Dict[str, Any]]:
    """Convert MuSiQue format.

    Extracts the gold supporting paragraphs as ``context`` for offline
    evaluation. MuSiQue rows have a ``paragraphs`` field:
        [{"idx": int, "title": str, "paragraph_text": str,
          "is_supporting": bool}, ...]
    """
    samples = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("answer", "")
        answerable = row.get("answerable", True)
        if question and answerable:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": answer,
                "hops": len(row.get("question_decomposition", [])) or 2,
            }
            # Extract gold supporting paragraphs as context
            paragraphs = row.get("paragraphs", [])
            if paragraphs:
                gold_passages = [
                    f"{p['title']}: {p['paragraph_text']}"
                    for p in paragraphs
                    if p.get("is_supporting") and p.get("paragraph_text")
                ]
                if gold_passages:
                    sample["context"] = "\n\n".join(gold_passages)
            samples.append(sample)
    return samples


def _convert_wiki_multihop(ds) -> List[Dict[str, Any]]:
    """Convert 2WikiMultiHopQA format."""
    samples = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("answer", "")
        q_type = row.get("type", "")
        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": answer,
                "hops": 2,
            }
            if q_type:
                sample["type"] = q_type
            samples.append(sample)
    return samples


def _convert_multihop_rag(ds) -> List[Dict[str, Any]]:
    """Convert MultiHop-RAG format."""
    samples = []
    for row in ds:
        question = row.get("query", row.get("question", ""))
        answer = row.get("answer", "")
        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": answer,
                "hops": 2,
            }
            evidence = row.get("evidence_list", [])
            if evidence:
                sample["evidence_count"] = len(evidence)
            samples.append(sample)
    return samples


def _convert_bamboogle(ds) -> List[Dict[str, Any]]:
    """Convert Bamboogle format (125 multi-hop questions)."""
    samples = []
    for row in ds:
        question = row.get("question", row.get("Question", ""))
        answer = row.get("answer", row.get("Answer", ""))
        num_hops = row.get("num_hops", row.get("Num Hops", 2))
        if question:
            samples.append(
                {
                    "question": question,
                    "ground_truth": answer,
                    "hops": int(num_hops) if num_hops else 2,
                }
            )
    return samples


def _convert_popqa(ds) -> List[Dict[str, Any]]:
    """Convert PopQA format (entity-centric QA with popularity scores).

    The ``possible_answers`` field is a JSON-encoded list string.  We parse
    it safely and take the first element as the canonical ground truth.
    Popularity scores and entity metadata are passed through for the adapter
    to store in ``BenchmarkSample.metadata``.
    """
    samples = []
    for row in ds:
        question = row.get("question", "")
        if not question:
            continue

        # Parse possible_answers — JSON list string like '["Paris", "City of Light"]'
        raw_answers = row.get("possible_answers", "")
        if isinstance(raw_answers, list):
            answers = raw_answers
        elif isinstance(raw_answers, str) and raw_answers:
            try:
                answers = json.loads(raw_answers)
                if not isinstance(answers, list):
                    answers = [str(answers)]
            except (json.JSONDecodeError, TypeError):
                answers = [raw_answers]
        else:
            answers = []

        ground_truth = answers[0] if answers else ""

        sample: Dict[str, Any] = {
            "question": question,
            "ground_truth": ground_truth,
        }

        # Preserve all answer aliases for downstream use
        if len(answers) > 1:
            sample["ground_truth_aliases"] = answers

        # Preserve popularity scores and entity metadata for the adapter
        s_pop = row.get("s_pop")
        o_pop = row.get("o_pop")
        try:
            if s_pop is not None and s_pop != "":
                sample["s_pop"] = int(s_pop)
        except (ValueError, TypeError):
            pass
        try:
            if o_pop is not None and o_pop != "":
                sample["o_pop"] = int(o_pop)
        except (ValueError, TypeError):
            pass

        prop = row.get("prop", "")
        if prop:
            sample["prop"] = prop

        subj = row.get("subj", "")
        if subj:
            sample["subj"] = subj

        obj = row.get("obj", "")
        if obj:
            sample["obj"] = obj

        samples.append(sample)
    return samples


def _convert_ragbench(ds) -> List[Dict[str, Any]]:
    """Convert RAGBench format (TRACe-style evaluation)."""
    samples = []
    for row in ds:
        question = row.get("question", "")
        answer = row.get("response", row.get("answer", ""))
        contexts = row.get("documents", row.get("contexts", []))
        if question:
            sample: Dict[str, Any] = {
                "question": question,
                "ground_truth": answer,
            }
            if contexts:
                sample["context"] = (
                    "\n\n".join(contexts) if isinstance(contexts, list) else contexts
                )
            samples.append(sample)
    return samples


def _convert_kilt(ds) -> List[Dict[str, Any]]:
    """Convert KILT format (knowledge-intensive language tasks)."""
    samples = []
    for row in ds:
        question = row.get("input", "")
        output = row.get("output", [])
        ground_truth = ""
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                ground_truth = first.get("answer", "")
            elif isinstance(first, str):
                ground_truth = first
        if question:
            samples.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )
    return samples


def _convert_fever(ds) -> List[Dict[str, Any]]:
    """Convert FEVER format (fact verification)."""
    samples = []
    for row in ds:
        claim = row.get("claim", "")
        label = row.get("label", "")
        if claim:
            samples.append(
                {
                    "question": claim,
                    "ground_truth": label,
                }
            )
    return samples


_FORMAT_CONVERTERS = {
    "beir": _convert_beir,
    "squad": _convert_squad,
    "nq": _convert_nq,
    "msmarco": _convert_msmarco,
    "triviaqa": _convert_triviaqa,
    "hotpotqa": _convert_hotpotqa,
    "musique": _convert_musique,
    "wiki_multihop": _convert_wiki_multihop,
    "multihop_rag": _convert_multihop_rag,
    "bamboogle": _convert_bamboogle,
    "popqa": _convert_popqa,
    "ragbench": _convert_ragbench,
    "kilt": _convert_kilt,
    "fever": _convert_fever,
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _available_splits(hf_name: str, config: Optional[str] = None) -> List[str]:
    """Split names HuggingFace reports for a dataset, best effort.

    Only used to make the "split not available" error actionable. Any failure
    here must not mask the original error, so it returns an empty list rather
    than raising.
    """
    try:
        from datasets import get_dataset_split_names

        return list(get_dataset_split_names(hf_name, config_name=config))
    except Exception:
        return []


def _cache_path(
    name: str,
    split: str,
    n_samples: Optional[int],
    seed: int,
    cache_dir: Optional[str],
) -> Path:
    """Deterministic cache file path for a benchmark configuration."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "lasagna", "benchmarks")

    # Create a deterministic filename from the config.
    # Include _CACHE_VERSION so converter changes invalidate old caches.
    key = f"v{_CACHE_VERSION}:{name}:{split}:{n_samples}:{seed}"
    hash_suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
    safe_name = name.replace("/", "_")
    filename = f"{safe_name}_{split}_{hash_suffix}.json"

    return Path(cache_dir) / filename
