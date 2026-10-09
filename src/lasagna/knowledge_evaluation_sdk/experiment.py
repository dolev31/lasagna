"""Experiment orchestrator — config loading, task generation, and execution.

Supports YAML and TOML experiment configs (auto-detected by file extension).
Generates the Cartesian product of benchmarks x pipeline configs x runs
and delegates execution to :class:`BenchmarkRunner`.

Example::

    from lasagna.knowledge_evaluation_sdk import Experiment

    # From a config file
    exp = Experiment.from_file("beir_sweep.yaml")
    results = await exp.run()
    print(results.to_markdown())

    # Programmatic
    exp = Experiment(
        config=ExperimentConfig(
            name="Quick Test",
            benchmarks=[BenchmarkSpec(name="beir/nfcorpus", n_samples=50)],
        )
    )
    results = await exp.run()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .benchmarks import BenchmarkSample, benchmark_registry
from .benchmarks._base import BaseBenchmarkAdapter as _BaseBenchmarkAdapter
from .checkpoint import CheckpointManager
from .results import BenchmarkResults
from .runner import HARNESS_KEYS, BenchmarkRunner, BenchmarkTask, TaskResult

#: The unoverridden ``load_corpus``. An adapter still pointing at it ships no
#: corpus of its own, whatever else it can do.
_BASE_LOAD_CORPUS = _BaseBenchmarkAdapter.load_corpus


# ---------------------------------------------------------------------------
# Ingestion-time config keys
# ---------------------------------------------------------------------------

# Keys that affect document processing / chunking at ingestion time.
# Experiments that include any of these keys in config_overrides require a
# separate corpus session per config variant (ingestion_per_config mode).
_INGESTION_TIME_KEYS = frozenset(
    {
        "docling_enabled",
        "docling_mode",
        "docling_pdf_backend",
        "docling_ocr_enabled",
        "docling_table_structure",
        "docling_max_tokens",
        "chunking_strategy",
        "chunk_size",
        "chunk_overlap",
        "enable_hierarchical",
        "parent_chunk_size",
        "child_chunk_size",
        "window_size",
        "semantic_threshold",
        "embedding_model",
        "embedding_dim",
        "embedding_truncate_dim",
        "enable_contextual_retrieval",
        "enable_late_chunking",
        "enable_raptor",
        "raptor_max_cluster_size",
        "raptor_max_levels",
        "enable_graph_retrieval",
        "collection_name",
    }
)

# Keys that affect ingestion *behavior* (e.g., whether BM25 tokenization
# happens during upload) but do NOT affect corpus identity.  These are
# propagated as ingestion overrides so DocumentPipeline respects them, but
# are NOT included in the corpus cache signature — different values share
# the same indexed corpus.
_INGESTION_RUNTIME_KEYS = frozenset(
    {
        "enable_hybrid_search",
    }
)

# Benchmarks that share the same corpus.jsonl file can reuse the same indexed
# session.  This avoids re-ingesting 3K+ docs for every new benchmark track.
_SHARED_CORPUS_KEY: Dict[str, str] = {
    "private_eval/retrieval": "private_eval/retrieval_corpus",
    "private_eval/e2e_generation": "private_eval/e2e_corpus",
    "private_eval/hallucination": "private_eval/retrieval_corpus",
    # Deprecated: kept for backward compat with old YAML configs
    "private_eval/generation": "private_eval/e2e_corpus",
}

# Maps benchmark names to corpus identifiers for ingestion signature.
# Benchmarks sharing the same corpus_id reuse the same ingestion.
_CORPUS_ID_MAP: Dict[str, str] = {
    "private_eval/retrieval": "retrieval_corpus",
    "private_eval/e2e_generation": "e2e_corpus",
    "private_eval/hallucination": "retrieval_corpus",
    "private_eval/generation": "e2e_corpus",
}


from lasagna.utils.token_usage import EVAL, SYSTEM, TokenUsage, UsageMeter  # noqa: E402

# There was a `_get_cost_rates` here, and a `_DEFAULT_COST_TABLE` behind it,
# converting token counts to dollars. Both are gone. The token counts they
# multiplied were always zero -- read from a `chunk.metadata` key nothing ever
# wrote -- so every dollar figure this package ever printed was `$0.00000`, and
# the rates were applied at the *judge's* price regardless of which model did
# the work. Prices are also not a property of a run: they change, they differ by
# tier and commitment, and a rate frozen into a published bundle is
# indistinguishable from a measurement to anyone reading it later. Tokens are
# reported; multiply by what you actually pay.


def _is_abstention_answer(text: str) -> bool:
    """Quick check whether an answer is an abstention (no LLM call needed)."""
    if not text or not text.strip():
        return True
    from .answer_metrics import is_abstention

    return is_abstention(text)


def _record_eval_usage(result: TaskResult, meter: UsageMeter) -> None:
    """Attach the judge's spend to a result, separately from the system's."""
    usage = meter.usage(EVAL)
    result.eval_input_tokens += usage.input_tokens
    result.eval_output_tokens += usage.output_tokens
    result.eval_llm_calls += usage.llm_calls


# ---------------------------------------------------------------------------
# Pure helper functions — imported from _helpers.py (zero third-party deps)
# ---------------------------------------------------------------------------
from ._helpers import (  # noqa: E402
    apply_canonical_metric_mapping,
    build_bm25_docs,
    filter_nonempty_agentic_indices,
)


# ---------------------------------------------------------------------------
# Config data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSpec:
    """Specifies a single benchmark to evaluate."""

    name: str
    split: str = "test"
    n_samples: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _boolean_config_flags() -> frozenset:
    """Every boolean setting the pipeline actually has.

    Read off a live default config rather than kept as a hand-written list.
    A list would be a second copy of the schema, it would drift, and the way it
    fails is a factorial rejecting a flag that exists or -- worse -- accepting
    one that does not and reporting an interaction between two settings the
    pipeline ignored.
    """
    try:
        from lasagna.core.config import RAGConfig

        return frozenset(k for k, v in RAGConfig()._config.items() if isinstance(v, bool))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not enumerate config flags for validation: {exc}")
        return frozenset()


#: Flag pairs that route to the same branch, so a 2x2 over them cannot produce
#: four distinct behaviours.
#:
#: A curated list, and honestly so: whether two flags are independent is a
#: property of the code paths they gate, and no static check proves it. What
#: this does is stop the traps already paid for.
#:
#: ``enable_rag_fusion`` and ``enable_query_expansion`` are OR'd into one
#: condition (`_search_orchestrator.py:319`) that delegates to
#: ``search_multi_query``; the only difference is which setting supplies
#: ``n_variations``, and both default to 3. So "expansion only", "fusion only"
#: and "both" all run the identical search -- three of the four cells collapse,
#: and the interaction comes out exactly 0 with a confident interval around it.
#: This shipped as a worked example twice: once as `query_enhancers: [hyde,
#: null]`, and once as the "fix" for it.
_COUPLED_FLAG_PAIRS: Dict[frozenset, str] = {
    frozenset({"enable_rag_fusion", "enable_query_expansion"}): (
        "both are OR'd into the same branch in SearchOrchestrator.search "
        "(_search_orchestrator.py:319) and delegate to search_multi_query; "
        "with rag_fusion_n_queries == max_query_variations == 3 the 'expansion', "
        "'fusion' and 'both' cells run identical searches"
    ),
}


def _short_factor_name(flag: str) -> str:
    """`enable_reranking` -> `reranking`, for readable arm names."""
    for prefix in ("enable_", "use_"):
        if flag.startswith(prefix):
            return flag[len(prefix) :]
    return flag


@dataclass
class FactorialSpec:
    """A 2x2: two techniques, each on and off, all four combinations.

    The design that answers "do these two compose?", which neither a component
    sweep nor a list of override dicts could express here before.

    The dimension lists (``rerankers``, ``chunkers``, ...) have no *off* level
    -- a ``None`` entry is a silent no-op, not a disabled feature -- and the
    ``config_overrides`` branch that can express "off" disables the dimension
    lists and names its arms ``config_0``, ``config_1``. The shipped
    ``beir_sweep.yaml`` demonstrated the trap: ``query_enhancers: [hyde, null]``
    produced two arms that were byte-identical, and it had been in the
    repository as an example.

    Factors are boolean **config flags**, not component slots. A slot has no
    off level, so "reranker off" is not something the four cells can express.
    """

    #: Two boolean settings, e.g. ``enable_hyde`` and ``enable_reranking``.
    factor_a: str = ""
    factor_b: str = ""
    #: Applied to all four cells. Use for settings that must not vary, such as
    #: ``enable_query_classification: false``, which otherwise overwrites
    #: ``hybrid_alpha`` per query and silently blurs the arms together.
    held_fixed: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.factor_a and self.factor_b)

    def validate(self) -> None:
        """Refuse a design that cannot measure what it claims to.

        Checked here rather than at run time. A factorial that is wrong costs a
        full sweep before anyone finds out, and what it reports on the way out
        is an interaction estimate -- a number that looks exactly like a real
        one.
        """
        if not self.factor_a or not self.factor_b:
            raise ValueError("A factorial needs both `factor_a` and `factor_b`.")
        if self.factor_a == self.factor_b:
            raise ValueError(
                f"factor_a and factor_b are both {self.factor_a!r}. A 2x2 over one "
                "factor has an interaction of zero by construction."
            )

        coupled = _COUPLED_FLAG_PAIRS.get(frozenset({self.factor_a, self.factor_b}))
        if coupled:
            raise ValueError(
                f"{self.factor_a!r} and {self.factor_b!r} cannot form a 2x2: {coupled}. "
                "The interaction would be zero by construction."
            )

        flags = _boolean_config_flags()
        for role, name in (("factor_a", self.factor_a), ("factor_b", self.factor_b)):
            if name in _COMPONENT_KEYS:
                raise ValueError(
                    f"{role}={name!r} is a component slot, not a boolean flag. A "
                    f"slot has no 'off' level, so the 2x2 cannot be built from "
                    f"it. Use the flag that disables the stage instead -- for "
                    f"example `enable_reranking` rather than `reranker`."
                )
            if name not in flags:
                suggestions = sorted(f for f in flags if f.startswith(("enable_", "use_")))
                raise ValueError(
                    f"{role}={name!r} is not a known boolean setting. "
                    f"Candidates: {', '.join(suggestions[:12])}, ... "
                    f"({len(suggestions)} in total)."
                )

    def cells(self) -> List[Dict[str, Any]]:
        """The four arms, named and with their levels recorded in the overrides.

        ``config_overrides`` travels onto every ``TaskResult``, so the levels
        can be read back out of a results file without parsing config names.
        """
        self.validate()
        short_a = _short_factor_name(self.factor_a)
        short_b = _short_factor_name(self.factor_b)

        return [
            {
                "name": name,
                "overrides": {
                    **dict(self.held_fixed),
                    self.factor_a: level_a,
                    self.factor_b: level_b,
                },
            }
            for name, level_a, level_b in (
                ("neither", False, False),
                (short_a, True, False),
                (short_b, False, True),
                (f"{short_a}+{short_b}", True, True),
            )
        ]


@dataclass
class PipelineSweepSpec:
    """Specifies which pipeline components to sweep."""

    base_profile: str = "standard"
    rerankers: List[str] = field(default_factory=list)
    embedders: List[str] = field(default_factory=list)
    chunkers: List[str] = field(default_factory=list)
    query_enhancers: List[str] = field(default_factory=list)
    config_overrides: List[Dict[str, Any]] = field(default_factory=list)
    factorial: FactorialSpec = field(default_factory=FactorialSpec)


@dataclass
class ExperimentConfig:
    """Full experiment specification."""

    name: str = "experiment"
    description: str = ""
    benchmarks: List[BenchmarkSpec] = field(default_factory=list)
    pipeline: PipelineSweepSpec = field(default_factory=PipelineSweepSpec)
    mode: str = "auto"
    n_runs: int = 1
    #: What repeated runs vary. "run" (default) keeps the query subset
    #: fixed so +/- std reflects pipeline nondeterminism -- which is what
    #: reports have always implied. "sample" redraws each run, the old
    #: behaviour, which measures sampling variance instead.
    variance_source: str = "run"
    n_samples: Optional[int] = None
    seed: int = 42
    max_concurrency: int = 4
    checkpoint_dir: str = ""
    output_dir: str = ""
    output_formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    judge_model: str = ""  # LLM model for RAGAS judge (e.g. "gpt-4o-mini")
    ingestion_per_config: bool = False  # True = each config variant gets its own corpus session
    max_corpus_docs: int = 0  # 0 = no limit; >0 = cap corpus size (e.g. 500 for pilot runs)
    #: Evaluation tracks to run: "retrieval", "generation", or both (the
    #: default). A component sweep scored on retrieval needs no LLM or RAGAS.
    tracks: List[str] = field(default_factory=lambda: ["retrieval", "generation"])
    _resolved_output_dir: Optional[str] = field(default=None, repr=False)

    def effective_output_dir(self) -> str:
        """Return the resolved output directory (stable across calls).

        When no explicit ``output_dir`` is set, generates a timestamped
        path once and caches it so that checkpoint clearing, the run
        itself, and the final display all point to the same directory.
        """
        if self.output_dir:
            return os.path.expanduser(self.output_dir)
        # Cache on first call so repeated accesses return the same path
        if self._resolved_output_dir is None:
            self._resolved_output_dir = os.path.join(
                os.path.expanduser("~"),
                ".lasagna",
                "research",
                "results",
                f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        return self._resolved_output_dir

    def config_hash(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "benchmarks": [b.name for b in self.benchmarks],
                "pipeline": {
                    "base_profile": self.pipeline.base_profile,
                    "rerankers": self.pipeline.rerankers,
                    "embedders": self.pipeline.embedders,
                },
                "mode": self.mode,
                "n_runs": self.n_runs,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # Python 3.10 fallback
        except ImportError:
            raise ImportError("TOML parsing requires tomli for Python <3.11: pip install tomli")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _embedder_collection_name(model: Optional[str], dim: Any) -> str:
    """The benchmark collection for one embedder (model and dimension)."""
    import hashlib

    emb_id = f"{model or 'default'}|{dim or 'default'}"
    return f"rag_bench_emb_{hashlib.md5(emb_id.encode()).hexdigest()[:8]}"


#: Registered rerankers that score with the configured cross-encoder model.
_CROSS_ENCODER_RERANKERS = frozenset(
    {"cross_encoder", "cross_encoder_adapted", "combined", "combined_adapted"}
)


def _ran_system(
    embedder_model: Any,
    embedder_dim: Any,
    ingestion: Optional[Dict[str, Any]],
    task_config: Dict[str, Any],
    reranker_component: Optional[str] = None,
) -> Dict[str, Any]:
    """What a live task actually ran with, for its held-fixed block."""
    from lasagna.core.config import rag_config

    ingestion = ingestion or {}

    def _ingest(name: str) -> Any:
        value = ingestion.get(name)
        return value if value is not None else getattr(rag_config, name, None)

    # The strategy that ran, not the one requested: structural settings
    # override it at ingest, in the order resolve_effective_strategy applies.
    chunker = getattr(_ingest("chunking_strategy"), "value", _ingest("chunking_strategy"))
    if _ingest("enable_hierarchical"):
        chunker = "hierarchical"
    elif _ingest("preserve_headers") or _ingest("preserve_tables"):
        chunker = "structure_aware"
    try:
        from lasagna.retrieval.search.hybrid import STEMMER_AVAILABLE as bm25_stemming
    except Exception:  # pragma: no cover - hybrid search always importable here
        bm25_stemming = None
    return {
        "embedder_model": embedder_model,
        "embedder_dim": embedder_dim,
        "chunker": chunker,
        "chunk_size": _ingest("child_chunk_size")
        if chunker == "hierarchical"
        else _ingest("chunk_size"),
        "retriever_top_k": task_config.get("default_top_k"),
        # A named component that is not a cross-encoder reranks on its own,
        # so the configured cross-encoder model is not what ran.
        "reranker_model": (
            reranker_component
            if reranker_component and reranker_component not in _CROSS_ENCODER_RERANKERS
            else task_config.get("reranker_model")
        )
        if task_config.get("enable_reranking")
        else None,
        "vector_backend": getattr(rag_config, "vector_backend", None),
        "bm25_stemming": bm25_stemming,
    }


#: Rows one presence query may return; below Milvus's limit+offset ceiling.
_PRESENCE_QUERY_ROWS = 16_000


def _present_document_ids(
    backend: Any, collection_name: str, session_id: str, expected_ids: Any
) -> set:
    """Which of ``expected_ids`` already have chunks in this session.

    Asked in bounded batches (``document_id in [...]``) instead of listing the
    whole collection. The listing streamed every row through Milvus Lite's
    query iterator, which on a re-run or a resume could block forever at 0%
    CPU; this asks the question the pre-flight actually has, with a known
    number of plain queries. A batch whose reply fills the row cap is split
    and asked again, so a document with many chunks is never reported missing.
    """
    from lasagna.vector_backends.filters import FilterExpression as _FE

    ids = sorted(expected_ids)
    if not ids:
        return set()
    try:
        present: set = set()
        pending = [ids[i : i + 400] for i in range(0, len(ids), 400)]
        while pending:
            batch = pending.pop()
            rows = backend.query(
                collection_name=collection_name,
                filter_expr=_FE.and_(
                    _FE.eq("session_id", session_id), _FE.in_list("document_id", batch)
                ),
                output_fields=["document_id"],
                limit=_PRESENCE_QUERY_ROWS,
            )
            if len(rows) >= _PRESENCE_QUERY_ROWS and len(batch) > 1:
                half = len(batch) // 2
                pending.extend([batch[:half], batch[half:]])
                continue
            present.update(r.get("document_id") for r in rows if r.get("document_id"))
        return present
    except NotImplementedError:
        return set(
            backend.list_document_ids(
                collection_name=collection_name, session_id=session_id, limit=len(ids) + 1000
            )
        )


def _check_components_exist(pipeline_configs: List[Dict[str, Any]]) -> None:
    """Refuse a sweep that names a component the registry does not have.

    Checked before any corpus is ingested. The command line only sees
    components installed as ``lasagna_components`` entry points (or built in);
    one defined in a script the CLI never imports is not registered.
    """
    from lasagna.core.registry import registry

    unknown = []
    for pc in pipeline_configs:
        for key, value in (pc.get("overrides") or {}).items():
            if key in _COMPONENT_KEYS and isinstance(value, str):
                known = registry.list(key).get(key, [])
                if value not in known:
                    unknown.append(
                        f"{pc.get('name')}: {key}={value!r} (registered: {sorted(known)})"
                    )
    if unknown:
        raise ValueError(
            "These arms name components that are not registered, so they would "
            "measure the default component instead: " + "; ".join(unknown) + ". "
            "Install your component as a lasagna_components entry point "
            "(see docs/contributing-components.md)."
        )


def _fix_retrieval_depth(task_config: Dict[str, Any], depth: int) -> None:
    """Make retrieval evaluation return ``depth`` uncached, untruncated results.

    Answer-time policies are right for answering and wrong for measuring
    retrieval at k: the search cache can serve a different arm's lists, the
    relative similarity threshold and adaptive-k cut lists well below k, and
    default_top_k can cap the candidates below the requested depth.
    """
    task_config["enable_search_cache"] = False
    task_config["similarity_threshold"] = 0.0
    ccr = dict(task_config.get("chunk_cite_retrieval") or {})
    ccr["strategy"] = "fixed"
    task_config["chunk_cite_retrieval"] = ccr
    task_config["default_top_k"] = max(int(task_config.get("default_top_k") or 0), depth)


#: Above this share of failed answer generations, a task is failed, not scored.
_MAX_FAILED_GENERATION_RATE = 0.1


class _StaleCorpusCleanupError(RuntimeError):
    """A partial corpus could not be removed before re-ingesting it."""


def _agentic_fell_back(agentic_requested: bool, search_results: List[Any]) -> bool:
    """Whether an agentic request was served by standard retrieval instead.

    The orchestrator stamps the tier that ran on the first result. A failed
    agentic run falls back to standard search and still returns results, so an
    empty list is not the only sign of a fallback.
    """
    if not agentic_requested:
        return False
    if not search_results:
        return True
    first_chunk = getattr(search_results[0], "chunk", None)
    tier_used = (getattr(first_chunk, "metadata", None) or {}).get("_routing_tier_used")
    return tier_used == "standard_fallback"


_TOP_LEVEL_KEYS = frozenset(
    {
        "name",
        "description",
        "benchmarks",
        "pipeline",
        "mode",
        "n_runs",
        "variance_source",
        "n_samples",
        "seed",
        "max_concurrency",
        "checkpoint_dir",
        "output_dir",
        "output_formats",
        "judge_model",
        "ingestion_per_config",
        "max_corpus_docs",
        "tracks",
    }
)
_PIPELINE_KEYS = frozenset(
    {
        "base_profile",
        "rerankers",
        "embedders",
        "chunkers",
        "query_enhancers",
        "config_overrides",
        "factorial",
    }
)


def _parse_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Parse a raw config dict into :class:`ExperimentConfig`."""
    # A pipeline key at the top level used to vanish without a word, and the
    # sweep silently collapsed to the baseline configuration.
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    misplaced = [k for k in unknown if k in _PIPELINE_KEYS]
    if misplaced:
        logger.warning(
            f"Ignoring top-level keys {misplaced}: pipeline settings belong under `pipeline:`"
        )
    if set(unknown) - set(misplaced):
        logger.warning(f"Ignoring unknown experiment keys {sorted(set(unknown) - set(misplaced))}")
    benchmarks = []
    for b in data.get("benchmarks", []):
        if isinstance(b, str):
            benchmarks.append(BenchmarkSpec(name=b))
        elif isinstance(b, dict):
            benchmarks.append(
                BenchmarkSpec(
                    name=b["name"],
                    split=b.get("split", "test"),
                    n_samples=b.get("n_samples"),
                    filters=b.get("filters", {}),
                )
            )

    pipeline_data = data.get("pipeline", {})
    factorial_data = pipeline_data.get("factorial") or {}
    factorial = FactorialSpec(
        factor_a=factorial_data.get("factor_a", ""),
        factor_b=factorial_data.get("factor_b", ""),
        held_fixed=factorial_data.get("held_fixed", {}) or {},
    )
    # Validated while reading the file, so a bad design fails before the first
    # corpus is ingested rather than after the last one.
    if factorial:
        factorial.validate()
    elif factorial_data:
        raise ValueError(
            "`pipeline.factorial` needs both `factor_a` and `factor_b`; "
            f"got {sorted(factorial_data)}."
        )

    pipeline = PipelineSweepSpec(
        base_profile=pipeline_data.get("base_profile", "standard"),
        rerankers=pipeline_data.get("rerankers", []),
        embedders=pipeline_data.get("embedders", []),
        chunkers=pipeline_data.get("chunkers", []),
        query_enhancers=pipeline_data.get("query_enhancers", []),
        config_overrides=pipeline_data.get("config_overrides", []),
        factorial=factorial,
    )

    return ExperimentConfig(
        name=data.get("name", "experiment"),
        description=data.get("description", ""),
        benchmarks=benchmarks,
        pipeline=pipeline,
        # `auto`, matching ExperimentConfig.mode, research/README.md and the
        # eval skill -- all three of which document `auto` as the default.
        #
        # This read `"offline"`, the deprecated alias for `generation_only`. A
        # YAML that omitted `mode` therefore never constructed the pipeline and
        # never measured retrieval, so a reranker or embedder sweep written from
        # the documented config reference returned four identical numbers with
        # no indication why.
        mode=data.get("mode", "auto"),
        n_runs=data.get("n_runs", 1),
        # What repeated runs vary. Declared on ExperimentConfig, documented in
        # the shipped plugin template with a comment explaining why it matters,
        # read in three places -- and never parsed, so setting it in YAML did
        # nothing and every run measured pipeline nondeterminism whatever the
        # file said.
        variance_source=data.get("variance_source", "run"),
        n_samples=data.get("n_samples"),
        seed=data.get("seed", 42),
        max_concurrency=data.get("max_concurrency", 4),
        checkpoint_dir=data.get("checkpoint_dir", ""),
        output_dir=data.get("output_dir", ""),
        output_formats=data.get("output_formats", ["markdown", "json"]),
        judge_model=data.get("judge_model", ""),
        ingestion_per_config=data.get("ingestion_per_config", False),
        max_corpus_docs=data.get("max_corpus_docs", 0),
        tracks=_parse_tracks(data.get("tracks")),
    )


# ---------------------------------------------------------------------------
# Pipeline config generation
# ---------------------------------------------------------------------------


_TRACKS = ("retrieval", "generation")


def _parse_tracks(value: Any) -> List[str]:
    """The ``tracks:`` list, checked: a typo must not silently run nothing."""
    if value is None:
        return list(_TRACKS)
    tracks = [value] if isinstance(value, str) else list(value)
    unknown = sorted(set(tracks) - set(_TRACKS))
    if unknown or not tracks:
        raise ValueError(f"tracks must be a non-empty subset of {list(_TRACKS)}; got {value!r}")
    return [t for t in _TRACKS if t in tracks]


def _generate_pipeline_configs(spec: PipelineSweepSpec) -> List[Dict[str, Any]]:
    """Generate Cartesian product of pipeline component variants.

    Returns a list of ``{name: str, overrides: dict, profile: str}`` dicts.
    The ``profile`` key tells downstream code which base profile to load.
    """
    profile = spec.base_profile or "standard"

    # A factorial is its own branch, ahead of both the override list and the
    # dimension lists. It emits override-style configs -- the only shape that
    # can express a boolean "off" -- but with chosen names, so the four cells
    # are identifiable in a leaderboard instead of arriving as `config_0..3`.
    if spec.factorial:
        return [
            {"name": cell["name"], "overrides": cell["overrides"], "profile": profile}
            for cell in spec.factorial.cells()
        ]

    if spec.config_overrides:
        # The component lists used to be dropped here without a word when
        # config_overrides was also given (the natural way to add
        # enable_reranking for a reranker sweep), so the sweep never ran.
        _lists = {
            name: getattr(spec, name)
            for name in ("rerankers", "embedders", "chunkers", "query_enhancers")
            if getattr(spec, name)
        }
        if _lists:
            raise ValueError(
                f"pipeline sets both config_overrides and {sorted(_lists)}; the lists would "
                f"be ignored. Put each component in its own config_overrides entry "
                f"(for example {{name: mine, reranker: my_reranker}})."
            )
        # Explicit overrides take precedence — copy to avoid mutating caller
        configs = []
        seen_names: Dict[str, int] = {}  # (name, profile) → first index
        for i, override in enumerate(spec.config_overrides):
            override = dict(override)  # shallow copy to avoid mutating original
            name = override.pop("name", f"config_{i}")
            # Allow per-override profile; fall back to spec-level
            cfg_profile = override.pop("profile", profile)

            # Auto-derive per-model collection name for embedding experiments.
            # Must happen BEFORE config_hash is computed so that corpus cache
            # keys and session IDs are unique per embedding model.
            if "embedding_model" in override or "embedding_dim" in override:
                override.setdefault(
                    "collection_name",
                    _embedder_collection_name(
                        override.get("embedding_model"), override.get("embedding_dim")
                    ),
                )

            # Validate name uniqueness — duplicate names cause checkpoint
            # collisions (config_hash in task_key guards correctness, but
            # human-readable names should still be distinct).
            identity = f"{name}|{cfg_profile}"
            if identity in seen_names:
                logger.warning(
                    f"Duplicate pipeline config name '{name}' (profile='{cfg_profile}') "
                    f"at index {i} (first seen at index {seen_names[identity]}). "
                    f"Checkpoint keys include a content hash so correctness is "
                    f"preserved, but distinct names improve readability."
                )
            else:
                seen_names[identity] = i

            configs.append({"name": name, "overrides": override, "profile": cfg_profile})
        return configs

    # Build sweep dimensions
    dimensions: Dict[str, List[str]] = {}
    if spec.rerankers:
        dimensions["reranker"] = spec.rerankers
    if spec.embedders:
        dimensions["embedder"] = spec.embedders
    if spec.chunkers:
        dimensions["chunker"] = spec.chunkers
    if spec.query_enhancers:
        dimensions["query_enhancer"] = spec.query_enhancers

    # `null` in a dimension list reads as "the feature off" and is not.
    #
    # It survives into the config name and into `config_overrides`, so the arm
    # looks like a treatment, but `_apply_overrides_to_pipeline` skips a None
    # component and the pipeline keeps the profile's default. When the profile
    # already has the feature off, the two arms are byte-identical -- and the
    # sweep reads as evidence that the technique does not help. The shipped
    # `beir_sweep.yaml` had exactly this in it.
    #
    # A feature is turned off by its flag, in a `factorial` or an explicit
    # `config_overrides` entry, and then the off level is recorded.
    for key, values in dimensions.items():
        if any(v is None for v in values):
            raise ValueError(
                f"`{key}` contains a null entry. A null does not disable the stage "
                f"-- it leaves the profile's default in place, so that arm may be "
                f"identical to another one and nothing will say so. To compare "
                f"on against off, use a `factorial` over the boolean flag "
                f"(e.g. `enable_reranking`), which records both levels."
            )

    if not dimensions:
        return [{"name": profile, "overrides": {}, "profile": profile}]

    # Cartesian product
    keys = sorted(dimensions.keys())
    value_lists = [dimensions[k] for k in keys]
    configs = []
    for combo in product(*value_lists):
        overrides = dict(zip(keys, combo))
        name = "_".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        configs.append({"name": name, "overrides": overrides, "profile": profile})

    return configs


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class Experiment:
    """Orchestrates a benchmark experiment from config to results.

    Generates the Cartesian product of (benchmarks x pipeline_configs x runs),
    executes them via :class:`BenchmarkRunner`, and returns
    :class:`BenchmarkResults`.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        eval_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self._eval_fn = eval_fn

    @classmethod
    def from_file(cls, path: str, eval_fn: Optional[Callable] = None) -> "Experiment":
        """Load experiment config from YAML or TOML (auto-detected by extension)."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext in (".yaml", ".yml"):
            data = _load_yaml(str(p))
        elif ext == ".toml":
            data = _load_toml(str(p))
        else:
            raise ValueError(f"Unsupported config format: {ext} (use .yaml or .toml)")

        config = _parse_config(data)
        return cls(config=config, eval_fn=eval_fn)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], eval_fn: Optional[Callable] = None) -> "Experiment":
        """Create experiment from a raw config dict."""
        return cls(config=_parse_config(data), eval_fn=eval_fn)

    def generate_tasks(self) -> List[BenchmarkTask]:
        """Generate all benchmark tasks (Cartesian product)."""
        cfg = self.config
        pipeline_configs = _generate_pipeline_configs(cfg.pipeline)
        tasks: List[BenchmarkTask] = []

        # Ingestion always runs the built-in chunker and document processor,
        # so an arm naming another one shared the default corpus and reported
        # its number. This was a warning in live mode only, and the default
        # mode (auto) printed nothing.
        all_override_keys: set = set()
        for pc in pipeline_configs:
            all_override_keys.update(pc["overrides"].keys())
        unsweepable = sorted(all_override_keys & _UNSWEEPABLE_COMPONENTS)
        if unsweepable:
            raise ValueError(
                f"Experiments cannot swap {unsweepable}: ingestion always uses the "
                f"built-in chunker and document processor, so every arm would measure "
                f"the same corpus. Sweep the ingestion settings instead "
                f"(chunking_strategy, chunk_size, chunk_overlap, enable_hierarchical), "
                f"which re-ingest per arm, or compare your chunker with "
                f"`lasagna component validate`."
            )
        if "agentic_pipeline" in all_override_keys:
            logger.warning(
                "agentic_pipeline arms change generation metrics only: retrieval "
                "metrics are measured on the standard tier for every arm."
            )
        _check_components_exist(pipeline_configs)

        for bench_spec in cfg.benchmarks:
            for pipe_cfg in pipeline_configs:
                for run_idx in range(cfg.n_runs):
                    n_samples = bench_spec.n_samples or cfg.n_samples
                    task = BenchmarkTask(
                        benchmark_name=bench_spec.name,
                        config_name=pipe_cfg["name"],
                        config_overrides=dict(
                            pipe_cfg["overrides"]
                        ),  # copy to avoid cross-task mutation
                        run_index=run_idx,
                        split=bench_spec.split,
                        n_samples=n_samples,
                        seed=cfg.seed,
                        mode=cfg.mode,
                        profile=pipe_cfg.get("profile", "standard"),
                        filters=bench_spec.filters,
                        max_corpus_docs=cfg.max_corpus_docs,
                        variance_source=cfg.variance_source,
                        tracks=(tuple(cfg.tracks) if list(cfg.tracks) != list(_TRACKS) else None),
                    )
                    tasks.append(task)

        return tasks

    async def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BenchmarkResults:
        """Execute the experiment and return results.

        Args:
            progress_callback: Optional ``(completed, total, task_key) -> None``.

        Returns:
            :class:`BenchmarkResults` with all task outcomes.
        """
        cfg = self.config
        output_dir = cfg.effective_output_dir()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting experiment '{cfg.name}' "
            f"({len(cfg.benchmarks)} benchmarks, "
            f"{cfg.n_runs} run(s), mode={cfg.mode})"
        )

        # Set judge model for offline eval (module-level so _eval_offline can read it)
        # RAG_EVAL_JUDGE_MODEL env var takes precedence over YAML judge_model so
        # operators can override the model without editing YAML files (e.g. for OpenRouter).
        global _active_judge_model
        _active_judge_model = os.environ.get("RAG_EVAL_JUDGE_MODEL") or cfg.judge_model or None
        if _active_judge_model:
            logger.info(f"Judge model: {_active_judge_model}")

        # Checkpoint manager
        cp_dir = cfg.checkpoint_dir or output_dir
        checkpoint = CheckpointManager(cp_dir)

        # Generate tasks
        tasks = self.generate_tasks()
        logger.info(f"Generated {len(tasks)} tasks")

        # Pre-register corpus ref counts for live mode.
        # With max_concurrency=1 tasks run sequentially; without this,
        # task N releases (and deletes) the corpus before task N+1 runs.
        if cfg.mode == "live":
            await _pre_register_corpus_refs(tasks)

        # Select eval function
        eval_fn = self._eval_fn or _default_eval_fn

        # Run
        runner = BenchmarkRunner(max_concurrency=cfg.max_concurrency)
        task_results = await runner.run_tasks(
            tasks=tasks,
            eval_fn=eval_fn,
            checkpoint=checkpoint,
            progress_callback=progress_callback,
        )

        # Build results
        results = BenchmarkResults(
            task_results=task_results,
            experiment_name=cfg.name,
            experiment_config=_run_metadata(cfg),
        )

        # Save outputs
        _save_outputs(results, output_dir, cfg.output_formats)

        logger.info(
            f"Experiment '{cfg.name}' complete: "
            f"{results.total_tasks} tasks, "
            f"{len(results.failed_tasks)} failed"
        )

        return results


_PREFERRED_METRICS = [
    # Generation metrics — faithfulness is most meaningful (context_precision is trivially 1.0)
    "faithfulness",
    "context_recall",
    # Retrieval IR metrics — prefer NDCG then F1
    "ndcg@10",
    "ndcg_at_10",
    "f1@10",
    "f1_at_10",
    "mrr",
    "map",
]


def _pick_primary_metric(available: List[str]) -> str:
    """Pick the most informative metric to use as the report's primary column."""
    for preferred in _PREFERRED_METRICS:
        if preferred in available:
            return preferred
    return available[0] if available else "ndcg@10"


def _save_outputs(
    results: BenchmarkResults,
    output_dir: str,
    formats: List[str],
) -> None:
    """Write experiment outputs to the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "json" in formats:
        results.save(str(out / "results.json"))

    if "markdown" in formats:
        # Write a report for each available metric
        metrics = results.get_metric_names()
        primary_metric = _pick_primary_metric(metrics)
        report = results.to_markdown(primary_metric)
        (out / "report.md").write_text(report)
        logger.info(f"Markdown report saved to {out / 'report.md'}")

    if "latex" in formats:
        metrics = results.get_metric_names()
        primary_metric = _pick_primary_metric(metrics)
        latex = results.to_latex(primary_metric)
        (out / "table.tex").write_text(latex)
        logger.info(f"LaTeX table saved to {out / 'table.tex'}")


#: ``generation_only`` is the honest name for what used to be called
#: ``offline``: it does not run the retrieval pipeline at all, it scores gold
#: context with an LLM judge. The old name implied "no network", which it never
#: meant -- it calls a judge model -- and implied the pipeline was involved,
#: which it never was. ``offline`` stays accepted as an alias.
#:
#: ``auto`` picks ``live`` when the benchmark can supply a corpus and
#: ``generation_only`` when it cannot, and says which it chose and why. It is
#: the default because the previous default silently produced numbers that did
#: not measure what the caller asked about.
def _run_metadata(cfg: "ExperimentConfig") -> Dict[str, Any]:
    """Everything needed to situate this run, beside the numbers.

    The previous version recorded four fields -- mode, n_runs, seed, and the
    benchmark names -- which is not enough to tell whether two results files
    are describing the same experiment. Missing were the library versions and
    hardware that move a metric, the resolved sweep (a one-way config hash
    cannot be read back), the judge model, and the split.
    """
    from datetime import datetime, timezone

    from lasagna.core.provenance import capture, redact

    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg.name,
        "description": cfg.description,
        # Requested mode; each result carries what it actually resolved to.
        "mode": cfg.mode,
        "n_runs": cfg.n_runs,
        "seed": cfg.seed,
        "variance_source": cfg.variance_source,
        "determinism": "seeded",  # never "reproducible" -- see seeding.py
        "max_concurrency": cfg.max_concurrency,
        "benchmarks": [
            {"name": b.name, "split": b.split, "n_samples": b.n_samples, "filters": b.filters}
            for b in cfg.benchmarks
        ],
        # The sweep as resolved, not as a hash. A config_hash is one-way, so a
        # reader of the results file could not recover what was varied.
        "pipeline": {
            "base_profile": cfg.pipeline.base_profile,
            "rerankers": list(cfg.pipeline.rerankers),
            "embedders": list(cfg.pipeline.embedders),
            "chunkers": list(cfg.pipeline.chunkers),
            "query_enhancers": list(cfg.pipeline.query_enhancers),
            "config_overrides": list(cfg.pipeline.config_overrides),
        },
        "judge_model": _active_judge_model or cfg.judge_model or None,
        "max_corpus_docs": cfg.max_corpus_docs,
        "tracks": list(cfg.tracks),
        "environment": capture(),
        # The system under test. Five of the fourteen comparability-key fields
        # are sourced from here, and without it they are all None -- which means
        # two runs on *different embedding models* hash to the same key and get
        # ranked in the same leaderboard table. A key that cannot tell those
        # apart is worse than no key, because it looks like it can.
        #
        # These are the base-profile values, resolved from the profile this run
        # actually named -- not from the process-global config, which no run
        # uses and which made a speed-profile run report the operator's
        # settings file instead of its own embedder.
        #
        # When a sweep varies one of them the arms still share this block, which
        # is correct: the varied stage is deliberately excluded from the key and
        # recorded in varied_dimensions, so the arms belong in one group by
        # construction.
        "system": _system_under_test(cfg.pipeline.base_profile),
    }
    return redact(meta)


def _resolve_profile_config(base_profile: Optional[str]) -> Optional[Any]:
    """The named profile's configuration, or ``None`` if it cannot be resolved.

    ``_system_under_test`` used to read the process-global ``rag_config``, which
    is not what any run uses: every override path in this harness is
    session-scoped (``sp.set_config_overrides``), and ``from_profile`` stores
    the profile on the pipeline instance. The global is never mutated. So a
    ``speed``-profile run (all-MiniLM-L6-v2 @ 384) published
    ``held_fixed.embedder_model = BAAI/bge-m3, embedder_dim = 1024`` -- whatever
    happened to be in the operator's settings file.

    Five of the fourteen comparability-key fields come from that block, so two
    runs on *different embedding models* hashed to the same key, passed
    ``lasagna results validate``, and were ranked in one leaderboard table. The
    comment six lines above the call site says exactly why that is worse than
    having no key at all.
    """
    if not base_profile:
        return None
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager()
        # `_default_profiles` and not `list_profiles()`: the latter returns
        # {name, description, builtin} metadata dicts with no config on them, so
        # reading `.config` off one silently yields None for every profile.
        # This is the same lookup `KnowledgePipeline.from_profile` performs.
        profile = next(
            (p for p in mgr._default_profiles if getattr(p, "name", None) == base_profile),
            None,
        )
    except Exception:  # pragma: no cover - provenance must never break a bundle
        return None
    if profile is None:
        return None
    return getattr(profile, "config", None)


def _system_under_test(base_profile: Optional[str] = None) -> Dict[str, Any]:
    """Resolved pipeline configuration, for the held-fixed block.

    Read defensively: a missing field must land as ``None`` rather than raise,
    because a bundle that fails to build is worse than one with a gap a reader
    can see. The same principle decides the fallback below: when the profile
    cannot be resolved these fields are ``None`` rather than the machine's
    global defaults, because a visible gap is honest and a confident wrong
    value is not.
    """
    profile_config = _resolve_profile_config(base_profile)

    try:
        from lasagna.core.config import rag_config
    except Exception:  # pragma: no cover - config import is exercised everywhere
        return {}

    def _from_profile(name: str) -> Any:
        """`RAGProfile.config` is a plain dict, not a config object.

        Reading it with `getattr` returns None for every field, which is how
        this looked fixed while still reporting the global: `speed` pins
        ``all-MiniLM-L6-v2`` at 384 and `standard` pins ``BAAI/bge-m3`` at 1024,
        and both published 1024.
        """
        if isinstance(profile_config, dict):
            return profile_config.get(name)
        return getattr(profile_config, name, None)

    def _get(name: str) -> Any:
        if profile_config is not None:
            value = _from_profile(name)
            if value is not None:
                return value
            # A profile that genuinely does not pin this field inherits it, so
            # the global is the right answer for that field specifically.
            return getattr(rag_config, name, None)
        if base_profile:
            # A profile was named and could not be resolved. Reporting the
            # global here is what produced the wrong key; say nothing instead.
            return None
        return getattr(rag_config, name, None)

    # Agentic prompt and per-node model overrides. Without these, two agentic
    # runs differing in the prompt that decides whether an answer is grounded
    # record nothing to tell them apart and compare as the same system.
    try:
        from lasagna.agentic.prompts import active_prompt_fingerprint

        prompt_fp = active_prompt_fingerprint(rag_config)
    except Exception:  # pragma: no cover - never break a bundle over provenance
        prompt_fp = {}

    try:
        from lasagna.retrieval.search.hybrid import STEMMER_AVAILABLE as bm25_stemming
    except Exception:  # pragma: no cover - hybrid search always importable here
        bm25_stemming = None

    return {
        "bm25_stemming": bm25_stemming,
        "agentic_prompts": prompt_fp,
        "agentic_models": dict(_get("agentic_models") or {}),
        "embedder_model": _get("embedding_model"),
        "embedder_dim": _get("embedding_dim"),
        "chunker": _get("chunking_strategy"),
        "chunk_size": _get("chunk_size"),
        "chunk_overlap": _get("chunk_overlap"),
        "retriever_top_k": _get("rerank_top_k"),
        "vector_backend": _get("vector_backend"),
    }


_VALID_MODES = frozenset({"auto", "live", "generation_only", "offline"})

#: Deprecated spellings mapped to their replacement.
_MODE_ALIASES = {"offline": "generation_only"}


def resolve_mode(mode: str, benchmark_name: str) -> tuple[str, str]:
    """Resolve ``mode`` for one benchmark. Returns ``(mode, reason)``.

    The reason is recorded alongside the results, because "which mode did this
    number come from" is the first question anyone comparing two runs has to
    answer, and inferring it later from a metric name is guesswork.
    """
    canonical = _MODE_ALIASES.get(mode, mode)

    if canonical != "auto":
        if canonical == "offline":  # pragma: no cover - alias maps above
            canonical = "generation_only"
        return canonical, "requested explicitly"

    try:
        from .benchmarks import benchmark_registry

        adapter = benchmark_registry.get(benchmark_name)
        supports_live = adapter.load_corpus is not None and _adapter_has_corpus(adapter)
    except Exception as exc:  # unknown benchmark: let the task fail with a real error
        return "generation_only", f"could not inspect benchmark ({type(exc).__name__})"

    if supports_live:
        if getattr(adapter, "pools_gold_context", False) and (
            type(adapter).load_corpus is _BASE_LOAD_CORPUS
        ):
            return (
                "live",
                "corpus pooled from the samples' own gold context, so retrieval "
                "can be measured -- but against gold passages only, which is an "
                "easier task than the published benchmark",
            )
        return "live", "benchmark provides a corpus, so retrieval can be measured"
    return (
        "generation_only",
        "benchmark provides no corpus, so retrieval cannot be measured; "
        "scoring generation against gold context instead",
    )


def _adapter_has_corpus(adapter: Any) -> bool:
    """Whether retrieval can be measured on this benchmark at all.

    Two ways to qualify: the adapter ships a real corpus (it overrides
    ``load_corpus``), or its samples carry inline gold context that pools into
    one (``pools_gold_context``). The second takes eight benchmarks from
    generation-only to something a retriever can be measured on -- at the cost
    of an easier task, which ``corpus_provenance`` records and the comparability
    key enforces.

    Checked by comparing against the base implementation rather than by calling
    it: calling downloads the corpus, and mode resolution has to be cheap enough
    to run while planning a sweep.
    """
    if type(adapter).load_corpus is not _BASE_LOAD_CORPUS:
        return True
    return bool(getattr(adapter, "pools_gold_context", False))


# Module-level judge model set by Experiment.run() for _eval_offline to read.
# Resolved order: ExperimentConfig.judge_model → env RAG_EVAL_JUDGE_MODEL → None (RAGAS default).
_active_judge_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Judge and generator LLMs are resolved SEPARATELY.
#
# These are two different jobs. The judge scores answers (RAGAS, via an
# OpenAI-compatible endpoint); the generator writes them (whatever platform the
# package is configured for). They previously shared one model string, which
# could not work: a judge id valid for the scoring endpoint is generally not a
# valid model for the generator's platform, so the generator silently failed to
# build, produced an empty answer, and faithfulness scored 0.0 with no error.
#
# They also must not share *credentials*. Provider auto-detection in
# ``providers/__init__.py`` scans a priority-ordered table of ``*_API_KEY``
# variables, so pointing the judge at an OpenAI-compatible proxy by exporting
# ``OPENAI_API_KEY`` would also re-route the generator to that proxy. The judge
# therefore reads its own variables first and only falls back to ``OPENAI_*``.
# ---------------------------------------------------------------------------


def _resolve_generator_model() -> Optional[str]:
    """Model the *generator* (answer LLM) should use.

    Order: ``RAG_EVAL_GENERATOR_MODEL`` → ``MODEL_NAME`` → None, where None
    means "let the configured platform pick its default". Deliberately ignores
    the judge model.
    """
    return os.environ.get("RAG_EVAL_GENERATOR_MODEL") or os.environ.get("MODEL_NAME") or None


def _build_judge_llm(model_name: Optional[str] = None):
    """Build the RAGAS judge LLM against an OpenAI-compatible endpoint.

    Credentials resolve judge-first so they cannot disturb the generator's
    platform detection:

    ==========  ==========================================================
    api key     ``RAG_EVAL_JUDGE_API_KEY`` → ``OPENAI_API_KEY``
    base url    ``RAG_EVAL_JUDGE_BASE_URL`` → ``OPENAI_BASE_URL``
    model       argument → ``_active_judge_model`` → ``RAG_EVAL_JUDGE_MODEL``
    ==========  ==========================================================

    Returns None when no judge model is configured, or when construction
    fails — callers treat that as "fall back to the RAGAS default judge".
    """
    resolved_model = model_name or _active_judge_model or os.environ.get("RAG_EVAL_JUDGE_MODEL")
    if not resolved_model:
        return None

    try:
        from ragas.llms import llm_factory
        from openai import OpenAI

        client_kwargs = {}
        api_key = os.environ.get("RAG_EVAL_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("RAG_EVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        # Counted under `eval`, separately from the system under test. RAGAS
        # builds its judge on a raw OpenAI client rather than through the
        # LangChain provider, so without this wrapper the entire cost of
        # *measuring* -- often larger than the cost of the pipeline -- was
        # invisible, and `eval_cost_usd` had no producer at all.
        from lasagna.utils.token_usage import EVAL, instrument_openai_client

        judge = llm_factory(
            resolved_model,
            client=instrument_openai_client(OpenAI(**client_kwargs), EVAL),
        )
        logger.info(f"RAGAS judge using model: {resolved_model}")
        return judge
    except Exception as exc:
        logger.warning(
            f"Could not create judge LLM ({resolved_model}) [{type(exc).__name__}]: {exc}"
        )
        logger.exception("Judge LLM construction failed; RAGAS will use its default")
        return None


async def _default_eval_fn(
    task: BenchmarkTask,
    samples: List[BenchmarkSample],
) -> TaskResult:
    """Default evaluation function using the knowledge pipeline.

    For offline mode: uses existing evaluator with gold labels.
    For live mode: ingests corpus, searches, computes IR metrics.
    """
    if task.mode not in _VALID_MODES:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=(f"Unknown mode '{task.mode}'. Valid modes: {sorted(_VALID_MODES)}"),
        )

    resolved, reason = resolve_mode(task.mode, task.benchmark_name)
    # Announced before any work: a run whose mode was chosen for it must say so
    # while there is still time to disagree, not in a footnote afterwards.
    logger.info(f"[{task.benchmark_name}] mode={resolved} ({reason})")
    task.resolved_mode = resolved
    task.mode_resolution_reason = reason

    if resolved == "generation_only":
        return await _eval_offline(task, samples)
    return await _eval_live(task, samples)


# Component override keys that should use pipeline.replace() rather than
# being set as config dict entries.
_COMPONENT_KEYS = frozenset(
    {
        "reranker",
        "embedder",
        "chunker",
        "retriever",
        "document_processor",
        "query_enhancer",
        "agentic_pipeline",
    }
)

#: Component slots the harness cannot swap: ingestion always runs the built-in
#: chunker and document processor, so an arm naming one measured the default.
_UNSWEEPABLE_COMPONENTS = frozenset({"chunker", "document_processor"})

# Defined in `runner`, beside `BenchmarkTask.config_hash`, which must exclude
# the same keys: a setting that steers reporting rather than the pipeline must
# not change the session id, the checkpoint key or the comparability of a run.
_HARNESS_KEYS = HARNESS_KEYS


def _apply_overrides_to_pipeline(
    pipeline: Any,
    overrides: Dict[str, Any],
    session_id: str,
) -> None:
    """Apply config overrides to a KnowledgePipeline.

    - Component names (reranker, embedder, etc.) → ``pipeline.replace()``
    - ``ensemble_reranker_models`` → create an EnsembleReranker instance
      and inject it into pipeline._instances (session-scoped, not per-query)
    - Config keys (enable_hybrid_search, chunk_size, etc.) → dict update
      on ``pipeline._config``, then ``_apply_session_config(session_id)``
    """
    # Defensive copy — from_profile() passes profile.config by reference
    # into pipeline._config.  Without this copy, .update() below would
    # mutate the config-manager singleton's stored profile dict, leaking
    # overrides into every subsequent task that loads the same profile.
    pipeline._config = dict(pipeline._config)

    # Handle ensemble reranker: create once at task init (not per-query).
    # Intercept before normal processing so these keys don't leak into config dict.
    if "ensemble_reranker_models" in overrides:
        overrides = dict(overrides)  # defensive copy
        models = overrides.pop("ensemble_reranker_models")
        weights = overrides.pop("ensemble_reranker_weights", [1.0] * len(models))
        if len(models) != len(weights):
            logger.warning(
                f"ensemble_reranker_models/weights length mismatch: "
                f"{len(models)} vs {len(weights)} — using equal weights"
            )
            weights = [1.0] * len(models)
        try:
            from ..retrieval.search.ranker import EnsembleReranker, RerankerModelConfig

            model_configs = [
                RerankerModelConfig(model_name=m, weight=w) for m, w in zip(models, weights)
            ]
            from ..api.rag_backend import get_rag_backend as _get_backend

            _backend = _get_backend()
            ensemble = EnsembleReranker(
                model_configs=model_configs,
                embedding_manager=_backend.embedding_manager,
            )
            # Inject directly into the pipeline instance cache — get_component()
            # checks _instances first, bypassing registry lookup.
            pipeline._instances["reranker"] = ensemble
            logger.info(
                f"Injected EnsembleReranker with {len(ensemble.rerankers)} models: "
                f"{ensemble.model_name}"
            )
        except Exception as exc:
            logger.warning(f"Failed to create EnsembleReranker: {exc}")

    config_overrides: Dict[str, Any] = {}

    for key, value in overrides.items():
        if key in _HARNESS_KEYS:
            continue  # steers the harness, never a pipeline setting
        if key in _COMPONENT_KEYS:
            if value is not None:
                try:
                    pipeline.replace(key, value)
                except (ValueError, KeyError) as exc:
                    # This used to log one warning and run the profile's own
                    # component, reporting its number under this arm's name.
                    raise ValueError(f"Cannot use {key}={value!r}: {exc}") from exc
                if key == "reranker" and "enable_reranking" not in overrides:
                    # Naming a reranker means measuring it: under a profile
                    # with reranking off (speed) it was never called.
                    config_overrides["enable_reranking"] = True
                if key == "agentic_pipeline" and "use_agentic_rag" not in overrides:
                    config_overrides["use_agentic_rag"] = True
        else:
            config_overrides[key] = value

    # Merge into pipeline's (now-copied) config dict
    if config_overrides:
        pipeline._config.update(config_overrides)

    # Push config to session-scoped overrides so SearchOrchestrator sees them.
    # NOTE: pipeline._apply_session_config writes to KnowledgeConfigManager
    # which is *disconnected* from InMemorySessionProvider — the store that
    # SearchOrchestrator actually reads via rag_config.with_session().
    # We must also write directly to InMemorySessionProvider.
    pipeline._apply_session_config(session_id)
    try:
        from ..providers import get_session_provider

        sp = get_session_provider()
        sp.set_config_overrides(session_id, dict(pipeline._config))
    except Exception as exc:
        logger.warning(f"Failed to write session config overrides: {exc}")


async def _eval_offline(
    task: BenchmarkTask,
    samples: List[BenchmarkSample],
) -> TaskResult:
    """Offline evaluation — score against gold labels without vector DB.

    Uses gold context and ground_truth from the dataset directly with RAGAS.
    No retrieval from vector DB is needed — this measures answer quality
    independent of retrieval performance.

    For retrieval-only benchmarks with expected_docs: offline mode is N/A
    (need live mode for IR metrics from real retrieval). Returns a warning.
    """
    start = time.monotonic()

    # Warn when pipeline config overrides are non-trivial but will be ignored.
    # Offline mode bypasses the RAG pipeline entirely (no retrieval, no agentic
    # graph) — it only runs LLM answer generation from gold context + RAGAS scoring.
    # Retrieval-dependent overrides (use_agentic_rag, max_agent_iterations,
    # enable_hybrid_search, enable_reranking, etc.) have NO effect here.
    _RETRIEVAL_OVERRIDES = {
        "use_agentic_rag",
        "max_agent_iterations",
        "chunk_refinement_method",
        "enable_hybrid_search",
        "enable_reranking",
        "enable_mmr",
        "enable_hyde",
        "enable_rag_fusion",
        "enable_query_expansion",
        "enable_query_routing",
        "coverage_scorer_mode",
        "use_structured_graph",
        "enable_hallucination_check",
        "enable_answer_quality_check",
    }
    # Component swaps (`reranker=`, `chunker=`, …) are the whole reason someone
    # runs this harness, and this mode cannot measure them: it scores gold
    # context and never constructs the pipeline. Returning a number here is the
    # worst outcome available -- it is plausible, it is precise, and it has
    # nothing to do with the component under test. So it is an error, not a log
    # line that scrolls past.
    #
    # `_RETRIEVAL_OVERRIDES` alone was not enough: it lists 14 feature flags and
    # omitted every component key, which is exactly the set a researcher passes.
    inert_keys = (_RETRIEVAL_OVERRIDES | _COMPONENT_KEYS) & set(task.config_overrides.keys())
    if inert_keys and not task.config_overrides.get("allow_inert_overrides"):
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=(
                f"Config overrides {sorted(inert_keys)} have no effect in "
                f"'{task.mode}' mode, which scores gold context and never runs the "
                f"retrieval pipeline — any metric produced here would be unrelated "
                f"to them. Use mode='live' (or mode='auto') to measure these. "
                f"Set allow_inert_overrides=True to score gold context anyway."
            ),
        )
    if inert_keys:
        logger.warning(
            f"[{task.mode}] Overrides {sorted(inert_keys)} are inert in this mode "
            f"and allow_inert_overrides=True was set. The resulting metrics do not "
            f"measure them. Task: {task.task_key}"
        )

    # Check if this is a retrieval-only benchmark (expected_docs but no ground_truth)
    has_expected_docs = any(s.expected_docs for s in samples)
    has_ground_truth = any(s.ground_truth for s in samples)

    if has_expected_docs and not has_ground_truth:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=(
                f"Benchmark '{task.benchmark_name}' has expected_docs but no "
                f"ground_truth answers. Use mode='live' for IR metric evaluation."
            ),
        )

    # Filter to samples with both ground_truth and context (required for
    # RAGAS metrics like faithfulness, context_recall, context_precision).
    # Samples without gold context need live-mode retrieval.
    eval_samples = [s for s in samples if s.ground_truth and s.context]
    if not eval_samples:
        # Fall back to samples with just ground_truth + factual_correctness
        eval_samples = [s for s in samples if s.ground_truth]
    if not eval_samples:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=f"No samples with ground_truth in '{task.benchmark_name}' for offline eval",
        )

    skipped = len(samples) - len(eval_samples)
    if skipped:
        logger.info(
            f"Offline eval: using {len(eval_samples)}/{len(samples)} samples "
            f"(skipped {skipped} without gold context)"
        )

    # Build RAGAS judge LLM
    judge_llm = _build_judge_llm()

    # Build the answer LLM (generator). Resolved independently of the judge —
    # see the note beside _resolve_generator_model.
    answer_llm = None
    try:
        from ..providers import get_llm_provider

        llm_provider = get_llm_provider()
        if llm_provider:
            answer_llm = llm_provider.get_llm(model_name=_resolve_generator_model())
    except Exception as exc:
        logger.warning(f"Could not create answer LLM [{type(exc).__name__}]: {exc}")
        logger.exception("Answer LLM construction failed; answers will be empty")

    # Generate answers from gold context using the LLM, then score with RAGAS
    questions: List[str] = []
    answers: List[str] = []
    _offline_failed_generations = 0
    detailed = None
    contexts: List[List[str]] = []
    ground_truths: List[str] = []
    per_sample_latencies: List[float] = []
    per_sample_usage: List[TokenUsage] = []

    for sample in eval_samples:
        questions.append(sample.question)
        ground_truths.append(sample.ground_truth)

        # Use gold context if available
        ctx = [sample.context] if sample.context else []
        contexts.append(ctx)

        # Generate answer from context using LLM
        sample_meter = UsageMeter()
        if answer_llm and ctx:
            q_start = time.monotonic()
            try:
                context_str = "\n\n".join(ctx)
                prompt = (
                    f"Answer the following question based on the given context.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {sample.question}\n\n"
                    f"Answer:"
                )
                with sample_meter, sample_meter.scope(SYSTEM):
                    response = await asyncio.to_thread(
                        answer_llm.invoke,
                        prompt,
                    )
                answer = response.content if hasattr(response, "content") else str(response)
                per_sample_latencies.append((time.monotonic() - q_start) * 1000)
            except Exception as exc:
                # The gold context used to stand in as the answer here, which a
                # judge then scored as perfectly faithful. No answer is a failure.
                logger.warning(f"Answer generation failed: {exc}")
                answer = ""
                _offline_failed_generations += 1
                per_sample_latencies.append(0.0)
        else:
            # No generator configured, or no context: nothing was generated.
            answer = ""
            _offline_failed_generations += 1
            per_sample_latencies.append(0.0)

        answers.append(answer)
        per_sample_usage.append(sample_meter.usage(SYSTEM))

    logger.info(
        f"Offline eval: {len(eval_samples)} samples, "
        f"{sum(1 for c in contexts if c)} with context, "
        f"{'LLM' if answer_llm else 'passthrough'} answers"
    )

    # Score with RAGAS
    metrics: Dict[str, float] = {}
    per_sample_scores: List[Dict[str, float]] = []
    eval_meter = UsageMeter()

    try:
        from .ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(llm=judge_llm)

        # Select metrics based on available data
        has_context = any(c for c in contexts)
        if has_context:
            eval_metrics = ["faithfulness", "context_recall", "context_precision"]
        else:
            eval_metrics = ["factual_correctness"]

        # The judge's own spend, kept separate from the system's. It is often
        # the larger of the two, and reporting one total lets an expensive
        # judge masquerade as an expensive pipeline.
        with eval_meter, eval_meter.scope(EVAL):
            detailed = evaluator.evaluate_detailed(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
                metrics=eval_metrics,
            )
        metrics = detailed.aggregate
        per_sample_scores = detailed.per_sample

        # Add latency, tokens, EM/F1, and trajectory to per-sample scores
        for i, latency in enumerate(per_sample_latencies):
            if i >= len(per_sample_scores):
                per_sample_scores.append({})
            per_sample_scores[i]["latency_ms"] = latency
            if i < len(per_sample_usage):
                per_sample_scores[i].update(per_sample_usage[i].to_dict())

        # EM/F1 + trajectory for offline path (mirrors live path)
        from .answer_metrics import exact_match as _em_fn, token_f1 as _tf1_fn

        _save_traj = (task.config_overrides or {}).get("save_trajectories", "minimal")
        for i, sample in enumerate(eval_samples):
            if i >= len(per_sample_scores):
                per_sample_scores.append({})
            ps = per_sample_scores[i]

            # Sample metadata
            ps["_sample_id"] = getattr(sample, "query_id", "")
            _meta = getattr(sample, "metadata", {}) or {}
            ps["_source_benchmark"] = _meta.get("source_benchmark", "")

            # EM/F1
            _gt = ground_truths[i] if i < len(ground_truths) else ""
            _ans = answers[i] if i < len(answers) else ""
            _aliases = getattr(sample, "ground_truth_answers", None) or []
            if _gt.strip() and _ans.strip():
                ps["exact_match"] = _em_fn(_ans, _gt, aliases=_aliases)
                ps["token_f1"] = _tf1_fn(_ans, _gt, aliases=_aliases)
            elif not _gt.strip() and (not _ans.strip() or _is_abstention_answer(_ans)):
                ps["exact_match"] = 1.0
                ps["token_f1"] = 1.0
            else:
                ps["exact_match"] = 0.0
                ps["token_f1"] = 0.0

            # Trajectory
            if _save_traj == "full":
                ps["_generated_answer"] = _ans[:2000]
                ps["_retrieved_contexts"] = [
                    c[:500] for c in (contexts[i] if i < len(contexts) else [])[:5]
                ]
                ps["_ground_truth"] = _gt[:500]
                ps["_question"] = questions[i][:500] if i < len(questions) else ""
            elif _save_traj == "minimal":
                ps["_generated_answer"] = _ans[:500]
                ps["_question"] = questions[i][:200] if i < len(questions) else ""

        # Aggregate EM/F1
        _em_scores = [s.get("exact_match") for s in per_sample_scores if "exact_match" in s]
        if _em_scores:
            metrics["exact_match"] = sum(_em_scores) / len(_em_scores)
            _f1_scores = [s.get("token_f1", 0.0) for s in per_sample_scores if "token_f1" in s]
            metrics["token_f1"] = sum(_f1_scores) / len(_f1_scores)

    except ImportError:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error="RAGAS not installed (pip install lasagna-rag[eval])",
        )
    except Exception as exc:
        # Include the exception type: several RAGAS failures stringify to an
        # empty message, which produced the uninformative "RAGAS evaluation
        # failed: " with nothing after the colon.
        logger.error(f"RAGAS evaluation failed [{type(exc).__name__}]: {exc}")
        logger.exception("RAGAS evaluation failed")
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=f"RAGAS evaluation failed: {exc}",
        )

    result = TaskResult(
        task_key=task.task_key,
        benchmark_name=task.benchmark_name,
        config_name=task.config_name,
        run_index=task.run_index,
        metrics=metrics,
        per_sample_scores=per_sample_scores,
        duration_seconds=time.monotonic() - start,
        n_samples=len(eval_samples),
    )
    _offline_failures: List[str] = []
    if getattr(detailed, "failed_metrics", None):
        _offline_failures.append(
            "judge produced no usable scores for " + ", ".join(detailed.failed_metrics)
        )
    if eval_samples:
        result.metrics["generations_failed"] = float(_offline_failed_generations)
        if _offline_failed_generations / len(eval_samples) > _MAX_FAILED_GENERATION_RATE:
            _offline_failures.append(
                f"{_offline_failed_generations} of {len(eval_samples)} answers could not be "
                f"generated (failed, or no generator configured)"
            )
    result.error = "; ".join(_offline_failures) or None
    result.compute_latency_stats()
    _record_eval_usage(result, eval_meter)
    return result


# ---------------------------------------------------------------------------
# Shared corpus cache for live-mode evaluation.
# Ingest once per benchmark, reuse for all config × run sweep tasks.
# ---------------------------------------------------------------------------
_corpus_cache_lock = asyncio.Lock()
# {benchmark_name: {"session_id": str, "corpus_to_valid": dict, "valid_to_corpus": dict,
#                    "ref_count": int, "ready": asyncio.Event}}
_corpus_sessions: Dict[str, Dict[str, Any]] = {}

# Threading lock to serialize pipeline.query() calls.  The query() method
# writes profile config to the session (write) and then immediately reads
# it back during search (read).  Without serialization, concurrent tasks
# sharing the same corpus session overwrite each other's config, producing
# identical metrics regardless of profile.
import threading as _threading  # noqa: E402

_search_config_lock = _threading.Lock()


def _corpus_cache_key(task: "BenchmarkTask") -> Optional[str]:
    """Compute the corpus cache key for a task.

    Returns None when the task uses the default shared corpus (no per-config
    ingestion and no corpus cap).  Returns an explicit key otherwise.

    Uses an **ingestion signature** that hashes only ingestion-time params
    (embedding model, chunking, hierarchical, etc.) + corpus identity.
    Query-time params (hybrid alpha, reranking, agentic, etc.) are excluded
    so configs differing only at query time correctly share the same corpus.

    This is the single source of truth for cache-key construction — used by
    _pre_register_corpus_refs, _eval_live, and _release_corpus.
    """
    # Build ingestion signature from corpus_id + ingestion-time params only.
    # This includes both profile defaults AND explicit overrides.
    corpus_id = _CORPUS_ID_MAP.get(task.benchmark_name, task.benchmark_name)

    # Resolve profile ingestion defaults (embedding model, chunking, etc.)
    _profile_ingestion: Dict[str, Any] = {}
    if task.profile:
        try:
            from ..core.session_config._profiles import resolve_ingestion_defaults

            _all_profile = resolve_ingestion_defaults(task.profile)
            _profile_ingestion = {
                k: v for k, v in _all_profile.items() if k in _INGESTION_TIME_KEYS
            }
        except Exception:
            pass  # fall back to empty

    # Explicit ingestion overrides take precedence over profile defaults
    _explicit_ingestion = {
        k: v for k, v in (task.config_overrides or {}).items() if k in _INGESTION_TIME_KEYS
    }
    merged = {**_profile_ingestion, **_explicit_ingestion}

    # A pooled-gold corpus IS the sample set -- it is built from those samples'
    # own context -- so two tasks drawing different samples are looking at
    # different corpora and must never share a session. Everywhere else the
    # corpus is a fixed artefact the sampling cannot affect, which is why this
    # is conditional rather than always on: adding the sampling hash
    # unconditionally would re-ingest BEIR once per (split, n_samples, seed).
    _pooled = False
    try:
        _adapter = benchmark_registry.get(task.benchmark_name)
        _pooled = bool(getattr(_adapter, "pools_gold_context", False)) and (
            type(_adapter).load_corpus is _BASE_LOAD_CORPUS
        )
    except Exception:  # noqa: BLE001 - an unknown benchmark fails later, properly
        pass

    # If no ingestion-time params differ from default AND no corpus cap,
    # use the shared corpus key.
    if not merged and task.max_corpus_docs <= 0 and not _pooled:
        return None  # uses default shared corpus

    import hashlib as _hashlib

    parts = [
        corpus_id,
        getattr(task, "corpus_fingerprint", "") or "",
        str(task.max_corpus_docs) if task.max_corpus_docs > 0 else "",
        json.dumps(merged, sort_keys=True),
    ]
    if _pooled:
        # Appended only for a pooled corpus, so a native key is byte-identical
        # to what it was before this existed.
        #
        # An earlier version appended `"" if not pooled`, which still adds a
        # separator to the joined string -- every native key changed, and the
        # persistent vector-DB session id derives from it, so every previously
        # ingested corpus was orphaned and silently re-ingested.
        #
        # `sampling_hash` covers split, n_samples, seed and filters. With
        # `variance_source="sample"` the seed is offset per run *after* this is
        # computed, so two runs of one task still share a key -- which is
        # correct for a native corpus and wrong for a pooled one. The run index
        # closes that, and is inert when the sample set is fixed across runs.
        parts.append(task.sampling_hash)
        if task.variance_source == "sample":
            parts.append(f"run{task.run_index}")
    sig = _hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"{corpus_id}|{sig}"


async def _pre_register_corpus_refs(tasks: List["BenchmarkTask"]) -> None:
    """Pre-register corpus session slots with the total consumer count.

    With max_concurrency=1, tasks run sequentially: without pre-registration
    the first task releases the corpus (ref 1→0 → deletes session data) before
    the second task can use it, causing re-ingestion to fail on missing temp files.

    By setting ref_count = number of tasks that share a corpus upfront,
    _release_corpus will only clean up after ALL tasks have finished.
    """
    from collections import Counter

    counts: Counter = Counter()
    for task in tasks:
        explicit = _corpus_cache_key(task)
        ck = (
            explicit
            if explicit is not None
            else _SHARED_CORPUS_KEY.get(task.benchmark_name, task.benchmark_name)
        )
        counts[ck] += 1

    async with _corpus_cache_lock:
        for ck, count in counts.items():
            if ck not in _corpus_sessions:
                _corpus_sessions[ck] = {
                    "session_id": None,
                    "corpus_to_valid": {},
                    "valid_to_corpus": {},
                    "ref_count": count,
                    "ready": asyncio.Event(),
                    "_pre_registered": True,
                    "_ingester_claimed": False,
                }
                logger.debug(f"Pre-registered corpus '{ck}' for {count} task(s)")
            else:
                # Key already exists (another experiment pre-registered it).
                # ADD our task count so the corpus stays alive until all
                # experiments' tasks complete (not just the first registrant's).
                _corpus_sessions[ck]["ref_count"] += count
                logger.debug(
                    f"Added {count} ref(s) to existing corpus '{ck}' "
                    f"(total now {_corpus_sessions[ck]['ref_count']})"
                )


def _mint_pooled_qrels(adapter: Any, samples: List[BenchmarkSample], provenance: str) -> None:
    """Give pooled-corpus samples the labels a pooled corpus cannot ship.

    Called by *every* task on a pooled corpus, not only the one that ingests it.
    It lived inside the ingester branch, so the first task got labels and every
    other task on the same shared session got samples with empty
    ``expected_docs`` -- and an empty qrel set scores 0.0 on every IR metric
    without erroring, so a sweep reported one arm working and the rest at zero.

    Idempotent: a sample that already has labels keeps them.
    """
    if provenance != "pooled_gold":
        return
    try:
        pooled = adapter.pooled_gold_expected_docs(samples)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not mint pooled-gold qrels: {exc}")
        return
    for sample in samples:
        if not sample.expected_docs and sample.query_id in pooled:
            sample.expected_docs = pooled[sample.query_id]


async def _ensure_corpus_ingested(
    benchmark_name: str,
    samples: List[BenchmarkSample],
    corpus_key: Optional[str] = None,
    ingestion_overrides: Optional[Dict[str, Any]] = None,
    max_corpus_docs: int = 0,
) -> Optional[Dict[str, Any]]:
    """Ingest corpus once and return shared session info.

    First caller ingests; subsequent callers wait on an asyncio.Event.
    Returns dict with session_id, corpus_to_valid, valid_to_corpus,
    or None if corpus unavailable.

    Args:
        benchmark_name: Benchmark name for loading the adapter and corpus.
        samples: Samples for this task (used to pre-check qrels coverage).
        corpus_key: Cache key override. When provided, multiple benchmark names
            can share one indexed session (e.g. private_eval/retrieval and
            private_eval/generation share the same corpus.jsonl). When None, uses
            the _SHARED_CORPUS_KEY map, then falls back to benchmark_name.
        ingestion_overrides: Config overrides applied to the session before
            document upload (for per-config ingestion of docling/chunking sweeps).
    """
    import hashlib as _hashlib

    # Resolve cache key — shared corpus map takes priority, then explicit override
    if corpus_key is not None:
        cache_key = corpus_key
    else:
        cache_key = _SHARED_CORPUS_KEY.get(benchmark_name, benchmark_name)

    async with _corpus_cache_lock:
        if cache_key in _corpus_sessions:
            entry = _corpus_sessions[cache_key]
            # Pre-registered slots already hold the correct total ref count
            # for all tasks; do NOT double-count. Non-pre-registered slots
            # must increment for each new caller so _release_corpus knows
            # when all tasks are done.
            if not entry.get("_pre_registered"):
                entry["ref_count"] += 1
            ready_event = entry["ready"]
            if ready_event.is_set():
                logger.info(
                    f"Reusing pre-ingested corpus for '{benchmark_name}' "
                    f"(cache_key='{cache_key}', session={entry['session_id']}, "
                    f"ref_count={entry['ref_count']})"
                )
                return entry
            elif not entry.get("_ingester_claimed", True):
                # Pre-registered slot — no ingester claimed yet.
                # This task becomes the ingester.
                entry["_ingester_claimed"] = True
                logger.info(
                    f"Claiming ingester for pre-registered corpus '{cache_key}' "
                    f"(ref_count={entry['ref_count']})"
                )
                entry = None  # signal: we are the ingester
            else:
                # Another task is ingesting — wait outside the lock.
                logger.info(
                    f"Waiting for corpus ingestion by another task "
                    f"(benchmark={benchmark_name}, cache_key='{cache_key}', "
                    f"ref_count={entry['ref_count']})"
                )
        else:
            # First caller — claim the slot (no pre-registration path).
            _corpus_sessions[cache_key] = {
                "session_id": None,
                "corpus_to_valid": {},
                "valid_to_corpus": {},
                "ref_count": 1,
                "ready": asyncio.Event(),
                "_ingester_claimed": True,
            }
            entry = None  # signal: we are the ingester

    # If we are a waiter, block until the ingester finishes.
    if entry is not None:
        await entry["ready"].wait()
        # Read the slot we already hold, not the dict.
        #
        # A failed ingest removes its slot, so `_corpus_sessions[cache_key]`
        # would raise KeyError here -- turning one benchmark's failure into an
        # unrelated crash in every sibling task. `entry` is the same object the
        # ingester sets `ready` and `error` on.
        if entry.get("error"):
            return {"error": entry["error"]}
        return entry

    # --- We are the ingester ---
    try:
        adapter = benchmark_registry.get(benchmark_name)
        # Always pass it. Omitting the kwarg when 0 let the adapter's own
        # signature default (100_000) apply, so `max_corpus_docs: 0` --
        # documented as "no limit" in two places -- silently capped at
        # 100,000 and truncated 9 of the 13 BEIR corpora. The adapters treat
        # 0 as unlimited (`if max_docs and ...`).
        load_kwargs = {"max_docs": max(0, int(max_corpus_docs or 0))}
        if load_kwargs["max_docs"] > 0:
            logger.info(f"Corpus size capped at {load_kwargs['max_docs']} docs")
        else:
            logger.info(
                "No corpus cap (max_corpus_docs=0). Large benchmarks ingest in "
                "full -- beir/msmarco is 8.8M documents, which is hours of "
                "embedding and tens of GB of index. Set max_corpus_docs to cap it."
            )
        corpus = adapter.load_corpus(**load_kwargs)
        corpus_provenance = "native"
        if corpus is None and getattr(adapter, "pools_gold_context", False):
            # No corpus shipped, but the samples carry their own gold context.
            # Pooled across *all* the samples in this task, so each query has to
            # find its passages among every other query's -- pooling per query
            # would score 1.0 by construction.
            corpus = adapter.build_pooled_gold_corpus(samples)
            corpus_provenance = "pooled_gold"
            if corpus:
                logger.warning(
                    f"'{benchmark_name}' ships no corpus; retrieval will be measured "
                    f"against {len(corpus)} passages pooled from the samples' own gold "
                    f"context. This is an easier task than the published benchmark -- "
                    f"the pool holds no hard negatives -- and these numbers are not "
                    f"comparable with a native-corpus run."
                )
        if corpus is None:
            # Wake the waiters before the slot disappears.
            #
            # Every task after the first parks on `slot["ready"]`, an Event
            # object it captured by reference. Deleting the slot and returning
            # without setting it leaves them awaiting an Event nobody will ever
            # set -- a permanent hang, not a failure. The success and exception
            # paths below both call `set()` for exactly this reason.
            async with _corpus_cache_lock:
                _slot = _corpus_sessions.pop(cache_key, None)
            if _slot is not None:
                _slot["error"] = f"No corpus available for '{benchmark_name}'"
                _slot["ready"].set()
            return None

        # Compute corpus fingerprint (stable hash of sorted doc IDs + sizes)
        _fp_parts = sorted(f"{k}:{len(v)}" for k, v in corpus.items())
        _corpus_fp = _hashlib.sha256("|".join(_fp_parts).encode()).hexdigest()[:16]

        # Build doc ID mappings
        corpus_to_valid: Dict[str, str] = {}
        valid_to_corpus: Dict[str, str] = {}
        for corpus_id in corpus:
            hex_hash = _hashlib.md5(corpus_id.encode()).hexdigest()[:16]
            valid_id = f"doc_{hex_hash}"
            corpus_to_valid[corpus_id] = valid_id
            valid_to_corpus[valid_id] = corpus_id

        _mint_pooled_qrels(adapter, samples, corpus_provenance)

        # Pre-check missing qrels docs
        total_qrel_docs = set()
        for sample in samples:
            total_qrel_docs.update(sample.expected_docs.keys())
        missing_qrel_docs = total_qrel_docs - set(corpus.keys())
        if missing_qrel_docs:
            pct = len(missing_qrel_docs) / len(total_qrel_docs) * 100 if total_qrel_docs else 0
            logger.warning(
                f"Benchmark '{benchmark_name}': {len(missing_qrel_docs)} of "
                f"{len(total_qrel_docs)} qrels docs ({pct:.1f}%) are missing from "
                f"the loaded corpus (likely truncated by max_docs). "
                f"IR metrics will be deflated."
            )
            if pct > 50:
                _msg = (
                    f"Aborting: {pct:.0f}% of qrels docs missing from loaded "
                    f"corpus — results would be unreliable. Raise "
                    f"`max_corpus_docs` (0 = no limit) so the judged documents "
                    f"are actually ingested."
                )
                # Wake the waiters. Without this every sibling task on this
                # corpus hangs forever on an Event nobody sets -- so one
                # benchmark aborting took down a whole sweep with no output.
                async with _corpus_cache_lock:
                    _slot = _corpus_sessions.pop(cache_key, None)
                if _slot is not None:
                    _slot["error"] = _msg
                    _slot["ready"].set()
                return {"error": _msg}

        # Session ID derived from cache key (not benchmark_name) so shared-corpus
        # variants always resolve to the same vector DB session.
        corpus_hash = _hashlib.md5(cache_key.encode()).hexdigest()[:12]
        shared_session_id = f"bench_corpus_{corpus_hash}"

        # Apply ingestion-time overrides (docling, chunking, embedding) BEFORE
        # uploading so the document pipeline uses the correct settings.
        if ingestion_overrides:
            try:
                from ..providers import get_session_provider

                get_session_provider().set_config_overrides(shared_session_id, ingestion_overrides)
                logger.info(
                    f"Applied ingestion overrides for session {shared_session_id}: "
                    f"{list(ingestion_overrides.keys())}"
                )
            except Exception as exc:
                logger.warning(f"Could not apply ingestion overrides: {exc}")

            # Ensure per-model Milvus collection exists before ingestion.
            # The default collection (rag_documents_v2) is created at backend
            # startup, but per-embedding-model collections are not.
            if "collection_name" in ingestion_overrides:
                try:
                    from ..api.rag_backend import get_rag_backend as _get_backend
                    from ..core.config import rag_config

                    _backend = _get_backend()
                    _session_cfg = rag_config.with_session(shared_session_id)
                    _backend._collection_manager.ensure_collection_for_config(_session_cfg)
                    logger.info(
                        f"Ensured collection '{_session_cfg.collection_name}' exists "
                        f"(dim={_session_cfg.embedding_dim})"
                    )
                except Exception as exc:
                    logger.warning(f"Could not ensure collection for config: {exc}")

        # Ingest — sequential upload (MilvusLite embedded DB doesn't support
        # concurrent writes; concurrent uploads cause deadlocks and data loss).

        logger.info(
            f"Ingesting corpus for '{benchmark_name}' (cache_key='{cache_key}'): "
            f"{len(corpus)} docs into session {shared_session_id}"
        )
        # Fast-path ingestion: bypass client.upload() and use direct
        # chunk → embed → insert pipeline for eval corpus.
        from ..api.rag_backend import get_rag_backend as _get_ingest_backend
        from ..core.config import rag_config as _rag_cfg
        from ..vector_backends.filters import FilterExpression as _FE
        from ._fast_ingest import fast_ingest_corpus as _fast_ingest

        _ingest_backend = _get_ingest_backend()
        _resolved = _ingest_backend._collection_manager.resolve_backend(shared_session_id)
        _coll_name = (
            _rag_cfg.collection_name
            if not ingestion_overrides
            else (ingestion_overrides.get("collection_name", _rag_cfg.collection_name))
        )

        # Pre-flight: check if corpus already fully indexed via vector store
        expected_valid_ids = {corpus_to_valid[cid] for cid in corpus}
        ready_ids: set = set()
        _skip_ingest = False
        try:
            # limit must exceed corpus size to avoid false "partial" detection.
            # It was min(max(n + 1000, 50_000), 16_000), which is 16,000 for
            # every n, so any corpus above 16K documents looked partial and was
            # deleted and re-ingested on every run. list_document_ids pages
            # with a cursor, so asking for the true size is safe.
            existing_doc_ids = _present_document_ids(
                _resolved, _coll_name, shared_session_id, expected_valid_ids
            )
            missing_ids = expected_valid_ids - existing_doc_ids

            if not missing_ids:
                logger.info(
                    f"Corpus already fully indexed ({len(existing_doc_ids)} docs), "
                    f"reusing session {shared_session_id}"
                )
                ready_ids = existing_doc_ids
                _skip_ingest = True
            elif existing_doc_ids:
                # Load allowlist for known errors from previous runs
                _allowlist_path = (
                    Path.home() / ".lasagna" / "research" / "results" / "_ingest_allowlist.json"
                )
                _known_errors: set = set()
                if _allowlist_path.exists():
                    try:
                        _al = json.loads(_allowlist_path.read_text())
                        _known_errors = set(_al.get(shared_session_id, []))
                    except Exception:
                        pass

                missing_non_error = missing_ids - _known_errors
                if not missing_non_error:
                    logger.info(
                        f"Corpus indexed with {len(_known_errors)} known errors, "
                        f"reusing session {shared_session_id}"
                    )
                    ready_ids = existing_doc_ids
                    _skip_ingest = True  # all missing are allowlisted errors
                else:
                    # Stale/partial — clean up and re-ingest
                    logger.warning(
                        f"Stale partial corpus ({len(existing_doc_ids)}/{len(expected_valid_ids)} docs, "
                        f"{len(missing_non_error)} missing). Cleaning up and re-ingesting."
                    )
                    try:
                        if _resolved.backend_name == "milvus":
                            _filt = _FE.json_eq("metadata", "session_id", shared_session_id)
                        else:
                            _filt = _FE.eq("session_id", shared_session_id)
                        _resolved.delete(collection_name=_coll_name, filter_expr=_filt)
                    except Exception as _del_exc:
                        # Re-ingesting on top of vectors that could not be
                        # deleted duplicates chunks and changes rankings.
                        raise _StaleCorpusCleanupError(
                            f"Could not delete stale vectors for session "
                            f"{shared_session_id}: {_del_exc}. Delete that session "
                            f"or collection and rerun."
                        ) from _del_exc
                    _ingest_backend.hybrid_searcher.clear_session(shared_session_id)
                    ready_ids = set()
        except _StaleCorpusCleanupError:
            raise
        except Exception as exc:
            logger.warning(f"Could not check existing vectors (will ingest fresh): {exc}")
            ready_ids = set()

        if not _skip_ingest:
            _session_cfg = _rag_cfg.with_session(shared_session_id)
            total_chunks, ingested_ids, error_ids = await _fast_ingest(
                corpus=corpus,
                corpus_to_valid=corpus_to_valid,
                ready_ids=ready_ids,
                session_id=shared_session_id,
                config=_session_cfg,
                backend=_ingest_backend,
            )
            logger.info(
                f"Fast ingestion complete: {total_chunks} chunks from "
                f"{len(ingested_ids)} docs, {len(error_ids)} errors"
            )

            # Persist error allowlist for future resume
            if error_ids:
                _allowlist_path = (
                    Path.home() / ".lasagna" / "research" / "results" / "_ingest_allowlist.json"
                )
                _allowlist_path.parent.mkdir(parents=True, exist_ok=True)
                _al: dict = {}
                if _allowlist_path.exists():
                    try:
                        _al = json.loads(_allowlist_path.read_text())
                    except Exception:
                        pass
                _al[shared_session_id] = sorted(error_ids)
                _allowlist_path.write_text(json.dumps(_al, indent=2))

        logger.info(f"Corpus ingestion complete for '{benchmark_name}' (cache_key='{cache_key}')")

        # Publish result and wake waiters.
        async with _corpus_cache_lock:
            slot = _corpus_sessions[cache_key]
            slot["session_id"] = shared_session_id
            slot["corpus_to_valid"] = corpus_to_valid
            slot["valid_to_corpus"] = valid_to_corpus
            slot["corpus_fingerprint"] = _corpus_fp
            slot["corpus_provenance"] = corpus_provenance
            # Whether the adapter actually left documents behind. The adapter
            # set this on itself and nothing carried it to the task, so
            # `TRUNCATED_CORPUS` -- one of only two blocking integrity codes --
            # could never fire and truncated bundles validated clean.
            slot["corpus_truncated"] = bool(getattr(adapter, "corpus_truncated", False))
            slot["ready"].set()

        return _corpus_sessions[cache_key]

    except Exception as exc:
        # On failure, clean up the slot and let waiters fail gracefully.
        async with _corpus_cache_lock:
            slot = _corpus_sessions.pop(cache_key, None)
            if slot:
                # The reason, not just the wake-up. Waking a waiter with an
                # empty slot sends it on to search a session that was never
                # ingested, which reads as a retrieval failure rather than as
                # an ingestion one.
                slot["error"] = f"Corpus ingestion failed: {type(exc).__name__}: {exc}"
                slot["ready"].set()  # unblock waiters
        raise


async def _release_corpus(benchmark_name: str, corpus_key: Optional[str] = None) -> None:
    """Decrement ref count; clean up when last task is done."""
    cache_key = (
        corpus_key
        if corpus_key is not None
        else _SHARED_CORPUS_KEY.get(benchmark_name, benchmark_name)
    )
    async with _corpus_cache_lock:
        info = _corpus_sessions.get(cache_key)
        if info is None:
            return
        info["ref_count"] -= 1
        if info["ref_count"] > 0:
            return
        # Last user — remove in-memory state only.
        # The Milvus corpus data is intentionally kept on disk so the next
        # experiment (or re-run) can find it via `len(ready_ids) >= len(corpus)`
        # and skip re-ingestion.  This is safe because the session_id is a
        # deterministic hash of the corpus content, so stale data from a
        # different corpus would never be matched.
        session_id = info["session_id"]  # noqa: F841 (kept for log)
        del _corpus_sessions[cache_key]
        logger.debug(
            f"Released in-memory corpus entry '{cache_key}' (Milvus data retained for reuse)"
        )


async def _eval_live(
    task: BenchmarkTask,
    samples: List[BenchmarkSample],
) -> TaskResult:
    """Live evaluation — ingest corpus once, search, compute IR metrics.

    Uses a shared corpus cache: the first task for a benchmark ingests the
    corpus; subsequent tasks (different configs/runs) reuse the same indexed
    data.  This eliminates redundant ingestion (~10 min per task) and enables
    safe parallelism for the search phase (read-only).

    Uses KnowledgeClient for ingestion (document_id support) and
    KnowledgePipeline for search (component override support).
    """
    start = time.monotonic()

    try:
        from ..sdk.client import KnowledgeClient  # noqa: F401
        from ..core.pipeline import KnowledgePipeline
        from .ir_metrics import corpus_metrics
    except ImportError as exc:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=f"Live mode requires KnowledgeClient + KnowledgePipeline: {exc}",
        )

    # Per-config ingestion: tasks that vary ingestion-time parameters (docling,
    # chunking, embedding) each need their own indexed corpus session so that
    # different chunking strategies don't bleed into each other's results.
    # _corpus_cache_key() is the single source of truth for key construction
    # (shared with _pre_register_corpus_refs to keep keys consistent).
    _corpus_key: Optional[str] = _corpus_cache_key(task)
    _ingestion_overrides: Optional[Dict[str, Any]] = None

    # Build ingestion overrides from two sources:
    # 1. _INGESTION_TIME_KEYS: affect corpus identity (chunking, embedding, etc.)
    # 2. _INGESTION_RUNTIME_KEYS: affect ingestion behavior but NOT corpus identity
    #    (e.g., enable_hybrid_search controls whether BM25 tokenization happens)
    _combined_ingestion_keys = _INGESTION_TIME_KEYS | _INGESTION_RUNTIME_KEYS
    _task_overrides = task.config_overrides or {}

    # Pull the profile's ingestion keys -- corpus-identity keys (embedder,
    # chunking) as well as runtime ones -- so the profile's corpus is the one
    # actually built. Only runtime keys used to be applied, while the corpus
    # cache key hashed all of them: `speed` declared all-MiniLM-L6-v2 / 384,
    # was cached under that, and ingested and queried with the global bge-m3.
    _from_profile: Dict[str, Any] = {}
    if task.profile:
        from ..core.session_config._profiles import resolve_ingestion_defaults as _resolve

        _all_profile = _resolve(task.profile)
        _from_profile = {k: v for k, v in _all_profile.items() if k in _combined_ingestion_keys}
    _merged_keys = {**_from_profile, **_task_overrides}
    _has_signature_keys = any(k in _INGESTION_TIME_KEYS for k in _merged_keys)
    _has_runtime_keys = any(k in _INGESTION_RUNTIME_KEYS for k in _merged_keys)

    if _has_signature_keys or _has_runtime_keys:
        _ingestion_overrides = {}
        # 1. Profile defaults for ingestion keys (base)
        _ingestion_overrides.update(_from_profile)
        # 2. Explicit task overrides for all ingestion keys (override)
        _ingestion_overrides.update(
            {k: v for k, v in _task_overrides.items() if k in _combined_ingestion_keys}
        )
        # One collection per embedder: vectors of different dimensions cannot
        # share one, and evaluation corpora stay out of the user's documents.
        if (
            "embedding_model" in _ingestion_overrides or "embedding_dim" in _ingestion_overrides
        ) and "collection_name" not in _ingestion_overrides:
            _ingestion_overrides["collection_name"] = _embedder_collection_name(
                _ingestion_overrides.get("embedding_model"),
                _ingestion_overrides.get("embedding_dim"),
            )

    if _has_signature_keys and _ingestion_overrides:
        logger.info(
            f"Per-config corpus session for '{task.config_name}': "
            f"cache_key='{_corpus_key}', "
            f"ingestion_overrides={list(_ingestion_overrides.keys())}"
        )
    elif _ingestion_overrides:
        logger.debug(
            f"Runtime ingestion overrides for '{task.config_name}': "
            f"{list(_ingestion_overrides.keys())}"
        )

    # Ensure corpus is ingested (first caller ingests, rest reuse)
    corpus_info = await _ensure_corpus_ingested(
        task.benchmark_name,
        samples,
        corpus_key=_corpus_key,
        ingestion_overrides=_ingestion_overrides,
        max_corpus_docs=task.max_corpus_docs,
    )

    if corpus_info is None:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=f"Benchmark '{task.benchmark_name}' does not provide a corpus for live mode",
        )
    if "error" in corpus_info:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=corpus_info["error"],
        )

    shared_session_id = corpus_info["session_id"]
    corpus_to_valid = corpus_info["corpus_to_valid"]

    # Populate corpus_fingerprint on the task (used by ingestion signature)
    if hasattr(task, "corpus_fingerprint"):
        task.corpus_fingerprint = corpus_info.get("corpus_fingerprint", "")
    # Where the corpus came from. Hashed into the comparability key, so a
    # pooled-gold number can never share a leaderboard row with a real one.
    task.corpus_provenance = corpus_info.get("corpus_provenance", "native")
    task.corpus_truncated = bool(corpus_info.get("corpus_truncated", False))

    # Every task on a pooled corpus needs the labels, not just the one that
    # ingested it. A waiter otherwise gets samples with empty expected_docs and
    # scores 0.0 on every IR metric without erroring -- so a sweep reports the
    # first arm working and the rest at zero, which reads as a catastrophic
    # regression rather than as missing labels.
    _mint_pooled_qrels(benchmark_registry.get(task.benchmark_name), samples, task.corpus_provenance)

    # Split samples by evaluation type.
    # IR samples have qrels (expected_docs) — evaluate retrieval metrics.
    # Gen samples have ground_truth — evaluate generation quality (RAGAS + EM/F1).
    # E2E samples (both expected_docs AND ground_truth) appear in BOTH lists.
    # They are queried twice: once with IR-forced config, once with gen config.
    ir_samples = [s for s in samples if s.expected_docs]
    gen_samples = [s for s in samples if s.ground_truth]
    if task.tracks:
        # Skipped by request, not by failure: nothing is scored for the
        # track that did not run.
        if "retrieval" not in task.tracks:
            ir_samples = []
        if "generation" not in task.tracks:
            gen_samples = []

    # Build pipeline from profile and apply task-level overrides.
    pipeline = KnowledgePipeline.from_profile(task.profile)
    _apply_overrides_to_pipeline(pipeline, task.config_overrides, shared_session_id)

    # Live-mode fix 1: Force embedding model/dim to match the ingested corpus.
    # For embedding experiments, use the task's own ingestion overrides (the global
    # EmbeddingManager singleton may hold a previously-loaded model). The existing
    # config-driven reload path (maybe_reload_embedding_manager) handles the actual
    # model switch at query time — we just need correct values in pipeline._config.
    from ..api.rag_backend import get_rag_backend as _get_backend

    _backend = _get_backend()
    if _ingestion_overrides and "embedding_model" in _ingestion_overrides:
        _corpus_model = _ingestion_overrides["embedding_model"]
        _corpus_dim = _ingestion_overrides.get("embedding_dim", 1024)
    else:
        _corpus_model = _backend.embedding_manager.model_name
        _corpus_dim = _backend.embedding_manager.dimension
    pipeline._config = dict(pipeline._config) if pipeline._config else {}
    pipeline._config["embedding_model"] = _corpus_model
    pipeline._config["embedding_dim"] = _corpus_dim
    # Propagate collection_name for per-embedding-model collections
    if _ingestion_overrides and "collection_name" in _ingestion_overrides:
        pipeline._config["collection_name"] = _ingestion_overrides["collection_name"]

    # Live-mode fix 3: Ensure BM25 fallback index is built for hybrid search.
    # The in-memory rank_bm25 index is not persisted across process restarts
    # and is not populated when reusing a previously ingested corpus (the
    # skip-ingestion path never calls DocumentPipeline, so
    # HybridSearcher.index_chunks() is never invoked).  Without this fix,
    # _fallback_hybrid_search returns dense-only results regardless of alpha.
    #
    # Three-tier guard:
    #   1. Skip if native hybrid available (no fallback BM25 needed).
    #   2. Skip if rank_bm25 not installed.
    #   3. Fast-path: if in-memory docs exist from ingestion, just build.
    #   4. Slow-path: corpus-reuse / skip-ingestion — fetch from Milvus.
    from ..core.config import rag_config as _rag_cfg
    from ..retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE as _BM25_AVAIL

    # Check pipeline config first (has profile applied), then overrides, then global default
    _enable_hybrid = pipeline._config.get(
        "enable_hybrid_search",
        (task.config_overrides or {}).get("enable_hybrid_search", _rag_cfg.enable_hybrid_search),
    )
    if _enable_hybrid:
        _hybrid = _backend.hybrid_searcher
        # Guard 1: Skip entirely if native hybrid is available (no fallback needed)
        if _hybrid.is_native_mode_available() or _hybrid._backend_supports_native_hybrid(
            shared_session_id
        ):
            logger.debug("Native hybrid available — skipping BM25 fallback prebuild")
        elif not _BM25_AVAIL:
            # Guard 2: rank_bm25 not installed
            logger.debug("BM25 fallback not available — skipping prebuild")
        else:
            _bm25_idx = _hybrid.get_session_index(shared_session_id)
            if _bm25_idx.is_current():
                logger.debug(f"BM25 index already built for session {shared_session_id}")
            elif _bm25_idx.tokenized_corpus:
                # Fast path: docs accumulated in-memory during ingestion — just build.
                # Skips the expensive Milvus paginated fetch entirely.
                logger.info(
                    f"Building BM25 index from {len(_bm25_idx.documents)} "
                    f"in-memory chunks (session {shared_session_id})"
                )
                _hybrid.build_session_index(shared_session_id)
            else:
                # Slow path: corpus-reuse / skip-ingestion — fetch from Milvus.
                logger.info(
                    f"Building BM25 fallback index for session {shared_session_id} "
                    f"(fetching from vector store)"
                )
                from ..vector_backends.filters import FilterExpression

                _vec = _backend._vector_backend
                _coll = pipeline._config.get("collection_name", _rag_cfg.collection_name)
                # Fetch every chunk for this session. This must be exhaustive:
                # a BM25 index built from part of the corpus still returns
                # results, so a truncated scan produces a plausible retrieval
                # number from a corpus that was never fully indexed. `iter_all`
                # uses a native cursor rather than limit+offset paging, which
                # previously failed outright past 16384 rows.
                if _vec.backend_name == "milvus":
                    _filt = FilterExpression.json_eq("metadata", "session_id", shared_session_id)
                else:
                    _filt = FilterExpression.eq("session_id", shared_session_id)
                _all_chunks: list = list(
                    _vec.iter_all(
                        collection_name=_coll,
                        filter_expr=_filt,
                        output_fields=["chunk_id", "content", "document_id", "metadata"],
                    )
                )

                # Convert to BM25 chunk format and build index.
                _bm25_docs = build_bm25_docs(_all_chunks)
                if _bm25_docs:
                    _hybrid.index_chunks(_bm25_docs, shared_session_id)
                    logger.info(
                        f"BM25 fallback index built: {len(_bm25_docs)} chunks "
                        f"for session {shared_session_id}"
                    )
                else:
                    logger.warning(
                        f"No chunks found in vector store for session "
                        f"{shared_session_id} — BM25 hybrid search will use "
                        f"dense-only fallback"
                    )

    # Save config for generation path — includes task overrides + embedding fixes,
    # but NOT the IR-specific overrides that disable agentic routing.
    gen_config = dict(pipeline._config)

    # Live-mode fix 2 (IR only): Disable agentic/comprehensive routing.
    # The agentic pipeline runs multi-hop retrieval + LLM synthesis. For IR
    # metric computation we need direct chunk-level SearchResult objects with
    # stable document_ids — forcing standard tier gives the cleanest signal.
    # (Note: agentic DOES return SearchResult objects via _convert_agentic_results,
    # but their document_ids may differ and latency is high — not suitable for IR eval.)
    if ir_samples:
        pipeline._config["use_agentic_rag"] = False
        pipeline._config["agentic_pipeline"] = "legacy_react"
        pipeline._config["routing_default_tier"] = "standard"
        # Clear component name so pipeline.query() doesn't force comprehensive
        # tier; the generation path below puts it back.
        _agentic_name = pipeline._component_names.pop("agentic_pipeline", None)
    else:
        _agentic_name = None

    # IR config — has all forced overrides applied. Passed as per-query ephemeral
    # overrides via pipeline.query(config_overrides=…) → RAGConfig.with_query_overrides(…).
    # This bypasses the session store entirely — concurrent tasks never clobber each other.
    task_config = dict(pipeline._config)

    # Query via KnowledgePipeline (supports component overrides).
    from .ir_metrics import (
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        mrr as mrr_score,
        map_score,
        f1_at_k,
        hit_rate_at_k,
    )

    relevances: List[Dict[str, int]] = []
    retrieved_lists: List[List[str]] = []
    per_sample_scores: List[Dict[str, float]] = []
    #: Reasons this task's numbers cannot be trusted. Any entry fails the task
    #: (TaskResult.error) instead of letting a failure stand in as a score.
    _eval_failures: List[str] = []
    _n_generations = 0
    _failed_generations = 0
    total_dropped = 0
    #: The judge's spend, accumulated across every RAGAS pass in this task.
    eval_meter = UsageMeter()

    # The cutoff the IR metrics are reported at. Was hard-coded to 10 in six
    # places, so NDCG@100 and Recall@1000 -- the cutoffs much of the retrieval
    # literature reports -- could not be produced at all.
    #
    # Read from the task's overrides, not from `task_config`: both of these are
    # HARNESS_KEYS, so `_apply_overrides_to_pipeline` deliberately keeps them
    # out of `pipeline._config` (they steer measurement, not retrieval), and
    # `task_config` is a copy of that config.
    _harness = task.config_overrides or {}
    _metric_k = int(_harness.get("metric_k", 10) or 10)

    # How many *chunks* to ask the pipeline for.
    #
    # The metrics are computed over parent documents: search results are
    # de-duplicated by `document_id` below, because several chunks of one
    # document would otherwise let NDCG exceed 1.0. But the request was for
    # `k` chunks, so with `chunk_size: 800` a top-10 chunk list collapsed to
    # 4-7 distinct documents and NDCG@10 was computed over a list shorter than
    # 10 -- a systematic downward bias on every dataset, present whether or not
    # the corpus was truncated, and invisible because the number looked
    # plausible.
    #
    # Over-fetch by a factor and then cut to `k` documents. The factor is
    # configurable because the right value depends on chunk size relative to
    # document length; the default of 4 covers a document splitting into four
    # chunks, which `chunk_size: 800` makes common on BEIR.
    _chunk_overfetch = max(1, int(_harness.get("retrieval_chunk_overfetch", 4) or 4))
    _retrieval_depth = _metric_k * _chunk_overfetch
    logger.info(
        f"IR metrics at k={_metric_k}; requesting {_retrieval_depth} chunks per query "
        f"to yield {_metric_k} distinct documents after de-duplication"
    )

    # Retrieval metrics measure the pipeline at a fixed depth, so the
    # answer-time policies that shorten or reuse result lists are neutralized
    # here. Each one changed what a sweep measured, without an error:
    # - the search cache keyed on a dozen fields (not the reranker model), so
    #   one arm was served another arm's lists, and repeated runs replayed run 0;
    # - similarity_threshold is applied to per-query min-max-normalized scores,
    #   a relative cut that left about 4 documents per query on SciFact;
    # - adaptive-k cut reranked lists by score gap, to about 7 documents;
    # - default_top_k capped the dense candidates below the requested depth.
    _fix_retrieval_depth(task_config, _retrieval_depth)
    # Generation keeps its answer-time policies (they are the system under
    # test), but every run must execute rather than replay the cache.
    gen_config["enable_search_cache"] = False

    try:
        # ── IR path ─────────────────────────────────────────────────────────
        for sample in ir_samples:
            # Pass task_config as per-query ephemeral overrides — no session
            # store write needed, no lock required.  Concurrent tasks are fully
            # independent because each call gets its own RAGConfig snapshot via
            # with_query_overrides().
            query_start = time.monotonic()
            # Count what this query spends. The meter is a contextvar, which
            # `asyncio.to_thread` copies into the worker thread, so LLM calls
            # made deep inside the pipeline -- HyDE, query expansion, the whole
            # agentic graph -- are attributed to this sample.
            sample_meter = UsageMeter()
            with sample_meter, sample_meter.scope(SYSTEM):
                search_results = await asyncio.to_thread(
                    pipeline.query,
                    sample.question,
                    _retrieval_depth,
                    shared_session_id,
                    task_config,
                )
            query_latency_ms = (time.monotonic() - query_start) * 1000

            # Extract doc IDs from search results.
            # De-duplicate by parent document ID to avoid ndcg > 1.0 when
            # multiple chunks of the same document appear in top-k results.
            retrieved_ids: List[str] = []
            seen_doc_ids: set = set()
            #: (doc_id, score, rank) for what was actually returned.
            #:
            #: The list of ids alone cannot answer the first question anyone
            #: asks about a lost query -- "was the right document retrieved and
            #: ranked badly, or never retrieved at all?" -- because it carries
            #: no scores, and because de-duplicating by parent document means
            #: position in the list is not the rank the retriever assigned.
            retrieved_detail: List[Dict[str, Any]] = []
            for chunk_index, r in enumerate(search_results):
                doc_id = None
                if hasattr(r, "chunk") and hasattr(r.chunk, "document_id"):
                    doc_id = str(r.chunk.document_id)
                elif hasattr(r, "document") and hasattr(r.document, "document_id"):
                    doc_id = str(r.document.document_id)
                elif hasattr(r, "document_id") and r.document_id:
                    doc_id = str(r.document_id)
                if doc_id and doc_id not in seen_doc_ids:
                    retrieved_ids.append(doc_id)
                    seen_doc_ids.add(doc_id)
                    retrieved_detail.append(
                        {
                            "doc_id": doc_id,
                            "score": float(getattr(r, "score", 0.0) or 0.0),
                            # Position in the de-duplicated document list, which
                            # is the rank the IR metrics are computed against.
                            #
                            # This comment used to claim "the retriever's own
                            # rank, before de-duplication" -- it never was, since
                            # the list is only appended to for documents not seen
                            # before. `chunk_rank` is the pre-dedup position, so
                            # a reader can see how far down the chunk list the
                            # first hit for this document appeared.
                            "rank": len(retrieved_detail) + 1,
                            "chunk_rank": chunk_index + 1,
                        }
                    )

            # Map expected_docs corpus IDs to validated IDs for comparison.
            mapped_expected: Dict[str, int] = {}
            for corpus_id, score in sample.expected_docs.items():
                valid_id = corpus_to_valid.get(corpus_id)
                if valid_id:
                    mapped_expected[valid_id] = score
                else:
                    total_dropped += 1

            # Cut to k *documents*, which is what the metrics are reported at.
            # Over-fetching chunks above is what makes this list long enough to
            # reach k in the first place; without the cut, a run that
            # over-fetched would score a longer list than the cutoff names.
            retrieved_ids = retrieved_ids[:_metric_k]
            retrieved_detail = retrieved_detail[:_metric_k]

            relevances.append(mapped_expected)
            retrieved_lists.append(retrieved_ids)

            # Per-sample IR scores for significance testing + latency tracking.
            # _sample_id and _source_benchmark (underscore prefix) are metadata
            # read by BenchmarkResults.domain_stratified_significance_test,
            # which needs to know which source benchmark each query came from
            # to test per domain rather than pooling them.
            _sid = (
                getattr(sample, "sample_id", None) or sample.metadata.get("sample_id", "")
                if hasattr(sample, "metadata")
                else ""
            ) or sample.query_id
            _src = (
                sample.metadata.get("source_benchmark", "") if hasattr(sample, "metadata") else ""
            )
            sample_metrics: Dict[str, float] = {
                "_sample_id": _sid,
                "_source_benchmark": _src,
                # Keyed by the cutoff actually used. Hard-coding `@10` in the
                # name while the cutoff was configurable would relabel a
                # NDCG@100 as NDCG@10 -- a wrong number under a right-looking
                # key, which is worse than not offering the option.
                f"ndcg@{_metric_k}": ndcg_at_k(mapped_expected, retrieved_ids, k=_metric_k),
                f"precision@{_metric_k}": precision_at_k(
                    mapped_expected, retrieved_ids, k=_metric_k
                ),
                f"recall@{_metric_k}": recall_at_k(mapped_expected, retrieved_ids, k=_metric_k),
                f"f1@{_metric_k}": f1_at_k(mapped_expected, retrieved_ids, k=_metric_k),
                f"mrr@{_metric_k}": mrr_score(mapped_expected, retrieved_ids, k=_metric_k),
                f"map@{_metric_k}": map_score(mapped_expected, retrieved_ids, k=_metric_k),
                f"hit_rate@{_metric_k}": hit_rate_at_k(mapped_expected, retrieved_ids, k=_metric_k),
                "latency_ms": query_latency_ms,
                # The full list, not the first 20. The cap predated any reader:
                # nothing consumed these, so nobody noticed that a run with
                # top_k above 20 recorded a truncated view of its own results.
                "_retrieved_ids": retrieved_ids,
                "_retrieved": retrieved_detail,
                # What the query was *supposed* to find, so a per-query row can
                # be read without the dataset alongside it.
                "_expected_docs": sorted(mapped_expected),
            }

            # What this query actually spent, measured at the provider rather
            # than read off chunk metadata. The old code summed
            # `chunk.metadata["input_tokens"]`, a key nothing in the package
            # ever wrote, so every token count and every derived dollar figure
            # was zero -- which made a technique that quietly calls an LLM per
            # query look exactly as cheap as one that calls none.
            sample_metrics.update(sample_meter.usage(SYSTEM).to_dict())

            per_sample_scores.append(sample_metrics)

        if total_dropped > 0:
            logger.warning(
                f"Dropped {total_dropped} qrels entries whose corpus docs "
                f"were not loaded (truncated corpus). Metrics may be deflated."
            )

        # Compute corpus-level IR metrics.
        if relevances:
            metrics = corpus_metrics(relevances, retrieved_lists, k=_metric_k)
        else:
            metrics = {}

        # ── Generation path ──────────────────────────────────────────────────
        # Uses gen_config (task overrides + embedding fixes, NOT IR-forced
        # overrides) so use_agentic_rag / routing_default_tier respect the
        # task's config_overrides and the agentic path is exercised properly.
        if gen_samples:
            if _agentic_name:
                # A named agentic pipeline is what this arm measures; the
                # retrieval pass above had to drop it.
                pipeline._component_names["agentic_pipeline"] = _agentic_name
            gen_questions: List[str] = []
            gen_answers: List[str] = []
            gen_contexts: List[List[str]] = []
            gen_ground_truths: List[str] = []
            gen_agentic_answers: List[str] = []  # agentic pipeline's own answers
            # One meter for the whole generation phase, read as a delta at the
            # end of each sample. Re-entered around each region that can spend
            # tokens rather than wrapping the loop body, so the accounting does
            # not depend on remembering to re-indent two hundred lines.
            gen_meter = UsageMeter()
            gen_spent = TokenUsage()

            # Answer LLM for generating responses from retrieved context.
            # Resolved independently of the judge — see _resolve_generator_model.
            answer_llm = None
            try:
                from ..providers import get_llm_provider

                answer_llm = get_llm_provider().get_llm(model_name=_resolve_generator_model())
            except Exception as exc:
                logger.warning(
                    f"Could not get answer LLM for generation eval [{type(exc).__name__}]: {exc}"
                )
                logger.exception("Answer LLM construction failed; generated answers will be empty")

            for gen_sample in gen_samples:
                query_start = time.monotonic()
                try:
                    # 5-min per-sample timeout: guards against silent LLM gateway
                    # hangs where the TCP connection is established but the server
                    # never sends a response (no exception, just indefinite block).
                    with gen_meter, gen_meter.scope(SYSTEM):
                        search_results = await asyncio.wait_for(
                            asyncio.to_thread(
                                pipeline.query,
                                gen_sample.question,
                                10,
                                shared_session_id,
                                gen_config,
                            ),
                            timeout=300,  # 5 min
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Query timed out after 5 min for "
                        f"'{gen_sample.question[:50]}': skipping sample."
                    )
                    search_results = []
                query_latency_ms = (time.monotonic() - query_start) * 1000

                # Extract text from retrieved chunks.
                contexts = [
                    r.chunk.content
                    for r in search_results
                    if hasattr(r, "chunk") and hasattr(r.chunk, "content") and r.chunk.content
                ]

                # Track agentic fallback. The orchestrator stamps the tier that
                # actually ran on the first result; a failed agentic run falls
                # back to standard retrieval and returns results, so an empty
                # list is not the only sign of it.
                agentic_requested = gen_config.get("use_agentic_rag", False)
                fallback = _agentic_fell_back(agentic_requested, search_results)

                # Extract agentic pipeline's own answer if available.
                # _convert_agentic_results() stashes it in the first chunk's
                # metadata as "_agentic_answer".
                _agentic_answer = ""
                if agentic_requested and search_results:
                    first_meta = (
                        getattr(getattr(search_results[0], "chunk", None), "metadata", {}) or {}
                    )
                    _agentic_answer = first_meta.get("_agentic_answer", "")

                # Agentic pipeline makes many parallel LLM calls which can
                # saturate a hosted endpoint's rate limit. An optional pause
                # before the answer-generation call (pipeline setting
                # post_agentic_cooldown_s, or LASAGNA_AGENTIC_COOLDOWN_S). It
                # was a fixed 10 s for every agentic sample, which costs 2.8 h
                # per 1,000 samples even against a local model; the default is
                # now 0 and the generation call's own retry handles 429s.
                _cooldown = float(
                    gen_config.get(
                        "post_agentic_cooldown_s",
                        os.environ.get("LASAGNA_AGENTIC_COOLDOWN_S", 0),
                    )
                    or 0
                )
                if agentic_requested and _cooldown > 0:
                    await asyncio.sleep(_cooldown)

                # Generate answer from retrieved context.  The generation prompt
                # is identical across configs — what differs is WHAT gets retrieved.
                answer = ""

                # Speculative RAG: parallel draft generation + verification
                _speculative = gen_config.get("enable_speculative_rag", False)
                if _speculative and answer_llm and contexts:
                    try:
                        from lasagna.agentic.speculative_rag import SpeculativeRAG

                        spec_rag = SpeculativeRAG(
                            draft_llm=answer_llm,
                            verifier_llm=answer_llm,
                            n_drafts=int(gen_config.get("speculative_n_drafts", 3)),
                            subset_size=int(gen_config.get("speculative_subset_size", 3)),
                        )
                        with gen_meter, gen_meter.scope(SYSTEM):
                            answer, _drafts = await spec_rag.generate(
                                query=gen_sample.question,
                                contexts=contexts[:10],
                            )
                        logger.debug(
                            f"Speculative RAG: {len(_drafts)} drafts for "
                            f"'{gen_sample.question[:40]}', answer='{answer[:80]}'"
                        )
                    except Exception as exc:
                        logger.warning(f"Speculative RAG failed, falling back: {exc}")
                        _speculative = False  # fall through to standard generation
                    # If speculative succeeded but returned empty, fall through
                    if _speculative and not answer.strip():
                        _speculative = False

                if not _speculative and answer_llm and contexts:
                    ctx_text = "\n\n".join(contexts[:5])
                    prompt = (
                        "Answer the following question in ONE concise sentence "
                        "based only on the provided context. "
                        "Do not explain or elaborate.\n\n"
                        f"Context:\n{ctx_text}\n\n"
                        f"Question: {gen_sample.question}\n\nAnswer:"
                    )
                    # Retry up to 5 times with aggressive backoff — agentic
                    # pipeline's parallel LLM calls trigger gateway rate limits
                    # that can take 30-60 s to recover (5s, 15s, 30s, 60s waits).
                    _backoffs = [5, 15, 30, 60]
                    for _attempt in range(5):
                        try:
                            with gen_meter, gen_meter.scope(SYSTEM):
                                if hasattr(answer_llm, "ainvoke"):
                                    # 2-min timeout: same silent-hang guard as pipeline.query
                                    raw = await asyncio.wait_for(
                                        answer_llm.ainvoke(prompt),
                                        timeout=120,
                                    )
                                else:
                                    raw = await asyncio.wait_for(
                                        asyncio.to_thread(answer_llm.invoke, prompt),
                                        timeout=120,
                                    )
                            answer = raw.content if hasattr(raw, "content") else str(raw)
                            logger.debug(
                                f"Generated answer for '{gen_sample.question[:40]}': "
                                f"'{answer[:80]}'"
                            )
                            break  # success
                        except Exception as exc:
                            exc_str = str(exc)
                            # Do NOT retry content-policy / bad-request errors
                            # (400) — same prompt will be rejected again.
                            if (
                                "400" in exc_str
                                or "content_filter" in exc_str
                                or "ContentPolicy" in exc_str
                            ):
                                logger.warning(
                                    f"Answer generation blocked by content filter "
                                    f"for '{gen_sample.question[:50]}': skipping."
                                )
                                break
                            if _attempt < 4:
                                _backoff = _backoffs[_attempt]
                                logger.warning(
                                    f"Answer generation attempt {_attempt + 1}/5 failed "
                                    f"('{gen_sample.question[:40]}'): {exc}. "
                                    f"Retrying in {_backoff}s..."
                                )
                                await asyncio.sleep(_backoff)
                            else:
                                logger.warning(
                                    f"Answer generation failed after 5 attempts for "
                                    f"'{gen_sample.question[:50]}': {exc}"
                                )

                _n_generations += 1
                if not (answer or "").strip():
                    # No answer: no context, no generator, or every attempt
                    # failed. Scored, it would read as an unfaithful answer.
                    _failed_generations += 1
                gen_questions.append(gen_sample.question)
                gen_answers.append(answer)
                # Pass the SAME contexts used for generation to RAGAS.
                # Standard generation uses top 5; speculative RAG uses top 10.
                # Must match to avoid faithfulness metric mismatch.
                _ctx_k = 10 if _speculative else 5
                gen_contexts.append(contexts[:_ctx_k])
                gen_ground_truths.append(gen_sample.ground_truth)
                gen_agentic_answers.append(_agentic_answer)

                _src = (
                    gen_sample.metadata.get("source_benchmark", "")
                    if hasattr(gen_sample, "metadata")
                    else ""
                )
                # ── EM / Token-F1 (free, no LLM calls) ──────────────
                _em_score = 0.0
                _f1_score_val = 0.0
                _gt = gen_sample.ground_truth or ""
                _gt_aliases = getattr(gen_sample, "ground_truth_answers", None) or []
                if _gt.strip() and answer.strip():
                    from .answer_metrics import exact_match as _em_fn, token_f1 as _tf1_fn

                    _em_score = _em_fn(answer, _gt, aliases=_gt_aliases)
                    _f1_score_val = _tf1_fn(answer, _gt, aliases=_gt_aliases)
                elif not _gt.strip() and (not answer.strip() or _is_abstention_answer(answer)):
                    # Both empty / abstention on unanswerable → correct
                    _em_score = 1.0
                    _f1_score_val = 1.0

                _gen_sample_metrics: Dict[str, Any] = {
                    "_sample_id": gen_sample.query_id,
                    "_source_benchmark": _src,
                    "latency_ms": query_latency_ms,
                    "n_contexts": float(len(contexts)),
                    "agentic_fallback": 1.0 if fallback else 0.0,
                    "exact_match": _em_score,
                    "token_f1": _f1_score_val,
                }

                # What this sample spent, as a delta against the phase meter.
                _running = gen_meter.usage(SYSTEM)
                _gen_sample_metrics.update(
                    TokenUsage(
                        input_tokens=_running.input_tokens - gen_spent.input_tokens,
                        output_tokens=_running.output_tokens - gen_spent.output_tokens,
                        llm_calls=_running.llm_calls - gen_spent.llm_calls,
                    ).to_dict()
                )
                gen_spent = TokenUsage(
                    input_tokens=_running.input_tokens,
                    output_tokens=_running.output_tokens,
                    llm_calls=_running.llm_calls,
                )

                # ── Trajectory logging ────────────────────────────────
                # Save generated answer + contexts for post-hoc analysis.
                # Controlled by save_trajectories config (default: minimal).
                _save_traj = (task.config_overrides or {}).get("save_trajectories", "minimal")
                if _save_traj == "full":
                    _gen_sample_metrics["_generated_answer"] = answer[:2000]
                    _gen_sample_metrics["_retrieved_contexts"] = [c[:500] for c in contexts[:5]]
                    _gen_sample_metrics["_ground_truth"] = _gt[:500]
                    _gen_sample_metrics["_question"] = gen_sample.question[:500]
                elif _save_traj == "minimal":
                    _gen_sample_metrics["_generated_answer"] = answer[:500]
                    _gen_sample_metrics["_question"] = gen_sample.question[:200]

                per_sample_scores.append(_gen_sample_metrics)

            # Run RAGAS on all collected generation samples.
            if gen_questions:
                try:
                    from .ragas_evaluator import RAGASEvaluator

                    # RAGAS runs in its own thread (see below), so the judge is
                    # built per call via the module-level _build_judge_llm()
                    # rather than shared across threads.

                    # --- Evaluate re-generated answers (faithfulness_regen) ---
                    # Run RAGAS in a thread — ragas.evaluate() calls
                    # asyncio.run() internally which deadlocks when called
                    # from within an already-running event loop.
                    # batch_size=4: hosted LLM gateways rate-limit aggressively.
                    def _ragas_regen_in_thread():
                        # The meter is entered inside the worker thread because
                        # that is where the judge's calls happen; it accumulates
                        # into the shared object the caller reads afterwards.
                        with eval_meter, eval_meter.scope(EVAL):
                            ragas_eval = RAGASEvaluator(llm=_build_judge_llm())
                            return ragas_eval.evaluate_detailed(
                                questions=gen_questions,
                                answers=gen_answers,
                                contexts=gen_contexts,
                                ground_truths=gen_ground_truths,
                                batch_size=4,
                            )

                    # 45-min timeout: guards against hung LLM gateway connections.
                    regen_result = await asyncio.wait_for(
                        asyncio.to_thread(_ragas_regen_in_thread),
                        timeout=2700,  # 45 min
                    )
                    if getattr(regen_result, "failed_metrics", None):
                        _eval_failures.append(
                            "judge produced no usable scores for "
                            + ", ".join(regen_result.failed_metrics)
                        )
                    # Store aggregate scores (prefix with _regen when agentic
                    # answers are also available, otherwise use plain names).
                    has_agentic_answers = any(a for a in gen_agentic_answers)
                    if has_agentic_answers:
                        for k, v in regen_result.aggregate.items():
                            metrics[f"{k}_regen"] = v
                    else:
                        metrics.update(regen_result.aggregate)

                    # Merge per-sample RAGAS scores into per_sample_scores.
                    # Gen entries start AFTER IR entries in per_sample_scores,
                    # so use _gen_ps_offset to align indices correctly.
                    _regen_prefix = "_regen" if has_agentic_answers else ""
                    _gen_ps_offset = len(ir_samples)  # gen entries start after IR
                    if regen_result.per_sample:
                        _n_gen_ps = len(regen_result.per_sample)
                        for i, ps in enumerate(regen_result.per_sample):
                            _target_idx = _gen_ps_offset + i
                            if _target_idx < len(per_sample_scores):
                                for metric_name, score_val in ps.items():
                                    per_sample_scores[_target_idx][
                                        f"{metric_name}{_regen_prefix}"
                                    ] = score_val

                    logger.info(f"RAGAS regen scores for {task.task_key}: {regen_result.aggregate}")

                    # --- Evaluate agentic pipeline answers (faithfulness_agentic) ---
                    # Finding 1 fix: Only evaluate samples with NON-EMPTY agentic
                    # answers.  Empty answers (e.g. agentic returned 0 chunks for
                    # that sample) would score 0.0, systematically depressing the
                    # aggregate agentic metric.
                    if has_agentic_answers:
                        # Build index of samples that actually have agentic answers.
                        _ag_indices = filter_nonempty_agentic_indices(gen_agentic_answers)
                        _ag_questions = [gen_questions[i] for i in _ag_indices]
                        _ag_answers = [gen_agentic_answers[i] for i in _ag_indices]
                        _ag_contexts = [gen_contexts[i] for i in _ag_indices]
                        _ag_truths = [gen_ground_truths[i] for i in _ag_indices]

                        if _ag_questions:
                            logger.info(
                                f"Evaluating agentic answers: {len(_ag_questions)} "
                                f"of {len(gen_questions)} samples have non-empty "
                                f"agentic answers"
                            )

                            def _ragas_agentic_in_thread():
                                with eval_meter, eval_meter.scope(EVAL):
                                    ragas_eval = RAGASEvaluator(llm=_build_judge_llm())
                                    return ragas_eval.evaluate_detailed(
                                        questions=_ag_questions,
                                        answers=_ag_answers,
                                        contexts=_ag_contexts,
                                        ground_truths=_ag_truths,
                                        batch_size=4,
                                    )

                            agentic_eval_result = await asyncio.wait_for(
                                asyncio.to_thread(_ragas_agentic_in_thread),
                                timeout=2700,
                            )
                            for k, v in agentic_eval_result.aggregate.items():
                                metrics[f"{k}_agentic"] = v

                            # Merge agentic per-sample scores back to original
                            # indices.  _ag_indices are 0-based within gen_samples;
                            # offset by _gen_ps_offset to align with per_sample_scores.
                            if agentic_eval_result.per_sample:
                                for j, orig_idx in enumerate(_ag_indices):
                                    _target_idx = _gen_ps_offset + orig_idx
                                    if j < len(
                                        agentic_eval_result.per_sample
                                    ) and _target_idx < len(per_sample_scores):
                                        ps = agentic_eval_result.per_sample[j]
                                        for metric_name, score_val in ps.items():
                                            per_sample_scores[_target_idx][
                                                f"{metric_name}_agentic"
                                            ] = score_val

                            logger.info(
                                f"RAGAS agentic scores for {task.task_key}: "
                                f"{agentic_eval_result.aggregate}"
                            )
                        else:
                            logger.warning(
                                f"All agentic answers were empty for "
                                f"{task.task_key} — skipping agentic eval"
                            )

                    # Emit canonical unsuffixed metrics so downstream
                    # consumers (ablation.py, BenchmarkResults.significance_test
                    # and pairwise_significance) can compare agentic vs
                    # non-agentic configs using the same metric key.
                    #
                    # For agentic configs the canonical per-sample score is:
                    #   agentic answer score if available, else regen score.
                    # The canonical AGGREGATE is the mean of these per-sample
                    # canonical values over ALL N samples — not the subset
                    # average from faithfulness_agentic (which may cover
                    # fewer samples when some agentic answers are empty).
                    if has_agentic_answers:
                        apply_canonical_metric_mapping(
                            per_sample_scores,
                            metrics,
                        )

                    agentic_fallbacks = sum(
                        1 for s in per_sample_scores if s.get("agentic_fallback", 0.0) > 0
                    )
                    if agentic_fallbacks:
                        metrics["agentic_fallback_count"] = float(agentic_fallbacks)
                    if gen_config.get("use_agentic_rag", False) and per_sample_scores:
                        # Always reported for agentic configs, so a 0 is a
                        # measurement rather than a missing key.
                        metrics["agentic_fallback_rate"] = agentic_fallbacks / len(
                            per_sample_scores
                        )

                except Exception as exc:
                    logger.error(
                        f"RAGAS evaluation failed for {task.task_key} [{type(exc).__name__}]: {exc}"
                    )
                    logger.exception(f"RAGAS evaluation failed for {task.task_key}")
                    _eval_failures.append(f"RAGAS evaluation failed: {type(exc).__name__}: {exc}")

            # ── Aggregate EM / Token-F1 (outside RAGAS try block) ─────────
            # Computed per-sample above; now aggregate across gen samples.
            _gen_ps = [s for s in per_sample_scores if "exact_match" in s]
            if _gen_ps:
                metrics["exact_match"] = sum(s["exact_match"] for s in _gen_ps) / len(_gen_ps)
                metrics["token_f1"] = sum(s["token_f1"] for s in _gen_ps) / len(_gen_ps)

    finally:
        # Release corpus ref — always called even if evaluation raises, so the
        # shared corpus is not leaked when a task errors or is cancelled.
        await _release_corpus(task.benchmark_name, corpus_key=_corpus_key)

    if _n_generations:
        metrics["generations_failed"] = float(_failed_generations)
        # A few failures (a content filter, one timeout) are reported; more than
        # a tenth means the generation metrics describe the outage, not the system.
        if _failed_generations / _n_generations > _MAX_FAILED_GENERATION_RATE:
            _eval_failures.append(
                f"{_failed_generations} of {_n_generations} answers could not be generated"
            )

    result = TaskResult(
        task_key=task.task_key,
        benchmark_name=task.benchmark_name,
        config_name=task.config_name,
        run_index=task.run_index,
        metrics=metrics,
        per_sample_scores=per_sample_scores,
        duration_seconds=time.monotonic() - start,
        # Deduplicate: e2e samples appear in both ir_samples and gen_samples
        n_samples=len({s.query_id for s in ir_samples} | {s.query_id for s in gen_samples}),
        error="; ".join(_eval_failures) or None,
        system=_ran_system(
            _corpus_model,
            _corpus_dim,
            _ingestion_overrides,
            task_config,
            reranker_component=(task.config_overrides or {}).get("reranker"),
        ),
    )
    result.compute_latency_stats()
    _record_eval_usage(result, eval_meter)
    return result


async def run_benchmark(
    benchmark: str,
    n_samples: Optional[int] = 100,
    mode: str = "auto",
    n_runs: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
    eval_fn: Optional[Callable] = None,
    **kwargs: Any,
) -> BenchmarkResults:
    """One-liner benchmark execution.

    Example::

        from lasagna.sdk import run_benchmark

        results = await run_benchmark("beir/nfcorpus", n_samples=100)
        print(results.to_markdown())

    Args:
        benchmark: Benchmark name (e.g. ``"beir/nfcorpus"``).
        n_samples: Number of samples to evaluate.
        mode: ``"offline"`` or ``"live"``.
        n_runs: Number of runs for variance estimation.
        seed: Random seed.
        output_dir: Optional output directory.
        eval_fn: Optional custom evaluation function.
        **kwargs: Additional pipeline config overrides.

    Returns:
        :class:`BenchmarkResults`.
    """
    config = ExperimentConfig(
        name=f"benchmark_{benchmark.replace('/', '_')}",
        benchmarks=[BenchmarkSpec(name=benchmark, n_samples=n_samples)],
        mode=mode,
        n_runs=n_runs,
        seed=seed,
        output_dir=output_dir or "",
        pipeline=PipelineSweepSpec(
            config_overrides=[kwargs] if kwargs else [],
        ),
    )

    exp = Experiment(config=config, eval_fn=eval_fn)
    try:
        return await exp.run()
    finally:
        # Milvus Lite's threads are not daemons; without this a script that
        # ends with run_benchmark() never exits. A later call reopens it.
        from ..api.rag_backend import close_backend

        close_backend()
