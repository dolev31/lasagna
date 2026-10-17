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
import functools
import json
import os
import re
import warnings
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from loguru import logger

from .benchmarks import BenchmarkSample, benchmark_registry
from .benchmarks._base import BaseBenchmarkAdapter as _BaseBenchmarkAdapter
from .checkpoint import CheckpointManager
from .results import BenchmarkResults, primary_metric
from .runner import HARNESS_KEYS, BenchmarkRunner, BenchmarkTask, TaskResult
from ..utils.llm_cache import CacheStats, LLMCacheSettings, llm_cache_scope
from .cost import PriceTable

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


@functools.lru_cache(maxsize=1)
def _ingestion_time_keys() -> frozenset:
    """Every setting that changes what gets ingested: the list above plus each
    setting the stage map files under chunking, embedding or document
    processing.

    The list alone went stale: settings added with new chunkers
    (semantic_breakpoint_method, chunk_unit, ...) were missing, so an arm
    sweeping only one of them reused the control's corpus and measured nothing.
    """
    from lasagna.core.config import RAGConfig, rag_config

    from .stages import stage_of

    known = set(getattr(rag_config, "_config", {}) or {})
    known |= {n for n, v in vars(RAGConfig).items() if isinstance(v, property)}
    derived = {k for k in known if stage_of(k) in ("chunker", "embedder", "document_processor")}
    return frozenset(_INGESTION_TIME_KEYS | derived)


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
from lasagna.core.strict import (  # noqa: E402
    ContextPresentation,
    lost_in_the_middle,
    observe_reranking,
    present_context_order,
)

# There was a `_get_cost_rates` here, and a `_DEFAULT_COST_TABLE` behind it,
# converting token counts to dollars. Both are gone. The token counts they
# multiplied were always zero -- read from a `chunk.metadata` key nothing ever
# wrote -- so every dollar figure this package ever printed was `$0.00000`, and
# the rates were applied at the *judge's* price regardless of which model did
# the work. Prices are also not a property of a run: they change, they differ by
# tier and commitment, and a rate frozen into a published bundle is
# indistinguishable from a measurement to anyone reading it later. Tokens are
# reported; multiply by what you actually pay. A sweep that asks for a dollar
# figure (`prices:`) gets `cost_usd_estimate` from `cost.py`, labelled as an
# estimate, with the table and rates recorded beside it.


def _is_abstention_answer(text: str) -> bool:
    """Quick check whether an answer is an abstention (no LLM call needed)."""
    if not text or not text.strip():
        return True
    if text.strip().strip(".!").strip().lower() in ("unanswerable", "no answer"):
        return True  # the reply _ABSTAIN_INSTRUCTION asks for
    from .answer_metrics import is_abstention

    return is_abstention(text)


#: Added to the answer prompt only for a benchmark that has unanswerable
#: questions (SQuAD v2), so every other benchmark's prompt is unchanged.
_ABSTAIN_INSTRUCTION = "If the context does not contain the answer, reply exactly: unanswerable. "


def _is_unanswerable(sample: Any) -> bool:
    """Whether the benchmark marks this question as having no answer.

    Marked by the adapter, not inferred from an empty ``ground_truth``: an empty
    answer elsewhere is missing data, while SQuAD v2's unanswerable half is the
    skill the benchmark exists to measure. Both used to be dropped, so a
    system that never abstains scored EM 1.000 where the official rule gives
    0.505.
    """
    return bool((getattr(sample, "metadata", None) or {}).get("unanswerable"))


def _no_answer_em(answer: str) -> float:
    """SQuAD v2's rule for an unanswerable question, made explicit: correct only
    if the system abstained. An empty answer here means generation failed (it
    is counted as a failure), so it does not count as an abstention."""
    return 1.0 if answer.strip() and _is_abstention_answer(answer) else 0.0


def _answerability_split(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """The official script's has_ans / no_ans breakdown, when there are both."""
    no = [r for r in rows if r.get("_answerable") == 0.0 and "exact_match" in r]
    if not no:
        return {}
    has = [r for r in rows if r.get("_answerable") == 1.0 and "exact_match" in r]
    out: Dict[str, float] = {"n_unanswerable": float(len(no))}
    for name, group in (("has_ans", has), ("no_ans", no)):
        if group:
            out[f"exact_match_{name}"] = sum(r["exact_match"] for r in group) / len(group)
            out[f"token_f1_{name}"] = sum(r.get("token_f1", 0.0) for r in group) / len(group)
            # Abstaining on an answerable question is over-refusal; on an
            # unanswerable one it is RGB's negative rejection.
            out.update(
                {f"{k}_{name}": v for k, v in _mean_present(group, ("abstention_rate",)).items()}
            )
    return out


#: Per-query answer metrics beside exact match and token F1, each averaged over
#: the rows that carry it. A row without one was not scored for it (no
#: reference, no citations, a judge that could not decide), which is not 0.
_ANSWER_METRICS = (
    "rouge_l",
    "contains_answer",
    "abstention_rate",
    "answer_similarity",
    "citation_recall",
    "citation_precision",
)


def _mean_present(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    """The mean of each key over the rows that have it; keys no row has are left out."""
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in rows if key in r]
        if values:
            out[key] = sum(values) / len(values)
    return out


def _answer_scores(answer: str, sample: Any) -> Dict[str, float]:
    """ROUGE-L, answer containment and abstention for one generated answer.

    The same branches as exact match and token F1 beside them: an
    unanswerable question is right only if the system abstained, and an empty
    answer (generation failed) scores 0. Abstention is left out for an empty
    answer, since nothing was chosen.
    """
    from .answer_metrics import contains_answer, rouge_l

    gt = getattr(sample, "ground_truth", "") or ""
    aliases = getattr(sample, "ground_truth_answers", None) or []
    if _is_unanswerable(sample):
        out = dict.fromkeys(("rouge_l", "contains_answer"), _no_answer_em(answer))
    elif gt.strip() and answer.strip():
        out = {
            "rouge_l": rouge_l(answer, gt, aliases=aliases),
            "contains_answer": contains_answer(answer, gt, aliases=aliases),
        }
    else:
        out = {"rouge_l": 0.0, "contains_answer": 0.0}
    if answer.strip():
        out["abstention_rate"] = 1.0 if _is_abstention_answer(answer) else 0.0
    return out


def _add_answer_similarity(
    rows: List[Dict[str, Any]], answers: List[str], samples: List[Any]
) -> None:
    """``answer_similarity`` on each row that has a reference, in one batch.

    Left out, with a warning, when the embedder cannot be loaded (offline with
    no cached copy) or ``RAG_EVAL_SIMILARITY_MODEL=none``: the rest of the
    task's metrics do not depend on it.
    """
    from .answer_metrics import answer_similarity, load_similarity_encoder, similarity_model_name

    model = similarity_model_name()
    if model.lower() in ("none", "off", "false", "0"):
        return
    references = [
        []
        if _is_unanswerable(s)
        else [
            getattr(s, "ground_truth", "") or "",
            *(getattr(s, "ground_truth_answers", None) or []),
        ]
        for s in samples
    ]
    try:
        scores = answer_similarity(answers, references, load_similarity_encoder(model))
    except Exception as exc:
        logger.warning(
            f"answer_similarity not computed: could not run embedder {model!r} "
            f"[{type(exc).__name__}]: {exc}"
        )
        return
    for row, score in zip(rows, scores):
        if score is not None:
            row["answer_similarity"] = score


def _add_citation_scores(
    rows: List[Dict[str, Any]], answers: List[str], contexts: List[List[str]]
) -> Optional[str]:
    """ALCE citation recall and precision on the rows whose answer cites.

    ``[n]`` refers to the n-th context the generator was shown (``contexts``,
    the list RAGAS judges). An answer with no citation gets neither key, so a
    generator that does not cite reports no citation metric rather than 0.
    Returns a failure reason when answers cite and the judge scored none.
    """
    from .answer_metrics import citation_precision_recall, has_citations

    cited = [i for i, a in enumerate(answers) if has_citations(a)]
    if not cited:
        return None
    judge = _build_judge_llm()
    if judge is None:
        logger.warning(
            f"{len(cited)} answers cite passages but no judge could be built; "
            f"citation_recall/precision are not reported. Set OPENAI_API_KEY (or "
            f"RAG_EVAL_JUDGE_API_KEY), and RAG_EVAL_JUDGE_MODEL to choose the judge."
        )
        return None
    from .judge import entailment_judge

    entails = entailment_judge(judge)
    scored = 0
    for i in cited:
        result = citation_precision_recall(answers[i], contexts[i], entails)
        if result is not None:
            rows[i].update(result)
            scored += 1
    if scored < len(cited):
        logger.warning(
            f"citation metrics: the judge scored {scored} of {len(cited)} citing answers"
        )
    return None if scored else "judge produced no usable scores for citation_recall"


def _per_query_costs(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """System tokens and LLM calls per query, as named aggregates.

    Summed over a query's rows first (an end-to-end query has a retrieval row
    and a generation row, two stages of one answer), then averaged over
    queries. Named ``*_per_query`` because no single row carries that sum.
    """
    totals: Dict[Any, List[float]] = {}
    for i, row in enumerate(rows):
        if "llm_calls" not in row:
            continue
        acc = totals.setdefault(row.get("_sample_id", i), [0.0, 0.0, 0.0])
        acc[0] += float(row.get("input_tokens", 0.0))
        acc[1] += float(row.get("output_tokens", 0.0))
        acc[2] += float(row["llm_calls"])
    if not totals:
        return {}
    n = len(totals)
    return {
        "input_tokens_per_query": sum(t[0] for t in totals.values()) / n,
        "output_tokens_per_query": sum(t[1] for t in totals.values()) / n,
        "llm_calls_per_query": sum(t[2] for t in totals.values()) / n,
    }


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
    """Specifies a single benchmark to evaluate.

    ``path`` names your own data instead of a registered benchmark: a JSONL or
    CSV file, or a BEIR folder (``docs/own-data.md``). ``generate_tasks``
    registers it and replaces ``name`` with ``local/<stem>@<content hash>``, or
    ``<name>@<content hash>`` when ``name`` is given.
    """

    name: str = ""
    split: str = "test"
    n_samples: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    path: str = ""


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
                    f"it. Use the flag that disables the stage instead, for "
                    f"example `enable_reranking` rather than `reranker`."
                )
            if name not in flags:
                from lasagna.core._text import did_you_mean

                suggestions = sorted(f for f in flags if f.startswith(("enable_", "use_")))
                raise ValueError(
                    f"{role}={name!r} is not a known boolean setting"
                    f"{did_you_mean(str(name), flags)}. "
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
    generators: List[str] = field(default_factory=list)
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
    judge_model: str = ""  # LLM model for RAGAS judge (default: DEFAULT_JUDGE_MODEL)
    ingestion_per_config: bool = False  # True = each config variant gets its own corpus session
    max_corpus_docs: int = 0  # 0 = no limit; >0 = cap corpus size (e.g. 500 for pilot runs)
    #: The share of a corpus's documents that may fail to ingest (0 to 0.1)
    #: before every task on it fails. None by default: a partial corpus scores
    #: lower and the number would read as the method's.
    ingest_failure_tolerance: float = 0.0
    #: Evaluation tracks to run: "retrieval", "generation", or both (the
    #: default). A component sweep scored on retrieval needs no LLM or RAGAS.
    tracks: List[str] = field(default_factory=lambda: ["retrieval", "generation"])
    #: The LLM response cache (``llm_cache:``): on unless set false, the CLI
    #: passes ``--no-cache`` or ``LASAGNA_LLM_CACHE=0``. Replays deterministic
    #: calls only; see :mod:`lasagna.utils.llm_cache`.
    llm_cache: LLMCacheSettings = field(default_factory=LLMCacheSettings)
    #: Rates for a dollar estimate (``prices:``, opt-in): ``builtin`` or a
    #: mapping of model to rates over the built-in table. None, the default,
    #: reports tokens only. See :mod:`lasagna.knowledge_evaluation_sdk.cost`.
    prices: Optional[PriceTable] = None
    #: Folder of the sweep file, where `module:function` components are found.
    #: Set by `Experiment.from_file`; not a YAML key.
    base_dir: str = ""
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
                _results_root(), f"{_slug(self.name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        return self._resolved_output_dir

    def latest_output_dir(self) -> Optional[str]:
        """The newest default output directory of this experiment that holds
        checkpoints, for ``--resume``; None when there is nothing to resume.

        Each run without ``output_dir`` writes to a new timestamped folder, so
        ``--resume`` used to look in a fresh, empty one and re-run every task.
        """
        root = Path(_results_root())
        if not root.is_dir():
            return None
        prefixes = {f"{_slug(self.name)}_", f"{self.name}_"}  # older runs kept the raw name
        candidates = [
            d
            for d in root.iterdir()
            if d.is_dir()
            and any(
                d.name.startswith(p) and re.fullmatch(r"\d{8}_\d{6}", d.name[len(p) :])
                for p in prefixes
            )
            and any((d / ".checkpoints").glob("*.json"))
        ]
        if not candidates:
            return None
        return str(max(candidates, key=lambda d: d.name[-15:]))

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
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            # A ValueError, so the CLI prints this one line rather than a
            # 164-line traceback of the parser.
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            problem = getattr(exc, "problem", None) or str(exc)
            raise ValueError(f"not valid YAML{where}: {problem}") from None


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


#: The setting that names the model a built-in reranker component loads.
_RERANKER_MODEL_KEYS = {
    "colbert": "colbert_model",
    "llm_listwise": "llm_reranker_model",
    "llm_pointwise": "llm_reranker_model",
    "monot5": "monot5_model",
    "flashrank": "flashrank_model",
    "cohere": "cohere_rerank_model",
}


def _reranker_that_ran(component: Optional[str], task_config: Dict[str, Any]) -> Any:
    """The reranker a task ran, down to the model where the component has one.

    Two colbert arms with different checkpoints both recorded ``colbert``, so
    the held-fixed block could not tell them apart.
    """
    if not component or component in _CROSS_ENCODER_RERANKERS:
        return task_config.get("reranker_model")
    if component in ("ensemble", "ensemble_adapted"):
        models = task_config.get("ensemble_reranker_models") or [task_config.get("reranker_model")]
        return "ensemble:" + "+".join(str(m) for m in models)
    key = _RERANKER_MODEL_KEYS.get(component)
    model = task_config.get(key) if key else None
    return f"{component}:{model}" if model else component


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
    from .stages import CHUNKER_DISPATCH_KEYS, effective_chunking_strategy

    chunker = effective_chunking_strategy({k: _ingest(k) for k in CHUNKER_DISPATCH_KEYS})
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
        "reranker_model": _reranker_that_ran(reranker_component, task_config)
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


#: Documents a corpus census reads at the end of each task (all, up to this).
_CENSUS_DOCUMENTS = 2_000


def _corpus_census(
    backend: Any, collection_name: str, session_id: str, document_ids: Any
) -> Dict[str, Any]:
    """What a published corpus holds, to compare at the end of each task.

    The documents (all, or an evenly spread ``_CENSUS_DOCUMENTS`` of them)
    and their chunks: a corpus another process deleted, or ingested again on
    top, differs in one or the other. A few queries, as the pre-flight asks.
    """
    ids = sorted(document_ids)
    if len(ids) > _CENSUS_DOCUMENTS:
        step = len(ids) / _CENSUS_DOCUMENTS
        ids = [ids[int(i * step)] for i in range(_CENSUS_DOCUMENTS)]
    present, chunks = _count_present(backend, collection_name, session_id, ids)
    return {
        "collection": collection_name,
        "documents": ids,
        "present": present,
        "chunks": chunks,
    }


def _count_present(backend: Any, collection_name: str, session_id: str, ids: List[str]) -> tuple:
    """``(documents with chunks, chunks)`` among ``ids`` in this session."""
    from lasagna.vector_backends.filters import FilterExpression as _FE

    documents: set = set()
    chunks = 0
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
        hits = [r.get("document_id") for r in rows if r.get("document_id")]
        documents.update(hits)
        chunks += len(hits)
    return len(documents), chunks


def _corpus_changed(corpus_info: Dict[str, Any]) -> Optional[str]:
    """Why the corpus a task searched is not the one published to it, or None.

    Another process on the same vector store (a Milvus or Qdrant server
    that two experiments share) could delete the corpus, or ingest it again
    on top, while this one evaluated against it, and the numbers it recorded
    described neither.
    """
    census = corpus_info.get("corpus_census")
    store = corpus_info.get("_census_store")
    if not census or store is None:
        return None
    try:
        present, chunks = _count_present(
            store(), census["collection"], corpus_info["session_id"], census["documents"]
        )
    except NotImplementedError:
        return None  # a store that cannot be asked; nothing was counted at the start either
    except Exception as exc:  # noqa: BLE001 - unknown is not "unchanged"
        return (
            f"The corpus of session {corpus_info['session_id']} could not be counted at the "
            f"end of this task ({type(exc).__name__}: {exc}), so whether it changed while "
            f"this run used it is unknown; its numbers are not recorded."
        )
    if (present, chunks) == (census["present"], census["chunks"]):
        return None
    return (
        f"The corpus of session {corpus_info['session_id']} changed while this run used it: "
        f"{present} of {len(census['documents'])} documents checked have chunks now "
        f"({chunks} chunks), against {census['present']} ({census['chunks']} chunks) when it "
        f"was published. Another process on the same vector store deleted it or ingested "
        f"it again, so this task's numbers are not recorded. Run one experiment on a "
        f"corpus at a time per vector store, then run this one again."
    )


class _CorpusLock:
    """The corpus's store lock, held from the check of what is stored until it is published.

    ``corpus:<session>:<collection>`` in the metadata store: on PostgreSQL an
    advisory lock every host shares, on SQLite a lock file for the processes
    of one host. Without it a second experiment process took the first's
    ingest in progress for a stale partial corpus and deleted its vectors.
    Waited for without a limit (an ingest takes hours), saying so once a
    minute; a process that dies releases it. Milvus Lite admits one process
    per store, so there it is never contended. Machines that share a vector
    store but not a metadata store are not covered by it; the census at the
    end of each task (``_corpus_changed``) catches what changed under them.
    """

    def __init__(self):
        self._held = None

    async def take(self, session_id: str, collection: str) -> None:
        from lasagna.core.store_locks import StoreLockTimeout, store_lock

        name = f"corpus:{session_id}:{collection}"
        started, said = time.monotonic(), None
        while True:
            lock = store_lock(name, timeout=0)
            try:
                lock.__enter__()
            except StoreLockTimeout:
                now = time.monotonic()
                if said is None or now - said >= 60:
                    logger.warning(
                        f"Waiting for another process ingesting the corpus of session "
                        f"{session_id} into {collection!r} ({now - started:.0f} s so far); "
                        f"this run uses that corpus once it is published."
                    )
                    said = now
                await asyncio.sleep(1.0)
                continue
            self._held = lock
            return

    def release(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            held.__exit__(None, None, None)


def _check_components_exist(pipeline_configs: List[Dict[str, Any]]) -> None:
    """Refuse a sweep that names a component the registry does not have.

    Checked before any corpus is ingested. The command line only sees
    components installed as ``lasagna_components`` entry points (or built in);
    one defined in a script the CLI never imports is not registered.
    """
    from lasagna.core.registry import registry

    unknown = []
    likely_typo = False
    for pc in pipeline_configs:
        for key, value in (pc.get("overrides") or {}).items():
            if key in _COMPONENT_KEYS and isinstance(value, str):
                known = registry.list(key).get(key, [])
                if not registry.has(key, value):
                    hint = _did_you_mean(value, known)
                    likely_typo = likely_typo or bool(hint)
                    unknown.append(
                        f"{pc.get('name')}: {key} {value!r}{hint} (registered: "
                        f"{', '.join(sorted(known))})"
                    )
    if unknown:
        # A misspelled built-in was told to install an entry point.
        advice = (
            "Correct the name, or name your own component by its import path "
            "(my_ideas:my_function)."
            if likely_typo
            else "Name your own component by its import path (my_ideas:my_function), "
            "or install it as a lasagna_components entry point "
            "(see docs/contributing-components.md)."
        )
        raise ValueError(
            "These arms name components that are not registered, so they would "
            "measure the default component instead: " + "; ".join(unknown) + ". " + advice
        )


def _check_benchmarks_exist(specs: List[BenchmarkSpec]) -> None:
    """Refuse a benchmark name nobody registered, before any task starts.

    ``benchmarks: [{name: smok}]`` printed "Starting experiment", then failed
    every task with a quoted ``KeyError`` listing every benchmark, and the dry
    run printed that list three times and exited 0. Every unknown name is
    named once, with the closest registered one. Entries given as a ``path``
    are registered before this runs, so their names are known.
    """
    from .benchmarks import benchmark_registry

    known = benchmark_registry.names()
    unknown = [spec.name for spec in specs if spec.name not in known]
    if unknown:
        names = ", ".join(f"{name!r}{_did_you_mean(name, known)}" for name in unknown)
        noun = "benchmark" if len(unknown) == 1 else "benchmarks"
        raise ValueError(f"Unknown {noun} {names}. `lasagna benchmark list` shows them all.")


def _check_profiles(pipeline_configs: List[Dict[str, Any]], base_profile: str = "") -> None:
    """Refuse an unknown profile name before any corpus is ingested.

    A misspelled ``--profile`` resolved to no ingestion defaults, so the run
    fell back to the settings-file embedder and failed much later on that
    model's download ("Could not load any embedding model ... bge-m3"),
    naming the wrong problem. The message names the key the name came from:
    a ``base_profile`` typo was reported "in arm 'reranker=cross_encoder'",
    an arm that names no profile.
    """
    from lasagna.core.session_config._profiles import ProfileMixin

    known = set(ProfileMixin._PROFILE_SPECS) | {"default"}
    profiles = f"Profiles: {', '.join(sorted(known))}."
    base = base_profile or "standard"
    if base not in known:
        raise ValueError(
            f"Unknown profile {base!r}{_did_you_mean(base, known)} in the sweep's "
            f"pipeline.base_profile. {profiles}"
        )
    for pc in pipeline_configs:
        name = pc.get("profile") or "standard"
        if name not in known:
            raise ValueError(
                f"Unknown profile {name!r}{_did_you_mean(name, known)} in the profile: "
                f"key of arm {pc.get('name')!r}. {profiles}"
            )


def _known_override_keys() -> set:
    """Every key an arm can set: settings, component slots and harness keys."""
    from lasagna.core.config import RAGConfig, rag_config

    keys = set(getattr(rag_config, "_config", {}) or {})
    keys |= {n for n, v in vars(RAGConfig).items() if isinstance(v, property)}
    return keys | set(_COMPONENT_KEYS) | set(HARNESS_KEYS) | _COMPONENT_SETTING_KEYS


#: Settings read by one component only, with no global default.
_COMPONENT_SETTING_KEYS = frozenset({"ensemble_reranker_models", "ensemble_reranker_weights"})


def _check_override_keys(pipeline_configs: List[Dict[str, Any]]) -> None:
    """Refuse an arm setting a key nothing reads, before any corpus is ingested.

    ``enable_rerankng: true`` reached no stage, so the arm ran the control's
    pipeline under its own name; the isolation check could not see it, since
    an unknown key belongs to no stage whose output it could compare.
    """
    known = _known_override_keys()
    for pc in pipeline_configs:
        unknown = sorted(set(pc.get("overrides") or {}) - known)
        if not unknown:
            continue
        # The likely key, and where the rest are listed: printing all ~400 known
        # keys buried it among celery_broker_url and the like.
        parts = ", ".join(f"{k!r}{_did_you_mean(k, known)}" for k in unknown)
        raise ValueError(
            f"Unknown key(s) in arm {pc.get('name')!r}: {parts}. An arm names components "
            f"({', '.join(sorted(_COMPONENT_KEYS))}), harness keys "
            f"({', '.join(sorted(HARNESS_KEYS))}) or settings, listed in "
            f"docs/reference/settings.md."
        )


def _check_override_values(pipeline_configs: List[Dict[str, Any]]) -> None:
    """Refuse an arm whose settings the pipeline would refuse, before any corpus is ingested.

    An arm's settings are validated when its task applies them, so
    ``chunk_size: -5`` and ``hybrid_alpha: 7`` passed the dry run, failed two
    tasks of three mid-sweep, and the run still printed a "Next:" step. The
    same validator now checks every arm first (``validate_config``: the ranges
    and types in ``configurator/schema.py``, which the environment's ranges
    match), over the profile's settings as the task merges them. It checks
    validity only: the caps on what a session may store are not applied, so
    an arm may go past them.
    """
    import copy

    from lasagna.configurator.validation import validate_config
    from lasagna.core.pipeline import KnowledgePipeline

    profiles: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    for pc in pipeline_configs:
        settings = {
            k: v
            for k, v in (pc.get("overrides") or {}).items()
            if k not in _COMPONENT_KEYS and k not in HARNESS_KEYS
        }
        if not settings:
            continue
        profile = pc.get("profile") or "standard"
        if profile not in profiles:
            profiles[profile] = dict(KnowledgePipeline.from_profile(profile)._config)
        merged = copy.deepcopy(profiles[profile])
        merged.update(settings)
        result = validate_config(merged)
        problems.extend(
            f"arm {pc.get('name')!r}: {e['key']}: {e['message']}" for e in result.errors
        )
    if problems:
        raise ValueError(
            "These arms have settings the pipeline refuses, so their tasks would fail: "
            + "; ".join(problems)
            + ". Fix them in the sweep file; docs/reference/settings.md gives each "
            "setting's range."
        )


def _refuse_a_held_store() -> None:
    """Raise before a run makes its folder when another process holds the local store.

    With `lasagna serve` running, every task failed when it opened the store,
    after the run had printed its output folder. The CLI's benchmark commands
    move to a store of their own when the default one is held; a run started
    from Python is told how to do the same.
    """
    from lasagna.core.local_server import local_store_held, local_store_path

    if local_store_held():
        raise RuntimeError(
            f"Another process is using the local store at {local_store_path()} (a running "
            f"`lasagna serve`, an MCP server or another benchmark), so this experiment "
            f"cannot open it. Stop that process, or give the experiment a store of its "
            f"own: set RAG_DB_DIR=<another directory> before Python starts. `lasagna "
            f"benchmark experiment` does this itself when the default store is held."
        )


def _load_component_references(
    pipeline_configs: List[Dict[str, Any]], search_paths: List[str]
) -> None:
    """Import and register every ``module:function`` an arm names.

    One line in the sweep (``rerankers: ["my_ideas:keyword_overlap"]``) is the
    whole integration: no package, no entry point, no registration call.
    """
    from lasagna import components

    for pc in pipeline_configs:
        for key, value in (pc.get("overrides") or {}).items():
            if key in _COMPONENT_KEYS and components.is_reference(value):
                try:
                    components.load(key, value, search_paths)
                except Exception as exc:
                    raise ValueError(
                        f"{pc.get('name')}: cannot load {key}={value!r}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc


def _check_components_conform(pipeline_configs: List[Dict[str, Any]]) -> None:
    """Refuse an arm whose component would not be measured like-for-like.

    Two checks, before any corpus is ingested. The component must implement
    its stage's interface, or the comparison also measures an interface
    difference. And a component that is not a built-in must pass its stage's
    behavioural contract (return types, top_k, empty input, no mutation of what
    it was handed), or it fails mid-run with a number already half-recorded.
    """
    from lasagna.core.registry import _get_expected_abc, registry
    from lasagna.core.testing import ComponentTestHarness

    checked: set = set()
    problems: List[str] = []
    for pc in pipeline_configs:
        for key, value in (pc.get("overrides") or {}).items():
            if key not in _COMPONENT_KEYS or not isinstance(value, str):
                continue
            if (key, value) in checked:
                continue
            checked.add((key, value))
            cls = registry.get_class(key, value)
            abc = _get_expected_abc(key)
            if abc is not None and not (isinstance(cls, type) and issubclass(cls, abc)):
                problems.append(
                    f"{key}={value!r} does not implement {abc.__name__}; subclass it, or "
                    f"register a plain function with @lasagna.{key}"
                )
                continue
            if registry.is_builtin(key, value) or key == "agentic_pipeline":
                continue
            try:
                report = ComponentTestHarness().validate(registry.get(key, value))
                failures = [] if report.passed else list(report.failures)
            except Exception as exc:  # a crash is a failed contract, not a crash here
                failures = [f"raised {type(exc).__name__}: {exc}"]
            if failures:
                problems.append(f"{key}={value!r} fails its contract: {'; '.join(failures)}")
    if problems:
        raise ValueError("These components cannot be measured: " + " | ".join(problems))


def _check_no_rag_arms(pipeline_configs: List[Dict[str, Any]], tracks: List[str]) -> None:
    """A ``no_rag`` arm needs a generator that reads no passages and a
    generation track to be scored on; refused before anything is ingested."""
    for pc in pipeline_configs:
        overrides = pc["overrides"]
        if not overrides.get("no_rag"):
            continue
        if "generation" not in tracks:
            raise ValueError(
                f"Arm {pc['name']!r} sets no_rag: true, which is scored on generation "
                f"only, and this sweep runs tracks: {list(tracks)}. Add generation to "
                f"tracks:, or remove the arm."
            )
        if overrides.get("generator"):
            from lasagna.core.registry import registry

            _no_rag_generator(registry.get("generator", overrides["generator"]), overrides)


def _fix_retrieval_depth(task_config: Dict[str, Any], depth: int) -> None:
    """Make retrieval evaluation return ``depth`` uncached, untruncated results, in rank order.

    Answer-time policies are right for answering and wrong for measuring
    retrieval at k: the search cache can serve a different arm's lists, the
    relative similarity threshold and adaptive-k cut lists well below k, and
    default_top_k can cap the candidates below the requested depth. Context
    reordering ("lost in the middle") is how the generator is shown its
    passages, not a ranking: scored, it moved the second-ranked document to
    the end of the list, below the cut to k.
    """
    task_config["enable_search_cache"] = False
    task_config["enable_context_reordering"] = False
    task_config["similarity_threshold"] = 0.0
    ccr = dict(task_config.get("chunk_cite_retrieval") or {})
    ccr["strategy"] = "fixed"
    task_config["chunk_cite_retrieval"] = ccr
    task_config["default_top_k"] = max(int(task_config.get("default_top_k") or 0), depth)


def _context_order(gen_config: Any) -> Dict[str, str]:
    """The order each track reads its results in, for the stage manifest.

    ``ranked`` is the retriever's (or reranker's) order: the retrieval track
    always scores it. ``lost_in_the_middle`` is context reordering's, best
    passages at both ends of the generator's context, which the generation
    track uses when the arm turns reordering on. A function of the arm's
    setting alone, so an arm that skips a track or reads no passages digests
    like its control in this stage.
    """
    reordered = bool(getattr(gen_config, "enable_context_reordering", False))
    return {
        "retrieval": "ranked",
        "generation": "lost_in_the_middle" if reordered else "ranked",
    }


#: Retrievers that return only passages sharing a term with the query.
_LEXICAL_RETRIEVERS = frozenset({"bm25", "splade", "sparse"})


def _short_retrieval(config: Dict[str, Any], retriever: Optional[str] = None) -> str:
    """What ran out of candidates in a short list, named for the arm that ran.

    The warning blamed "BM25 and SPLADE" on a dense speed run, where neither
    ran: there the corpus held fewer chunks than were requested.
    """
    if retriever:
        if retriever.lower() in _LEXICAL_RETRIEVERS:
            return (
                f"the {retriever} retriever returns only passages that share a term with the query"
            )
        return f"the {retriever} retriever returned all it found"
    if config.get("enable_hybrid_search"):
        lexical = (
            "SPLADE" if str(config.get("sparse_method", "bm25")).lower() == "splade" else "BM25"
        )
        return (
            f"hybrid search returns the dense and {lexical} candidates it finds, and the "
            f"corpus held fewer than were requested"
        )
    return (
        "dense search returns at most the chunks the corpus holds, and it held fewer "
        "than were requested (or similarity_threshold dropped some)"
    )


def _short_list_warning(
    short: int,
    exhausted: int,
    n_queries: int,
    k: int,
    overfetch: int,
    ran_short: str = "",
) -> str:
    """Why ``short`` queries scored fewer than ``k`` documents, and what would help.

    The warning used to blame merged chunks and advise raising the over-fetch
    for every short list. A lexical retriever returns only passages that share
    a term with the query, so a one-word NFCorpus query can match three
    documents in the whole corpus; no over-fetch changes that, and the advice
    made a correct BEIR number look deflated. ``ran_short`` names the
    retrieval that ran out (:func:`_short_retrieval`).
    """
    parts = [
        f"{short} of {n_queries} queries had fewer than {k} distinct documents, "
        f"which lowers @{k} metrics."
    ]
    if exhausted:
        why = ran_short or "the retriever found no more candidates"
        parts.append(
            f"For {exhausted} of them the retriever returned fewer results than were "
            f"requested: it ran out of candidates ({why}), and over-fetching does "
            f"not change that."
        )
    if short - exhausted:
        parts.append(
            f"For {short - exhausted} of them the full request came back and merged "
            f"into fewer documents; raise retrieval_chunk_overfetch (now "
            f"{overfetch}) if the corpus has more documents than that."
        )
    return " ".join(parts)


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
        "ingest_failure_tolerance",
        "tracks",
        "llm_cache",
        "prices",
    }
)
_PIPELINE_KEYS = frozenset(
    {
        "base_profile",
        "rerankers",
        "embedders",
        "chunkers",
        "query_enhancers",
        "generators",
        "config_overrides",
        "factorial",
    }
)


_FACTORIAL_KEYS = frozenset({"factor_a", "factor_b", "held_fixed"})
_BENCHMARK_KEYS = frozenset({"name", "split", "n_samples", "filters", "path"})


def runs_were_identical(task_results: Iterable[Any]) -> bool:
    """Whether every arm's repeated runs produced the same per-query scores.

    Retrieval with fixed models is deterministic, and significance tests pair
    by query, so then ``n_runs > 1`` multiplied the wall time and added no
    information. It matters when a stage samples: an LLM or an agentic step.
    """
    groups: Dict[tuple, List[Any]] = {}
    for r in task_results:
        if getattr(r, "error", None):
            continue
        groups.setdefault((r.benchmark_name, r.config_name), []).append(r)
    repeated = [g for g in groups.values() if len(g) > 1]
    if not repeated:
        return False

    def scores(r):
        return [
            {k: v for k, v in s.items() if not k.startswith("_") and "latency" not in k}
            for s in (r.per_sample_scores or [])
        ]

    return all(all(scores(r) == scores(g[0]) for r in g[1:]) for g in repeated)


def nothing_samples(task_results: Iterable[Any], tracks: Iterable[str]) -> bool:
    """Whether repeating this run would give the same numbers, by construction.

    True when only the retrieval track ran and no task made an LLM call (no
    agentic step, LLM reranker or query rewriter): retrieval with fixed models
    is deterministic. When the run has repeats, they must also have come out
    identical (`runs_were_identical`, the end-of-run note). Recorded as the
    run's ``nothing_samples``, so a single run of such a pipeline is a complete
    result: the docs recommend ``n_runs: 1`` for it, and the bundle's
    ``SINGLE_RUN`` warning, which ``--strict`` and the hub refused, is for runs
    with an LLM stage in the loop.
    """
    done = [r for r in task_results if not getattr(r, "error", None)]
    if not done or set(tracks or ()) != {"retrieval"}:
        return False
    if any(getattr(r, "system_llm_calls", 0) or getattr(r, "eval_llm_calls", 0) for r in done):
        return False
    groups: Dict[tuple, int] = {}
    for r in done:
        key = (r.benchmark_name, r.config_name)
        groups[key] = groups.get(key, 0) + 1
    return runs_were_identical(done) if any(n > 1 for n in groups.values()) else True


def _results_root() -> str:
    """``<data dir>/research/results``: it ignored ``LASAGNA_DATA_DIR`` and
    stayed under the home directory."""
    from lasagna.core import paths

    return str(paths.results_dir())


def _slug(name: str) -> str:
    """A folder name a shell can take unquoted: the free-text experiment name
    ("does it help?") made the printed path fail in zsh as a glob."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return slug or "experiment"


# Moved to lasagna.core._text so the CLI can check names before importing this
# module; kept here under its old name for the callers below.
from lasagna.core._text import did_you_mean as _did_you_mean  # noqa: E402


def _refuse_unknown(keys: Iterable[str], known: Iterable[str], where: str) -> None:
    """Refuse keys a section does not read, naming the likely intended one.

    An unknown key used to be dropped, at most with a warning among hundreds of
    log lines: ``rerankrs: [...]`` gave a one-arm run of the stock pipeline with
    exit 0, which reads exactly like a score for the component and is not one.
    """
    known = set(known)
    unknown = sorted(set(keys) - known)
    if not unknown:
        return
    parts = []
    for key in unknown:
        if where == "the top level" and key in _PIPELINE_KEYS:
            parts.append(f"{key!r} (a pipeline setting: put it under `pipeline:`)")
        else:
            parts.append(f"{key!r}{_did_you_mean(key, known)}")
    raise ValueError(
        f"Unknown key(s) at {where}: {', '.join(parts)}. Known: {', '.join(sorted(known))}."
    )


#: Keys whose value has one type, with the least a number may be. Checked
#: while the file is read: ``n_runs: lots`` failed later with "'str' object
#: cannot be interpreted as an integer", naming neither the file nor the key.
_TYPED_KEYS: Dict[str, tuple] = {
    "name": (str, None),
    "description": (str, None),
    "mode": (str, None),
    "n_runs": (int, 1),
    "n_samples": (int, 1),
    "seed": (int, None),
    "max_concurrency": (int, 1),
    "max_corpus_docs": (int, 0),
    "output_dir": (str, None),
    "checkpoint_dir": (str, None),
    "judge_model": (str, None),
    "ingestion_per_config": (bool, None),
    "benchmarks": (list, None),
    "output_formats": (list, None),
    "split": (str, None),
}

_TYPE_WORDS = {int: "a whole number", str: "a string", bool: "true or false", list: "a list"}


def _check_types(data: Dict[str, Any], where: str) -> None:
    """Raise ``ValueError`` for a key in ``data`` whose value has the wrong type or size."""
    for key, (kind, least) in _TYPED_KEYS.items():
        value = data.get(key)
        if value is None:
            continue
        wrong = isinstance(value, bool) and kind is not bool or not isinstance(value, kind)
        if wrong:
            raise ValueError(
                f"`{key}:` {where} takes {_TYPE_WORDS[kind]}, got {value!r}"
                + (f" (at least {least})" if least is not None else "")
                + "."
            )
        if least is not None and value < least:
            raise ValueError(f"`{key}:` {where} must be at least {least}, got {value}.")


def _parse_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Parse a raw config dict into :class:`ExperimentConfig`."""
    if not isinstance(data, dict):
        # An empty file failed with "'NoneType' object is not iterable".
        what = (
            "the file is empty"
            if data is None
            else f"its top level is a {type(data).__name__}, not a mapping of keys"
        )
        raise ValueError(
            f"{what}. A sweep names at least `benchmarks:` and `pipeline:`; "
            "https://github.com/dolev31/lasagna/blob/main/examples/lasagna-plugin-template/sweep.yaml "
            "is a complete one."
        )
    _refuse_unknown(data, _TOP_LEVEL_KEYS, "the top level")
    _check_types(data, "at the top level")
    pipeline_raw = data.get("pipeline") or {}
    if not isinstance(pipeline_raw, dict):
        raise ValueError(f"`pipeline:` must be a mapping, got {type(pipeline_raw).__name__}")
    _refuse_unknown(pipeline_raw, _PIPELINE_KEYS, "`pipeline:`")
    if isinstance(pipeline_raw.get("factorial"), dict):
        _refuse_unknown(pipeline_raw["factorial"], _FACTORIAL_KEYS, "`pipeline.factorial:`")
    benchmarks = []
    for b in data.get("benchmarks", []):
        if isinstance(b, str):
            benchmarks.append(BenchmarkSpec(name=b))
        elif isinstance(b, dict):
            _refuse_unknown(
                b, _BENCHMARK_KEYS, f"benchmark {b.get('name') or b.get('path') or '?'!r}"
            )
            if "name" not in b and "path" not in b:
                raise ValueError(
                    f"A benchmark entry needs a `name` (a registered benchmark) or a "
                    f"`path` (a JSONL or CSV file or a BEIR folder of your own); got {b!r}"
                )
            if "path" in b and not (isinstance(b["path"], str) and b["path"].strip()):
                raise ValueError(f"A benchmark `path` must be a file or folder; got {b!r}")
            _check_types(b, f"in benchmark {b.get('name') or b.get('path')!r}")
            benchmarks.append(
                BenchmarkSpec(
                    name=b.get("name", ""),
                    split=b.get("split", "test"),
                    n_samples=b.get("n_samples"),
                    filters=b.get("filters", {}),
                    path=b.get("path", ""),
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
        generators=pipeline_data.get("generators", []),
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
        ingest_failure_tolerance=_parse_ingest_tolerance(data.get("ingest_failure_tolerance")),
        tracks=_parse_tracks(data.get("tracks")),
        llm_cache=LLMCacheSettings.parse(data.get("llm_cache")),
        prices=PriceTable.parse(data.get("prices")),
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
    unknown = sorted(set(tracks) - set(_TRACKS), key=str)
    if unknown or not tracks:
        from lasagna.core._text import did_you_mean

        named = "; ".join(f"{t!r} is not a track{did_you_mean(str(t), _TRACKS)}" for t in unknown)
        raise ValueError(
            f"tracks must be a non-empty subset of {list(_TRACKS)}; got {value!r}"
            + (f" ({named})" if named else "")
        )
    return [t for t in _TRACKS if t in tracks]


def _parse_ingest_tolerance(value: Any) -> float:
    """``ingest_failure_tolerance:``, a share of the corpus from 0 to 0.1."""
    from ._fast_ingest import _MAX_CHUNK_FAILURE_RATE

    if value is None:
        return 0.0
    try:
        share = float(value)
    except (TypeError, ValueError):
        share = -1.0
    if isinstance(value, bool) or not 0.0 <= share <= _MAX_CHUNK_FAILURE_RATE:
        raise ValueError(
            f"ingest_failure_tolerance is the share of corpus documents that may fail to "
            f"ingest, from 0 to {_MAX_CHUNK_FAILURE_RATE} (ingestion stops above that); "
            f"got {value!r}"
        )
    return share


def _ingest_gap(
    corpus_ids: Iterable[str], valid_to_corpus: Dict[str, str], failed_valid_ids: Iterable[str]
) -> Optional[Dict[str, Any]]:
    """What a corpus is missing after ingestion, or None when nothing is.

    The counts and a digest of the ingested ids go into the comparability key
    (``corpus_ingest_gap``), so a partial corpus never shares a leaderboard
    group with the complete one; the first failures are for the reader.
    """
    failed = sorted({valid_to_corpus.get(v, v) for v in failed_valid_ids})
    if not failed:
        return None
    every = sorted(set(corpus_ids))
    ingested = sorted(set(every) - set(failed))
    return {
        "documents": len(every),
        "ingested": len(ingested),
        "failed": len(failed),
        "ingested_digest": hashlib.sha256("\n".join(ingested).encode()).hexdigest()[:16],
        "first_failed": failed[:5],
    }


def _partial_corpus_refusal(gap: Optional[Dict[str, Any]], tolerance: float) -> Optional[str]:
    """Why a task on a partial corpus must not run, or None.

    Under evaluation (strict stages) a corpus missing more documents than the
    experiment's ``ingest_failure_tolerance`` allows fails the task: up to 10%
    of documents could fail to chunk, and the run scored the rest under the
    benchmark's name, and later runs reused that partial corpus.
    """
    from lasagna.core.strict import strict_stages

    from ._fast_ingest import _MAX_CHUNK_FAILURE_RATE

    if not gap or not strict_stages() or gap["failed"] <= tolerance * gap["documents"]:
        return None
    needed = -(-gap["failed"] * 1000 // gap["documents"]) / 1000  # rounded up
    accept = (
        f", or accept the partial corpus with `ingest_failure_tolerance: {needed}` in the "
        f"experiment file; its results then get a comparability key of their own"
        if needed <= _MAX_CHUNK_FAILURE_RATE
        else ""
    )
    return (
        f"{gap['failed']} of {gap['documents']} corpus documents could not be ingested "
        f"(first: {', '.join(gap['first_failed'])}; the log has each reason), so this task "
        f"would score a partial corpus. Fix those documents or the chunker and run "
        f"again{accept}."
    )


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
            for name in ("rerankers", "embedders", "chunkers", "query_enhancers", "generators")
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
    if spec.generators:
        dimensions["generator"] = spec.generators

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
                f"`{key}` contains a null entry. A null does not disable the stage: "
                f"it leaves the profile's default in place, so that arm may be "
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
        config.base_dir = str(p.resolve().parent)
        return cls(config=config, eval_fn=eval_fn)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], eval_fn: Optional[Callable] = None) -> "Experiment":
        """Create experiment from a raw config dict."""
        return cls(config=_parse_config(data), eval_fn=eval_fn)

    def generate_tasks(self) -> List[BenchmarkTask]:
        """Generate all benchmark tasks (Cartesian product)."""
        cfg = self.config
        # Entries given as a `path` are read, validated and registered under a
        # name carrying their content hash, before any model loads.
        from .benchmarks.adapters.local_files import prepare_local_benchmarks

        prepare_local_benchmarks(cfg.benchmarks, cfg.base_dir, cfg.tracks)
        _check_benchmarks_exist(cfg.benchmarks)
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
                f"Experiments cannot swap {unsweepable}: benchmark corpora are loaded "
                f"as text, so a document processor never runs on them and every arm "
                f"would measure the same corpus. Check one on your own files with "
                f"`lasagna component validate`."
            )
        if "agentic_pipeline" in all_override_keys:
            logger.warning(
                "agentic_pipeline arms change generation metrics only: retrieval "
                "metrics are measured on the standard tier for every arm."
            )
        _check_profiles(pipeline_configs, cfg.pipeline.base_profile)
        _check_override_keys(pipeline_configs)
        _check_override_values(pipeline_configs)
        _load_component_references(pipeline_configs, [cfg.base_dir] if cfg.base_dir else [])
        _check_components_exist(pipeline_configs)
        _check_components_conform(pipeline_configs)
        _check_no_rag_arms(pipeline_configs, cfg.tracks)

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
                        ingest_failure_tolerance=cfg.ingest_failure_tolerance,
                        variance_source=cfg.variance_source,
                        tracks=(tuple(cfg.tracks) if list(cfg.tracks) != list(_TRACKS) else None),
                    )
                    tasks.append(task)

        _set_resume_fingerprints(cfg, tasks)
        return tasks

    async def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BenchmarkResults:
        """Execute the experiment and return results.

        A run that stops before writing results leaves no empty folder behind
        (see :func:`_tidy_run_folder`).

        Args:
            progress_callback: Optional ``(completed, total, task_key) -> None``.

        Returns:
            :class:`BenchmarkResults` with all task outcomes.
        """
        try:
            return await self._run(progress_callback)
        finally:
            _tidy_run_folder(self.config)

    async def _run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BenchmarkResults:
        cfg = self.config
        _refuse_a_held_store()
        output_dir = cfg.effective_output_dir()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting experiment '{cfg.name}' "
            f"({len(cfg.benchmarks)} benchmarks, "
            f"{cfg.n_runs} {'run' if cfg.n_runs == 1 else 'runs'}, mode={cfg.mode})"
        )

        # Set judge model for offline eval (module-level so _eval_offline can read it)
        # RAG_EVAL_JUDGE_MODEL env var takes precedence over YAML judge_model so
        # operators can override the model without editing YAML files (e.g. for OpenRouter).
        global _active_judge_model
        _active_judge_model = _resolve_judge_model(cfg)
        if _active_judge_model:
            logger.info(f"Judge model: {_active_judge_model}")
        # Said once, before any work: a model that takes no temperature makes
        # repeated runs differ, and n_runs is how a reader learns by how much.
        if "generation" in _parse_tracks(cfg.tracks):
            for note in _sampling_notes():
                logger.warning(note)

        # Checkpoint manager
        cp_dir = cfg.checkpoint_dir or output_dir
        checkpoint = CheckpointManager(cp_dir)

        # Generate tasks
        tasks = self.generate_tasks()
        logger.info(f"Generated {len(tasks)} tasks")

        # Offline, a benchmark whose data is not cached fails here, once, rather
        # than in every task with the same "dataset_info.json not found".
        problem = _offline_dataset_problem([t.benchmark_name for t in tasks])
        if problem is not None:
            raise problem

        # Pre-register corpus ref counts for live mode.
        # With max_concurrency=1 tasks run sequentially; without this,
        # task N releases (and deletes) the corpus before task N+1 runs.
        if cfg.mode == "live":
            await _pre_register_corpus_refs(tasks)

        # Select eval function
        eval_fn = self._eval_fn or _default_eval_fn
        cache_settings = LLMCacheSettings.parse(cfg.llm_cache).resolved()
        cache_totals = CacheStats()
        eval_fn = _with_llm_cache(eval_fn, cache_settings, cache_totals)
        prices = PriceTable.parse(cfg.prices)
        if prices is not None:
            from .cost import with_cost_estimate

            eval_fn = with_cost_estimate(eval_fn, prices)

        # Agentic search reads part of its settings from the corpus session,
        # which the arms of a sweep share, so two agentic arms running at once
        # could read each other's settings. Run those sweeps one task at a time.
        concurrency = cfg.max_concurrency
        if concurrency > 1 and any(
            (t.config_overrides or {}).get("use_agentic_rag")
            or (t.config_overrides or {}).get("agentic_pipeline")
            or _profile_uses_agentic(t.profile)
            for t in tasks
        ):
            logger.warning(
                f"Running agentic arms one at a time (max_concurrency {concurrency} -> 1): "
                f"they share the corpus session's settings."
            )
            concurrency = 1

        # Run
        runner = BenchmarkRunner(max_concurrency=concurrency)
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
            experiment_config=_run_metadata(cfg, concurrency=concurrency),
        )
        # The commit of each model the tasks ran, read from the local cache.
        from .run_provenance import component_provenance, model_revisions, models_used

        task_rows = [r.to_dict() for r in task_results]
        results.experiment_config["model_revisions"] = model_revisions(
            models_used(task_rows, results.experiment_config)
        )
        # The paper and claimed numbers of each component that declared them,
        # so a results table can show the claim beside what was measured.
        results.experiment_config["components"] = component_provenance(task_rows)
        # How much of this run was replayed, so a fast run can be read as one.
        results.experiment_config["llm_cache"] = cache_settings.record(cache_totals)
        # Whether one run is the whole answer (see nothing_samples).
        results.experiment_config["nothing_samples"] = nothing_samples(task_results, cfg.tracks)
        if prices is not None:
            # The table and the rates behind every cost_usd_estimate.
            from .cost import generator_model_name, judge_model_name

            results.experiment_config["cost_estimate"] = prices.record(
                [generator_model_name(), judge_model_name()]
            )

        # Save outputs
        _save_outputs(results, output_dir, cfg.output_formats)

        if task_results and len(results.failed_tasks) == len(task_results):
            # Library logs are off by default, so from Python a run where every
            # task failed returned in seconds with nothing on screen.
            first = results.failed_tasks[0]
            warnings.warn(
                f"All {len(task_results)} task(s) of '{cfg.name}' failed. First: "
                f"{first.task_key}: {first.error}. See results.failed_tasks.",
                RuntimeWarning,
                stacklevel=3,  # past run(), as before it delegated to _run()
            )

        logger.info(
            f"Experiment '{cfg.name}' complete: "
            f"{results.total_tasks} tasks, "
            f"{len(results.failed_tasks)} failed"
        )

        return results


#: The report's primary column, chosen as `results per-query` chooses its first.
_pick_primary_metric = primary_metric


def base_profiles_line(tasks: Iterable["BenchmarkTask"]) -> str:
    """``Base profile: standard, for every arm ...``: the profile each arm starts from.

    `lasagna benchmark run` starts from speed and a sweep file without
    ``base_profile:`` from standard, and neither the dry run nor the run
    header said which an arm used.
    """
    profiles = {t.config_name: t.profile or "standard" for t in tasks}
    if len(set(profiles.values())) == 1:
        (profile,) = set(profiles.values())
        return (
            f"Base profile: {profile}, for every arm (pipeline.base_profile, standard when "
            f"unset; an arm's own profile: wins)"
        )
    return "Base profiles: " + ", ".join(f"{arm} from {p}" for arm, p in profiles.items())


def _tidy_run_folder(cfg: "ExperimentConfig") -> None:
    """After a run: drop what holds nothing, and say what a stopped run left.

    A run refused before any task (a dataset not cached offline, say) left
    ``results/<name>_<time>/.checkpoints``, empty, for each attempt. An empty
    checkpoint folder is removed; so is the run's own timestamped folder when
    it holds nothing else (a folder named by ``output_dir`` is kept). A run
    that stopped with finished tasks checkpointed and no results.json keeps
    them, and says so: ``--resume`` continues from them.
    """
    out = Path(cfg.effective_output_dir())
    checkpoints = Path(cfg.checkpoint_dir or out) / ".checkpoints"
    finished = len(list(checkpoints.glob("*.json"))) if checkpoints.is_dir() else 0
    try:
        if checkpoints.is_dir() and not finished:
            checkpoints.rmdir()
        if (out / "results.json").exists():
            return
        if finished:
            logger.warning(
                f"The run stopped before writing results. {out} holds {finished} finished "
                f"task(s) as checkpoints: run the same command with --resume to continue "
                f"from them."
            )
        elif not cfg.output_dir and out.is_dir() and not any(out.iterdir()):
            out.rmdir()
    except OSError as exc:  # tidying must never hide why the run stopped
        logger.debug(f"Could not tidy {out}: {exc}")


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
def _run_metadata(cfg: "ExperimentConfig", concurrency: Optional[int] = None) -> Dict[str, Any]:
    """Everything needed to situate this run, beside the numbers.

    The previous version recorded four fields -- mode, n_runs, seed, and the
    benchmark names -- which is not enough to tell whether two results files
    are describing the same experiment. Missing were the library versions and
    hardware that move a metric, the resolved sweep (a one-way config hash
    cannot be read back), the judge model, and the split.
    """
    from datetime import datetime, timezone

    from lasagna.core.provenance import capture, redact

    from .bundle import METRICS_VERSION
    from .run_provenance import determinism_label, package_version, rag_overrides

    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg.name,
        "description": cfg.description,
        # Requested mode; each result carries what it actually resolved to.
        "mode": cfg.mode,
        "n_runs": cfg.n_runs,
        "seed": cfg.seed,
        "variance_source": cfg.variance_source,
        # Never "reproducible" (see seeding.py). Tasks seed the process-wide
        # generators, so only one task at a time keeps each task's draws its own.
        "determinism": determinism_label(
            cfg.max_concurrency if concurrency is None else concurrency
        ),
        # What the numbers were computed under: the metric definitions, the
        # package, and the settings the environment overrode (redacted).
        "metrics_version": METRICS_VERSION,
        "package_version": package_version(),
        "rag_overrides": rag_overrides(),
        "max_concurrency": cfg.max_concurrency,
        "benchmarks": [
            {
                "name": b.name,
                "split": b.split,
                "n_samples": b.n_samples,
                "filters": b.filters,
                # Data of your own: where it was read from. The hash in the
                # name says what it was; this says which file to open again.
                **({"path": _recorded_data_path(b.path, cfg.base_dir)} if b.path else {}),
            }
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
            "generators": list(cfg.pipeline.generators),
            "config_overrides": list(cfg.pipeline.config_overrides),
        },
        "judge_model": _active_judge_model or cfg.judge_model or None,
        # Whether temperature reached the generator and the judge. A model that
        # takes none ("provider-default") samples between runs however the
        # pipeline is configured. None for a run that generates nothing.
        "sampling": _sampling_record() if "generation" in _parse_tracks(cfg.tracks) else None,
        # The embedder behind answer_similarity: a different one gives
        # different numbers, so it belongs in the comparability key.
        "similarity_model": _similarity_model_for_record(),
        "max_corpus_docs": cfg.max_corpus_docs,
        "ingest_failure_tolerance": cfg.ingest_failure_tolerance,
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


def _offline_dataset_problem(benchmark_names: List[str]):
    """The `DatasetNotCached` error for these benchmarks, or None (see `missing_offline_datasets`)."""
    from .datasets import hf_repos, missing_offline_datasets

    repos = {}
    for name in dict.fromkeys(benchmark_names):
        try:
            adapter = benchmark_registry.get(name)
        except Exception:  # noqa: BLE001 - local data has no registry entry to look up
            adapter = None
        repos[name] = hf_repos(name, adapter)
    return missing_offline_datasets(repos)


def _recorded_data_path(path: str, base_dir: str) -> str:
    """A local benchmark's ``path`` as the run metadata records it.

    Relative to the sweep file's folder (the working directory for `benchmark
    run`), however it was written. Never absolute: the run metadata goes into
    bundles, which are published, and the path redaction only knows the home
    and temp folders, so ``/data/acme/faq.jsonl`` would have been published as
    written. A file on another Windows drive is recorded by its name alone.
    """
    from .benchmarks.adapters.local_files import resolve_data_path

    full = resolve_data_path(path, base_dir)
    root = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    try:
        return Path(os.path.relpath(full, root)).as_posix()
    except ValueError:  # no relative path between two drives
        return full.name


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
    from .stages import CHUNKER_DISPATCH_KEYS, effective_chunking_strategy

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
        # What runs, as a task records it: a structural setting decides it.
        "chunker": effective_chunking_strategy({k: _get(k) for k in CHUNKER_DISPATCH_KEYS}),
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
                "can be measured, but against gold passages only, which is an "
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
# Resolved order: env RAG_EVAL_JUDGE_MODEL → ExperimentConfig.judge_model → DEFAULT_JUDGE_MODEL.
_active_judge_model: Optional[str] = None

#: The judge when neither ``judge_model:`` nor ``RAG_EVAL_JUDGE_MODEL`` names
#: one. Without it RAGAS chose GPT-4o mini itself, on a client this harness
#: neither metered nor cached; OpenAI shuts that model down on 2027-01-14 and
#: names gpt-6-luna as the replacement.
#: https://developers.openai.com/api/docs/deprecations
DEFAULT_JUDGE_MODEL = "gpt-6-luna"

#: The temperature RAGAS asks its judge for (``InstructorModelArgs``). It
#: forces 1.0 on gpt-5 and later; the judge client restores this one where the
#: model takes it. See ``pin_openai_client``.
_JUDGE_TEMPERATURE = 0.01


def _set_resume_fingerprints(cfg: "ExperimentConfig", tasks: List[BenchmarkTask]) -> None:
    """Put what a checkpoint depends on beyond its task into every task key.

    Metric definitions, package version, judge and its endpoint, environment
    settings, the LLM provider and model, and the local commit of each model
    the arm loads, so --resume never restores a result computed under other
    ones. The LLM counts on the retrieval track too: query expansion, HyDE and
    the LLM rerankers call it there. A model not yet in the local cache has no
    commit to hash, so after a run that downloaded one, --resume reruns its
    tasks once.
    """
    from .cost import _arm_models, _platform, generator_model_name
    from .run_provenance import model_revisions, resume_fingerprint

    judge = _resolve_judge_model(cfg)
    llms: Dict[str, Any] = {"provider": _platform(), "model": generator_model_name()}
    if judge:
        llms["judge_base_url"] = (
            os.environ.get("RAG_EVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
        )
    scorers = [_similarity_model_for_record()] if "generation" in _parse_tracks(cfg.tracks) else []
    by_arm: Dict[str, str] = {}
    for task in tasks:
        if task.config_hash not in by_arm:
            loaded = [m for _, m, _ in _arm_models(task) if not m.startswith("component ")]
            by_arm[task.config_hash] = resume_fingerprint(
                judge,
                variance_source=cfg.variance_source,
                max_concurrency=cfg.max_concurrency,
                llms=llms,
                model_revisions=model_revisions(n for n in loaded + scorers if n),
            )
        task.resume_fingerprint = by_arm[task.config_hash]


def _resolve_judge_model(cfg: "ExperimentConfig") -> Optional[str]:
    """``RAG_EVAL_JUDGE_MODEL``, then ``judge_model:``, then the default.

    A run with no generation track has no judge: None, so its comparability
    key stays what it was.
    """
    named = os.environ.get("RAG_EVAL_JUDGE_MODEL") or cfg.judge_model or None
    if named or "generation" not in _parse_tracks(cfg.tracks):
        return named
    return DEFAULT_JUDGE_MODEL


def _judge_platform() -> str:
    """The judge is an OpenAI client; a base URL can point it at OpenRouter."""
    base_url = os.environ.get("RAG_EVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    return "openrouter" if "openrouter.ai" in base_url else "openai"


def _sampling_record() -> Dict[str, Any]:
    """Whether the generator and the judge could be pinned, for the run metadata."""
    from lasagna.providers.model_capabilities import sampling_label

    from .cost import _platform, generator_model_name, judge_model_name

    generator, platform = generator_model_name(), _platform()
    judge = judge_model_name() or DEFAULT_JUDGE_MODEL
    return {
        "generator": {
            "model": generator,
            "sampling": sampling_label(generator, platform) if generator else None,
        },
        "judge": {"model": judge, "sampling": sampling_label(judge, _judge_platform())},
    }


def _sampling_notes() -> List[str]:
    """One line each for a generator or judge that cannot be pinned."""
    from lasagna.providers.model_capabilities import unpinned_note

    from .cost import _platform, generator_model_name, judge_model_name

    judge = judge_model_name() or DEFAULT_JUDGE_MODEL
    notes = [
        unpinned_note(generator_model_name(), _platform(), "answers"),
        unpinned_note(judge, _judge_platform(), "judge scores"),
    ]
    return [n for n in notes if n]


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
                → ``DEFAULT_JUDGE_MODEL``
    ==========  ==========================================================

    Returns None when construction fails, or when nothing names a judge and
    there is no key for the default one; callers then fall back to the RAGAS
    default judge.

    Each request is rebuilt for the model it names (``pin_openai_client``):
    RAGAS sends gpt-5 and later temperature 1.0 at their default reasoning
    effort, so gpt-6-luna gets ``reasoning_effort="none"`` and RAGAS's usual
    0.01 instead, and a model that takes no temperature gets none.
    """
    resolved_model = model_name or _active_judge_model or os.environ.get("RAG_EVAL_JUDGE_MODEL")
    client_kwargs = {}
    api_key = os.environ.get("RAG_EVAL_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("RAG_EVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    if not resolved_model:
        resolved_model = DEFAULT_JUDGE_MODEL
    named = model_name or os.environ.get("RAG_EVAL_JUDGE_MODEL")
    if resolved_model == DEFAULT_JUDGE_MODEL and not named and not api_key:
        # The default judge needs a key, and nobody asked for it: no judge is
        # not an error here, as it was not before there was a default.
        return None

    try:
        from ragas.llms import llm_factory
        from openai import OpenAI

        # Counted under `eval`, separately from the system under test. RAGAS
        # builds its judge on a raw OpenAI client rather than through the
        # LangChain provider, so without this wrapper the entire cost of
        # *measuring* -- often larger than the cost of the pipeline -- was
        # invisible, and `eval_cost_usd` had no producer at all.
        #
        # The cache goes on first, so the meter sees a replayed response too
        # and counts it at its cold-run usage.
        #
        # The sampling rewrite goes on last, so the cache keys on, and the
        # meter counts, the request that is actually sent.
        from lasagna.providers.model_capabilities import pin_openai_client
        from lasagna.utils.llm_cache import cache_openai_client
        from lasagna.utils.token_usage import EVAL, instrument_openai_client

        client = instrument_openai_client(cache_openai_client(OpenAI(**client_kwargs)), EVAL)
        judge = llm_factory(resolved_model, client=pin_openai_client(client, _JUDGE_TEMPERATURE))
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
    # Under evaluation a stage that cannot run fails the task instead of
    # falling back: an arm whose query expansion or HyDE never ran scored
    # exactly the control, with Failed: 0.
    from lasagna.core.strict import strict

    with strict():
        return await _eval_live(task, samples)


def _with_llm_cache(eval_fn: Callable, settings: LLMCacheSettings, totals: CacheStats) -> Callable:
    """Run each task in a cache scope of its own, and record what it replayed.

    The scope's seed is the task's process seed (``seed + run_index``, what
    ``seed_everything`` sets), so each run of an arm keeps its own responses.
    The counts land beside ``generations_failed``, and only on a task that
    looked something up: a task that made no LLM call, or ran with the cache
    off, carries no cache field at all. They are task metrics only, never
    per-query scores, so no significance test or comparability key sees them.
    """

    async def run(task: BenchmarkTask, samples: List[BenchmarkSample]) -> TaskResult:
        with llm_cache_scope(settings, seed=task.seed + task.run_index) as scope:
            result = await eval_fn(task, samples)
        totals.add(scope)
        if scope.lookups:
            result.metrics["llm_cache_hits"] = float(scope.hits)
            result.metrics["llm_cache_misses"] = float(scope.misses)
        return result

    return run


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
        "generator",
    }
)

#: Component slots the harness cannot swap: ingestion always runs the built-in
#: chunker and document processor, so an arm naming one measured the default.
#: Ingestion-time stages. The corpus is ingested once per ingestion setting,
#: always with the built-ins, so an arm swapping one of these would measure
#: queries embedded (or nothing changed) against the profile's corpus.
_UNSWEEPABLE_COMPONENTS = frozenset({"document_processor"})

#: Ingestion-time slots an arm can fill with a component: the arm's corpus is
#: chunked or embedded with it, in its own corpus session (and, for an
#: embedder, its own collection), keyed by the component's identity.
_INGEST_COMPONENTS = ("chunker", "embedder")

# Defined in `runner`, beside `BenchmarkTask.config_hash`, which must exclude
# the same keys: a setting that steers reporting rather than the pipeline must
# not change the session id, the checkpoint key or the comparability of a run.
_HARNESS_KEYS = HARNESS_KEYS


def _profile_uses_agentic(profile: Optional[str]) -> bool:
    try:
        from lasagna.core.session_config._profiles import resolve_ingestion_defaults

        return bool(resolve_ingestion_defaults(profile or "standard").get("use_agentic_rag"))
    except Exception:
        return False


def _complete_stage_settings(config: Dict[str, Any]) -> None:
    """Fill every stage setting the config lacks with its global value.

    Operational settings (stores, caches, credentials) are left out: they do
    not change what a run computes, and they must not be copied into a
    session store.
    """
    from lasagna.core.config import rag_config

    from .stages import stage_of

    for key in getattr(rag_config, "_config", {}) or {}:
        if key in config or stage_of(key) is None:
            continue
        if hasattr(type(rag_config), key):
            try:
                config[key] = getattr(rag_config, key)
                continue
            except Exception:
                pass
        config[key] = rag_config._config.get(key)


def _apply_overrides_to_pipeline(
    pipeline: Any,
    overrides: Dict[str, Any],
    session_id: str,
) -> None:
    """Apply config overrides to a KnowledgePipeline.

    - Component names (reranker, embedder, etc.) → ``pipeline.replace()``
    - ``ensemble_reranker_models`` → the ``ensemble`` reranker, which reads
      them (and ``ensemble_reranker_weights``) from the arm's config
    - Config keys (enable_hybrid_search, chunk_size, etc.) → dict update
      on ``pipeline._config``, then ``_apply_session_config(session_id)``
    """
    # Defensive copy — from_profile() passes profile.config by reference
    # into pipeline._config.  Without this copy, .update() below would
    # mutate the config-manager singleton's stored profile dict, leaking
    # overrides into every subsequent task that loads the same profile.
    pipeline._config = dict(pipeline._config)
    # Complete it: every setting that reaches a pipeline stage gets an explicit
    # value (the global one where the profile is silent). The arms of a sweep
    # share one corpus session, and a key the profile does not set used to
    # resolve from that session, which still held the previous arm's value:
    # one arm's rrf_k changed the next arm's number, and the manifest agreed.
    _complete_stage_settings(pipeline._config)

    # The ensemble reads its members from the arm's config on each call, so
    # naming them is enough to measure it. They used to build an ensemble here
    # that `reranker: ensemble` then replaced with one of the settings file's
    # model, and the YAML refused the keys as unknown anyway.
    if "ensemble_reranker_models" in overrides and not overrides.get("reranker"):
        overrides = {**overrides, "reranker": "ensemble"}

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

    # Never through the shared session store (see _ensure_corpus_ingested).
    from ..core.session_config import keep_in_process

    keep_in_process(session_id)
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
        # The agentic tier reads the session store: an arm whose settings
        # could not be written there would run its loop on the previous
        # arm's settings.
        raise ValueError(f"Could not apply this arm's settings: {exc}") from exc


def _no_rag(task: BenchmarkTask) -> bool:
    """Whether the arm answers with no retrieval at all (``no_rag: true``)."""
    return bool((task.config_overrides or {}).get("no_rag"))


def _no_rag_generator(generator: Any, overrides: Dict[str, Any]) -> Any:
    """The generator a ``no_rag`` arm answers with: ``closed_book`` unless it
    names one that reads no passages. One that reads them would be handed none,
    so that arm is refused rather than scored on empty context."""
    if generator is None:
        from lasagna.core.registry import registry

        return registry.get("generator", "closed_book")
    if getattr(generator, "needs_context", True):
        raise ValueError(
            f"no_rag: true gives the generator no passages, but generator "
            f"{overrides.get('generator')!r} writes its answer from them. Use "
            f"generator: closed_book, or remove no_rag from this arm."
        )
    return generator


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
    # A generator writes the answer from the gold context, so it acts here too.
    inert_keys = ((_RETRIEVAL_OVERRIDES | _COMPONENT_KEYS) - {"generator"}) & set(
        task.config_overrides.keys()
    )
    if inert_keys and not task.config_overrides.get("allow_inert_overrides"):
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=(
                f"Config overrides {sorted(inert_keys)} have no effect in "
                f"'{task.mode}' mode, which scores gold context and never runs the "
                f"retrieval pipeline: any metric produced here would be unrelated "
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

    # A named generator writes the answers; with none, the prompt below does.
    _gen_name = (task.config_overrides or {}).get("generator")
    _offline_generator = None
    if _gen_name:
        from lasagna.core.registry import registry as _registry

        _offline_generator = _registry.get("generator", _gen_name)
    if _no_rag(task):
        _offline_generator = _no_rag_generator(_offline_generator, task.config_overrides)
    # A generator that reads no passages (closed_book, a no_rag arm) is handed
    # no gold context and scored without it, so it needs none to be scored.
    _needs_context = getattr(_offline_generator, "needs_context", True)

    # Filter to samples with both ground_truth and context (required for
    # RAGAS metrics like faithfulness, context_recall, context_precision).
    # Samples without gold context need live-mode retrieval.
    eval_samples = [
        s
        for s in samples
        if (s.ground_truth or _is_unanswerable(s)) and (s.context or not _needs_context)
    ]
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
            f"(skipped {skipped} without a gold answer or gold context)"
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
    _has_unanswerable = any(_is_unanswerable(s) for s in eval_samples)
    _abstain = _ABSTAIN_INSTRUCTION if _has_unanswerable else ""
    # The judge needs a reference answer, so RAGAS scores the answerable ones.
    _ans_idx = [i for i, s in enumerate(eval_samples) if not _is_unanswerable(s)]

    for sample in eval_samples:
        questions.append(sample.question)
        ground_truths.append(sample.ground_truth)

        # Use gold context if available
        ctx = [sample.context] if sample.context and _needs_context else []
        contexts.append(ctx)

        # Generate answer from context using LLM
        sample_meter = UsageMeter()
        if (answer_llm or _offline_generator) and (ctx or not _needs_context):
            q_start = time.monotonic()
            try:
                if _offline_generator is not None:
                    with sample_meter, sample_meter.scope(SYSTEM):
                        answer = await asyncio.to_thread(
                            _offline_generator.generate,
                            sample.question,
                            list(ctx),
                            llm=answer_llm,
                            instructions=_abstain,
                        )
                else:
                    context_str = "\n\n".join(ctx)
                    prompt = (
                        f"Answer the following question based on the given context. "
                        f"{_abstain}\n\n"
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
    _citation_failure: Optional[str] = None

    try:
        from .ragas_evaluator import RAGASEvaluator, harness_metrics

        evaluator = RAGASEvaluator(llm=judge_llm)

        # RAG_EVAL_JUDGE_METRICS or the harness set; with no context, only
        # the metrics that do not judge one.
        eval_metrics = harness_metrics(has_context=any(c for c in contexts))

        # The judge's own spend, kept separate from the system's. It is often
        # the larger of the two, and reporting one total lets an expensive
        # judge masquerade as an expensive pipeline.
        if _ans_idx:
            with eval_meter, eval_meter.scope(EVAL):
                detailed = evaluator.evaluate_detailed(
                    questions=[questions[i] for i in _ans_idx],
                    answers=[answers[i] for i in _ans_idx],
                    contexts=[contexts[i] for i in _ans_idx],
                    ground_truths=[ground_truths[i] for i in _ans_idx],
                    metrics=eval_metrics,
                )
            metrics = dict(detailed.aggregate)
        # One row per sample, in sample order; the judge's rows go to the
        # answerable samples they scored.
        per_sample_scores = [{} for _ in eval_samples]
        for j, i in enumerate(_ans_idx):
            if detailed is not None and j < len(detailed.per_sample):
                per_sample_scores[i] = dict(detailed.per_sample[j])

        # Add latency, tokens, EM/F1, and trajectory to per-sample scores
        for i, latency in enumerate(per_sample_latencies):
            if i >= len(per_sample_scores):
                per_sample_scores.append({})
            per_sample_scores[i]["latency_ms"] = latency
            if i < len(per_sample_usage):
                per_sample_scores[i].update(per_sample_usage[i].to_dict())

        # EM/F1 + trajectory for offline path (mirrors live path)
        from .answer_metrics import exact_match as _em_fn, strip_citations, token_f1 as _tf1_fn

        _save_traj = (task.config_overrides or {}).get("save_trajectories", "minimal")
        for i, sample in enumerate(eval_samples):
            if i >= len(per_sample_scores):
                per_sample_scores.append({})
            ps = per_sample_scores[i]

            # Sample metadata
            ps["_sample_id"] = getattr(sample, "query_id", "")
            _meta = getattr(sample, "metadata", {}) or {}
            ps["_source_benchmark"] = _meta.get("source_benchmark", "")

            # EM/F1, on the answer without citation markers
            _gt = ground_truths[i] if i < len(ground_truths) else ""
            _ans = answers[i] if i < len(answers) else ""
            _scored = strip_citations(_ans)
            _aliases = getattr(sample, "ground_truth_answers", None) or []
            if _is_unanswerable(sample):
                ps["exact_match"] = ps["token_f1"] = _no_answer_em(_scored)
            elif _gt.strip() and _scored.strip():
                ps["exact_match"] = _em_fn(_scored, _gt, aliases=_aliases)
                ps["token_f1"] = _tf1_fn(_scored, _gt, aliases=_aliases)
            else:
                ps["exact_match"] = 0.0
                ps["token_f1"] = 0.0
            ps.update(_answer_scores(_scored, sample))
            if _has_unanswerable:
                ps["_answerable"] = 0.0 if _is_unanswerable(sample) else 1.0

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

        # Semantic similarity (local embedder) and, for answers that cite,
        # citation quality (the judge, counted as eval spend).
        await asyncio.to_thread(
            _add_answer_similarity,
            per_sample_scores,
            [strip_citations(a) for a in answers],
            eval_samples,
        )

        def _citations_in_thread():
            with eval_meter, eval_meter.scope(EVAL):
                return _add_citation_scores(per_sample_scores, answers, contexts)

        _citation_failure = await asyncio.to_thread(_citations_in_thread)

        # Aggregate EM/F1
        _em_scores = [s.get("exact_match") for s in per_sample_scores if "exact_match" in s]
        if _em_scores:
            metrics["exact_match"] = sum(_em_scores) / len(_em_scores)
            _f1_scores = [s.get("token_f1", 0.0) for s in per_sample_scores if "token_f1" in s]
            metrics["token_f1"] = sum(_f1_scores) / len(_f1_scores)
            metrics.update(_answerability_split(per_sample_scores))
        metrics.update(_mean_present(per_sample_scores, _ANSWER_METRICS))
        metrics.update(_per_query_costs(per_sample_scores))

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
    if not _needs_context:
        # Read by the reports, which say why this arm has no faithfulness.
        result.system = {"generator_context": "none"}
    _offline_failures: List[str] = []
    if getattr(detailed, "failed_metrics", None):
        _offline_failures.append(
            "judge produced no usable scores for " + ", ".join(detailed.failed_metrics)
        )
    if _citation_failure:
        _offline_failures.append(_citation_failure)
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


def _similarity_model_for_record() -> Optional[str]:
    try:
        from .answer_metrics import similarity_model_name

        name = similarity_model_name()
    except Exception:
        return None
    return None if str(name).lower() == "none" else name


def _corpus_doc_id(corpus_id: str, session_key: str = "") -> str:
    """The stored document id for a benchmark document in one corpus session.

    The shared corpus (no ``session_key``) keeps the original form, so corpora
    ingested before per-session ids existed are still found.
    """
    import hashlib

    raw = f"{session_key}|{corpus_id}" if session_key else corpus_id
    return f"doc_{hashlib.md5(raw.encode()).hexdigest()[:16]}"


def _ingest_component_fingerprint(slot: str, name: str) -> str:
    """What a chunker or embedder component is, for corpus keys and collections.

    The registered name and class, and for a function its code and simple
    captured values (an edited function must not reuse the old corpus).
    """
    from lasagna.core.registry import registry

    try:
        cls = registry.get_class(slot, name)
    except Exception:
        return name  # unknown names are refused before any task runs
    fn = getattr(cls, "_fn", None)
    if fn is not None:
        from lasagna.components import _code_digest, _identity

        return f"{name}|{_identity(fn)}#{_code_digest(fn)}"
    return f"{name}|{cls.__module__}.{cls.__qualname__}"


def _ingest_components_for(task: "BenchmarkTask", settings: Dict[str, Any]) -> Dict[str, Any]:
    """The arm's chunker and embedder components, instantiated, by slot.

    A component that takes a config gets the arm's (profile defaults plus the
    arm's own settings): built without one, the sentence-transformers embedder
    read the global embedding_model and ignored the arm's.
    """
    import inspect

    from lasagna.core.config import rag_config
    from lasagna.core.registry import registry

    out: Dict[str, Any] = {}
    for slot in _INGEST_COMPONENTS:
        name = (task.config_overrides or {}).get(slot)
        if not (isinstance(name, str) and name):
            continue
        cls = registry.get_class(slot, name)
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            params = {}
        takes_config = "config" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if takes_config:
            out[slot] = registry.get(slot, name, config=rag_config.with_query_overrides(settings))
        else:
            out[slot] = registry.get(slot, name)
    return out


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
                k: v for k, v in _all_profile.items() if k in _ingestion_time_keys()
            }
        except Exception:
            pass  # fall back to empty

    # Explicit ingestion overrides take precedence over profile defaults
    _explicit_ingestion = {
        k: v for k, v in (task.config_overrides or {}).items() if k in _ingestion_time_keys()
    }
    merged = {**_profile_ingestion, **_explicit_ingestion}
    # A chunker or embedder component makes its own corpus: keyed by what the
    # component is, so a changed function is never served an old corpus.
    for _slot in _INGEST_COMPONENTS:
        _name = (task.config_overrides or {}).get(_slot)
        if isinstance(_name, str) and _name:
            merged[f"component:{_slot}"] = _ingest_component_fingerprint(_slot, _name)

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


async def _abandon_corpus(cache_key: str, message: str) -> Dict[str, Any]:
    """Fail every task waiting on this corpus with ``message``.

    The waiters block on the slot's event; leaving it unset hung a whole sweep.
    """
    async with _corpus_cache_lock:
        slot = _corpus_sessions.pop(cache_key, None)
    if slot is not None:
        slot["error"] = message
        slot["ready"].set()
    return {"error": message}


async def _ensure_corpus_ingested(
    benchmark_name: str,
    samples: List[BenchmarkSample],
    corpus_key: Optional[str] = None,
    ingestion_overrides: Optional[Dict[str, Any]] = None,
    max_corpus_docs: int = 0,
    components: Optional[Dict[str, Any]] = None,
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
        components: The arm's chunker and embedder components, by slot, when it
            named them; ``corpus_key`` must already be keyed on them.
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
    corpus_lock = _CorpusLock()
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
                "full: beir/msmarco is 8.8M documents, which is hours of "
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
            from .benchmarks.adapters.local_files import LocalFileAdapter

            # Your own data has no published benchmark to be easier than: its
            # contexts are the corpus you asked about.
            if corpus and not isinstance(adapter, LocalFileAdapter):
                logger.warning(
                    f"'{benchmark_name}' ships no corpus; retrieval will be measured "
                    f"against {len(corpus)} passages pooled from the samples' own gold "
                    f"context. This is an easier task than the published benchmark (the "
                    f"pool holds no hard negatives), and these numbers are not comparable "
                    f"with a native-corpus run."
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
        # A per-arm corpus session gets ids of its own. Derived from the corpus
        # id alone, two sessions in one collection shared document and chunk
        # ids: the documents table keeps one session per id and the vector
        # store one row per chunk id, so the arm that ingested last took the
        # other's documents and that arm's searches came back empty (0.0).
        # The shared corpus keeps the plain ids, so nothing already ingested
        # is orphaned.
        _id_ns = cache_key if corpus_key is not None else ""
        for corpus_id in corpus:
            valid_id = _corpus_doc_id(corpus_id, _id_ns)
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
                    f"corpus: results would be unreliable. Raise "
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
        # Its settings stay in this process: every run on this corpus uses this
        # session id, and two experiment processes sharing its row in the
        # session store read each other's arm settings.
        from ..core.session_config import keep_in_process

        keep_in_process(shared_session_id)

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
                # Ingesting anyway built the corpus with the global settings
                # (its embedder, its chunking) under this arm's name.
                return await _abandon_corpus(
                    cache_key, f"Could not apply the arm's ingestion settings: {exc}"
                )

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
                    return await _abandon_corpus(
                        cache_key, f"Could not create the arm's collection: {exc}"
                    )

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
        from ._fast_ingest import fast_ingest_corpus as _fast_ingest

        _ingest_backend = _get_ingest_backend()
        _resolved = _ingest_backend._collection_manager.resolve_backend(shared_session_id)
        _coll_name = (
            _rag_cfg.collection_name
            if not ingestion_overrides
            else (ingestion_overrides.get("collection_name", _rag_cfg.collection_name))
        )

        # From the check of what is stored until the corpus is published, held
        # against every process sharing the metadata store (_CorpusLock).
        await corpus_lock.take(shared_session_id, _coll_name)
        # Pre-flight: check if corpus already fully indexed via vector store
        expected_valid_ids = {corpus_to_valid[cid] for cid in corpus}
        ready_ids: set = set()
        _skip_ingest = False
        # Documents that could not be ingested, in this run or (allowlisted)
        # in the run that built the corpus being reused.
        failed_valid_ids: set = set()
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
                _allowlist_path = Path(_results_root()) / "_ingest_allowlist.json"
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
                    failed_valid_ids = set(missing_ids)
                else:
                    # Stale/partial — clean up and re-ingest
                    logger.warning(
                        f"Stale partial corpus ({len(existing_doc_ids)}/{len(expected_valid_ids)} docs, "
                        f"{len(missing_non_error)} missing). Cleaning up and re-ingesting."
                    )
                    try:
                        _filt = _resolved.session_filter(_coll_name, shared_session_id)
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
            # Ingested fresh on top of a corpus that was there, every chunk was
            # stored twice (Milvus `insert` keeps both rows of an id) and the
            # dense ranked lists held each one twice. What is there goes first.
            logger.warning(
                f"Could not check existing vectors ({exc}); removing session "
                f"{shared_session_id}'s vectors, if any, and ingesting fresh."
            )
            try:
                if _resolved.has_collection(_coll_name):
                    _filt = _resolved.session_filter(_coll_name, shared_session_id)
                    _resolved.delete(collection_name=_coll_name, filter_expr=_filt)
            except Exception as _del_exc:
                raise _StaleCorpusCleanupError(
                    f"Could not check or delete the vectors of session {shared_session_id} "
                    f"({_del_exc}), so the corpus was not ingested on top of them. Delete "
                    f"that session or collection and rerun."
                ) from _del_exc
            _ingest_backend.hybrid_searcher.clear_session(shared_session_id)
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
                chunker=(components or {}).get("chunker"),
                embedder=(components or {}).get("embedder"),
            )
            logger.info(
                f"Fast ingestion complete: {total_chunks} chunks from "
                f"{len(ingested_ids)} docs, {len(error_ids)} errors"
            )
            failed_valid_ids = set(error_ids)

            # Persist error allowlist for future resume
            if error_ids:
                _allowlist_path = Path(_results_root()) / "_ingest_allowlist.json"
                _allowlist_path.parent.mkdir(parents=True, exist_ok=True)
                _al: dict = {}
                if _allowlist_path.exists():
                    try:
                        _al = json.loads(_allowlist_path.read_text())
                    except Exception:
                        pass
                _al[shared_session_id] = sorted(error_ids)
                _allowlist_path.write_text(json.dumps(_al, indent=2))

        # What is published, counted under the lock: each task compares it at its end.
        census = None
        try:
            census = _corpus_census(
                _resolved,
                _coll_name,
                shared_session_id,
                {corpus_to_valid[cid] for cid in corpus} - failed_valid_ids,
            )
        except NotImplementedError:
            pass  # a store without filtered queries: nothing to compare
        except Exception as exc:  # noqa: BLE001 - the census only guards the result
            logger.warning(f"Could not count the published corpus: {exc}")
        corpus_lock.release()
        logger.info(f"Corpus ingestion complete for '{benchmark_name}' (cache_key='{cache_key}')")
        ingest_gap = _ingest_gap(corpus, valid_to_corpus, failed_valid_ids)
        if ingest_gap:
            # Once per corpus, not per task: every task on it reads the slot.
            logger.warning(
                f"{ingest_gap['failed']} of {ingest_gap['documents']} documents of "
                f"'{benchmark_name}' could not be ingested (first: "
                f"{', '.join(ingest_gap['first_failed'])}). Results on this corpus record "
                f"corpus_ingest_gap and do not share a comparability key with the complete "
                f"corpus; under evaluation a task fails unless ingest_failure_tolerance "
                f"allows it."
            )

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
            slot["corpus_ingest_gap"] = ingest_gap
            slot["corpus_census"] = census
            slot["_census_store"] = (
                lambda: _get_ingest_backend()._collection_manager.resolve_backend(shared_session_id)
            )
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
    finally:
        corpus_lock.release()


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
    _combined_ingestion_keys = _ingestion_time_keys() | _INGESTION_RUNTIME_KEYS
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
    _has_signature_keys = any(k in _ingestion_time_keys() for k in _merged_keys)
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

    # Chunker and embedder components chunk and embed this arm's own corpus.
    # The embedder gets a collection of its own at its own dimension; the
    # query side already encodes with it through the pipeline's override.
    _ingest_components = _ingest_components_for(
        task,
        {
            k: v
            for k, v in {**_from_profile, **_task_overrides}.items()
            if k not in _COMPONENT_KEYS and k not in HARNESS_KEYS
        },
    )
    if "embedder" in _ingest_components:
        _ingestion_overrides = dict(_ingestion_overrides or {})
        _emb_dim = int(_ingest_components["embedder"].dimension)
        _ingestion_overrides["embedding_dim"] = _emb_dim
        _ingestion_overrides["collection_name"] = _embedder_collection_name(
            "component:"
            + _ingest_component_fingerprint("embedder", task.config_overrides["embedder"]),
            _emb_dim,
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

    # Generation needs an LLM. Checked before the corpus is ingested: without
    # one every answer failed after the whole run, with a traceback, and the
    # task's retrieval numbers were discarded with the generation ones.
    _generation_skipped: Optional[str] = None
    if (not task.tracks or "generation" in task.tracks) and any(
        s.ground_truth or _is_unanswerable(s) for s in samples
    ):
        from .synthesize import llm_configured, missing_llm_message

        if not llm_configured():
            _generation_skipped = missing_llm_message(
                "The generation track",
                "To measure retrieval only, run with --tracks retrieval (tracks: [retrieval]).",
            )
            if task.tracks and "retrieval" not in task.tracks:
                return TaskResult(
                    task_key=task.task_key,
                    benchmark_name=task.benchmark_name,
                    config_name=task.config_name,
                    run_index=task.run_index,
                    error=_generation_skipped,
                )

    # Ensure corpus is ingested (first caller ingests, rest reuse)
    corpus_info = await _ensure_corpus_ingested(
        task.benchmark_name,
        samples,
        corpus_key=_corpus_key,
        ingestion_overrides=_ingestion_overrides,
        max_corpus_docs=task.max_corpus_docs,
        components=_ingest_components,
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
    # Results name documents by the benchmark's own ids: the internal ids are
    # per corpus session, and arms compared query by query need the same ids.
    valid_to_corpus = corpus_info.get("valid_to_corpus") or {
        v: c for c, v in corpus_to_valid.items()
    }

    # Populate corpus_fingerprint on the task (used by ingestion signature)
    if hasattr(task, "corpus_fingerprint"):
        task.corpus_fingerprint = corpus_info.get("corpus_fingerprint", "")
    # Where the corpus came from. Hashed into the comparability key, so a
    # pooled-gold number can never share a leaderboard row with a real one.
    task.corpus_provenance = corpus_info.get("corpus_provenance", "native")
    task.corpus_truncated = bool(corpus_info.get("corpus_truncated", False))
    task.corpus_ingest_gap = corpus_info.get("corpus_ingest_gap")
    refusal = _partial_corpus_refusal(task.corpus_ingest_gap, task.ingest_failure_tolerance)
    if refusal:
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=refusal,
        )

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
    gen_samples = [s for s in samples if s.ground_truth or _is_unanswerable(s)]
    if task.tracks:
        # Skipped by request, not by failure: nothing is scored for the
        # track that did not run.
        if "retrieval" not in task.tracks:
            ir_samples = []
        if "generation" not in task.tracks:
            gen_samples = []
    if _generation_skipped and gen_samples:
        gen_samples = []
        if not ir_samples:
            await _release_corpus(task.benchmark_name, corpus_key=_corpus_key)
            return TaskResult(
                task_key=task.task_key,
                benchmark_name=task.benchmark_name,
                config_name=task.config_name,
                run_index=task.run_index,
                error=_generation_skipped,
            )

    # Build pipeline from profile and apply task-level overrides.
    pipeline = KnowledgePipeline.from_profile(task.profile)
    _apply_overrides_to_pipeline(pipeline, task.config_overrides, shared_session_id)
    # The pipeline queries with these same instances: a second one built by the
    # pipeline had no arm settings, so a sentence-transformers embedder arm
    # embedded its corpus with its model and its queries with the global one.
    for _slot, _instance in _ingest_components.items():
        pipeline._instances[_slot] = _instance

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
            logger.debug("Native hybrid available: skipping BM25 fallback prebuild")
        elif not _BM25_AVAIL:
            # Guard 2: rank_bm25 not installed
            logger.debug("BM25 fallback not available: skipping prebuild")
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
                _vec = _backend._vector_backend
                _coll = pipeline._config.get("collection_name", _rag_cfg.collection_name)
                # Fetch every chunk for this session. This must be exhaustive:
                # a BM25 index built from part of the corpus still returns
                # results, so a truncated scan produces a plausible retrieval
                # number from a corpus that was never fully indexed. `iter_all`
                # uses a native cursor rather than limit+offset paging, which
                # previously failed outright past 16384 rows.
                _filt = _vec.session_filter(_coll, shared_session_id)
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
                        f"{shared_session_id}: BM25 hybrid search will use "
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
        r_precision,
        complete_recall_at_k,
    )

    relevances: List[Dict[str, int]] = []
    retrieved_lists: List[List[str]] = []
    _short_lists = 0  # queries left with fewer than k distinct documents
    _exhausted_lists = 0  # ...of which the retriever returned fewer than requested
    per_sample_scores: List[Dict[str, float]] = []
    #: Reasons this task's numbers cannot be trusted. Any entry fails the task
    #: (TaskResult.error) instead of letting a failure stand in as a score.
    _eval_failures: List[str] = []
    _n_generations = 0
    _failed_generations = 0
    total_dropped = 0
    #: The judge's spend, accumulated across every RAGAS pass in this task.
    eval_meter = UsageMeter()
    #: Set when the generator read no passages, for the reports' footnote.
    _context_free = False

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
    # - default_top_k capped the dense candidates below the requested depth;
    # - context reordering (on in standard, balanced and max_quality) put even
    #   ranks first and odd ranks reversed at the end, so the metrics scored
    #   the order the generator reads, with rank 2 last.
    _fix_retrieval_depth(task_config, _retrieval_depth)
    # Generation keeps its answer-time policies (they are the system under
    # test), but every run must execute rather than replay the cache.
    gen_config["enable_search_cache"] = False

    # Dropped only here, after the retrieval settings above, so the arm's
    # recorded configuration matches its control's in every stage but the
    # retriever, which is the one it names.
    if _no_rag(task) and ir_samples:
        logger.info(
            f"[{task.benchmark_name}] arm '{task.config_name}' sets no_rag: retrieval "
            f"track skipped, since the arm retrieves nothing"
        )
        ir_samples = []

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
            with sample_meter, sample_meter.scope(SYSTEM), observe_reranking() as moved:
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
            if len(retrieved_ids) < _metric_k:
                _short_lists += 1
                if len(search_results) < _retrieval_depth:
                    _exhausted_lists += 1

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
                # Over the list already cut to k, so the key names the cutoff.
                f"r_precision@{_metric_k}": r_precision(mapped_expected, retrieved_ids),
                f"complete_recall@{_metric_k}": complete_recall_at_k(
                    mapped_expected, retrieved_ids, k=_metric_k
                ),
                "latency_ms": query_latency_ms,
                # The full list, not the first 20. The cap predated any reader:
                # nothing consumed these, so nobody noticed that a run with
                # top_k above 20 recorded a truncated view of its own results.
                "_retrieved_ids": [valid_to_corpus.get(d, d) for d in retrieved_ids],
                "_retrieved": [
                    {**r, "doc_id": valid_to_corpus.get(r["doc_id"], r["doc_id"])}
                    for r in retrieved_detail
                ],
                # What the query was *supposed* to find, so a per-query row can
                # be read without the dataset alongside it.
                "_expected_docs": sorted(valid_to_corpus.get(d, d) for d in mapped_expected),
            }

            if moved:
                # Whether the reranker component changed its candidates' order:
                # the isolation check flags an arm whose reranker never did.
                sample_metrics["_reranker_moved"] = any(moved)

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
        if _short_lists:
            # A list shorter than k is scored as if the missing ranks held
            # nothing relevant, so the metric drops with no other sign. Long
            # documents split into more chunks than the over-fetch covers.
            metrics["short_lists"] = float(_short_lists)
            logger.warning(
                _short_list_warning(
                    _short_lists,
                    _exhausted_lists,
                    len(retrieved_lists),
                    _metric_k,
                    _chunk_overfetch,
                    _short_retrieval(
                        {**task_config, "enable_hybrid_search": _enable_hybrid},
                        pipeline._component_names.get("retriever"),
                    ),
                )
            )

        # What retrieval scored, kept if generation fails below.
        _retrieval_metric_keys = set(metrics) if ir_samples else None
        _retrieval_rows = len(per_sample_scores)

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
            gen_answerable: List[bool] = []
            gen_agentic_answers: List[str] = []  # agentic pipeline's own answers
            _has_unanswerable = any(_is_unanswerable(s) for s in gen_samples)
            _abstain = _ABSTAIN_INSTRUCTION if _has_unanswerable else ""
            # A named generator writes the answers; with none, the prompt
            # below does, unchanged, so earlier results stay reproducible.
            _generator = (
                pipeline.get_component("generator")
                if pipeline._component_names.get("generator")
                else None
            )
            if _no_rag(task):
                _generator = _no_rag_generator(_generator, task.config_overrides)
            # A generator that reads no passages skips retrieval here: nothing
            # it would return reaches the answer, and an agentic profile would
            # bill this arm for a search it never used. One that reads more
            # (long_context) gets that many, and the judge sees the same ones;
            # with none named, the default stays the first five.
            _needs_context = getattr(_generator, "needs_context", True)
            _context_k = getattr(_generator, "context_k", None) or 5
            _context_free = not _needs_context
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
                search_results = []
                # The search returns its list in ranked order and says whether
                # it would have reordered it; the passages kept are reordered
                # below. Reordered by the search, the first five of its ten
                # were ranks 1, 3, 5, 7 and 9.
                _presented = ContextPresentation()
                if _needs_context:
                    try:
                        # 5-min per-sample timeout: guards against silent LLM gateway
                        # hangs where the TCP connection is established but the server
                        # never sends a response (no exception, just indefinite block).
                        with (
                            gen_meter,
                            gen_meter.scope(SYSTEM),
                            present_context_order() as _presented,
                        ):
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
                            f"Query timed out after 5 min for sample "
                            f"{gen_sample.query_id}: skipping it."
                        )
                        search_results = []
                query_latency_ms = (time.monotonic() - query_start) * 1000

                # Extract text from retrieved chunks, in ranked order.
                contexts = [
                    r.chunk.content
                    for r in search_results
                    if hasattr(r, "chunk") and hasattr(r.chunk, "content") and r.chunk.content
                ]

                def _shown(k: int, _contexts=contexts, _reorder=_presented.reorder) -> List[str]:
                    """The first k passages, in the order the generator reads them."""
                    kept = list(_contexts[:k])
                    return lost_in_the_middle(kept) if _reorder else kept

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

                # Generate the answer from the retrieved context: the arm's
                # generator when it names one, else the fixed prompt below.
                answer = ""
                # The successful call only: back-off sleeps between retries are
                # the gateway's latency, not the system's.
                _generation_ms = 0.0

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
                        _attempt_t0 = time.monotonic()
                        with gen_meter, gen_meter.scope(SYSTEM):
                            answer, _drafts = await spec_rag.generate(
                                query=gen_sample.question,
                                contexts=_shown(10),
                            )
                        _generation_ms = (time.monotonic() - _attempt_t0) * 1000
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

                if (
                    not _speculative
                    and (answer_llm or _generator)
                    and (contexts or not _needs_context)
                ):
                    ctx_text = "\n\n".join(_shown(_context_k))
                    prompt = (
                        "Answer the following question in ONE concise sentence "
                        "based only on the provided context. "
                        "Do not explain or elaborate. "
                        f"{_abstain}\n\n"
                        f"Context:\n{ctx_text}\n\n"
                        f"Question: {gen_sample.question}\n\nAnswer:"
                    )
                    # Retry up to 5 times with aggressive backoff — agentic
                    # pipeline's parallel LLM calls trigger gateway rate limits
                    # that can take 30-60 s to recover (5s, 15s, 30s, 60s waits).
                    _backoffs = [5, 15, 30, 60]
                    for _attempt in range(5):
                        try:
                            _attempt_t0 = time.monotonic()
                            with gen_meter, gen_meter.scope(SYSTEM):
                                if _generator is not None:
                                    raw = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            _generator.generate,
                                            gen_sample.question,
                                            _shown(_context_k),
                                            llm=answer_llm,
                                            instructions=_abstain,
                                        ),
                                        timeout=120,
                                    )
                                elif hasattr(answer_llm, "ainvoke"):
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
                            _generation_ms = (time.monotonic() - _attempt_t0) * 1000
                            logger.debug(
                                f"Generated answer for '{gen_sample.question[:40]}': "
                                f"'{answer[:80]}'"
                            )
                            break  # success
                        except Exception as exc:
                            exc_str = str(exc)
                            # Do NOT retry content-policy / bad-request errors
                            # (400) — same prompt will be rejected again.
                            if "needs an LLM" in exc_str:
                                # Retrying cannot conjure a model: fail this
                                # answer now instead of after 110 s of backoff.
                                logger.warning(f"Answer generation impossible: {exc}")
                                break
                            if (
                                "400" in exc_str
                                or "content_filter" in exc_str
                                or "ContentPolicy" in exc_str
                            ):
                                logger.warning(
                                    f"Answer generation blocked by content filter "
                                    f"for sample {gen_sample.query_id}: skipping."
                                )
                                break
                            if _attempt < 4:
                                _backoff = _backoffs[_attempt]
                                logger.warning(
                                    f"Answer generation attempt {_attempt + 1}/5 failed "
                                    f"(sample {gen_sample.query_id}): {exc}. "
                                    f"Retrying in {_backoff}s..."
                                )
                                await asyncio.sleep(_backoff)
                            else:
                                logger.warning(
                                    f"Answer generation failed after 5 attempts for "
                                    f"sample {gen_sample.query_id}: {exc}"
                                )

                _n_generations += 1
                if not (answer or "").strip():
                    # No answer: no context, no generator, or every attempt
                    # failed. Scored, it would read as an unfaithful answer.
                    _failed_generations += 1
                gen_questions.append(gen_sample.question)
                gen_answers.append(answer)
                # Pass the SAME contexts used for generation to RAGAS.
                # Standard generation uses the generator's context_k (top 5
                # unless it says otherwise); speculative RAG uses top 10.
                # Must match to avoid faithfulness metric mismatch. In ranked
                # order, whatever order the generator read them in: context
                # precision weighs each passage by its rank.
                _ctx_k = 10 if _speculative else _context_k
                gen_contexts.append(contexts[:_ctx_k])
                gen_ground_truths.append(gen_sample.ground_truth)
                gen_answerable.append(not _is_unanswerable(gen_sample))
                gen_agentic_answers.append(_agentic_answer)

                _src = (
                    gen_sample.metadata.get("source_benchmark", "")
                    if hasattr(gen_sample, "metadata")
                    else ""
                )
                # ── EM / Token-F1 (free, no LLM calls) ──────────────
                # Scored on the answer without citation markers, as ALCE does.
                from .answer_metrics import strip_citations

                _scored_answer = strip_citations(answer)
                _em_score = 0.0
                _f1_score_val = 0.0
                _gt = gen_sample.ground_truth or ""
                _gt_aliases = getattr(gen_sample, "ground_truth_answers", None) or []
                if _is_unanswerable(gen_sample):
                    _em_score = _f1_score_val = _no_answer_em(_scored_answer)
                elif _gt.strip() and _scored_answer.strip():
                    from .answer_metrics import exact_match as _em_fn, token_f1 as _tf1_fn

                    _em_score = _em_fn(_scored_answer, _gt, aliases=_gt_aliases)
                    _f1_score_val = _tf1_fn(_scored_answer, _gt, aliases=_gt_aliases)

                _gen_sample_metrics: Dict[str, Any] = {
                    "_sample_id": gen_sample.query_id,
                    "_source_benchmark": _src,
                    # End to end for an answer row: it used to stop before the
                    # answer was written, so p50/p95 left out generation.
                    "latency_ms": query_latency_ms + _generation_ms,
                    "retrieval_ms": query_latency_ms,
                    "generation_ms": _generation_ms,
                    "n_contexts": float(len(contexts)),
                    "agentic_fallback": 1.0 if fallback else 0.0,
                    "exact_match": _em_score,
                    "token_f1": _f1_score_val,
                }
                _gen_sample_metrics.update(_answer_scores(_scored_answer, gen_sample))
                if _has_unanswerable:
                    _gen_sample_metrics["_answerable"] = (
                        0.0 if _is_unanswerable(gen_sample) else 1.0
                    )

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
                    _gen_sample_metrics["_retrieved_contexts"] = [
                        c[:500] for c in _shown(_context_k)
                    ]
                    _gen_sample_metrics["_ground_truth"] = _gt[:500]
                    _gen_sample_metrics["_question"] = gen_sample.question[:500]
                elif _save_traj == "minimal":
                    _gen_sample_metrics["_generated_answer"] = answer[:500]
                    _gen_sample_metrics["_question"] = gen_sample.question[:200]

                per_sample_scores.append(_gen_sample_metrics)

            # Semantic similarity on a local embedder, no judge. The generation
            # rows follow the retrieval rows, one per generation sample.
            _gen_rows = per_sample_scores[len(ir_samples) :]
            from .answer_metrics import strip_citations

            await asyncio.to_thread(
                _add_answer_similarity,
                _gen_rows,
                [strip_citations(a) for a in gen_answers],
                gen_samples,
            )

            # Run RAGAS on all collected generation samples that have a
            # reference answer (an unanswerable question has none).
            if any(gen_answerable):
                try:
                    from .ragas_evaluator import RAGASEvaluator, harness_metrics

                    # RAG_EVAL_JUDGE_METRICS or the harness set, for both the
                    # regenerated and the agentic answers; without passages,
                    # only the metrics that do not judge them.
                    _judged_names = harness_metrics(has_context=_needs_context)

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
                                questions=[gen_questions[i] for i in _judged],
                                answers=[gen_answers[i] for i in _judged],
                                contexts=[gen_contexts[i] for i in _judged],
                                ground_truths=[gen_ground_truths[i] for i in _judged],
                                metrics=_judged_names,
                                batch_size=4,
                            )

                    # The judge needs a reference answer: unanswerable
                    # questions are scored by EM/F1 above, not by RAGAS.
                    _judged = [i for i, ok in enumerate(gen_answerable) if ok]

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
                            if i >= len(_judged):
                                break
                            _target_idx = _gen_ps_offset + _judged[i]
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
                        _ag_indices = [
                            i
                            for i in filter_nonempty_agentic_indices(gen_agentic_answers)
                            if gen_answerable[i]
                        ]
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
                                        metrics=_judged_names,
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
                                f"{task.task_key}: skipping agentic eval"
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
                            base_keys=_judged_names,
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

            # ALCE citation quality, for answers that cite (a citing
            # generator); absent otherwise. Judge calls count as eval spend.
            def _citations_in_thread():
                with eval_meter, eval_meter.scope(EVAL):
                    return _add_citation_scores(_gen_rows, gen_answers, gen_contexts)

            _citation_failure = await asyncio.to_thread(_citations_in_thread)
            if _citation_failure:
                _eval_failures.append(_citation_failure)

            # ── Aggregate EM / Token-F1 (outside RAGAS try block) ─────────
            # Computed per-sample above; now aggregate across gen samples.
            _gen_ps = [s for s in per_sample_scores if "exact_match" in s]
            if _gen_ps:
                metrics["exact_match"] = sum(s["exact_match"] for s in _gen_ps) / len(_gen_ps)
                metrics["token_f1"] = sum(s["token_f1"] for s in _gen_ps) / len(_gen_ps)
                metrics.update(_answerability_split(_gen_ps))
                metrics.update(_mean_present(_gen_ps, _ANSWER_METRICS))

        # Tokens and LLM calls per query, over both tracks' rows.
        metrics.update(_per_query_costs(per_sample_scores))

    finally:
        # Release corpus ref — always called even if evaluation raises, so the
        # shared corpus is not leaked when a task errors or is cancelled.
        await _release_corpus(task.benchmark_name, corpus_key=_corpus_key)

    changed = await asyncio.to_thread(_corpus_changed, corpus_info)
    if changed:
        logger.error(changed)
        return TaskResult(
            task_key=task.task_key,
            benchmark_name=task.benchmark_name,
            config_name=task.config_name,
            run_index=task.run_index,
            error=changed,
        )

    if _n_generations:
        metrics["generations_failed"] = float(_failed_generations)
        # A few failures (a content filter, one timeout) are reported; more than
        # a tenth means the generation metrics describe the outage, not the system.
        if _failed_generations / _n_generations > _MAX_FAILED_GENERATION_RATE:
            _eval_failures.append(
                f"{_failed_generations} of {_n_generations} answers could not be generated"
            )

    _system = _ran_system(
        _corpus_model,
        _corpus_dim,
        _ingestion_overrides,
        task_config,
        reranker_component=(task.config_overrides or {}).get("reranker"),
    )
    if _generation_skipped:
        _eval_failures.append(_generation_skipped)
    if _eval_failures and _retrieval_metric_keys:
        # Retrieval ran and was scored; only generation failed. Its numbers
        # would describe the failure, so they go, and the retrieval ones stay
        # with the reason: the CLI names it and exits non-zero.
        _system["generation_failed"] = "; ".join(_eval_failures)
        metrics = {k: v for k, v in metrics.items() if k in _retrieval_metric_keys}
        per_sample_scores = per_sample_scores[:_retrieval_rows]
        _eval_failures = []
    # What reached each stage, from the config the search actually resolved
    # (session plus this arm's overrides) and the components it named; the
    # bundle's isolation check compares these across arms.
    try:
        from lasagna.core.config import rag_config as _rc

        from .stages import stage_manifest

        _system["stages"] = stage_manifest(
            _rc.with_session(shared_session_id).with_query_overrides(task_config),
            {
                k: v
                for k, v in (task.config_overrides or {}).items()
                if k in _COMPONENT_KEYS and isinstance(v, str)
            },
            harness={
                "metric_k": _metric_k,
                "retrieval_chunk_overfetch": _chunk_overfetch,
                # Only when set, so every other arm's retriever digest is the
                # one it always had.
                **({"no_rag": True} if _no_rag(task) else {}),
                # The order each track used: the retrieval track always scores
                # the ranked list; the generator reads its passages reordered
                # when the arm says so. Without this the manifest showed
                # reordering off (the retrieval setting) for an arm whose
                # generator read the reordered list.
                "context_order": _context_order(
                    _rc.with_session(shared_session_id).with_query_overrides(gen_config)
                ),
            },
        )
    except Exception as exc:  # the manifest must never cost a result
        logger.warning(f"Could not record the stage manifest: {exc}")
    if _context_free:
        # Read by the reports, which say why this arm has no faithfulness.
        _system["generator_context"] = "none"

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
        system=_system,
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
