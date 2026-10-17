"""Which pipeline stage each setting belongs to, and a check that a swap stayed in it.

A comparison between two arms is only attributable when the arms differ in the
stage under test and nowhere else. The registry gives every stage a fixed
interface; this module checks the configuration half of the claim, at runtime:
each task records a **stage manifest** (a digest of every setting and component
that reached each stage), and :func:`isolation_report` compares the manifests
of a sweep's arms against what each arm declared it would change.

* A stage whose configuration differs between two arms although neither arm
  named it is an **undeclared change**: a setting leaked across a stage
  boundary (a profile default, a harness bug). The run says so and
  ``lasagna results validate`` refuses the bundle.
* An arm that names a change its manifest does not show is **inert**: the
  setting or component never reached the pipeline, so its number is the
  baseline's under another name.

The manifest digests configuration, not behaviour. A component that changes
state outside its stage at runtime (patching another module, say) leaves every
digest unchanged and passes this check; the stage contracts and their
conformance tests are what guard against that.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

#: The eight swappable stages, plus "other" for settings outside them.
STAGES: Tuple[str, ...] = (
    "document_processor",
    "chunker",
    "embedder",
    "retriever",
    "query_enhancer",
    "reranker",
    "agentic_pipeline",
    "generator",
)
OTHER = "other"

#: Component slots an arm can name, by the stage they fill.
COMPONENT_STAGES: Dict[str, str] = {stage: stage for stage in STAGES}

# Settings that change how a run is operated, not what it computes. They are
# left out of the manifest so that, say, a cache or a log level never reads as a
# change to the system under test.
_OPERATIONAL = frozenset(
    {
        "allow_model_download_in_requests",
        "async_processing",
        "batch_size",
        "bm25_max_sessions",
        "bm25_session_ttl_seconds",
        "cache_backend",
        "celery_broker_url",
        "celery_enabled",
        "celery_result_backend",
        "citation_db_max_age_hours",
        "citation_db_max_per_session",
        "citation_db_path",
        "citation_format",
        "citation_id_length",
        "citation_normalize_scores",
        "citation_score_method",
        "connection_pool_size",
        "database_backend",
        "database_dsn",
        "database_pool_max",
        "database_pool_min",
        "db_name",
        "db_path",
        "docling_timeout",
        "embedding_cache_size",
        "enable_embedding_cache",
        "enable_prometheus",
        "enable_search_cache",
        "enable_tracing",
        "include_source_preview",
        "ingest_embed_batch_size",
        "ingest_extract_workers",
        "ingest_insert_batch_size",
        "lifecycle",
        "log_level",
        "max_concurrent_uploads",
        "max_document_ids_filter",
        "max_file_size",
        "max_query_length",
        "processing_timeout",
        "quota_enabled",
        "redis_pool_size",
        "redis_url",
        "reranker_batch_size",
        "routing_log_decisions",
        "search_cache_max_size",
        "search_cache_ttl",
        "show_confidence_scores",
        "show_page_numbers",
        "storage_backend",
        "storage_backend_config",
        "supported_extensions",
        "tenant_isolation",
        "track_citations_db",
        "tracing_endpoint",
        "tracing_exporter",
        "tracing_service_name",
        "upload_dir",
        "upload_dir_name",
        "use_gpu",
        "verify_citations",
    }
)

_EXACT: Dict[str, str] = {
    # document processing
    "enable_ocr": "document_processor",
    "extract_images": "document_processor",
    "extract_metadata": "document_processor",
    "preserve_formatting": "document_processor",
    # chunking and ingestion-time enrichment
    "chunking_strategy": "chunker",
    "chunk_size": "chunker",
    "chunk_overlap": "chunker",
    "min_chunk_size": "chunker",
    "max_chunk_size": "chunker",
    "enable_hierarchical": "chunker",
    "parent_chunk_size": "chunker",
    "child_chunk_size": "chunker",
    "window_size": "chunker",
    "sentence_overlap": "chunker",
    "semantic_threshold": "chunker",
    "preserve_headers": "chunker",
    "preserve_tables": "chunker",
    "enable_late_chunking": "chunker",
    "enable_contextual_retrieval": "chunker",
    "enable_raptor": "chunker",
    "document_awareness": "chunker",
    # Every chunk is cut to this many characters before it is embedded and
    # indexed, so it decides what the corpus holds; filed as operational, an
    # arm could not change it and the manifest did not record it.
    "max_content_length": "chunker",
    # embedding and the index it lives in
    "normalize_embeddings": "embedder",
    "collection_name": "embedder",
    "metric_type": "embedder",
    "index_type": "embedder",
    "index_params": "embedder",
    "vector_backend": "embedder",
    "vector_backend_config": "embedder",
    # retrieval
    "enable_hybrid_search": "retriever",
    "hybrid_alpha": "retriever",
    "rrf_k": "retriever",
    "sparse_method": "retriever",
    "enable_bm25_stemming": "retriever",
    "default_top_k": "retriever",
    "similarity_threshold": "retriever",
    "search_params": "retriever",
    "auto_merge_threshold": "retriever",
    "chunk_cite_retrieval": "retriever",
    "enable_graph_retrieval": "retriever",
    "final_search_mode": "retriever",
    "use_final_search": "retriever",
    # query enhancement and routing
    "enable_hyde": "query_enhancer",
    "enable_query_expansion": "query_enhancer",
    "max_query_variations": "query_enhancer",
    "enable_query_decomposition": "query_enhancer",
    "enable_rag_fusion": "query_enhancer",
    "rag_fusion_n_queries": "query_enhancer",
    "enable_query_classification": "query_enhancer",
    "enable_query_routing": "query_enhancer",
    "custom_query_enhancer": "query_enhancer",
    "agentic_hyde_strategy": "query_enhancer",
    # reranking and diversity
    "enable_reranking": "reranker",
    "reranker_model": "reranker",
    "rerank_top_k": "reranker",
    "enable_mmr": "reranker",
    "colbert_model": "reranker",
    "custom_reranker": "reranker",
    "enable_context_reordering": "reranker",
    # agentic retrieval and its quality gates
    "use_agentic_rag": "agentic_pipeline",
    "agentic_pipeline": "agentic_pipeline",
    "use_structured_graph": "agentic_pipeline",
    "agentic_models": "agentic_pipeline",
    "agentic_prompts": "agentic_pipeline",
    "agent_temperature": "agentic_pipeline",
    "max_agent_iterations": "agentic_pipeline",
    "enable_crag": "agentic_pipeline",
    "crag_web_search_fallback": "agentic_pipeline",
    "enable_hallucination_check": "agentic_pipeline",
    "enable_answer_quality_check": "agentic_pipeline",
    "max_quality_check_attempts": "agentic_pipeline",
    "chunk_refinement_method": "agentic_pipeline",
    "coverage_scorer_mode": "agentic_pipeline",
    "enable_speculative_rag": "agentic_pipeline",
    "enable_context_compression": "agentic_pipeline",
    "compression_method": "agentic_pipeline",
    "compression_ratio": "agentic_pipeline",
    # built-in retriever components
    "bm25_k1": "retriever",
    "bm25_b": "retriever",
    "splade_model": "retriever",
    "hybrid_fusion_depth": "retriever",
    "hybrid_convex_normalization": "retriever",
    # built-in reranker components
    "monot5_model": "reranker",
    "flashrank_model": "reranker",
    "cohere_rerank_model": "reranker",
    # size units for the fixed and recursive chunkers
    "chunk_unit": "chunker",
    "chunk_tokenizer": "chunker",
}

_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("docling_", "document_processor"),
    ("ocr_", "document_processor"),
    ("raptor_", "chunker"),
    ("embedding_", "embedder"),
    ("graph_", "retriever"),
    ("hyde_", "query_enhancer"),
    ("routing_", "query_enhancer"),
    ("reranker_", "reranker"),
    ("mmr_", "reranker"),
    ("llm_reranker_", "reranker"),
    ("ensemble_reranker_", "reranker"),
    ("auto_refine_", "agentic_pipeline"),
    ("speculative_", "agentic_pipeline"),
    ("generator_", "generator"),
    ("query2doc_", "query_enhancer"),
    ("self_ask_", "agentic_pipeline"),
    ("ircot_", "agentic_pipeline"),
    ("flare_", "agentic_pipeline"),
    ("adaptive_rag_", "agentic_pipeline"),
    ("self_rag_", "agentic_pipeline"),
    ("recursive_", "chunker"),
    ("markdown_", "chunker"),
    ("semantic_", "chunker"),
    ("proposition_", "chunker"),
    ("contextual_", "chunker"),
    ("late_chunking_", "chunker"),
)


#: Harness settings: the IR cutoff and the over-fetch set how deep retrieval
#: goes, so they count as retriever changes, and ``no_rag`` replaces retrieval
#: with nothing, so it is the retriever's change too. ``context_order`` (which
#: order each track read) belongs with ``enable_context_reordering``, under the
#: reranker. The rest only steer recording.
_HARNESS_STAGES: Dict[str, Optional[str]] = {
    "metric_k": "retriever",
    "retrieval_chunk_overfetch": "retriever",
    "no_rag": "retriever",
    "context_order": "reranker",
    "allow_inert_overrides": None,
    "save_trajectories": None,
}

#: Strategies named after a structural setting; with it off, ``semantic`` runs.
_STRUCTURAL = frozenset({"structure_aware", "hierarchical"})

#: The settings that decide which built-in chunker runs.
CHUNKER_DISPATCH_KEYS = (
    "chunking_strategy",
    "enable_hierarchical",
    "preserve_headers",
    "preserve_tables",
)


def effective_chunking_strategy(settings: Mapping[str, Any]) -> Optional[str]:
    """The built-in chunker that runs for these chunker settings.

    ``enable_hierarchical`` wins, then ``preserve_headers`` or
    ``preserve_tables`` (``structure_aware``), and a structural strategy whose
    setting is off runs ``semantic``. This is the dispatch order of
    ``processing.chunker.semantic.resolve_effective_strategy``, here without
    its warning or its import cost so records can name what ran; a test holds
    the two equal.
    """
    strategy = settings.get("chunking_strategy")
    strategy = getattr(strategy, "value", strategy)
    if settings.get("enable_hierarchical"):
        return "hierarchical"
    if settings.get("preserve_headers") or settings.get("preserve_tables"):
        return "structure_aware"
    if strategy in _STRUCTURAL:
        return "semantic"
    return None if strategy is None else str(strategy)


def stage_of(key: str) -> Optional[str]:
    """The stage a setting or component slot belongs to; None if operational."""
    if key in COMPONENT_STAGES:
        return COMPONENT_STAGES[key]
    if key in _HARNESS_STAGES:
        return _HARNESS_STAGES[key]
    if key in _OPERATIONAL:
        return None
    if key in _EXACT:
        return _EXACT[key]
    for prefix, stage in _PREFIXES:
        if key.startswith(prefix):
            return stage
    return OTHER


def _effective(config: Any, key: str) -> Any:
    # Properties resolve the full precedence chain (per-query overrides, then
    # the session, environment, settings file and defaults); the raw dict is
    # the fallback for keys without one.
    if hasattr(type(config), key):
        try:
            return getattr(config, key)
        except Exception:
            pass
    # No property: resolve by hand, in the same order. Reading the raw dict
    # alone missed the arm's own override (document_awareness, a new knob), so
    # the arm digested like the control and was called inert.
    lookup = getattr(config, "_get_session_override", None)
    if callable(lookup):
        try:
            value = lookup(key)
        except Exception:
            value = None
        if value is not None:
            return value
    env = getattr(config, "_env_overrides", None) or {}
    if key in env:
        return env[key]
    raw = getattr(config, "_config", None) or {}
    return raw.get(key)


def _digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def stage_manifest(
    config: Any,
    components: Optional[Mapping[str, Any]] = None,
    harness: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Digest every setting and component that reached each stage.

    Args:
        config: The effective config the run searched with (a ``RAGConfig``
            resolved for the session and the arm's overrides).
        components: Component names the arm put in their slots, with their
            identity (registry name or ``module:function``).
        harness: The harness settings the evaluation used (``metric_k``, the
            over-fetch). They never reach the pipeline config, so they are
            passed here and filed under their stage like any other setting.

    Returns:
        ``{stage: {"digest", "component", "settings"}}``. ``settings`` holds the
        values, so a mismatch can name the keys that differ.
    """
    # The arm's own overrides too: a key with no default is in no base dict.
    keys = sorted(
        set(getattr(config, "_config", {}) or {})
        | set(getattr(config, "_query_overrides", {}) or {})
    )
    by_stage: Dict[str, Dict[str, Any]] = {s: {} for s in (*STAGES, OTHER)}
    for key in keys:
        stage = stage_of(key)
        if stage is not None:
            by_stage[stage][key] = _effective(config, key)
    for key, value in (harness or {}).items():
        stage = stage_of(key)
        if stage is not None:
            by_stage[stage][key] = value
    chunking = by_stage["chunker"]
    if "chunking_strategy" in chunking and not (components or {}).get("chunker"):
        # The strategy that runs, not the one named: structural settings
        # shadow it, so two arms naming different strategies can chunk alike
        # (and should digest alike). A chunker component replaces the
        # built-in dispatch, so its arms keep the setting as given.
        chunking["chunking_strategy"] = effective_chunking_strategy(chunking)

    manifest: Dict[str, Any] = {}
    for stage, settings in by_stage.items():
        component = (components or {}).get(stage)
        manifest[stage] = {
            "digest": _digest({"component": component, "settings": settings}),
            "component": component,
            "settings": settings,
        }
    return manifest


def declared_stages(overrides: Optional[Mapping[str, Any]]) -> Set[str]:
    """The stages an arm says it changes, read off its overrides."""
    out: Set[str] = set()
    for key in overrides or {}:
        stage = stage_of(key)
        if stage is not None:
            out.add(stage)
    return out


def _arm_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    """One row per (benchmark, arm): the first run that carries a manifest."""
    arms: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        stages = (row.get("system") or {}).get("stages")
        if not stages or row.get("error"):
            continue
        key = (str(row.get("benchmark_name")), str(row.get("config_name")))
        arms.setdefault(key, row)
    return arms


def _differing_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    keys = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    if a.get("component") != b.get("component"):
        keys.insert(0, "component")
    return keys


#: The stages whose settings can change what retrieval returns. Two arms that
#: differ only here and return the same ranked lists on every query changed
#: nothing measurable. (Agentic settings and anything unclassified can change
#: the answer without touching retrieval, so they are not judged this way.)
_RETRIEVAL_STAGES = frozenset(
    {"document_processor", "chunker", "embedder", "retriever", "query_enhancer", "reranker"}
)


def _ranked_lists(row: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Each query's ranked document ids, by sample id, from a row's per-query scores."""
    out: Dict[str, Tuple[str, ...]] = {}
    for sample in row.get("per_sample_scores") or []:
        ids, sid = sample.get("_retrieved_ids"), sample.get("_sample_id")
        if ids is not None and sid not in (None, ""):
            out[str(sid)] = tuple(str(i) for i in ids)
    return out


def _differing_overrides(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    a, b = a or {}, b or {}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and stage_of(k) is not None)


#: Fewest queries on which an arm that never changed a ranking is called inert.
#: On two questions of one's own data a cross-encoder can keep the order it was
#: given because that order was already right, and the bundle was marked
#: INERT_ARM for a reranker that works.
MIN_QUERIES_TO_CALL_INERT = 10


def _too_few(bench: str, what: str, n: int) -> str:
    return (
        f"{bench}: {what} on all {n} queries, too few queries to tell whether it "
        f"changes anything (at least {MIN_QUERIES_TO_CALL_INERT} are needed)"
    )


def _same_output(
    entries: List[Tuple[str, Mapping[str, Any]]], bench: str, too_few: List[str]
) -> List[str]:
    """Pairs of arms whose differences should move a ranking and never did.

    This is the behavioural half of the inert check. A setting can reach its
    stage's config and still change nothing: MMR is read only when reranking
    is on, a technique can fail into a fallback, a component can be ignored.
    The config manifest cannot see that; the ranked lists can. A pair compared
    on fewer than ``MIN_QUERIES_TO_CALL_INERT`` queries goes to ``too_few``.
    """
    lines: List[str] = []
    lists = {name: _ranked_lists(row) for name, row in entries}
    for i, (name_a, a) in enumerate(entries):
        for name_b, b in entries[i + 1 :]:
            if a.get("profile") != b.get("profile"):
                continue
            keys = _differing_overrides(a.get("config_overrides"), b.get("config_overrides"))
            if not keys or not all(stage_of(k) in _RETRIEVAL_STAGES for k in keys):
                continue
            la, lb = lists[name_a], lists[name_b]
            shared = sorted(set(la) & set(lb))
            if shared and all(la[q] == lb[q] for q in shared):
                if len(shared) < MIN_QUERIES_TO_CALL_INERT:
                    too_few.append(
                        _too_few(
                            bench,
                            f"arms {name_a!r} and {name_b!r} returned identical ranked lists "
                            f"({', '.join(keys)} tells them apart)",
                            len(shared),
                        )
                    )
                    continue
                lines.append(
                    f"{bench}: arms {name_a!r} and {name_b!r} returned identical ranked lists "
                    f"on all {len(shared)} queries, so what tells them apart ({', '.join(keys)}) "
                    f"changed nothing here"
                )
    return lines


def _reranker_never_moved(
    entries: List[Tuple[str, Mapping[str, Any]]], bench: str, too_few: List[str]
) -> List[str]:
    """Arms whose reranker handed back its candidates in the order it got them.

    The pairwise check needs an arm without the change to compare against, and
    a reranker sweep has none: every arm names a reranker. A pass-through
    reranker's lists then differ from the other reranker's and nothing flagged
    it, though they are the base profile's lists under another name. Each
    query records whether its reranker moved anything; an arm where it never
    did is inert on its own.
    """
    lines: List[str] = []
    for name, row in entries:
        component = (row.get("config_overrides") or {}).get("reranker")
        moved = [
            s["_reranker_moved"]
            for s in row.get("per_sample_scores") or []
            if isinstance(s, Mapping) and "_reranker_moved" in s
        ]
        if component and moved and not any(moved):
            if len(moved) < MIN_QUERIES_TO_CALL_INERT:
                too_few.append(
                    _too_few(
                        bench,
                        f"arm {name!r} ran reranker {component!r}, which kept the order "
                        f"it was given",
                        len(moved),
                    )
                )
                continue
            lines.append(
                f"{bench}: arm {name!r} ran reranker {component!r}, which returned its "
                f"candidates in the order it was given on all {len(moved)} queries, so its "
                f"ranked lists are the {row.get('profile') or 'base'} profile's retrieval "
                f"order, not a reranking"
            )
    return lines


def compared_pairs(rows: Iterable[Mapping[str, Any]]) -> int:
    """How many pairs of arms the isolation check could compare.

    Zero when every arm but one failed, or none recorded a manifest: then
    "the arms differ only in the stages they name" would be a claim about
    nothing.
    """
    groups: Dict[Tuple[str, Any], int] = {}
    for (bench, _arm), row in _arm_rows(rows).items():
        key = (bench, row.get("profile"))
        groups[key] = groups.get(key, 0) + 1
    return sum(n * (n - 1) // 2 for n in groups.values())


def isolation_report(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Compare every pair of arms on each benchmark against what they declared.

    Returns ``{"undeclared": [...], "inert": [...], "unverified": [...]}`` as
    human-readable lines. Arms on different base profiles are compared only as
    whole pipelines, since a profile is a declared change to every stage.
    """
    rows = list(rows)
    arms = _arm_rows(rows)
    report: Dict[str, List[str]] = {"undeclared": [], "inert": [], "unverified": []}

    unverified = sorted(
        {
            str(r.get("config_name"))
            for r in rows
            if not r.get("error") and not (r.get("system") or {}).get("stages")
        }
    )
    if unverified:
        report["unverified"].append(
            f"no stage manifest recorded for {unverified}; isolation was not checked"
        )

    by_bench: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
    for (bench, arm), row in arms.items():
        by_bench.setdefault(bench, []).append((arm, row))

    for bench, entries in sorted(by_bench.items()):
        entries.sort()
        for i, (name_a, a) in enumerate(entries):
            for name_b, b in entries[i + 1 :]:
                if a.get("profile") != b.get("profile"):
                    continue
                declared = declared_stages(a.get("config_overrides")) | declared_stages(
                    b.get("config_overrides")
                )
                ma, mb = a["system"]["stages"], b["system"]["stages"]
                for stage in (*STAGES, OTHER):
                    sa, sb = ma.get(stage) or {}, mb.get(stage) or {}
                    if sa.get("digest") == sb.get("digest") or stage in declared:
                        continue
                    report["undeclared"].append(
                        f"{bench}: arms {name_a!r} and {name_b!r} differ at the {stage} "
                        f"stage, which neither names ({', '.join(_differing_keys(sa, sb))})"
                    )

        # An arm whose declared change shows up nowhere in its manifest, against
        # the arm that declares nothing (the control), never ran what it named.
        controls = [(n, r) for n, r in entries if not declared_stages(r.get("config_overrides"))]
        for name, row in entries:
            stages = declared_stages(row.get("config_overrides"))
            if not stages:
                continue
            for control_name, control in controls:
                if control.get("profile") != row.get("profile"):
                    continue
                mc, mr = control["system"]["stages"], row["system"]["stages"]
                if all(
                    (mc.get(s) or {}).get("digest") == (mr.get(s) or {}).get("digest")
                    for s in stages
                ):
                    report["inert"].append(
                        f"{bench}: arm {name!r} names {sorted(stages)} but its pipeline is "
                        f"identical to the control {control_name!r} there"
                    )
                break
        report["inert"].extend(_same_output(entries, bench, report["unverified"]))
        report["inert"].extend(_reranker_never_moved(entries, bench, report["unverified"]))
    return report
