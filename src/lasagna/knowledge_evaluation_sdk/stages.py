"""Which pipeline stage each setting belongs to, and a check that a swap stayed in it.

A comparison between two arms is only attributable when the arms differ in the
stage under test and nowhere else. The registry gives every stage a fixed
interface; this module checks the configuration half of the claim, at runtime:
each task records a **stage manifest** (a digest of every setting and component
that reached each stage), and :func:`isolation_report` compares the manifests
of a sweep's arms against what each arm declared it would change.

* A setting that differs between two arms although neither arm named it, and
  the harness does not set on a named change's account, is an **undeclared
  change**: a setting leaked across a stage boundary, or into a stage an arm
  names for another reason (a profile default, a harness bug). The run says
  so and ``lasagna results validate`` refuses the bundle.
* An arm that names a change its manifest does not show is **inert**: the
  setting or component never reached the pipeline, so its number is the
  baseline's under another name. Judged per track: the retrieval track forces
  the routing and depth policies on every arm, so an arm that names one of
  them is inert there and live on the generation track, and the report says
  which.

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

#: What the harness sets on an arm's account when the arm declares a key, so
#: the check charges it to the declaration rather than calling it a leak:
#: naming a reranker turns reranking on and naming an agentic pipeline the
#: loop; an embedder, named or set, gets a collection of its own at its
#: dimension; ``metric_k`` and the over-fetch set the depth retrieval is asked
#: for; a structural chunker setting decides which chunker runs; and context
#: reordering is recorded per track. Each is traced to ``experiment.py``. A
#: key the harness starts setting for a new declaration belongs here, or the
#: arms that declare it block their bundle.
_ENTAILED: Dict[str, Tuple[str, ...]] = {
    "reranker": ("enable_reranking",),
    "ensemble_reranker_models": ("enable_reranking",),
    "agentic_pipeline": ("use_agentic_rag",),
    "embedder": ("embedding_dim", "collection_name"),
    "embedding_model": ("embedding_dim", "collection_name"),
    "embedding_dim": ("collection_name",),
    "metric_k": ("default_top_k",),
    "retrieval_chunk_overfetch": ("default_top_k",),
    "enable_context_reordering": ("context_order",),
    **{key: ("chunking_strategy",) for key in CHUNKER_DISPATCH_KEYS},
}

#: The two tracks a task runs: retrieval scores ranked lists, generation
#: scores answers, and each searches with its own configuration.
TRACKS: Tuple[str, ...] = ("retrieval", "generation")

#: Settings recorded per track, as ``{"retrieval": ..., "generation": ...}``:
#: the routing and depth policies the retrieval track forces on every arm
#: (``experiment._fix_retrieval_routing`` and ``_fix_retrieval_depth``) and
#: the generation track reads from the arm, and the order each track read its
#: results in. Each track's value is a function of the arm's configuration
#: alone and is recorded whether or not the track ran, so an arm digests alike
#: on a retrieval-only run and a full one; ``system.tracks`` beside the
#: manifest says which tracks ran, and two arms differ in one of these only
#: on a track both ran. Digesting the retrieval track's values alone hid an
#: agentic arm on the generation track: forced off on retrieval, it digested
#: like its control and was called inert while its loop answered.
TRACK_SPLIT_KEYS = frozenset(
    {
        "use_agentic_rag",
        "agentic_pipeline",
        "routing_default_tier",
        "similarity_threshold",
        "chunk_cite_retrieval",
        "default_top_k",
        "context_order",
    }
)

#: Component slots a track empties for every arm: the retrieval track runs no
#: agentic loop, since its metrics need the chunk-level results of one search,
#: so a loop an arm names is inert there whatever the arm says.
TRACK_EMPTIES_SLOT: Dict[str, frozenset] = {"retrieval": frozenset({"agentic_pipeline"})}


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


def track_manifest(by_track: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """One manifest from each track's, with the split keys recorded per track.

    ``by_track`` maps each of :data:`TRACKS` to the :func:`stage_manifest` of
    the configuration that track searches with. A key in
    :data:`TRACK_SPLIT_KEYS` becomes ``{track: value}``; every other setting,
    and each component, is the retrieval track's, which the tracks share.
    Digests are recomputed over the merged settings.
    """
    tracks = [t for t in TRACKS if t in by_track]
    base = by_track[tracks[0]]

    def _settings(track: str, stage: str) -> Mapping[str, Any]:
        return (by_track[track].get(stage) or {}).get("settings") or {}

    out: Dict[str, Any] = {}
    for stage, entry in base.items():
        keys = sorted({k for t in tracks for k in _settings(t, stage)})
        settings = {
            key: (
                {t: _settings(t, stage).get(key) for t in tracks}
                if key in TRACK_SPLIT_KEYS
                else _settings(tracks[0], stage).get(key)
            )
            for key in keys
        }
        component = entry.get("component")
        out[stage] = {
            "digest": _digest({"component": component, "settings": settings}),
            "component": component,
            "settings": settings,
        }
    return out


def tracks_ran(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """The tracks a row's task ran.

    Empty for a result that did not record them (written before they were
    recorded): such a row is judged on its values as a whole, with no claim
    about a track. Taking both for granted told a retrieval-only run of 0.1.0
    that its generation track ran with the arm's own value.
    """
    ran = (row.get("system") or {}).get("tracks")
    if not ran:
        return ()
    return tuple(t for t in TRACKS if t in ran)


def _on_track(key: str, value: Any, track: Optional[str]) -> Any:
    """A split key's value on ``track``; any other value, or ``None`` for the
    track, is the value as a whole."""
    if track is not None and key in TRACK_SPLIT_KEYS and isinstance(value, Mapping):
        return value.get(track)
    return value


def _differs(key: str, a: Any, b: Any, tracks: Tuple[str, ...]) -> bool:
    """Whether two recorded values differ on a track both arms ran (as a
    whole, when the tracks are not known)."""
    if not tracks:
        return a != b
    return any(_on_track(key, a, t) != _on_track(key, b, t) for t in tracks)


def declared_stages(overrides: Optional[Mapping[str, Any]]) -> Set[str]:
    """The stages an arm says it changes, read off its overrides."""
    out: Set[str] = set()
    for key in overrides or {}:
        stage = stage_of(key)
        if stage is not None:
            out.add(stage)
    return out


def declared_keys(overrides: Optional[Mapping[str, Any]]) -> Set[str]:
    """The manifest keys an arm's overrides account for.

    Each override that belongs to a stage, and what the harness sets because
    of it (:data:`_ENTAILED`). A component slot is declared under its stage's
    name, which is the name the manifest files it under too.
    """
    out: Set[str] = set()
    for key in overrides or {}:
        if stage_of(key) is None:
            continue
        out.add(key)
        out.update(_ENTAILED.get(key, ()))
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


def _differing_keys(
    a: Mapping[str, Any], b: Mapping[str, Any], tracks: Tuple[str, ...] = TRACKS
) -> List[str]:
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    keys = sorted(k for k in set(sa) | set(sb) if _differs(k, sa.get(k), sb.get(k), tracks))
    if a.get("component") != b.get("component"):
        keys.insert(0, "component")
    return keys


def _recorded(manifest: Mapping[str, Any], key: str) -> Any:
    """What a manifest holds for ``key``: a slot's component, or a setting."""
    if key in COMPONENT_STAGES:
        return (manifest.get(key) or {}).get("component")
    stage = stage_of(key)
    if stage is None:
        return None
    return ((manifest.get(stage) or {}).get("settings") or {}).get(key)


def _inert_tracks(
    key: str,
    arm: Mapping[str, Any],
    control: Mapping[str, Any],
    tracks: Tuple[Optional[str], ...],
) -> List[Optional[str]]:
    """The tracks on which a key the arm declared ran as the control's.

    Inert on a track when the key, and everything the harness sets on its
    account (:data:`_ENTAILED`), reached its stage with the control's value
    there, or when the track empties the slot for every arm. ``(None,)``
    judges the values as a whole, for rows that did not record their tracks.
    """
    out: List[Optional[str]] = []
    for track in tracks:
        if track is not None and key in TRACK_EMPTIES_SLOT.get(track, ()):
            out.append(track)
            continue
        if all(
            _on_track(k, _recorded(arm, k), track) == _on_track(k, _recorded(control, k), track)
            for k in (key, *_ENTAILED.get(key, ()))
        ):
            out.append(track)
    return out


def _forced(key: str, track: Optional[str]) -> bool:
    """Whether ``track`` sets ``key`` to the same value for every arm."""
    if track is None:
        return False
    if track == "retrieval" and key in TRACK_SPLIT_KEYS:
        return True
    return key in TRACK_EMPTIES_SLOT.get(track, ())


def _inert_lines(
    bench: str,
    name: str,
    arm: Mapping[str, Any],
    control_name: str,
    control: Mapping[str, Any],
    keys: List[str],
    tracks: Tuple[str, ...],
) -> List[str]:
    """What an arm declared that ran as the control's, by key and by track.

    Compared setting by setting: judged by stage digests, an arm that declared
    ``hybrid_alpha`` and ``rrf_k`` passed when only ``rrf_k`` reached the
    retriever, since the stage's digest had moved.
    """
    ma, mc = arm["system"]["stages"], control["system"]["stages"]
    # Values as a whole when the rows did not record their tracks.
    judged: Tuple[Optional[str], ...] = tracks or (None,)
    status = {k: _inert_tracks(k, ma, mc, judged) for k in keys}
    dead = [k for k in keys if len(status[k]) == len(judged)]
    lines: List[str] = []
    if dead and len(dead) == len(keys):
        why = (
            f", since the {judged[0]} track, the only one that ran, forces "
            f"{'it' if len(dead) == 1 else 'them'} for every arm"
            if len(judged) == 1 and all(_forced(k, judged[0]) for k in dead)
            else ""
        )
        lines.append(
            f"{bench}: arm {name!r} names {dead} but its pipeline is identical to the "
            f"control {control_name!r} there{why}"
        )
    elif dead:
        live = [k for k in keys if k not in dead]
        lines.append(
            f"{bench}: arm {name!r} names {dead}, which reached the pipeline with the control "
            f"{control_name!r}'s value{'s' if len(dead) > 1 else ''}; of what it names, only "
            f"{live} changed anything"
        )
    for key in keys:
        inert_on = status[key]
        if not inert_on or len(inert_on) == len(judged):
            continue
        live_on = [t for t in judged if t not in inert_on]
        forced = " (the track forces it for every arm)" if _forced(key, inert_on[0]) else ""
        lines.append(
            f"{bench}: arm {name!r} names {key!r}, which the {inert_on[0]} track ran with the "
            f"control {control_name!r}'s value{forced}, so its {inert_on[0]} metrics are the "
            f"control's; the {live_on[0]} track ran with its own"
        )
    return lines


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
                declared = declared_keys(a.get("config_overrides")) | declared_keys(
                    b.get("config_overrides")
                )
                # Only on a track both arms ran: a no_rag arm skips retrieval,
                # and what its retrieval track would have searched with
                # changed nothing it measured.
                shared = tuple(t for t in tracks_ran(a) if t in tracks_ran(b))
                ma, mb = a["system"]["stages"], b["system"]["stages"]
                for stage in (*STAGES, OTHER):
                    sa, sb = ma.get(stage) or {}, mb.get(stage) or {}
                    if sa.get("digest") == sb.get("digest"):
                        continue
                    # The keys the arms declared are excused, not the stage
                    # they sit in. A stage either arm named was skipped whole:
                    # an arm naming hybrid_alpha hid a leaked default_top_k,
                    # and a reranker sweep, where every arm names a reranker,
                    # hid a leaked rerank_top_k.
                    leaked = [
                        key
                        for key in _differing_keys(sa, sb, shared)
                        if key not in declared and not (key == "component" and stage in declared)
                    ]
                    if not leaked:
                        continue
                    report["undeclared"].append(
                        f"{bench}: arms {name_a!r} and {name_b!r} differ at the {stage} "
                        f"stage in settings neither names ({', '.join(leaked)})"
                    )

        # A key an arm declared that its manifest shows as the control's (the
        # arm that declares nothing) never ran as named: judged key by key and
        # track by track, so an arm naming two settings of one stage is caught
        # when only one arrived, and an arm whose change the retrieval track
        # forces off is called inert there and live on generation.
        controls = [(n, r) for n, r in entries if not declared_keys(r.get("config_overrides"))]
        for name, row in entries:
            keys = sorted(k for k in (row.get("config_overrides") or {}) if stage_of(k) is not None)
            if not keys:
                continue
            for control_name, control in controls:
                if control.get("profile") != row.get("profile"):
                    continue
                shared = tuple(t for t in tracks_ran(row) if t in tracks_ran(control))
                report["inert"].extend(
                    _inert_lines(bench, name, row, control_name, control, keys, shared)
                )
                break
        report["inert"].extend(_same_output(entries, bench, report["unverified"]))
        report["inert"].extend(_reranker_never_moved(entries, bench, report["unverified"]))
    return report
