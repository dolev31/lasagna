"""Which settings a session may set for itself.

A session's settings come from whoever holds the session: an MCP client, a
REST caller, an imported file. They used to be stored whatever their keys, and
search resolved every one of them, so a session could name its own vector
store (``vector_backend_config`` with any URL, which the server then
connected to), another collection, a database or upload path, a Redis URL, or
``reranker_trust_remote_code``. A session may set retrieval and generation
settings: those that reach one of the eight pipeline stages, minus the ones
that say where data lives or how the server reaches it.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

#: Settings inside a stage that decide where vectors live or how the index is
#: built. They are the server's, like the rest of its infrastructure.
_SERVER_KEYS = frozenset(
    {
        "collection_name",
        "vector_backend",
        "vector_backend_config",
        "index_type",
        "index_params",
        "metric_type",
    }
)

#: Settings outside the stages that only change how results are presented.
_PRESENTATION_KEYS = frozenset(
    {
        "citation_format",
        "citation_id_length",
        "citation_normalize_scores",
        "citation_score_method",
        "include_source_preview",
        "refinement_model",
        "show_confidence_scores",
        "show_page_numbers",
        "verify_citations",
    }
)

#: Key names that hold a path, an address or a credential, whatever stage
#: they would fall in.
_LOCATION_OR_SECRET = re.compile(
    r"(_path|_dir|_dir_name|_url|_uri|_endpoint|_host|_hosts|_dsn|_bucket|_region"
    r"|_api_key|_key|_token|_secret|_password)$"
)


#: Settings that send documents or queries to a third-party service on the
#: server's API key: where embeddings are computed, and CRAG's web search.
_THIRD_PARTY_KEYS = frozenset({"embedding_provider", "crag_web_search_fallback"})

#: Components that do the same, by slot: embedders and rerankers behind an API.
_THIRD_PARTY_COMPONENTS = {"embedder": {"openai", "cohere", "voyage"}, "reranker": {"cohere"}}

#: A "." or ".." path segment anywhere in a value.
_DOT_SEGMENT = re.compile(r"(^|[\\/])\.{1,2}([\\/]|$)")


class SessionSettingRefused(ValueError):
    """A session tried to set something only the server's own settings may set."""


#: The largest value a session may give each numeric setting it may set
#: (``parent.child`` for a number inside a dict setting). Numbers had no upper
#: bound: ``ircot_max_steps``, ``speculative_n_drafts``, ``rag_fusion_n_queries``
#: or ``hybrid_fusion_depth`` at 10^6 made one search run a million LLM calls or
#: fetch a million candidates. Each maximum is at least what any shipped profile,
#: preset, example sweep or benchmark config uses, and no more than the
#: configurator's own bound where it has one. Session writes only: the
#: evaluation harness writes its arms in-process (``trusted``), and the server's
#: settings file and environment are not bounded here.
SESSION_MAXIMUMS: Dict[str, float] = {
    "adaptive_rag_single_step_docs": 100,
    "agent_temperature": 1.0,
    "auto_merge_threshold": 1.0,
    "auto_refine_avg_length_threshold": 100_000,
    "auto_refine_avg_score_threshold": 1.0,
    "auto_refine_chunk_count_threshold": 1000,
    "bm25_b": 1.0,
    "bm25_k1": 10.0,
    "child_chunk_size": 100_000,
    "chunk_cite_retrieval.fixed_size": 50,
    "chunk_cite_retrieval.max_chunks": 100,
    "chunk_cite_retrieval.min_chunks": 20,
    "chunk_cite_retrieval.min_gap_ratio": 0.5,
    "chunk_cite_retrieval.score_gap_buffer": 10,
    "chunk_cite_retrieval.threshold_score": 1.0,
    "chunk_overlap": 50_000,
    "chunk_size": 100_000,
    "citation_id_length": 16,
    "compression_ratio": 1.0,
    "contextual_max_document_chars": 1_000_000,
    "default_top_k": 1000,
    "docling_max_tokens": 8192,
    "document_awareness.first_tokens_count": 10_000,
    "document_awareness.hybrid_llm_threshold_kb": 100_000,
    "document_awareness.max_documents_in_context": 50,
    "document_awareness.snippet_cache_ttl": 86_400,
    "document_awareness.summary_max_tokens": 4096,
    "embedding_dim": 8192,
    "embedding_truncate_dim": 8192,
    "flare_docs_per_retrieval": 50,
    "flare_mask_threshold": 1.0,
    "flare_max_sentences": 64,
    "flare_retrieval_threshold": 1.0,
    "graph_max_hops": 5,
    "hybrid_alpha": 1.0,
    "hybrid_fusion_depth": 10_000,
    "hyde_n_docs": 10,
    "ircot_max_paragraphs": 100,
    "ircot_max_steps": 32,
    "ircot_paragraphs_per_step": 50,
    "late_chunking_overlap": 8192,
    "llm_reranker_step_size": 100,
    "llm_reranker_window_size": 100,
    "markdown_header_levels": 6,
    "max_agent_iterations": 10,
    "max_chunk_size": 100_000,
    "max_content_length": 65_535,
    "max_quality_check_attempts": 5,
    "max_query_variations": 10,
    "min_chunk_size": 100_000,
    "mmr_lambda": 1.0,
    "ocr_confidence_threshold": 1.0,
    "parent_chunk_size": 100_000,
    "proposition_passage_size": 10_000,
    "query2doc_max_tokens": 2048,
    "query2doc_query_repeats": 20,
    "rag_fusion_n_queries": 10,
    "raptor_max_cluster_size": 100,
    "raptor_max_levels": 10,
    "rerank_top_k": 1000,
    "routing_confidence_threshold": 1.0,
    "rrf_k": 10_000,
    "search_params.ef": 32_768,
    "search_params.hnsw_ef": 32_768,
    "search_params.nprobe": 65_536,
    "search_params.search_list": 32_768,
    "self_ask_max_follow_ups": 20,
    "self_rag_passages": 50,
    "self_rag_weight_rel": 10.0,
    "self_rag_weight_sup": 10.0,
    "self_rag_weight_use": 10.0,
    "semantic_breakpoint_amount": 100.0,
    "semantic_buffer_size": 20,
    "semantic_threshold": 1.0,
    "sentence_overlap": 20,
    "similarity_threshold": 1.0,
    "speculative_n_drafts": 10,
    "speculative_subset_size": 20,
    "window_size": 20,
}

#: Numbers inside ``search_params``: the vector store's own, with no default.
_SEARCH_PARAM_NUMBERS = ("ef", "hnsw_ef", "nprobe", "search_list")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_session_keys() -> List[str]:
    """Every numeric setting a session may set, as ``SESSION_MAXIMUMS`` names them."""
    from lasagna.core.config import _NULLABLE_TYPES, get_rag_config

    names = [f"search_params.{p}" for p in _SEARCH_PARAM_NUMBERS]
    for key, value in get_rag_config()._config.items():
        if _is_number(value) and session_key_problem(key, value, bounded=False) is None:
            names.append(key)
        elif isinstance(value, dict) and session_key_problem(key, {}, bounded=False) is None:
            names += [f"{key}.{child}" for child, v in value.items() if _is_number(v)]
    for key, kind in _NULLABLE_TYPES.items():
        if kind in (int, float) and session_key_problem(key, 1, bounded=False) is None:
            names.append(key)
    return sorted(set(names))


def _past_maximum(key: str, value: Any) -> Optional[str]:
    """Why ``value`` is too large for a session's ``key``, or None."""
    if _is_number(value):
        cap = SESSION_MAXIMUMS.get(key)
        if cap is not None and value > cap:
            return f"is at most {cap:g} for a session ({value:g} asked)"
    elif isinstance(value, dict):
        for child, item in value.items():
            cap = SESSION_MAXIMUMS.get(f"{key}.{child}")
            if cap is not None and _is_number(item) and item > cap:
                return f"sets {child} to {item:g}; {key}.{child} is at most {cap:g} for a session"
    return None


def _names_a_model(key: str) -> bool:
    """Keys whose value a library loads, from a local folder when one has that name."""
    return key.endswith(("_model", "_models", "_tokenizer")) or key == "agentic_models"


def _looks_like_location(value: str, *, model: bool = False) -> bool:
    """Whether ``value`` reads as a path or an address.

    For a value that names a model or tokenizer, a relative path counts too:
    ``models/x`` loads that folder when it exists on the server, whatever the
    Hugging Face id of the same name holds.
    """
    text = value.strip()
    if (
        "://" in text
        or text.startswith(("/", "~", "\\"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", text))
        or bool(_DOT_SEGMENT.search(text))
    ):
        return True
    return model and ("\\" in text or os.path.exists(text))


def _component_problem(slot: str, value: Any) -> Optional[str]:
    """Why a session may not put ``value`` in component slot ``slot``, or None."""
    if value is None:
        return None
    from lasagna.core.registry import registry

    third_party = _THIRD_PARTY_COMPONENTS.get(slot, set())
    try:
        registered = isinstance(value, str) and registry.has(slot, value)
        names = sorted(set(registry.list(slot).get(slot, [])) - third_party)
    except ValueError:  # not a registry category
        registered, names = False, []
    if not registered:
        listed = f": {', '.join(names)}" if names else ""
        return (
            f"names no registered {slot} component ({value!r}); a session picks a "
            f"component by its registered name{listed}"
        )
    if value in third_party:
        return "sends documents or queries to a third-party service on the server's API key"
    return None


def session_key_problem(key: str, value: Any, *, bounded: bool = True) -> Optional[str]:
    """Why a session may not set ``key`` to ``value``, or None when it may.

    ``bounded``: a number past its ``SESSION_MAXIMUMS`` entry is refused too.
    """
    from lasagna.knowledge_evaluation_sdk.stages import STAGES, stage_of

    if "trust_remote_code" in key:
        return "would run code from a model repository on the server"
    if key in _SERVER_KEYS or _LOCATION_OR_SECRET.search(key):
        return "is a server setting (where data lives or how the server reaches it)"
    if key in _THIRD_PARTY_KEYS:
        return "sends documents or queries to a third-party service on the server's API key"
    if key not in _PRESENTATION_KEYS and stage_of(key) not in STAGES:
        return "is not a retrieval or generation setting"
    if key in STAGES:
        return _component_problem(key, value)
    if key == "search_params" and isinstance(value, dict) and "metric_type" in value:
        return "sets metric_type, which is a server setting"
    if isinstance(value, dict):
        values = list(value.values()) if key == "agentic_models" else []
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    for item in values:
        if isinstance(item, str) and _looks_like_location(item, model=_names_a_model(key)):
            return (
                f"names a path or URL ({item!r}); a session names models by their "
                "Hugging Face id or provider model name"
            )
    return _past_maximum(key, value) if bounded else None


def _known_keys() -> Iterable[str]:
    from lasagna.core.config import get_rag_config

    return get_rag_config()._config


def check_session_settings(
    config: Mapping[str, Any], *, echoes: Iterable[Mapping[str, Any]] = ()
) -> Dict[str, Any]:
    """``config`` without the keys a session may not set; raises if any would change.

    A key a session may not set is dropped when its value is the one already in
    force or a packaged default (``echoes``: the server's own settings, a
    built-in profile's, the configurator page's), so an exported configuration
    imports again and a client that sends its whole configuration back can
    save it. Dropped, never stored: the server's value stays in force. Any other such key raises
    ``SessionSettingRefused`` naming each key, before anything is stored.
    """
    from lasagna.core._text import did_you_mean

    baselines = [dict(e) for e in echoes if e]
    kept: Dict[str, Any] = {}
    refused: List[Tuple[str, str]] = []
    for key, value in config.items():
        problem = session_key_problem(key, value)
        if problem is None:
            kept[key] = value
        elif any(key in b and b[key] == value for b in baselines):
            continue  # what is already in force: nothing changes, nothing stored
        else:
            if problem.startswith("is not a retrieval"):
                # A setting of the process (use_gpu) is spelled right; only an
                # unknown name has a closer one. It suggested the key itself.
                known = _known_keys()
                if key not in known:
                    problem += did_you_mean(key, known)
            refused.append((key, problem))
    if refused:
        named = "; ".join(f"{key} {why}" for key, why in refused)
        raise SessionSettingRefused(
            f"A session cannot set these: {named}. A session changes retrieval and "
            "generation settings only. Set server settings in settings.lasagna.toml or "
            "in the environment the server starts with (docs/reference/settings.md lists "
            "each one), then restart it."
        )
    return kept
