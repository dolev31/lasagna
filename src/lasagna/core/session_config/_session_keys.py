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


class SessionSettingRefused(ValueError):
    """A session tried to set something only the server's own settings may set."""


def _looks_like_location(value: str) -> bool:
    text = value.strip()
    return (
        "://" in text
        or text.startswith(("/", "~", "\\", "./", "../"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", text))
        or "/../" in text
    )


def session_key_problem(key: str, value: Any) -> Optional[str]:
    """Why a session may not set ``key`` to ``value``, or None when it may."""
    from lasagna.knowledge_evaluation_sdk.stages import STAGES, stage_of

    if "trust_remote_code" in key:
        return "would run code from a model repository on the server"
    if key in _SERVER_KEYS or _LOCATION_OR_SECRET.search(key):
        return "is a server setting (where data lives or how the server reaches it)"
    if key not in _PRESENTATION_KEYS and stage_of(key) not in STAGES:
        return "is not a retrieval or generation setting"
    values = (
        list(value.values()) if key == "agentic_models" and isinstance(value, dict) else [value]
    )
    for item in values:
        if isinstance(item, str) and _looks_like_location(item):
            return (
                f"names a path or URL ({item!r}); a session names models by their "
                "Hugging Face id or provider model name"
            )
    return None


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
                problem += did_you_mean(key, _known_keys())
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
