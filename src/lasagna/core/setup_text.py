"""What the setup commands say: download sizes, API key checks, the active setup.

``lasagna init`` and ``lasagna doctor`` live in ``cli.py``. What they tell a
user about model downloads, a pasted API key and the configuration in force is
computed here, from the profiles themselves, so it stays true when a profile
changes its models.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#: Size in bytes of each model's weights (``model.safetensors``), read from a
#: Hugging Face cache on 2026-10-15. Tokenizer and config files add a few MB.
MODEL_WEIGHT_BYTES = {
    "all-MiniLM-L6-v2": 90_868_376,
    "sentence-transformers/all-MiniLM-L6-v2": 90_868_376,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 90_870_598,
    "BAAI/bge-large-en-v1.5": 1_340_616_616,
    "BAAI/bge-m3": 2_271_064_456,
    "mixedbread-ai/mxbai-rerank-large-v1": 870_174_306,
}


def human_size(n_bytes: int) -> str:
    """``91 MB``, ``2.3 GB``: decimal units, as download pages show them."""
    if n_bytes >= 1_000_000_000:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{round(n_bytes / 1e6)} MB"


def profile_models(profile: str) -> List[str]:
    """The models a built-in profile downloads: its embedder, and its reranker if on."""
    from lasagna.core.session_config._profiles import ProfileMixin

    spec = ProfileMixin._PROFILE_SPECS.get(profile, {})
    models = [spec.get("embedding_model")]
    if spec.get("enable_reranking"):
        models.append(spec.get("reranker_model"))
    return [m for m in models if m]


#: Retrieval settings that call an LLM. A configuration with all of them off
#: searches without an API key; ``use_agentic_rag`` sends the queries that
#: query routing finds complex through LLM steps.
LLM_RETRIEVAL_SETTINGS = (
    "use_agentic_rag",
    "enable_hyde",
    "enable_query_expansion",
    "enable_query_decomposition",
    "enable_rag_fusion",
    "enable_crag",
    "enable_contextual_retrieval",
    "enable_hallucination_check",
)

#: What ``lasagna doctor`` calls the configuration no profile was chosen for.
DEFAULT_CONFIGURATION = "the default configuration"


def _setting(profile: Optional[str], key: str) -> object:
    """``key`` under a built-in profile (its spec over the settings file), or the
    settings file's own value when ``profile`` is None."""
    from lasagna.core.config import rag_config
    from lasagna.core.session_config._profiles import ProfileMixin

    spec = ProfileMixin._PROFILE_SPECS.get(profile, {}) if profile else {}
    return spec[key] if key in spec else getattr(rag_config, key, None)


def needs_llm_key(profile: Optional[str]) -> bool:
    """Whether retrieval under ``profile`` (None: the settings file) calls an LLM."""
    return any(bool(_setting(profile, key)) for key in LLM_RETRIEVAL_SETTINGS)


def keyless_profiles() -> List[str]:
    """The built-in profiles whose retrieval makes no LLM call."""
    from lasagna.core.session_config._profiles import ProfileMixin

    return [name for name in ProfileMixin._PROFILE_SPECS if not needs_llm_key(name)]


def _join(names: List[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _profiles_phrase(names: List[str]) -> str:
    """``the speed profile``, ``the standard, balanced and max_quality profiles``."""
    return f"the {_join(names)} profile{'s' if len(names) > 1 else ''}" if names else ""


def llm_key_hint() -> str:
    """What ``doctor`` says when no LLM API key is set: where retrieval still works."""
    from lasagna.core.session_config._profiles import ProfileMixin

    keyless = keyless_profiles()
    needing = [n for n in ProfileMixin._PROFILE_SPECS if n not in keyless]
    default_needs = needs_llm_key(None)
    works = [
        x for x in (not default_needs and DEFAULT_CONFIGURATION, _profiles_phrase(keyless)) if x
    ]
    need = [x for x in (default_needs and DEFAULT_CONFIGURATION, _profiles_phrase(needing)) if x]
    commands = ", ".join(f"`lasagna configure --profile {n}`" for n in keyless)
    first = (
        f"Retrieval works without one only on {' and '.join(works)}"
        + (f" ({commands})." if commands else ".")
        if works
        else "No profile retrieves without one."
    )
    need_text = " and ".join(need)
    second = (
        f" {need_text[:1].upper() + need_text[1:]} send complex queries through LLM steps, "
        f"so they need one, as do generation, LLM reranking and benchmark judging."
        if need
        else " Generation, LLM reranking and benchmark judging need one."
    )
    return first + second + " Set OPENAI_API_KEY (or your provider's variable)."


def model_cache_hint() -> str:
    """What ``doctor`` says before anything is downloaded: how much, and the lighter choice."""
    from lasagna.core.config import rag_config

    models = [rag_config.embedding_model]
    if rag_config.enable_reranking:
        models.append(rag_config.reranker_model)
    hint = f"The first search on {DEFAULT_CONFIGURATION} {download_text(models)}."
    light = keyless_profiles()
    if light:
        hint += (
            f" The {light[0]} profile (`lasagna configure --profile {light[0]}`) "
            f"{profile_download_text(light[0])}."
        )
    return hint


def profile_download_text(profile: str) -> str:
    """``downloads 3.1 GB: BAAI/bge-m3 (2.3 GB), ... (870 MB)`` for ``profile``."""
    return download_text(profile_models(profile))


def download_text(models: List[str]) -> str:
    """``downloads 91 MB: all-MiniLM-L6-v2 (91 MB)`` for these models."""
    if not models:
        return "downloads nothing beyond the settings file's models"
    known = [(m, MODEL_WEIGHT_BYTES[m]) for m in models if m in MODEL_WEIGHT_BYTES]
    parts = ", ".join(f"{m} ({human_size(b)})" for m, b in known)
    unknown = [m for m in models if m not in MODEL_WEIGHT_BYTES]
    text = f"downloads {human_size(sum(b for _, b in known))}: {parts}" if known else "downloads"
    if unknown:
        text += ("; " if known else " ") + ", ".join(unknown) + " (size not recorded)"
    return text


def doctor_context(session_id: str = "default") -> List[Tuple[str, str]]:
    """What ``doctor`` reports about the setup in force: session, profile, files.

    Paths are redacted (``<home>/...``), as everything doctor prints is.
    """
    from pathlib import Path

    from lasagna.core import paths
    from lasagna.core.provenance import redact

    rows: List[Tuple[str, str]] = [("session", session_id)]
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        profile = get_knowledge_config_manager(session_id=session_id).get_active_profile(
            session_id=session_id
        )
        rows.append(("profile", str(getattr(profile, "name", profile))))
    except Exception as exc:  # noqa: BLE001 - doctor reports, it does not fail
        rows.append(("profile", f"could not be read ({exc})"))
    settings = paths.env_get("SETTINGS_FILE") or str(
        Path(__file__).resolve().parents[1] / "settings.lasagna.toml"
    )
    rows.append(("settings file", redact(settings)))
    rows.append(("data directory", redact(str(paths.data_dir()))))
    return rows
