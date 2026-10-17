"""What the setup commands say: download sizes, API key checks, the active setup.

``lasagna init`` and ``lasagna doctor`` live in ``cli.py``. What they tell a
user about model downloads, a pasted API key and the configuration in force is
computed here, from the profiles themselves, so it stays true when a profile
changes its models.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

#: Bytes the first load of each model downloads: every file sentence-transformers
#: fetches (weights, tokenizer, configs and, for an embedder, the model card),
#: measured in an empty HF_HOME on 2026-10-17 with sentence-transformers 6.1 and
#: transformers 5.19. These figures are what the release gate's cold start
#: measures (``model_download`` in ``e2e/phases/p2_cold_start.py``).
MODEL_DOWNLOAD_BYTES = {
    "all-MiniLM-L6-v2": 91_578_415,
    "sentence-transformers/all-MiniLM-L6-v2": 91_578_415,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 91_815_758,
    "BAAI/bge-large-en-v1.5": 1_341_656_113,
    # pytorch_model.bin (2,271,145,830), tokenizer.json, sentencepiece.bpe.model
    "BAAI/bge-m3": 2_293_331_623,
    "mixedbread-ai/mxbai-rerank-large-v1": 881_291_471,
}

#: What FlagEmbedding adds for BAAI/bge-m3 (the ``flag-embedding`` extra, which
#: ``all`` includes): it downloads the model's whole repository, so on top of
#: the files above it fetches the ONNX export (``onnx/model.onnx_data``, 2.27 GB,
#: and its own tokenizer copy), the ColBERT and sparse heads and the model card's
#: images. The rest of the repository at revision 5617a9f, read from the Hub on
#: 2026-10-17.
FLAG_EMBEDDING_EXTRA_BYTES = 2_293_985_781


def human_size(n_bytes: int) -> str:
    """``92 MB``, ``2.3 GB``: decimal units, as download pages show them."""
    if n_bytes >= 1_000_000_000:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{round(n_bytes / 1e6)} MB"


def profile_models(profile: Optional[str]) -> List[str]:
    """The models a built-in profile downloads (None: the settings file's).

    Its embedder, on the first upload or search, and its reranker when
    reranking is on, on the first search that reranks. The fallback embedder is
    fetched only when the embedder cannot load, so it is not counted.
    """
    from lasagna.core.session_config._profiles import ProfileMixin

    if profile is not None and profile not in ProfileMixin._PROFILE_SPECS:
        return []
    models = [_setting(profile, "embedding_model")]
    if _setting(profile, "enable_reranking"):
        models.append(_setting(profile, "reranker_model"))
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


def model_cache_hint(models: Optional[List[str]] = None, session_id: str = "default") -> str:
    """What ``doctor`` says before anything is downloaded: how much, and the lighter choice.

    ``models`` are the ones still to download (default: the default
    configuration's embedder and reranker).
    """
    from lasagna.core.config import rag_config

    if models is None:
        models = [rag_config.embedding_model]
        if rag_config.enable_reranking:
            models.append(rag_config.reranker_model)
    where = DEFAULT_CONFIGURATION if session_id == "default" else f"session {session_id!r}"
    hint = f"The first run on {where} {download_text(models)}."
    light = keyless_profiles()
    if light:
        hint += (
            f" The {light[0]} profile (`lasagna configure --profile {light[0]}`) "
            f"{profile_download_text(light[0])}."
        )
    return hint


def profile_download_text(profile: Optional[str]) -> str:
    """``downloads 3.2 GB: BAAI/bge-m3 (2.3 GB), ... (881 MB)`` for ``profile``."""
    return download_text(profile_models(profile))


def profile_download_size(profile: Optional[str]) -> str:
    """``3.2 GB``: everything ``profile`` (None: the settings file) downloads."""
    return human_size(sum(MODEL_DOWNLOAD_BYTES.get(m, 0) for m in profile_models(profile)))


def download_text(models: List[str]) -> str:
    """``downloads 92 MB: all-MiniLM-L6-v2 (92 MB)`` for these models."""
    if not models:
        return "downloads nothing beyond the settings file's models"
    known = [(m, MODEL_DOWNLOAD_BYTES[m]) for m in models if m in MODEL_DOWNLOAD_BYTES]
    parts = ", ".join(f"{m} ({human_size(b)})" for m, b in known)
    unknown = [m for m in models if m not in MODEL_DOWNLOAD_BYTES]
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
