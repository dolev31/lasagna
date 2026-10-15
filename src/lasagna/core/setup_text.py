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


def profile_download_text(profile: str) -> str:
    """``downloads 3.1 GB: BAAI/bge-m3 (2.3 GB), ... (870 MB)`` for ``profile``."""
    models = profile_models(profile)
    if not models:
        return "downloads nothing beyond the settings file's models"
    known = [(m, MODEL_WEIGHT_BYTES[m]) for m in models if m in MODEL_WEIGHT_BYTES]
    parts = ", ".join(f"{m} ({human_size(b)})" for m, b in known)
    unknown = [m for m in models if m not in MODEL_WEIGHT_BYTES]
    text = f"downloads {human_size(sum(b for _, b in known))}: {parts}" if known else "downloads"
    if unknown:
        text += ("; " if known else " ") + ", ".join(unknown) + " (size not recorded)"
    return text


def api_key_problem(name: str, value: Optional[str]) -> Optional[str]:
    """Why ``value`` cannot be saved as ``name``, or None when it can."""
    if not value or not value.strip():
        return (
            f"{name} cannot be empty. Paste the key, or choose "
            f"'None, search only' to set up without an LLM."
        )
    if any(ch.isspace() for ch in value.strip()):
        return f"{name} contains a space or a line break; paste only the key."
    return None


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
