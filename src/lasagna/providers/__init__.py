"""Provider interfaces and registry for RAG system.

This module provides the abstraction layer that enables RAG to work
with different configuration sources, LLM providers, and session managers.

Default behavior (plug & play):
  - Auto-configures from environment variables on first access
  - Detects API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
  - Sets up SimpleConfigProvider with sensible defaults (~/.rag_knowledge/)
  - No configuration code required from the user

Explicit configuration (optional):
  configure_providers() overrides auto-config. Call it before using any
  RAG component if you need custom providers.

Quick Start::

    # Just have an API key set — auto-configures on first use
    export OPENAI_API_KEY=sk-...

    from lasagna.sdk import KnowledgeClient
    async with KnowledgeClient() as client:
        await client.upload("doc.pdf")
        results = await client.search_documents("query")
"""

import os
from pathlib import Path
from threading import Lock
from typing import Optional

from loguru import logger

from .config_provider import ConfigProvider, SimpleConfigProvider
from .llm_provider import LLMProvider, SimpleLLMProvider
from .session_provider import (
    SessionConfigProvider,
    InMemorySessionProvider,
    PersistentSessionProvider,
)

# Global provider instances (singleton pattern)
# Protected by lock for thread safety in multi-threaded environments
_provider_lock = Lock()
_config_provider: Optional[ConfigProvider] = None
_llm_provider: Optional[LLMProvider] = None
_session_provider: Optional[SessionConfigProvider] = None
_auto_config_done = False


# ---------------------------------------------------------------------------
# Auto-configuration from environment variables
# ---------------------------------------------------------------------------

# Ordered by popularity/likelihood — first match wins
_API_KEY_TO_PLATFORM = [
    ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("GROQ_API_KEY", "groq"),
    ("MISTRAL_API_KEY", "mistral"),
    ("TOGETHER_API_KEY", "together"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("GOOGLE_API_KEY", "google"),
    ("AZURE_OPENAI_API_KEY", "azure"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("FIREWORKS_API_KEY", "fireworks"),
    ("COHERE_API_KEY", "cohere"),
    ("NVIDIA_API_KEY", "nvidia"),
    ("PERPLEXITY_API_KEY", "perplexity"),
    # Both spellings. IBM's own docs and langchain_ibm use WATSONX_APIKEY
    # (no underscore); only checking WATSONX_API_KEY meant a user with
    # standard IBM credentials had watsonx silently skipped and fell through
    # to whatever else was configured.
    ("WATSONX_API_KEY", "watsonx"),
    ("WATSONX_APIKEY", "watsonx"),
    ("OLLAMA_BASE_URL", "ollama"),
    # Tier 3 — Local inference (lower priority than cloud APIs)
    ("VLLM_BASE_URL", "vllm"),
    ("LLAMACPP_MODEL_PATH", "llamacpp"),
    ("HF_MODEL_ID", "huggingface"),
]


def _detect_llm_platform() -> Optional[str]:
    """Detect LLM platform from environment API keys.

    Returns:
        Platform name (e.g. 'openai') or None if no key found
    """
    for env_var, platform in _API_KEY_TO_PLATFORM:
        if os.environ.get(env_var):
            logger.info(f"Auto-detected LLM platform: {platform} (from {env_var})")
            return platform
    return None


def _auto_configure_from_env() -> None:
    """Auto-configure providers from environment variables.

    Called lazily on first access. Providers set via configure_providers()
    take precedence (this only fills in unset providers).

    Thread-safe with double-checked locking.
    """
    global _config_provider, _llm_provider, _session_provider, _auto_config_done

    # Fast path: already done
    if _auto_config_done:
        return

    with _provider_lock:
        if _auto_config_done:
            return

        # Auto-configure config provider from env vars
        if _config_provider is None:
            db_dir = Path(
                os.environ.get(
                    "RAG_DB_DIR",
                    str(Path.home() / ".rag_knowledge" / "dbs"),
                )
            )
            upload_dir = Path(
                os.environ.get(
                    "RAG_UPLOAD_DIR",
                    str(Path.home() / ".rag_knowledge" / "uploads"),
                )
            )
            package_root = Path(__file__).parent.parent  # knowledge package root
            _config_provider = SimpleConfigProvider(
                db_dir=db_dir,
                upload_dir=upload_dir,
                package_root=package_root,
            )
            logger.debug(f"Auto-configured ConfigProvider: db_dir={db_dir}")

        # Auto-configure LLM provider from API key env vars
        if _llm_provider is None:
            platform = _detect_llm_platform()
            if platform:
                _llm_provider = SimpleLLMProvider(default_provider=platform)

        # Session provider: persistent (file-backed) or in-memory (default)
        if _session_provider is None:
            if os.environ.get("RAG_SESSION_BACKEND", "").lower() == "persistent":
                _session_provider = PersistentSessionProvider()
            else:
                _session_provider = InMemorySessionProvider()

        _auto_config_done = True


# ---------------------------------------------------------------------------
# Provider getters
# ---------------------------------------------------------------------------


def get_config_provider() -> ConfigProvider:
    """Get the current configuration provider.

    Auto-configures from environment on first call if no provider was
    explicitly set via configure_providers().

    Returns:
        ConfigProvider instance
    """
    _auto_configure_from_env()
    if _config_provider is None:
        raise RuntimeError(
            "No ConfigProvider configured. Set RAG_DB_DIR / RAG_UPLOAD_DIR "
            "environment variables or call configure_providers()."
        )
    return _config_provider


def set_config_provider(provider: ConfigProvider) -> None:
    """Set a custom configuration provider. Thread-safe."""
    global _config_provider
    with _provider_lock:
        _config_provider = provider


def get_llm_provider() -> LLMProvider:
    """Get the current LLM provider.

    Auto-configures from environment on first call if no provider was
    explicitly set via configure_providers().

    Returns:
        LLMProvider instance

    Raises:
        RuntimeError: If no API key found and no provider configured
    """
    _auto_configure_from_env()
    if _llm_provider is None:
        supported_keys = ", ".join(env for env, _ in _API_KEY_TO_PLATFORM)
        raise RuntimeError(
            f"No LLMProvider configured. Set one of these API key env vars: "
            f"{supported_keys} — or call configure_providers()."
        )
    return _llm_provider


def set_llm_provider(provider: LLMProvider) -> None:
    """Set a custom LLM provider. Thread-safe."""
    global _llm_provider
    with _provider_lock:
        _llm_provider = provider


def get_session_provider() -> SessionConfigProvider:
    """Get the current session configuration provider.

    Defaults to InMemorySessionProvider if none configured
    (created by _auto_configure_from_env).

    Returns:
        SessionConfigProvider instance
    """
    _auto_configure_from_env()
    # _auto_configure_from_env always creates InMemorySessionProvider
    # if none was set, so _session_provider is guaranteed non-None here.
    return _session_provider  # type: ignore[return-value]


def set_session_provider(provider: SessionConfigProvider) -> None:
    """Set a custom session configuration provider. Thread-safe."""
    global _session_provider
    with _provider_lock:
        _session_provider = provider


def configure_providers(
    config: Optional[ConfigProvider] = None,
    llm: Optional[LLMProvider] = None,
    session: Optional[SessionConfigProvider] = None,
) -> None:
    """Configure all providers at once.

    Providers set here take precedence over auto-configuration.
    Call this before using any RAG component if you want custom providers.

    Args:
        config: Configuration provider
        llm: LLM provider
        session: Session configuration provider
    """
    if config is not None:
        set_config_provider(config)
    if llm is not None:
        set_llm_provider(llm)
    if session is not None:
        set_session_provider(session)


def reset_providers() -> None:
    """Reset all providers to uninitialized state.

    Useful for testing or reconfiguring providers. Thread-safe.
    """
    global _config_provider, _llm_provider, _session_provider, _auto_config_done
    with _provider_lock:
        _config_provider = None
        _llm_provider = None
        _session_provider = None
        _auto_config_done = False


__all__ = [
    # Interfaces
    "ConfigProvider",
    "LLMProvider",
    "SessionConfigProvider",
    # Simple implementations (for standalone use)
    "SimpleConfigProvider",
    "SimpleLLMProvider",
    "InMemorySessionProvider",
    "PersistentSessionProvider",
    # Registry functions
    "get_config_provider",
    "set_config_provider",
    "get_llm_provider",
    "set_llm_provider",
    "get_session_provider",
    "set_session_provider",
    "configure_providers",
    "reset_providers",
]
