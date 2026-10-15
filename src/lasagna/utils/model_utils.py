"""Model utilities for RAG system using provider abstraction.

This module provides utilities for resolving model references to
LLM-compatible settings. It uses the provider abstraction layer
to support both Lasagna and standalone deployments.

Backward Compatibility:
- Same function signatures as before
- Same behavior when running in the host application
- Uses provider registry for model resolution
"""

import os
from typing import Any, Dict, Optional

from loguru import logger


def message_text(message: Any) -> str:
    """The text of a chat model's reply, whatever shape its content has.

    ``content`` is a string for most providers but a list of blocks for
    Anthropic models with thinking or tool use, and for OpenAI's Responses
    API: ``[{"type": "thinking", ...}, {"type": "text", "text": "Paris"}]``.
    Code that called ``.strip()`` on it crashed, or, inside a ``try``, quietly
    took its fallback on every call. Only text blocks are kept; a plain string
    is returned as it is.
    """
    content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def get_default_model_name(fallback: str = "gpt-6-luna") -> str:
    """Get the default model name from environment or config.

    Resolution order:
        1. MODEL_NAME environment variable
        2. Config provider's agent.action.model model_name
        3. Hardcoded fallback (gpt-6-luna)

    This centralises the model name resolution so that every component
    in the knowledge package picks up the same default.
    """
    env_model = os.environ.get("MODEL_NAME", "")
    if env_model:
        return env_model

    try:
        from lasagna.providers import get_config_provider

        provider = get_config_provider()
        settings = provider.get_llm_settings("agent.action.model")
        if settings and settings.get("model_name"):
            return settings["model_name"]
    except (RuntimeError, Exception):
        pass

    return fallback


def get_default_model_settings() -> Dict[str, Any]:
    """Get default model settings via the config provider.

    Resolves 'agent.action.model' through the provider abstraction layer.
    Returns empty dict if no provider is configured.
    """
    from lasagna.providers import get_config_provider

    try:
        provider = get_config_provider()
        settings = provider.get_llm_settings("agent.action.model")
        if settings:
            return settings
    except RuntimeError:
        pass
    return {}


def is_langfuse_tracing_enabled() -> bool:
    """Check if Langfuse tracing is enabled via environment variable.

    Set RAG_LANGFUSE_TRACING=true to enable.
    """
    env_val = os.environ.get("RAG_LANGFUSE_TRACING", "")
    return env_val.lower() in ("true", "1", "yes") if env_val else False


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert various object types to dictionary.

    Args:
        obj: Object to convert (dict, Dynaconf, or object with to_dict())

    Returns:
        Dictionary representation

    Raises:
        ValueError: If object type not supported
    """
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj.copy()
    raise ValueError(f"Unsupported model settings type: {type(obj)}")


def _resolve_settings_path_via_provider(path: str) -> Optional[Any]:
    """Resolve settings path using the config provider.

    Args:
        path: Dotted path like 'agent.action.model'

    Returns:
        Settings value or None if not found
    """
    from lasagna.providers import get_config_provider

    try:
        provider = get_config_provider()
        return provider.get_llm_settings(path)
    except RuntimeError:
        # No provider configured - will fall back to default_settings
        return None


def resolve_model_settings(
    model_ref: Any, default_settings: Any, max_tokens: Optional[int], context: str
) -> Dict[str, Any]:
    """Resolve a model reference to provider-compatible settings.

    This function maintains full backward compatibility while using
    the provider abstraction internally.

    Args:
        model_ref: Model reference - can be:
            - None: Use default_settings
            - str "default", "lasagna_default", "agent_default": Use default_settings
            - str with ".": Path like "agent.action.model" - resolve via provider
            - str with ":": Platform:model like "openai:gpt-6-luna"
            - str (other): Use as model_name override with default_settings
            - dict: Use directly as settings
        default_settings: Fallback settings if model_ref not resolved
        max_tokens: Required max tokens for output
        context: Description for error messages (e.g., "Query enhancement")

    Returns:
        Dictionary with model settings including 'platform', 'model_name', etc.

    Raises:
        ValueError: If required fields missing or model unavailable
    """
    from lasagna.providers import get_llm_provider

    model_name_override: Optional[str] = None
    model_settings_obj: Optional[Any] = None

    # Handle special default references
    if isinstance(model_ref, str) and model_ref.strip().lower() in {
        "default",
        "lasagna_default",
        "agent_default",
        "agent.action.model",
    }:
        model_ref = None

    # Resolve model reference to settings object
    if model_ref is None:
        model_settings_obj = default_settings
    elif isinstance(model_ref, str):
        # Try dotted path (e.g., "agent.action.model")
        if "." in model_ref:
            model_settings_obj = _resolve_settings_path_via_provider(model_ref)

        # Try platform:model format (e.g., "openai:gpt-6-luna")
        if model_settings_obj is None and ":" in model_ref:
            platform, model_name = model_ref.split(":", 1)
            model_settings_obj = {
                "platform": platform.strip(),
                "model_name": model_name.strip(),
            }

        # Use as model name override
        if model_settings_obj is None:
            model_settings_obj = default_settings
            model_name_override = model_ref
    else:
        # Assume dict-like object
        model_settings_obj = model_ref

    # Convert to dict
    model_settings = _to_dict(model_settings_obj)

    # Apply model name override
    if model_name_override:
        model_settings["model_name"] = model_name_override

    # Apply max_tokens
    if max_tokens is not None:
        model_settings["max_tokens"] = max_tokens

    # Default max_tokens if not set (common for standalone knowledge package)
    if "max_tokens" not in model_settings or model_settings["max_tokens"] is None:
        model_settings["max_tokens"] = 4096

    if "platform" not in model_settings:
        # Try to infer platform from the LLM provider's default
        try:
            llm_provider = get_llm_provider()
            if hasattr(llm_provider, "default_provider"):
                model_settings["platform"] = llm_provider.default_provider
                if "model_name" not in model_settings:
                    model_settings["model_name"] = get_default_model_name(
                        getattr(llm_provider, "SUPPORTED_PLATFORMS", {}).get(
                            llm_provider.default_provider, "gpt-6-luna"
                        )
                    )
                logger.debug(f"Inferred platform={model_settings['platform']} for {context}")
            else:
                raise ValueError(f"{context} model settings missing required 'platform' field")
        except RuntimeError:
            raise ValueError(f"{context} model settings missing required 'platform' field")

    # Validate model is available via provider
    try:
        llm_provider = get_llm_provider()
        if not llm_provider.validate_model_settings(model_settings):
            raise ValueError(f"{context} model validation failed")
    except RuntimeError as e:
        # No provider configured - skip validation
        logger.warning(f"Skipping model validation (no provider): {e}")
    except Exception as e:
        raise ValueError(f"{context} model unavailable: {e}") from e

    logger.debug(
        f"Resolved {context} model settings: platform={model_settings.get('platform')}, "
        f"model_name={model_settings.get('model_name')}"
    )
    return model_settings
