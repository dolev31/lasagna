"""Validation mixin for session configuration.

Enforces RAG configuration validation before every mutation (patch, save, import).
Uses the configurator's validate_config() as the single source of truth.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from loguru import logger

from ._data_models import _deep_merge


class ValidationMixin:
    """Validates config dicts before mutation. Mixed into KnowledgeConfigManager.

    Expects the host class to provide:
    - self.get_resolved_config(session_id): method (from SessionStateMixin)
    """

    def _validate_or_raise(
        self,
        config: Dict[str, Any],
        session_id: Optional[str] = None,
        *,
        is_patch: bool = False,
    ) -> None:
        """Validate a config dict. Raises ValueError on errors, logs warnings.

        Args:
            config: Config dict to validate.
            session_id: If provided and is_patch=True, merges patch with the
                effective config before cross-field validation.
            is_patch: If True, config is a partial patch (not a full config).

        Raises:
            ValueError: If the config contains validation errors.
        """
        from lasagna.configurator.validation import validate_config

        if is_patch and session_id:
            # Merge patch with current effective config so cross-field rules
            # (e.g. chunk_overlap < chunk_size) work on the full picture.
            effective = self.get_resolved_config(session_id)
            merged = copy.deepcopy(effective)
            _deep_merge(merged, config)
            result = validate_config(merged)
        else:
            result = validate_config(config)

        if result.warnings:
            logger.warning("Config validation warnings: %s", result.warnings)

        if not result.valid:
            messages = "; ".join(f"{e['key']}: {e['message']}" for e in result.errors)
            raise ValueError(f"Invalid configuration: {messages}")

    def _session_settings(
        self,
        config: Dict[str, Any],
        *,
        trusted: bool = False,
        builtin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """``config`` as a session may store it; raises ``SessionSettingRefused``.

        Called by every write that takes settings from a caller (patch, profile
        switch with settings, save, import), before anything is applied. A key
        only the server may set is dropped when it repeats what is already in
        force (the server's own setting, the default profile's, or ``builtin``,
        the built-in profile an import names) and refused otherwise.
        ``trusted`` skips the check, for in-process callers writing the
        process's own configuration (the evaluation harness, a local SDK
        client).
        """
        if trusted or not config:
            return config
        from lasagna.core.config import get_rag_config

        from ._session_keys import check_session_settings

        server = get_rag_config()
        in_force = {}
        for key in config:
            try:
                in_force[key] = getattr(server, key)
            except Exception:
                in_force[key] = server._config.get(key)
        default = next((p for p in self._default_profiles if p.name == "default"), None)
        try:
            from lasagna.configurator.schema import get_defaults

            # What the configurator page sends for a field nobody touched.
            page_defaults = get_defaults()
        except Exception:  # pragma: no cover - the page is optional
            page_defaults = {}
        return check_session_settings(
            config,
            echoes=(in_force, default.config if default else {}, builtin or {}, page_defaults),
        )

    def _refuse_unsearchable(
        self,
        state: Any,
        after: Dict[str, Any],
        *,
        change: str,
        force: bool = False,
    ) -> None:
        """Raise before a write that would embed at another size than what is stored.

        Called by every write that can change the session's resolved config,
        before anything is applied, with ``after`` the config it would leave.
        Raises ``EmbeddingSizeConflict`` (a ``ValueError``).
        """
        if force:
            return
        from lasagna.api._profile_guard import refuse_embedding_change

        refuse_embedding_change(
            before=state.get_resolved_config(),
            after=after,
            change=change,
            kept=state.active_profile,
        )
