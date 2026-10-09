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
