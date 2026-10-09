"""Backend configuration mixin for session configuration.

Handles vector backend selection and configuration (BYOVDB).
"""

from typing import Any, Dict, Optional

from loguru import logger


class BackendConfigMixin:
    """Mixin for vector backend configuration.

    Expects the host class to provide:
    - self.get_resolved_config(): method
    - self.patch_config(): method
    """

    def get_vector_backend_config(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get vector backend configuration for a session."""
        config = self.get_resolved_config(session_id)

        # Get available backends from registry
        try:
            from lasagna.vector_backends import get_available_backends

            available = get_available_backends()
        except ImportError:
            available = ["milvus"]

        return {
            "vector_backend": config.get("vector_backend", "milvus"),
            "vector_backend_config": config.get("vector_backend_config", {}),
            "available_backends": available,
        }

    def set_vector_backend(
        self,
        backend_name: str,
        backend_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set vector backend for a session."""
        # Validate backend is available
        try:
            from lasagna.vector_backends import get_available_backends

            available = get_available_backends()
        except ImportError:
            available = ["milvus"]

        if backend_name not in available:
            raise ValueError(
                f"Backend '{backend_name}' is not available. Available backends: {available}"
            )

        patch = {"vector_backend": backend_name}
        if backend_config is not None:
            patch["vector_backend_config"] = backend_config

        logger.info(f"Setting vector backend to '{backend_name}' for session {session_id}")
        return self.patch_config(patch, session_id)
