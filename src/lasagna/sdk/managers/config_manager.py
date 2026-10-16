"""Config Manager - Configuration and profile management.

Provides configuration updates, profile management, and preset application.
Wraps the underlying KnowledgeConfigManager.

Example:
    ```python
    # Get current config
    config = await client.config_manager.get_config()

    # Update settings
    await client.config_manager.set_config(enable_hyde=True)

    # Use profiles
    await client.config_manager.set_profile("deep_search")
    ```
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from loguru import logger

from ..config import KnowledgeConfig
from ..exceptions import (
    ConfigurationError,
    ProfileNotFoundError,
)

if TYPE_CHECKING:
    from ..client import KnowledgeClient


class ConfigManager:
    """Manages RAG configuration and profiles.

    Provides configuration updates, profile management, and reset functionality.
    Changes can be session-scoped or persistent.

    Example:
        ```python
        # Update configuration
        await client.config_manager.set_config(
            enable_reranking=True,
            chunk_size=1000,
        )

        # List profiles
        profiles = await client.config_manager.list_profiles()

        # Switch profile
        await client.config_manager.set_profile("quality")

        # Reset to defaults
        await client.config_manager.reset()
        ```
    """

    def __init__(self, client: "KnowledgeClient"):
        """Initialize config manager.

        Args:
            client: Parent KnowledgeClient instance
        """
        self._client = client

    # =========================================================================
    # Configuration Access
    # =========================================================================

    async def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.

        Returns:
            Dictionary with all configuration values

        Example:
            ```python
            config = await client.config_manager.get_config()
            print(f"Reranking: {config['enable_reranking']}")
            ```
        """
        return self._client.config.to_dict()

    async def set_config(self, **kwargs) -> Dict[str, Any]:
        """Update configuration options.

        Args:
            **kwargs: Configuration options to update

        Returns:
            Updated configuration dictionary

        Example:
            ```python
            config = await client.config_manager.set_config(
                enable_hyde=True,
                max_agent_iterations=5,
            )
            ```
        """
        await self._client.set_config(**kwargs)
        return self._client.config.to_dict()

    async def apply_config(self, config: KnowledgeConfig) -> Dict[str, Any]:
        """Apply a complete configuration object.

        Args:
            config: KnowledgeConfig to apply

        Returns:
            Applied configuration dictionary

        Example:
            ```python
            from lasagna.sdk import KnowledgeConfig

            config = KnowledgeConfig(
                enable_reranking=True,
                enable_hyde=True,
            )
            await client.config_manager.apply_config(config)
            ```
        """
        self._client._config = config
        logger.info("Configuration applied")
        return config.to_dict()

    # =========================================================================
    # Presets
    # =========================================================================

    async def use_preset(self, preset: str) -> Dict[str, Any]:
        """Apply a configuration preset.

        Args:
            preset: Preset name ('speed', 'quality', 'balanced')

        Returns:
            Applied configuration dictionary

        Raises:
            ValueError: If preset not recognized

        Example:
            ```python
            # For fast responses
            await client.config_manager.use_preset("speed")

            # For best quality
            await client.config_manager.use_preset("quality")
            ```
        """
        self._client.use_preset(preset)
        return self._client.config.to_dict()

    def list_presets(self) -> List[str]:
        """List available configuration presets.

        Returns:
            List of preset names

        Example:
            ```python
            presets = client.config_manager.list_presets()
            # ['speed', 'quality', 'balanced', 'standalone']
            ```
        """
        return ["speed", "standard", "balanced", "max_quality", "standalone"]

    # =========================================================================
    # Profile Management
    # =========================================================================

    async def list_profiles(self, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """List available RAG configuration profiles.

        Args:
            include_builtin: Include built-in profiles

        Returns:
            List of profile dictionaries with name and description

        Raises:
            ConfigurationError: If profile listing fails

        Example:
            ```python
            profiles = await client.config_manager.list_profiles()
            for p in profiles:
                print(f"{p['name']}: {p['description']}")
            ```
        """
        try:
            from ...core.session_config import get_knowledge_config_manager

            manager = get_knowledge_config_manager()
            return manager.list_profiles(include_builtin=include_builtin)
        except ImportError:
            # Return default profiles when backend not available
            from lasagna.core.session_config._profiles import BUILTIN_PROFILE_DESCRIPTIONS

            return [
                {"name": name, "description": description, "builtin": True}
                for name, description in BUILTIN_PROFILE_DESCRIPTIONS.items()
            ]
        except Exception as e:
            logger.error(f"Could not list profiles: {e}")
            raise ConfigurationError(f"Could not list profiles: {e}") from e

    async def get_active_profile(self) -> str:
        """Get the name of the active profile.

        Returns:
            Active profile name

        Raises:
            ConfigurationError: If profile retrieval fails

        Example:
            ```python
            profile = await client.config_manager.get_active_profile()
            print(f"Active profile: {profile}")
            ```
        """
        try:
            from ...core.session_config import get_knowledge_config_manager

            manager = get_knowledge_config_manager()
            return manager.get_active_profile(session_id=self._client.session_id)
        except ImportError:
            return "default"
        except Exception as e:
            logger.error(f"Could not get active profile: {e}")
            raise ConfigurationError(f"Could not get active profile: {e}") from e

    async def set_profile(self, profile_name: str) -> Dict[str, Any]:
        """Set the active RAG profile.

        Args:
            profile_name: Name of profile to activate

        Returns:
            Profile configuration dictionary

        Raises:
            ProfileNotFoundError: If profile does not exist
            ConfigurationError: If profile activation fails

        Example:
            ```python
            try:
                await client.config_manager.set_profile("deep_search")
            except ProfileNotFoundError:
                print("Profile not found")
            ```
        """
        from ...core.session_config import UnknownProfileError

        try:
            from ...core.session_config import get_knowledge_config_manager

            manager = get_knowledge_config_manager()
            manager.set_active_profile(profile_name, session_id=self._client.session_id)
            logger.info(f"Profile set to: {profile_name}")
            return {"profile": profile_name, "status": "active"}
        except ImportError:
            raise ProfileNotFoundError(
                f"Profile not found: {profile_name}",
                profile_name=profile_name,
            )
        except UnknownProfileError as e:
            # Matched on the text "not found" before; the manager says "Unknown
            # profile", so a typo surfaced as a generic ConfigurationError.
            raise ProfileNotFoundError(str(e), profile_name=profile_name) from e
        except Exception as e:
            logger.error(f"Could not set profile: {e}")
            raise ConfigurationError(
                f"Could not set profile {profile_name}: {e}",
                config_key="profile",
            ) from e

    async def save_profile(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        set_active: bool = True,
    ) -> Dict[str, Any]:
        """Save a custom profile.

        Args:
            name: Profile name (cannot be 'default')
            config: Configuration to save. If None, saves current config.
            set_active: Activate after saving

        Returns:
            Saved profile information

        Raises:
            ValueError: If trying to overwrite 'default' profile
            ConfigurationError: If save operation fails

        Example:
            ```python
            # Save current settings
            await client.config_manager.save_profile("my_profile")

            # Save specific config
            await client.config_manager.save_profile(
                "custom",
                config={"enable_hyde": True, "max_iterations": 5}
            )
            ```
        """
        if name == "default":
            raise ValueError("Cannot overwrite 'default' profile")

        save_config = config or self._client.config.to_dict()

        try:
            from ...core.session_config import get_knowledge_config_manager

            manager = get_knowledge_config_manager()
            manager.save_profile(name, save_config, session_id=self._client.session_id)
            if set_active:
                manager.set_active_profile(name, session_id=self._client.session_id)
            logger.info(f"Profile saved: {name}")
            return {"name": name, "status": "saved", "active": set_active}
        except ImportError as e:
            logger.error(f"Backend not available for profile save: {e}")
            raise ConfigurationError(
                f"Backend not available for saving profile: {e}",
                config_key="profile",
            ) from e
        except Exception as e:
            logger.error(f"Could not save profile: {e}")
            raise ConfigurationError(
                f"Could not save profile {name}: {e}",
                config_key="profile",
            ) from e

    async def delete_profile(self, name: str) -> bool:
        """Delete a custom profile.

        Built-in profiles cannot be deleted.

        Args:
            name: Profile name to delete

        Returns:
            True if deleted successfully

        Raises:
            ProfileNotFoundError: If profile does not exist
            ConfigurationError: If delete operation fails

        Example:
            ```python
            try:
                await client.config_manager.delete_profile("my_old_profile")
            except ProfileNotFoundError:
                print("Profile not found")
            ```
        """
        try:
            from ...core.session_config import get_knowledge_config_manager

            manager = get_knowledge_config_manager()
            manager.delete_profile(name)
            logger.info(f"Profile deleted: {name}")
            return True
        except ImportError as e:
            raise ConfigurationError(
                f"Backend not available for profile deletion: {e}",
                config_key="profile",
            ) from e
        except Exception as e:
            if "not found" in str(e).lower():
                raise ProfileNotFoundError(
                    f"Profile not found: {name}",
                    profile_name=name,
                ) from e
            logger.error(f"Could not delete profile: {e}")
            raise ConfigurationError(
                f"Could not delete profile {name}: {e}",
                config_key="profile",
            ) from e

    # =========================================================================
    # Reset
    # =========================================================================

    async def reset(self, scope: str = "session") -> Dict[str, Any]:
        """Reset configuration to defaults.

        Args:
            scope: Reset scope:
                - 'session': Clear session overrides only
                - 'all': Full reset to default profile

        Returns:
            Reset configuration dictionary

        Example:
            ```python
            # Clear session changes
            await client.config_manager.reset()

            # Full reset
            await client.config_manager.reset(scope="all")
            ```
        """
        from ..config import config_for_standard

        self._client._config = config_for_standard()
        logger.info(f"Configuration reset (scope={scope})")
        return self._client.config.to_dict()
