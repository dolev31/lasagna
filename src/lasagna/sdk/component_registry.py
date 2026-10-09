"""Re-export component registry for SDK access.

Allows ``from lasagna.sdk import component_registry``.
"""

from lasagna.core.registry import (
    ComponentRegistry,
    component_registry,
    get_component_registry,
)

__all__ = ["ComponentRegistry", "component_registry", "get_component_registry"]
