"""``client.config_manager`` returns the client's ConfigManager.

The client reserved the manager and ConfigManager's own docs used
``client.config_manager``, but no property returned it, so every documented
profile call raised AttributeError.
"""

from __future__ import annotations

from lasagna.sdk import KnowledgeClient
from lasagna.sdk.managers import ConfigManager


def test_config_manager_is_a_config_manager_bound_to_the_client():
    client = KnowledgeClient(session_id="cm-test")
    manager = client.config_manager
    assert isinstance(manager, ConfigManager)
    assert manager is client.config_manager
