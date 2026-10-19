"""A profile change whose second write fails says what the first one stored.

Creating a profile while activating it writes the profile's row and then the
session's row. When only the second failed, the error said "nothing was
changed" although the profile was saved. It now names what was kept, and the
profile is there for the next request.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._data_models import SettingsNotSaved


def test_a_new_profile_whose_session_write_fails_is_reported_as_saved(monkeypatch):
    from lasagna.core.database import get_rag_database

    sid = f"half-{uuid.uuid4().hex[:8]}"
    manager = KnowledgeConfigManager()
    manager.set_active_profile("speed", session_id=sid)

    def full(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(get_rag_database(), "save_session", full)
    with pytest.raises(SettingsNotSaved) as refused:
        manager.set_active_profile("halfway", config={"rerank_top_k": 4}, session_id=sid)
    message = str(refused.value)
    assert "nothing was changed" not in message
    assert "Profile 'halfway' was saved" in message and "still on profile 'speed'" in message
    monkeypatch.undo()

    other = KnowledgeConfigManager()
    assert other.get_active_profile(sid) == "speed"
    other.set_active_profile("halfway", session_id=sid)
    assert other.get_chosen_config(sid).get("rerank_top_k") == 4
