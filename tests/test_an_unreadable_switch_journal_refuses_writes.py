"""A session's writes are refused while the record of its profile switches cannot be read.

``unfinished_switch`` answered "no switch" when the store could not read the
``profile_switches`` table, so an upload, a reindex or a settings change went
through during a switch it could not see, which is the race the record exists
to stop. It fails closed now, with a plain error.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

from lasagna.api._profile_guard import refuse_during_switch, unfinished_switch
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._data_models import SettingsNotSaved


@pytest.fixture
def unreadable(monkeypatch):
    def locked(session_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(get_rag_database(), "get_profile_switch", locked)


def test_the_check_fails_closed_with_a_plain_error(unreadable):
    sid = f"journal-{uuid.uuid4().hex[:8]}"
    with pytest.raises(SettingsNotSaved) as refused:
        unfinished_switch(sid)
    message = str(refused.value)
    assert sid in message and "database is locked" in message and "nothing was changed" in message
    with pytest.raises(SettingsNotSaved):
        refuse_during_switch(sid, "This upload")


def test_a_settings_change_is_refused(unreadable):
    sid = f"journal-{uuid.uuid4().hex[:8]}"
    with pytest.raises(SettingsNotSaved):
        KnowledgeConfigManager().patch_config({"rerank_top_k": 4}, session_id=sid)


def test_a_store_without_the_record_has_no_switch(monkeypatch):
    from types import SimpleNamespace

    import lasagna.core.database as database

    monkeypatch.setattr(database, "get_rag_database", lambda: SimpleNamespace())
    assert unfinished_switch("s") is None
