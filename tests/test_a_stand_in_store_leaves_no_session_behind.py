"""A store that answers with something other than rows leaves no session state behind.

Tests (and embedding applications) put a stand-in in place of the metadata
store, a ``MagicMock`` among them, which answers every query with another
mock. The process's manager took that answer for a stored session and kept it:
its ``last_accessed`` was a mock, and every later read of any session failed in
the once-a-minute cleanup (``'>' not supported between instances of
'MagicMock' and 'int'``), so searches answered 503. Only a row that is a dict
counts now, and a stored session's ``last_accessed`` must be a number.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import lasagna.core.database as database
from lasagna.core.session_config import KnowledgeConfigManager


def test_a_mock_answer_is_not_a_stored_session(monkeypatch):
    manager = KnowledgeConfigManager()
    sid = f"stand-in-{uuid.uuid4().hex[:8]}"
    with monkeypatch.context() as m:
        m.setattr(database, "get_rag_database", lambda: MagicMock())
        manager.set_focused_document_ids([], session_id=sid)
        saved = list(manager._global_custom_profiles)
        manager._refresh_saved_profiles(force=True)
        assert manager._global_custom_profiles == saved  # not replaced by the mock's answer

    manager._last_cleanup_time = 0  # the next read sweeps
    assert isinstance(manager.get_session_state(sid).last_accessed, float)
    manager.get_session_state(f"{sid}-other")
