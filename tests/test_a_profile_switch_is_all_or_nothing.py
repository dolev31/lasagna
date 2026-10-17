"""A profile switch that reindexes either happens whole or leaves the session as it was.

`lasagna configure --profile standard -s r1 --reindex` moved each document to
the new model's collection, then switched the session. It undid the moves only
for an ``Exception``: Ctrl-C, or a pod stopped mid-switch, left moved documents
in the new collection while the session stayed on its old profile, so no
search found them. It held no lock either: an upload during the switch saved
its row under the old collection and was stranded when the switch ended, and
two switches of one session interleaved their moves.

Now the switch holds the session's lock in the store from start to end, records
itself in ``profile_switches`` while it runs, moves the documents back on any
interruption, and refuses uploads, reindexes, deletes and settings changes of
the session from anywhere else while it runs. One whose process died is
reported by ``doctor`` and finished by ``lasagna configure --resume-switch``.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.api import _profile_switch
from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _mgr(sid):
    return get_knowledge_config_manager(session_id=sid)


def _session_with_documents(*names: str) -> str:
    """A ``speed`` session holding ready documents in speed's collection."""
    sid = f"atomic-{uuid.uuid4().hex[:8]}"
    _mgr(sid).set_active_profile("speed", session_id=sid)
    collection = collection_name_for_settings(_mgr(sid).get_resolved_config(sid))
    for name in names:
        get_rag_database().save_document(
            {
                "document_id": f"{name}_{sid}",
                "session_id": sid,
                "filename": f"{name}.md",
                "file_size": 10,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
            },
            f"/nonexistent/{name}.md",
            collection_name=collection,
        )
    return sid


def _collections(sid: str) -> set:
    return {row["collection_name"] for row in get_rag_database().get_documents(sid)}


def _speed_collection(sid: str) -> str:
    return collection_name_for_settings(_mgr(sid).get_resolved_config(sid))


class _Mover:
    """Reindexes a document by moving its row to the collection its settings search."""

    def __init__(self, *, interrupt_at: str = "", interrupt_restore: bool = False, gate=None):
        self.interrupt_at, self.interrupt_restore, self.gate = (
            interrupt_at,
            interrupt_restore,
            gate,
        )
        self.calls = []

    def reindex_document(self, document_id, session_id, *, config=None):
        forward = config is not None
        name = document_id.split("_")[0]
        self.calls.append((name, "forward" if forward else "back", threading.get_ident()))
        if self.gate is not None:
            self.gate(name)
        if forward and name == self.interrupt_at:
            raise KeyboardInterrupt
        if not forward and self.interrupt_restore:
            raise KeyboardInterrupt
        searched = config or get_rag_config().with_session(session_id)
        with get_rag_database().transaction() as conn:
            conn.execute(
                "UPDATE rag_documents SET collection_name = ? WHERE document_id = ?",
                (searched.collection_name, document_id),
            )


def _switch(sid, backend, profile="standard"):
    return _profile_switch.switch_profile(
        _mgr(sid), profile, session_id=sid, reindex=True, backend=backend
    )


def test_ctrl_c_mid_switch_moves_the_documents_back_and_keeps_the_profile():
    sid = _session_with_documents("a", "b", "c")
    speed = _speed_collection(sid)
    backend = _Mover(interrupt_at="b")

    with pytest.raises(KeyboardInterrupt):
        _switch(sid, backend)

    assert [(n, d) for n, d, _ in backend.calls] == [
        ("a", "forward"),
        ("b", "forward"),
        ("a", "back"),
    ]
    assert _mgr(sid).get_active_profile(sid) == "speed"
    assert _collections(sid) == {speed}
    assert get_rag_database().get_profile_switch(sid) is None
    _mgr(sid).patch_config({"rerank_top_k": 7}, session_id=sid)  # not refused


def test_a_switch_whose_process_stopped_is_reported_refused_and_resumed(monkeypatch):
    from lasagna.api._profile_guard import ProfileSwitchInProgress

    sid = _session_with_documents("a", "b")
    # Stopped while it moved "b", and again while moving "a" back: what a
    # killed process leaves.
    with pytest.raises(KeyboardInterrupt):
        _switch(sid, _Mover(interrupt_at="b", interrupt_restore=True))
    record = get_rag_database().get_profile_switch(sid)
    assert record["profile"] == "standard" and record["previous_profile"] == "speed"

    from lasagna.core.diagnostics import _check_profile_switches

    check = _check_profile_switches(sid)
    assert check.status == "error"
    assert f"lasagna configure --resume-switch -s {sid}" in check.advice

    with pytest.raises(ProfileSwitchInProgress) as refused:
        _mgr(sid).patch_config({"rerank_top_k": 7}, session_id=sid)
    assert "is switching from profile 'speed' to 'standard'" in str(refused.value)
    assert f"lasagna configure --resume-switch -s {sid}" in str(refused.value)

    import lasagna.api.rag_backend as rag_backend
    from lasagna.core import local_server

    backend = _Mover()
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)
    monkeypatch.setattr(rag_backend, "close_backend", lambda: None)
    monkeypatch.setattr(local_server, "local_store_held", lambda: False)
    result = CliRunner().invoke(cli.app, ["configure", "--resume-switch", "-s", sid])
    out = " ".join(result.output.split())
    assert result.exit_code == 0, out
    assert "Applied profile: standard" in out
    assert _mgr(sid).get_active_profile(sid) == "standard"
    standard = collection_name_for_settings(_mgr(sid).get_resolved_config(sid))
    assert _collections(sid) == {standard}
    assert get_rag_database().get_profile_switch(sid) is None
    assert _check_profile_switches(sid) is None


def test_a_failed_switch_that_takes_over_moves_back_what_the_stopped_one_moved():
    sid = _session_with_documents("a", "b")
    speed = _speed_collection(sid)
    with pytest.raises(KeyboardInterrupt):
        _switch(sid, _Mover(interrupt_at="b", interrupt_restore=True))
    assert len(_collections(sid)) == 2  # "a" was left in standard's collection

    class _FailsOnB(_Mover):
        def reindex_document(self, document_id, session_id, *, config=None):
            if config is not None and document_id.startswith("b_"):
                raise RuntimeError("bad table")
            return super().reindex_document(document_id, session_id, config=config)

    with pytest.raises(_profile_switch.ReindexFailed):
        _switch(sid, _FailsOnB())
    assert _mgr(sid).get_active_profile(sid) == "speed"
    assert _collections(sid) == {speed}
    assert get_rag_database().get_profile_switch(sid) is None


def test_writes_of_the_session_from_elsewhere_wait_for_the_switch(tmp_path):
    from lasagna.api._profile_guard import ProfileSwitchInProgress

    sid = _session_with_documents("a")
    inside, release = threading.Event(), threading.Event()

    def gate(name):
        inside.set()
        assert release.wait(30)

    runner = threading.Thread(target=_switch, args=(sid, _Mover(gate=gate)))
    runner.start()
    try:
        assert inside.wait(30)
        with pytest.raises(ProfileSwitchInProgress, match="The upload was refused"):
            from lasagna.api._document_pipeline import DocumentPipeline

            DocumentPipeline.upload_document(object.__new__(DocumentPipeline), "x.md", sid, "x.md")
        with pytest.raises(ProfileSwitchInProgress, match="The reindex was refused"):
            DocumentPipeline.reindex_document(object.__new__(DocumentPipeline), f"a_{sid}", sid)
        with pytest.raises(ProfileSwitchInProgress, match="The delete was refused"):
            DocumentPipeline.delete_document(object.__new__(DocumentPipeline), f"a_{sid}", sid)
        with pytest.raises(ProfileSwitchInProgress, match="This change was refused"):
            _mgr(sid).set_focused_document_ids([f"a_{sid}"], session_id=sid)
    finally:
        release.set()
        runner.join(30)
    assert _mgr(sid).get_active_profile(sid) == "standard"
    _mgr(sid).set_focused_document_ids([f"a_{sid}"], session_id=sid)  # after it: accepted


def test_an_upload_refused_during_a_switch_answers_409():
    from fastapi import HTTPException

    from lasagna.api._profile_guard import ProfileSwitchInProgress

    from lasagna.api.rest.routes.documents import upload_failure

    record = {
        "session_id": "s1",
        "profile": "standard",
        "previous_profile": "speed",
        "started_at": time.time(),
    }
    from lasagna.api._profile_guard import switch_in_progress_message

    answer = upload_failure(
        ProfileSwitchInProgress(switch_in_progress_message(record, "The upload")), None, "s1"
    )
    assert isinstance(answer, HTTPException) and answer.status_code == 409
    assert "is switching from profile 'speed' to 'standard'" in answer.detail


def test_two_switches_of_one_session_run_one_after_the_other():
    sid = _session_with_documents("a", "b", "c")

    def slow(name):
        time.sleep(0.05)

    backend = _Mover(gate=slow)
    threads = [
        threading.Thread(target=_switch, args=(sid, backend, profile))
        for profile in ("standard", "balanced")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    order = [thread for _, _, thread in backend.calls]
    # Each switch's three moves are together: the second waited for the first.
    assert order[:3] == [order[0]] * 3 and order[3:] == [order[3]] * 3, backend.calls
    assert order[0] != order[3]
    assert get_rag_database().get_profile_switch(sid) is None
