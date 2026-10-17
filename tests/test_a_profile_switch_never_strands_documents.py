"""A profile switch that would stop searching a session's documents is refused, with the choices.

Each embedding model has its own collection. `lasagna configure --profile
standard -s e` on a session whose documents were embedded on `speed` printed
"Applied profile: standard" and exited 0; from then on search read bge-m3's
empty collection while `status` still said "Documents: 1". Switching
bge-small to `speed`, both 384 dimensions, did the same with no error at
all: the guard compared only embedding sizes, read the new model's (empty)
collection, so `--force` never had anything to override, and its advice was
to point RAG_DB_DIR at a new directory. The guard now compares the
collection the session's documents are in with the one the change would
search.
"""

from __future__ import annotations

import uuid

import pytest
from typer.testing import CliRunner

from lasagna.api._profile_guard import EmbeddingSizeConflict
from lasagna.cli import app
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

BGE_SMALL = {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384}


def _session_with_a_document(profile: str = "", saved: dict = None) -> str:
    """A session on ``profile`` (or on a profile saved as ``saved``) holding one READY document."""
    sid = f"strand-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    if profile:
        mgr.set_active_profile(profile, session_id=sid)
    if saved:
        mgr.save_profile(f"small_{sid}", saved, session_id=sid, set_active=True)
    collection = collection_name_for_settings(mgr.get_resolved_config(sid))
    get_rag_database().save_document(
        {
            "document_id": f"doc_{sid}",
            "session_id": sid,
            "filename": "glaze.txt",
            "file_size": 10,
            "file_type": "txt",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/glaze.txt",
        collection_name=collection,
    )
    return sid


@pytest.mark.parametrize(
    "start,saved,to",
    [("speed", None, "standard"), ("", BGE_SMALL, "speed")],
    ids=["another size", "the same size"],
)
def test_the_switch_is_refused_and_says_what_to_do(start, saved, to):
    sid = _session_with_a_document(start, saved)
    mgr = get_knowledge_config_manager(session_id=sid)
    before = mgr.get_active_profile(session_id=sid)
    with pytest.raises(EmbeddingSizeConflict) as caught:
        mgr.set_active_profile(to, session_id=sid)
    message = str(caught.value)
    assert "1 document" in message and sid in message
    assert "Nothing was changed" in message
    assert "new session" in message and "reindex" in message
    assert "RAG_DB_DIR" not in message
    assert mgr.get_active_profile(session_id=sid) == before


def test_force_switches_and_leaves_the_documents_where_they_are():
    sid = _session_with_a_document("speed")
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("standard", session_id=sid, force=True)
    assert mgr.get_active_profile(session_id=sid) == "standard"


def test_a_switch_that_keeps_the_collection_is_allowed():
    sid = _session_with_a_document("speed")
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.patch_config({"rerank_top_k": 7}, session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)


def test_a_session_with_no_documents_switches_freely():
    sid = f"empty-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
    mgr.set_active_profile("standard", session_id=sid)


@pytest.mark.parametrize("start,saved", [("speed", None), ("", BGE_SMALL)])
def test_configure_refuses_and_names_force(start, saved):
    sid = _session_with_a_document(start, saved)
    to = "standard" if start else "speed"
    runner = CliRunner()
    refused = runner.invoke(app, ["configure", "--profile", to, "-s", sid])
    words = " ".join(refused.output.split())
    assert refused.exit_code == 1, refused.output
    assert "Applied profile" not in words
    assert f"lasagna configure --profile {to} -s {sid} --force" in words

    from lasagna.core import model_downloads

    # A forced switch still checks that the new profile's models load; here they do.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_downloads, "refusal", lambda *a, **k: None)
        forced = runner.invoke(app, ["configure", "--profile", to, "-s", sid, "--force"])
    assert forced.exit_code == 0, forced.output
    assert f"Applied profile: {to}" in forced.output
