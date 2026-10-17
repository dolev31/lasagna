"""`lasagna configure --reset` drops a session's own settings, and every piece of advice works.

A forced SDK change (``set_config(embedding_model=..., force=True)``) left a session
whose own setting named another model than its documents'. The advice on every
surface was to reset the session's overrides over REST, the SDK or MCP and then
switch, which no CLI command could do; doctor's ``configure --profile speed`` kept
the override; that command was refused for documents the session already did not
search; and ``configure --force`` said the documents "stay searchable" when no
search found them. A switch that kept the override printed "Applied profile:
speed" and reindexed with the override's model without saying so.
"""

from __future__ import annotations

import uuid

import pytest
from typer.testing import CliRunner

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager

MINILM, SMALL = "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"
MINILM_COLLECTION = collection_name_for_settings({"embedding_model": MINILM, "embedding_dim": 384})
SMALL_COLLECTION = collection_name_for_settings({"embedding_model": SMALL, "embedding_dim": 384})


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _mgr(sid):
    return get_knowledge_config_manager(session_id=sid)


def _overridden_session(documents_in: str = MINILM_COLLECTION) -> str:
    """On speed, three documents in ``documents_in``, and its own embedding_model=bge-small."""
    sid = f"reset-{uuid.uuid4().hex[:8]}"
    _mgr(sid).set_active_profile("speed", session_id=sid, force=True)
    for name in ("a", "b", "c"):
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
            collection_name=documents_in,
        )
    _mgr(sid).patch_config({"embedding_model": SMALL, "embedding_dim": 384}, sid, force=True)
    return sid


def _configure(*args):
    from lasagna import cli

    return CliRunner().invoke(cli.app, ["configure", *args])


def _searches(sid):
    return get_rag_config().with_session(sid).embedding_model


def test_doctors_advice_is_one_command_that_works():
    from lasagna.api._profile_guard import forget_documents_elsewhere
    from lasagna.core.diagnostics import _check_session_documents

    sid = _overridden_session()
    check = _check_session_documents(sid)
    command = f"lasagna configure --profile speed --reset -s {sid}"
    assert command in check.advice
    assert "Reindex them with profile 'speed' with embedding_model=" in check.advice

    result = _configure("--profile", "speed", "--reset", "-s", sid)
    assert result.exit_code == 0, result.output
    assert _searches(sid) == MINILM
    assert _mgr(sid).get_session_state(sid).session_overrides == {}
    forget_documents_elsewhere(sid)
    assert _check_session_documents(sid).status == "ok"


def test_a_switch_that_keeps_the_searched_collection_is_not_blamed_for_documents_elsewhere():
    sid = _overridden_session()
    result = _configure("--profile", "speed", "-s", sid)
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert f"Applied profile: speed with embedding_model={SMALL} from this session's" in output
    assert f"lasagna configure --profile speed --reset -s {sid} --reindex" in output


def test_a_reset_that_would_leave_the_documents_is_refused_with_the_reindex(monkeypatch):
    sid = _overridden_session(documents_in=SMALL_COLLECTION)  # searched now
    result = _configure("--reset", "-s", sid)
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert f"lasagna configure --profile speed --reset -s {sid} --reindex" in output
    assert _searches(sid) == SMALL  # nothing changed


def test_a_forced_refusal_says_what_is_true_of_the_documents(monkeypatch):
    def refusal(model, kind="embedding model", *args, **kwargs):
        if model == SMALL:
            return model_downloads.ModelNotDownloaded(model, offline=True, kind=kind)
        return None

    monkeypatch.setattr(model_downloads, "refusal", refusal)
    sid = _overridden_session()
    result = _configure("--profile", "speed", "-s", sid, "--force")
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "stay searchable" not in output
    assert "As before this change: 3 of 3 documents" in output
    assert "so it cannot find them" in output
    assert f"lasagna configure --profile speed --reset -s {sid}" in output


def test_a_reindex_says_the_model_comes_from_the_sessions_settings():
    from lasagna.api import _profile_switch

    class _Backend:
        def __init__(self):
            self.models = []

        def reindex_document(self, document_id, session_id, *, config=None):
            config = config or get_rag_config().with_session(session_id)
            self.models.append(config.embedding_model)

    sid = _overridden_session()
    report = _profile_switch.switch_profile(
        _mgr(sid), "speed", session_id=sid, reindex=True, backend=_Backend()
    )
    assert f"with {SMALL} (embedding_model from this session's settings" in (
        _profile_switch.summary(report)
    )
    assert "from this session's settings" in report.applied()

    backend = _Backend()
    reset = _profile_switch.switch_profile(
        _mgr(sid), "speed", session_id=sid, reindex=True, backend=backend, reset=True
    )
    assert backend.models == [MINILM] * 3 and reset.own_settings == {}
    assert _searches(sid) == MINILM


def test_the_model_advice_offers_the_reset_when_the_session_names_its_own_model():
    sid = _overridden_session()
    advice = model_downloads.fix_for_model(sid, SMALL)
    assert f"sets embedding_model={SMALL}" in advice
    assert f"lasagna configure --profile speed --reset -s {sid}" in advice
