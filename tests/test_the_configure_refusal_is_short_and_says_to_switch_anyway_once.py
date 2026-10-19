"""``configure`` refuses a switch that would strand documents in a few lines, one per option.

The refusal ran to about 15 lines and said "To switch anyway" twice: once in the
guard's text for every surface (``configure --force``, ``force=true`` over REST
and MCP) and again in the CLI's own tail. The CLI now prints the problem and one
line per option, each with the command as typed there; the guard's text for the
other surfaces is unchanged.
"""

from __future__ import annotations

import uuid

from typer.testing import CliRunner

from lasagna import cli
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager


def _session_with_a_document() -> str:
    sid = f"short-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
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
        collection_name=collection_name_for_settings(mgr.get_resolved_config(sid)),
    )
    return sid


def _refused():
    sid = _session_with_a_document()
    result = CliRunner().invoke(
        cli.app,
        ["configure", "--profile", "standard", "-s", sid],
        env={"COLUMNS": "300"},
    )
    assert result.exit_code == 1, result.output
    return sid, result.output


def test_the_refusal_says_to_switch_anyway_once():
    _, output = _refused()
    assert " ".join(output.split()).count("To switch anyway") == 1, output


def test_the_refusal_is_at_most_six_lines_at_a_wide_terminal():
    _, output = _refused()
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) <= 6, output


def test_each_option_is_one_line_with_its_command():
    sid, output = _refused()
    options = [line.strip() for line in output.splitlines() if line.startswith("  ")]
    assert any(
        line.startswith("To reindex and switch:") and f"standard -s {sid} --reindex" in line
        for line in options
    ), output
    assert any(
        line.startswith("To switch anyway") and f"standard -s {sid} --force" in line
        for line in options
    ), output
    assert any("lasagna upload <file>" in line for line in options), output


def test_the_other_surfaces_still_get_their_commands():
    from lasagna.api._profile_guard import EmbeddingSizeConflict
    import pytest

    sid = _session_with_a_document()
    with pytest.raises(EmbeddingSizeConflict) as refused:
        get_knowledge_config_manager(session_id=sid).set_active_profile("standard", session_id=sid)
    message = str(refused.value)
    assert message.count("To switch anyway") == 1
    for way in ("reindex=true over REST and MCP", "force=true over REST and MCP"):
        assert way in message
