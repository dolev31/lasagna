"""What an agent builder meets first says what happened, once, in plain words.

Found by the sixth persona audit:

* `lasagna upload ./docs` on an existing folder said "File not found: ./docs".
* An upload of an unsupported type printed "Upload failed" three times.
* An MCP argument of the wrong type returned pydantic's text: `search
  {"top_k": "lots"}` gave "1 validation error for searchArguments ...
  errors.pydantic.dev".
* deploy/README.md wrote the REST health checks as `/health/ready` and
  `/health`, which `lasagna serve` answers with 404: they are under `/api/v1`.
* `serve --help` did not say `configure` goes through the server too.
* With HF_HUB_OFFLINE=1 and the session's models missing, `doctor` said
  "downloads 3.2 GB" and "0 blocking" while the first upload failed.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture
def uploads(monkeypatch):
    """`lasagna upload` with a client that records what it was given."""
    sent = []

    async def upload(path, allow_duplicates=False):
        sent.append(Path(path).name)
        return SimpleNamespace(duplicate_of="doc_seen", document_id="doc_seen")

    async def close():
        pass

    client = SimpleNamespace(upload=upload, close=close)
    monkeypatch.setattr(cli, "_serving_url", lambda: None)
    monkeypatch.setattr(cli, "_client_for", lambda session_id, url: client)
    return sent


def _docs(tmp_path) -> Path:
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "a.txt").write_text("Gallium melts at 29.76 C.")
    (docs / "sub" / "b.md").write_text("# Tin\nTin melts at 232 C.")
    (docs / "c.xyz").write_text("x")
    (docs / ".hidden.txt").write_text("x")
    return docs


def test_a_folder_uploads_its_readable_files_and_says_what_it_skipped(tmp_path, uploads):
    docs = _docs(tmp_path)
    result = runner.invoke(cli.app, ["upload", str(docs)], terminal_width=300)
    words = " ".join(result.output.split())
    assert result.exit_code == 0, words
    assert "File not found" not in words
    assert sorted(uploads) == ["a.txt", "b.md"]
    assert "is a folder: uploading 2 readable files, subfolders included" in words
    assert "skipping 1 (.xyz)" in words


def test_an_unreadable_file_is_refused_once_with_the_types_that_are_read(tmp_path, uploads):
    notes = tmp_path / "notes.xyz"
    notes.write_text("x")
    result = runner.invoke(cli.app, ["upload", str(notes)], terminal_width=300)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert words.count("cannot read") == 1 and "Upload failed" not in words
    assert ".pdf" in words and ".md" in words
    assert uploads == []


def test_a_failed_upload_says_upload_failed_once(tmp_path, monkeypatch):
    from lasagna.sdk.exceptions import UploadError

    async def upload(path, allow_duplicates=False):
        raise UploadError(f"Upload failed for {Path(path).name}: File too large")

    async def close():
        pass

    monkeypatch.setattr(cli, "_serving_url", lambda: None)
    monkeypatch.setattr(
        cli, "_client_for", lambda s, u: SimpleNamespace(upload=upload, close=close)
    )
    doc = tmp_path / "a.txt"
    doc.write_text("x")
    result = runner.invoke(cli.app, ["upload", str(doc)], terminal_width=300)
    assert result.exit_code == 1
    assert result.output.count("Upload failed") == 1, result.output


def test_the_sdks_log_line_for_a_failed_upload_is_left_to_the_cli():
    record = {"extra": {}, "name": "lasagna.sdk.managers.document_manager"}
    assert cli._reported_document_failure({**record, "message": "Upload failed: too large"})
    assert not cli._reported_document_failure({**record, "message": "Uploading a.txt"})


@pytest.fixture
def mcp_server():
    pytest.importorskip("mcp")
    from lasagna.mcp import server

    return server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")


@pytest.mark.parametrize(
    "tool, arguments, expected",
    [
        (
            "search",
            {"query": "q", "top_k": "lots"},
            "search: argument 'top_k' takes a whole number, not 'lots'.",
        ),
        ("search", {}, "search: argument 'query' is required (a string)."),
        (
            "get_document_info",
            {"document_id": 5},
            "get_document_info: argument 'document_id' takes a string, not 5.",
        ),
    ],
)
def test_an_mcp_argument_of_the_wrong_type_gets_a_plain_message(
    mcp_server, tool, arguments, expected
):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as raised:
        asyncio.run(mcp_server.call_tool(tool, arguments))
    assert str(raised.value) == expected
    assert "pydantic" not in str(raised.value)


def test_every_tool_gets_the_plain_message(mcp_server):
    """Each tool with a typed argument refuses a list in it the same way."""
    from mcp.server.fastmcp.exceptions import ToolError

    tools = asyncio.run(mcp_server.list_tools())
    checked = 0
    for tool in tools:
        typed = [
            name
            for name, spec in (tool.inputSchema.get("properties") or {}).items()
            if any(s.get("type") in ("integer", "string") for s in spec.get("anyOf", [spec]))
        ]
        if not typed:
            continue
        with pytest.raises(ToolError) as raised:
            asyncio.run(mcp_server.call_tool(tool.name, {typed[0]: [1, 2]}))
        assert str(raised.value).startswith(f"{tool.name}: argument "), tool.name
        assert "errors.pydantic.dev" not in str(raised.value)
        checked += 1
    assert checked > 20


def test_the_deploy_guide_writes_rest_health_paths_under_api_v1():
    from lasagna.api.rest.app import create_app

    paths = set(create_app().openapi()["paths"])
    assert {"/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live"} <= paths
    assert not {"/health", "/health/ready"} & paths
    for page in ("src/lasagna/deploy/README.md", "src/lasagna/api/docs/endpoints.md"):
        # A REST health path is written in full; /health alone is the MCP
        # service's, on port 8005, in the section about it or beside the port.
        for section in re.split(r"\n(?=## )", (REPO / page).read_text()):
            if section.startswith("## MCP service"):
                continue
            for line in section.splitlines():
                for path in re.findall(r"`(/health[\w/]*)`", line):
                    assert "8005" in line or "MCP" in line, f"{page}: {path!r} in: {line}"


def test_serve_help_names_every_command_that_goes_through_it():
    result = runner.invoke(cli.app, ["serve", "--help"], terminal_width=300)
    words = " ".join(result.output.split())
    assert "`lasagna status`, `search`, `upload` and `configure` go through it" in words


def test_doctor_blocks_when_offline_and_the_sessions_models_are_missing(monkeypatch):
    from lasagna.core import diagnostics, model_downloads

    monkeypatch.setattr(model_downloads, "is_cached", lambda model, filename="config.json": False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    check = diagnostics._check_model_cache("default")
    assert check.status == "error"
    assert "offline mode is on" in check.detail and "first upload or search fails" in check.detail
    assert "network access" in check.advice

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert diagnostics._check_model_cache("default").status == "warning"
