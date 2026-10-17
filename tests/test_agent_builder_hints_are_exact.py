"""What an agent builder is told to run is exact, and says only what is true.

* The ``hf download BAAI/bge-m3`` hint fetched the whole repository, 4.6 GB
  with the ONNX export nothing reads; the command now names the files the
  loader reads, with their size.
* MCP ``patch_rag_config`` given an object for a choice setting answered
  "unhashable type: 'dict'", and a success returned the session's whole
  configuration; it now names the key and the type, and returns what changed.
* The unknown-session hint told MCP callers to use "the X-Session-ID header
  ... or -s"; each surface now says how it names a session.
* Replacing a document printed pymilvus's "failed to get mvccTs" note.
* ``init`` offered five vector stores (no OpenSearch) and preselected
  "Standard ... recommended" while the README says to pick speed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from lasagna.core.setup_text import MODEL_DOWNLOAD_BYTES, MODEL_FILES, download_command


def test_the_download_command_fetches_only_what_the_loader_reads():
    command = download_command("BAAI/bge-m3")
    assert command.startswith("hf download BAAI/bge-m3 --include modules.json")
    assert "--include pytorch_model.bin" in command and "onnx" not in command
    assert command.endswith("(2.3 GB)")
    assert download_command("all-MiniLM-L6-v2").startswith(
        "hf download sentence-transformers/all-MiniLM-L6-v2 --include"
    )
    assert download_command("some/unknown") == "hf download some/unknown"


def test_every_sized_model_has_its_file_list():
    assert set(MODEL_DOWNLOAD_BYTES) <= set(MODEL_FILES)


def _snapshot(model: str):
    from huggingface_hub import try_to_load_from_cache

    repo = model if "/" in model else f"sentence-transformers/{model}"
    hit = try_to_load_from_cache(repo, "config.json")
    return Path(hit).parent if isinstance(hit, str) else None


@pytest.mark.parametrize("model", sorted(m for m in MODEL_FILES if "/" in m))
def test_a_cached_copy_holds_exactly_the_listed_files(model):
    """A first load writes these files and no others (checked where a copy is cached)."""
    folder = _snapshot(model)
    if folder is None:
        pytest.skip(f"{model} is not cached")
    cached = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if not p.is_dir()}
    assert set(MODEL_FILES[model]) == cached


def test_the_hints_name_the_exact_command(monkeypatch):
    from lasagna.core import model_downloads

    monkeypatch.setattr(model_downloads, "offline", lambda: True)
    hint = model_downloads.fix_for_model("s", "BAAI/bge-m3")
    assert "hf download BAAI/bge-m3 --include" in hint and "(2.3 GB)" in hint


def test_a_session_hint_speaks_each_surfaces_language(monkeypatch):
    from lasagna.core import model_downloads

    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"my-corpus": 3})
    mcp = model_downloads.fix_for_session("other", "BAAI/bge-m3", surface="mcp")
    assert "the tool's session_id argument, or RAG_SESSION_ID" in mcp
    assert "X-Session-ID" not in mcp and " -s." not in mcp
    rest = model_downloads.fix_for_session("other", "BAAI/bge-m3")
    assert "the X-Session-ID header" in rest


@pytest.fixture
def mcp_session():
    from lasagna.mcp import server

    return server, "patch-hints"


def test_a_wrong_type_is_named_with_the_key_and_the_type(mcp_session):
    server, sid = mcp_session
    for patch, key in (
        ({"chunking_strategy": {"a": 1}}, "chunking_strategy"),
        ({"sparse_method": [1, 2]}, "sparse_method"),
        ({"embedding_model": {"x": 1}}, "embedding_model"),
        ({"hyde_model": {"x": 1}}, "hyde_model"),
        ({"embedding_truncate_dim": "zzz"}, "embedding_truncate_dim"),
        ({"graph_entity_types": [1, 2]}, "graph_entity_types"),
    ):
        reply = json.loads(server.patch_rag_config(json.dumps(patch), session_id=sid))
        assert "unhashable" not in reply.get("error", ""), reply
        assert key in reply.get("error", ""), (patch, reply)
        assert "Expected" in reply["error"], reply


def test_a_patch_returns_what_it_changed(mcp_session):
    server, sid = mcp_session
    reply = json.loads(server.patch_rag_config('{"rerank_top_k": 9}', session_id=sid))
    assert reply["changed"] == {
        "rerank_top_k": {"before": reply["changed"]["rerank_top_k"]["before"], "after": 9}
    }
    assert reply["changed"]["rerank_top_k"]["before"] != 9
    assert "profiles" not in reply
    again = json.loads(server.patch_rag_config('{"rerank_top_k": 9}', session_id=sid))
    assert again["changed"] == {} and again["unchanged"] == ["rerank_top_k"]


def test_the_mvcc_note_is_dropped():
    import lasagna.vector_backends.milvus_backend  # noqa: F401 - installs the filter

    log = logging.getLogger("pymilvus.client.iterator.query_iterator")
    record = log.makeRecord(
        log.name,
        logging.WARNING,
        "query_iterator.py",
        306,
        "failed to get mvccTs from milvus server, use client-side ts instead",
        (),
        None,
    )
    assert not log.filter(record)
    other = log.makeRecord(log.name, logging.WARNING, "x.py", 1, "something else", (), None)
    assert log.filter(other)


def test_init_lists_every_store_and_preselects_what_the_readme_recommends(monkeypatch, tmp_path):
    questionary = pytest.importorskip("questionary")
    from typer.testing import CliRunner

    from lasagna import cli
    from lasagna.core.diagnostics import _BUILTIN_VECTOR_BACKENDS

    asked = []

    def select(prompt, choices, default=None, **kwargs):
        asked.append((prompt, [c.value for c in choices], default))
        return SimpleNamespace(ask=lambda: default or choices[0].value)

    monkeypatch.setattr(questionary, "select", select)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["init"])
    assert result.exit_code == 0, result.output
    (_, stores, _), (_, profiles, preselected) = asked
    assert set(stores) == set(_BUILTIN_VECTOR_BACKENDS)
    assert preselected == "speed"
    readme = (Path(cli.__file__).resolve().parents[2] / "README.md").read_text()
    assert (
        "lasagna init                       # interactive setup; pick the speed profile" in readme
    )
