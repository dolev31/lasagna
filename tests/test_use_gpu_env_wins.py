"""RAG_USE_GPU=false must hold even when a profile or query override says true.

Profiles inherit use_gpu from the settings file; applied as per-query
overrides they outranked the operator's RAG_USE_GPU=false, and models were
moved onto a GPU backend the operator had turned off.
"""

from __future__ import annotations

from lasagna.core.config import RAGConfig


def test_the_operator_setting_beats_an_override(monkeypatch):
    monkeypatch.setenv("RAG_USE_GPU", "false")
    assert RAGConfig().with_query_overrides({"use_gpu": True}).use_gpu is False


def test_without_the_env_var_the_override_applies(monkeypatch):
    monkeypatch.delenv("RAG_USE_GPU", raising=False)
    assert RAGConfig().with_query_overrides({"use_gpu": False}).use_gpu is False
