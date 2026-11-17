"""The stage-isolation check compares configuration, and must say only that.

`stages.py` listed "a component with side effects" among what the check
reports, and every run printed "arms differ only in the stages they name". A
reranker that patched `EmbeddingManager.encode_query` on first call dropped the
control arm from NDCG@10 0.9464 to 0.2744 and changed all 12 ranked lists,
while every stage digest matched the clean run: the manifest digests what was
configured, not what a component did at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.knowledge_evaluation_sdk import stages


def test_the_module_does_not_claim_to_catch_runtime_side_effects():
    doc = stages.__doc__
    assert "a component with side effects" not in doc
    assert "configuration, not behaviour" in doc


def test_the_cli_names_what_was_compared(capsys):
    from lasagna import cli

    row = {
        "benchmark_name": "b",
        "config_name": "base",
        "config_overrides": {},
        "system": {"stages": {"retriever": {"digest": "x"}}},
    }
    cli._print_isolation(SimpleNamespace(_results=[row, dict(row, config_name="other")]))
    out = " ".join(capsys.readouterr().out.split())
    # The claim is about configuration, and about the settings the arms name.
    assert "the configuration that reached each stage differs only in the settings" in out
    assert "the arms name (1 pair compared)" in out
