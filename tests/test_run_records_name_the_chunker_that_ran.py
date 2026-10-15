"""Run records name the chunker that ran, not the one the config names.

A structural strategy (`structure_aware`, `hierarchical`) with its setting off
runs `semantic`, and any strategy runs `structure_aware` while
`preserve_headers` or `preserve_tables` is on. The records said otherwise: the
held-fixed `chunker` (part of the comparability key) said `structure_aware`
for a run that chunked semantically; the bundle decided the run never split
sentences, so it dropped the punctuation-splitter fallback from the bundle; the
stage manifest digested two arms that chunked alike as different chunkers; and
the MCP status reported `semantic_chunking: false` for the speed and balanced
presets, which chunk each section semantically.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment
from lasagna.knowledge_evaluation_sdk.bundle import _splits_sentences, exercised_fallbacks
from lasagna.knowledge_evaluation_sdk.stages import (
    effective_chunking_strategy,
    isolation_report,
    stage_manifest,
)

STRUCTURE_OFF = {"enable_hierarchical": False, "preserve_headers": False, "preserve_tables": False}
STRATEGIES = (
    "fixed",
    "semantic",
    "recursive",
    "sentence_window",
    "structure_aware",
    "hierarchical",
)


def test_the_record_follows_the_chunkers_own_dispatch():
    from lasagna.processing.chunker.semantic import resolve_effective_strategy

    for strategy, hierarchical, headers, tables in itertools.product(
        STRATEGIES, (False, True), (False, True), (False, True)
    ):
        settings = {
            "chunking_strategy": strategy,
            "enable_hierarchical": hierarchical,
            "preserve_headers": headers,
            "preserve_tables": tables,
        }
        assert effective_chunking_strategy(settings) == resolve_effective_strategy(
            SimpleNamespace(**settings)
        ), settings


@pytest.mark.parametrize("declared", ["structure_aware", "hierarchical"])
def test_a_structural_strategy_with_its_setting_off_records_semantic(declared):
    ingestion = {"chunking_strategy": declared, **STRUCTURE_OFF}
    system = experiment._ran_system("m", 384, ingestion, {"default_top_k": 10})
    assert system["chunker"] == "semantic"


def test_the_declared_block_names_what_the_profile_runs(monkeypatch):
    profile = {"chunking_strategy": "structure_aware", **STRUCTURE_OFF}
    monkeypatch.setattr(experiment, "_resolve_profile_config", lambda name: profile)
    assert experiment._system_under_test("mine")["chunker"] == "semantic"

    profile = {
        "chunking_strategy": "semantic",
        "enable_hierarchical": False,
        "preserve_headers": True,
    }
    assert experiment._system_under_test("mine")["chunker"] == "structure_aware"


def _row(name, overrides, settings):
    return {
        "benchmark_name": "b",
        "config_name": name,
        "config_overrides": overrides,
        "profile": "speed",
        "error": None,
        "system": {"stages": stage_manifest(SimpleNamespace(_config=settings), {})},
    }


def test_the_bundle_knows_a_structural_strategy_split_sentences():
    def recorded(settings):
        # As a task recorded it, declared name and all: results written
        # before the manifest named the chunker that ran read the same way.
        return {"system": {"stages": {"chunker": {"settings": settings}}}}

    row = recorded({"chunking_strategy": "structure_aware", **STRUCTURE_OFF})
    assert _splits_sentences(row) is True
    fallbacks = ["sentence_split_without_nltk", "bm25_without_stemming"]
    assert "sentence_split_without_nltk" in exercised_fallbacks([row], fallbacks)

    assert _splits_sentences(recorded({"chunking_strategy": "fixed", **STRUCTURE_OFF})) is False
    hierarchical = {**STRUCTURE_OFF, "enable_hierarchical": True}
    assert _splits_sentences(recorded({"chunking_strategy": "semantic", **hierarchical})) is False


class TestTheStageManifest:
    def test_two_names_for_one_chunker_digest_alike(self):
        a = _row("a", {}, {"chunking_strategy": "structure_aware", **STRUCTURE_OFF})
        b = _row("b", {}, {"chunking_strategy": "semantic", **STRUCTURE_OFF})
        sa, sb = a["system"]["stages"]["chunker"], b["system"]["stages"]["chunker"]
        assert sa["settings"]["chunking_strategy"] == "semantic"
        assert sa["digest"] == sb["digest"]

    def test_an_arm_whose_strategy_is_shadowed_is_inert(self):
        on = {"enable_hierarchical": False, "preserve_headers": True, "preserve_tables": True}
        control = _row("control", {}, {"chunking_strategy": "semantic", **on})
        arm = _row("fixed", {"chunking_strategy": "fixed"}, {"chunking_strategy": "fixed", **on})
        report = isolation_report([control, arm])
        assert report["undeclared"] == []
        assert any("'fixed' names ['chunker']" in line for line in report["inert"])

    def test_a_chunker_component_keeps_the_setting_as_given(self):
        config = SimpleNamespace(_config={"chunking_strategy": "structure_aware", **STRUCTURE_OFF})
        manifest = stage_manifest(config, {"chunker": "fixed"})
        assert manifest["chunker"]["settings"]["chunking_strategy"] == "structure_aware"


class _Store:
    def has_collection(self, name):
        return False


class _Backend:
    embedding_manager = SimpleNamespace(
        model_name="all-MiniLM-L6-v2", dimension=384, get_cache_stats=lambda: None
    )

    def get_vector_backend(self):
        return _Store()


@pytest.mark.parametrize(
    "settings, strategy, semantic",
    [
        # speed and balanced: structure_aware with headers and tables kept
        ({"chunking_strategy": "structure_aware", "preserve_headers": True}, "structure_aware", 1),
        ({"chunking_strategy": "hierarchical", "enable_hierarchical": False}, "semantic", 1),
        ({"chunking_strategy": "hierarchical", "enable_hierarchical": True}, "hierarchical", 0),
        ({"chunking_strategy": "fixed"}, "fixed", 0),
    ],
)
def test_the_mcp_status_reports_the_chunker_that_runs(
    monkeypatch, tmp_path, settings, strategy, semantic
):
    from lasagna.mcp import server

    config = SimpleNamespace(
        collection_name="c",
        embedding_model="m",
        embedding_dim=384,
        db_path=str(tmp_path / "db"),
        upload_dir=str(tmp_path / "up"),
        chunk_size=800,
        chunk_overlap=0,
        default_top_k=10,
        similarity_threshold=0.0,
        metric_type="COSINE",
        max_file_size=1024 * 1024,
        supported_extensions=[".txt"],
        use_agentic_rag=False,
        chunk_refinement_method="vanilla",
        enable_hybrid_search=True,
        enable_reranking=False,
        enable_mmr=False,
        enable_hyde=False,
        enable_ocr=False,
        **{**STRUCTURE_OFF, **settings},
    )
    monkeypatch.setattr(server, "_get_backend", lambda: _Backend())
    monkeypatch.setattr(server, "_get_config", lambda: config)
    status = json.loads(server.get_system_status())
    assert status["configuration"]["chunking_strategy"] == strategy
    features = next(v for v in status.values() if isinstance(v, dict) and "semantic_chunking" in v)
    assert features["semantic_chunking"] is bool(semantic)
