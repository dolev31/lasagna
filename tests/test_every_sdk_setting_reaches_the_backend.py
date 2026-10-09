"""An SDK setting the caller writes down reaches the backend.

The bridge used to copy only fields whose value differed from the dataclass
default. ``docling_enabled`` defaults to False in the dataclass but is true in
the shipped settings, so ``KnowledgeConfig(docling_enabled=False)`` and
``config_for_speed()`` (which asks for no Docling) both left Docling on: an
explicit value equal to the default was indistinguishable from no value. Six
other ``docling_*`` fields appeared in no bridge table at all, so setting them
did nothing.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lasagna.sdk import _config_bridge as bridge
from lasagna.sdk.config import KnowledgeConfig, config_for_speed


def _bridged(config: KnowledgeConfig) -> dict:
    return {**bridge.get_init_overrides(config), **bridge.get_runtime_overrides(config)}


def test_a_value_equal_to_the_sdk_default_is_still_bridged_when_set():
    overrides = _bridged(KnowledgeConfig(enable_hyde=False))
    assert overrides == {"enable_hyde": False}


def test_docling_can_be_turned_off():
    assert _bridged(KnowledgeConfig(docling_enabled=False))["docling_enabled"] is False


def test_the_speed_preset_turns_docling_off():
    assert bridge.get_runtime_overrides(config_for_speed())["docling_enabled"] is False


def test_every_docling_setting_is_bridged():
    values = {
        "docling_mode": "local",
        "docling_pdf_backend": "pypdfium2",
        "docling_ocr_enabled": False,
        "docling_table_structure": False,
        "docling_max_tokens": 128,
        "docling_timeout": 5,
    }
    overrides = _bridged(KnowledgeConfig(**values))
    assert {k: overrides.get(k) for k in values} == values


def test_every_field_is_bridged_or_declared_sdk_only():
    names = {f.name for f in dataclasses.fields(KnowledgeConfig)}
    covered = (
        set(bridge.INIT_TIME_FIELDS)
        | set(bridge.RUNTIME_FIELD_MAP)
        | set(bridge.SPECIAL_MAPPINGS)
        | {k for fields in bridge.NESTED_MAPPINGS.values() for k in fields}
    )
    assert names - covered == set(bridge.SDK_ONLY_FIELDS)
    assert not (covered - names), "the bridge maps fields KnowledgeConfig does not have"


def test_an_untouched_config_bridges_nothing():
    assert _bridged(KnowledgeConfig()) == {}


def test_updating_a_config_bridges_only_what_was_set():
    config = KnowledgeConfig(enable_hyde=True).with_updates(chunk_size=900)
    assert _bridged(config) == {"enable_hyde": True, "chunk_size": 900}


def test_assigning_a_field_marks_it_set():
    config = KnowledgeConfig()
    config.enable_reranking = KnowledgeConfig().enable_reranking
    assert "enable_reranking" in _bridged(config)


def test_the_docling_default_matches_the_shipped_settings():
    from lasagna.core.config import get_rag_config

    assert KnowledgeConfig().docling_enabled is get_rag_config().docling_enabled


# ---------------------------------------------------------------------------
# db_path and upload_dir were read by nothing, so config_for_standalone(...)
# wrote to ~/.rag_knowledge without a word.
# ---------------------------------------------------------------------------

_STORAGE_PROBE = """
import sys
from lasagna.sdk import KnowledgeClient, KnowledgeConfig
from lasagna.core.config import get_rag_config

client = KnowledgeClient(config=KnowledgeConfig(db_path=sys.argv[1], upload_dir=sys.argv[2]))
client._apply_storage_paths()
config = get_rag_config()
print(config.db_path)
print(config.upload_dir)
"""


def test_db_path_and_upload_dir_choose_where_storage_lives(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("RAG_DB_DIR", "RAG_UPLOAD_DIR")}
    db_file, uploads = tmp_path / "mine" / "vectors.db", tmp_path / "mine" / "up"
    proc = subprocess.run(
        [sys.executable, "-c", _STORAGE_PROBE, str(db_file), str(uploads)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    db_path, upload_dir = proc.stdout.strip().splitlines()[-2:]
    assert Path(db_path).parent.resolve() == db_file.parent.resolve()
    assert Path(upload_dir).resolve() == uploads.resolve()


def test_storage_paths_that_cannot_take_effect_say_so(tmp_path):
    from lasagna.core.config import get_rag_config
    from lasagna.sdk import KnowledgeClient

    get_rag_config()  # the process has chosen its storage already
    client = KnowledgeClient(config=KnowledgeConfig(db_path=str(tmp_path / "elsewhere")))
    with pytest.warns(UserWarning, match="db_path"):
        client._apply_storage_paths()


def test_a_setting_nothing_reads_is_not_offered():
    """show_confidence_scores was bridged to a key no code read; setting it did nothing."""
    with pytest.raises(TypeError):
        KnowledgeConfig(show_confidence_scores=True)
