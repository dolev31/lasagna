"""Experiment results and the LLM cache live in the data directory.

``LASAGNA_DATA_DIR`` moved the metadata database and the uploads, but an
experiment without ``output_dir`` still wrote to ``~/.lasagna/research/results``
and the LLM cache to ``~/.lasagna/cache/llm.sqlite``. On a machine whose home
directory is small or shared, or in a container whose data volume is the only
writable place, the results went somewhere else. Both now follow
``core/paths.data_dir()``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.utils.llm_cache import default_cache_path


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_the_paths_follow_the_data_dir(data_dir):
    run_dir = Path(exp.ExperimentConfig(name="sweep").effective_output_dir())

    assert run_dir.parent == data_dir / "research" / "results"
    assert Path(default_cache_path()) == data_dir / "cache" / "llm.sqlite"


def test_resume_looks_in_the_data_dir(data_dir):
    folder = data_dir / "research" / "results" / "sweep_20261016_120000"
    (folder / ".checkpoints").mkdir(parents=True)
    (folder / ".checkpoints" / "t.json").write_text("{}")

    assert exp.ExperimentConfig(name="sweep").latest_output_dir() == str(folder)


def test_a_run_writes_its_results_and_cache_there(data_dir, monkeypatch):
    ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")
    from tests.test_llm_cache_is_invisible_on_a_cold_cache import _Judge, _StubChatModel

    from lasagna.providers.llm_provider import SimpleLLMProvider
    import lasagna.providers as providers

    provider = SimpleLLMProvider("openai")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _StubChatModel(log=[]))
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.delenv("LASAGNA_LLM_CACHE", raising=False)
    config = {
        "name": "sweep",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "mode": "generation_only",
    }

    experiment = exp.Experiment.from_dict(config)
    asyncio.run(experiment.run())

    written = Path(experiment.config.effective_output_dir())
    assert (written / "results.json").is_file()
    assert written.is_relative_to(data_dir / "research" / "results")
    assert (data_dir / "cache" / "llm.sqlite").is_file()
    assert not (Path.home() / ".lasagna").exists(), sorted(Path.home().rglob("*"))
