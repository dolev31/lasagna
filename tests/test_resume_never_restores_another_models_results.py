"""--resume never restores a result another model computed.

The resume key hashed the judge named in the config and the ``RAG_*`` overrides,
but not the model that wrote the answers: ``MODEL_NAME`` is not a ``RAG_``
variable, and the provider is picked from whichever API key is set. So a sweep
checkpointed with gpt-6-luna on OpenAI resumed under Claude on Anthropic and
restored the OpenAI numbers as if Claude had produced them. The judge's base
URL (OpenAI or OpenRouter) and the commit of each local model were missing too.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import run_provenance
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, Experiment, ExperimentConfig
from lasagna.providers import reset_providers

_LLM_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MODEL_NAME",
    "RAG_EVAL_GENERATOR_MODEL",
    "RAG_EVAL_JUDGE_MODEL",
    "RAG_EVAL_JUDGE_BASE_URL",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for var in _LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    reset_providers()
    yield
    reset_providers()


def _key(monkeypatch, tracks=None, **env):
    for var in _LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    reset_providers()
    cfg = ExperimentConfig(name="t", benchmarks=[BenchmarkSpec(name="smoke")])
    if tracks:
        cfg.tracks = list(tracks)
    return Experiment(cfg).generate_tasks()[0].task_key


def test_another_model_on_another_provider_is_another_key(monkeypatch):
    openai = _key(monkeypatch, OPENAI_API_KEY="k", MODEL_NAME="gpt-6-luna")
    anthropic = _key(monkeypatch, ANTHROPIC_API_KEY="k", MODEL_NAME="claude-sonnet-4-5")
    assert openai != anthropic
    assert openai == _key(monkeypatch, OPENAI_API_KEY="k", MODEL_NAME="gpt-6-luna")


def test_the_same_model_name_on_another_provider_is_another_key(monkeypatch):
    openai = _key(monkeypatch, OPENAI_API_KEY="k", MODEL_NAME="some-model")
    anthropic = _key(monkeypatch, ANTHROPIC_API_KEY="k", MODEL_NAME="some-model")
    assert openai != anthropic


def test_model_name_changes_the_key_with_no_rag_variable_set(monkeypatch):
    first = _key(monkeypatch, OPENAI_API_KEY="k", MODEL_NAME="model-a")
    assert first != _key(monkeypatch, OPENAI_API_KEY="k", MODEL_NAME="model-b")


def test_the_judge_base_url_changes_a_generation_key(monkeypatch):
    base = _key(monkeypatch, OPENAI_API_KEY="k")
    routed = _key(
        monkeypatch, OPENAI_API_KEY="k", RAG_EVAL_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
    )
    assert base != routed


def test_a_retrieval_only_key_still_hashes_the_pipeline_llm(monkeypatch):
    """Query expansion, HyDE and LLM rerankers call the pipeline's LLM in retrieval."""
    first = _key(monkeypatch, ("retrieval",), OPENAI_API_KEY="k", MODEL_NAME="model-a")
    assert first != _key(monkeypatch, ("retrieval",), ANTHROPIC_API_KEY="k", MODEL_NAME="model-a")


def test_a_retrieval_only_key_ignores_the_judge(monkeypatch):
    """No judge runs without the generation track, so its endpoint cannot move a number."""
    first = _key(monkeypatch, ("retrieval",), OPENAI_API_KEY="k")
    routed = _key(
        monkeypatch, ("retrieval",), OPENAI_API_KEY="k", RAG_EVAL_JUDGE_BASE_URL="https://x.test/v1"
    )
    assert first == routed


def test_another_commit_of_the_embedder_is_another_key(monkeypatch):
    commits = {}
    monkeypatch.setattr(
        run_provenance, "model_revision", lambda name, cache_dir=None: commits.get(name)
    )
    before = _key(monkeypatch, OPENAI_API_KEY="k")

    from lasagna.knowledge_evaluation_sdk.cost import _arm_models

    task = Experiment(
        ExperimentConfig(name="t", benchmarks=[BenchmarkSpec(name="smoke")])
    ).generate_tasks()[0]
    embedder = next(model for role, model, _ in _arm_models(task) if role == "embedder")
    commits[embedder] = "a" * 40
    cached = _key(monkeypatch, OPENAI_API_KEY="k")
    commits[embedder] = "b" * 40
    updated = _key(monkeypatch, OPENAI_API_KEY="k")
    assert len({before, cached, updated}) == 3
