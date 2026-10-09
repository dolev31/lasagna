"""Regression tests for infrastructure thread-safety fixes.

Covers:
- LLMManager._parse_max_models() env-var validation
- LLMManager._clone_model_instance() per-call cloning
- LLMManager set_llm/clear_pre_instantiated_model/get_model concurrency
- get_embedding_manager() never returns None under concurrent reset
- get_search_cache() never returns None under concurrent reset
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# LLMManager tests
# ---------------------------------------------------------------------------


class TestLLMManagerParseMaxModels:
    """_parse_max_models() must handle all invalid env values gracefully."""

    def _parse(self):
        from lasagna.infrastructure.llm_manager import LLMManager

        return LLMManager._parse_max_models()

    def test_default_when_unset(self):
        env = os.environ.copy()
        env.pop("RAG_LLM_CACHE_MAX_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            assert self._parse() == 50

    def test_valid_positive_value(self):
        with patch.dict(os.environ, {"RAG_LLM_CACHE_MAX_SIZE": "25"}):
            assert self._parse() == 25

    def test_zero_falls_back(self):
        with patch.dict(os.environ, {"RAG_LLM_CACHE_MAX_SIZE": "0"}):
            assert self._parse() == 50

    def test_negative_falls_back(self):
        with patch.dict(os.environ, {"RAG_LLM_CACHE_MAX_SIZE": "-5"}):
            assert self._parse() == 50

    def test_non_numeric_falls_back(self):
        with patch.dict(os.environ, {"RAG_LLM_CACHE_MAX_SIZE": "abc"}):
            assert self._parse() == 50

    def test_float_falls_back(self):
        with patch.dict(os.environ, {"RAG_LLM_CACHE_MAX_SIZE": "3.5"}):
            assert self._parse() == 50


class TestLLMManagerCloneModel:
    """_clone_model_instance() must return a distinct object."""

    def test_clone_returns_different_object(self):
        from lasagna.infrastructure.llm_manager import LLMManager

        # Use a mock with model_copy (Pydantic v2 style)
        model = MagicMock()
        model.model_copy.return_value = MagicMock()

        cloned = LLMManager._clone_model_instance(model)
        model.model_copy.assert_called_once_with(deep=True)
        assert cloned is not model

    def test_clone_falls_back_to_deepcopy(self):
        from lasagna.infrastructure.llm_manager import LLMManager

        # Model without model_copy attribute
        model = MagicMock(spec=[])  # empty spec -> no model_copy
        cloned = LLMManager._clone_model_instance(model)
        assert cloned is not model


class TestLLMManagerPreInstantiatedSync:
    """set_llm / clear / get_model with pre-instantiated model are thread-safe.

    Behavioral tests: exercise concurrent set_llm, clear, and get_model to
    verify no crashes, no None returns, and no parameter bleed.
    """

    def _make_fake_model(self, *, temperature=0.0, max_tokens=100):
        """Create a fake BaseChatModel-like mock suitable for set_llm/get_model."""
        from langchain_core.language_models.chat_models import BaseChatModel

        model = MagicMock(spec=BaseChatModel)
        model.temperature = temperature
        model.max_tokens = max_tokens
        model.max_completion_tokens = max_tokens
        model.model_kwargs = {}
        model.model_name = "test-model"
        # model_copy returns a fresh mock each time (simulating clone)
        model.model_copy = lambda deep=False: self._make_fake_model(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return model

    def test_concurrent_set_clear_get_no_crash(self):
        """Concurrent set_llm / clear / get_model must not raise."""
        from lasagna.infrastructure.llm_manager import LLMManager

        lm = LLMManager()
        fake_model = self._make_fake_model()
        errors = []
        settings = {"max_tokens": 200, "platform": "openai"}

        # Monkeypatch _create_llm_instance so the cache-miss path returns a
        # dummy model instead of hitting real APIs.  This removes the need to
        # tolerate ValueError from missing API keys, ensuring any unexpected
        # ValueError is surfaced as a test failure.
        original_create = lm._create_llm_instance
        lm._create_llm_instance = lambda s: self._make_fake_model()

        def setter():
            for _ in range(50):
                try:
                    lm.set_llm(fake_model)
                except Exception as e:
                    errors.append(("set_llm", type(e).__name__, str(e)))

        def clearer():
            for _ in range(50):
                try:
                    lm.clear_pre_instantiated_model()
                except Exception as e:
                    errors.append(("clear", type(e).__name__, str(e)))

        def getter():
            for _ in range(50):
                try:
                    result = lm.get_model(settings)
                    assert result is not None, "get_model() returned None"
                except AssertionError as e:
                    errors.append(("get_model", "AssertionError", str(e)))
                except Exception as e:
                    errors.append(("get_model", type(e).__name__, str(e)))

        # Pre-set the model so the pre-instantiated path is exercised
        lm.set_llm(fake_model)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=setter))
            threads.append(threading.Thread(target=clearer))
            threads.append(threading.Thread(target=getter))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cleanup
        lm._create_llm_instance = original_create
        lm.clear_pre_instantiated_model()

        assert not errors, f"Concurrent set/clear/get errors: {errors}"

    def test_pre_instantiated_clone_isolates_parameters(self):
        """Two get_model calls with different max_tokens must not bleed."""
        from lasagna.infrastructure.llm_manager import LLMManager

        lm = LLMManager()
        fake_model = self._make_fake_model()
        lm.set_llm(fake_model)

        results = {}
        errors = []

        def get_with_tokens(name, tokens):
            try:
                result = lm.get_model({"max_tokens": tokens, "platform": "openai"})
                assert result is not None
                results[name] = result.max_tokens
            except Exception as e:
                errors.append((name, type(e).__name__, str(e)))

        t1 = threading.Thread(target=get_with_tokens, args=("a", 100))
        t2 = threading.Thread(target=get_with_tokens, args=("b", 999))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        lm.clear_pre_instantiated_model()

        assert not errors, f"Parameter isolation errors: {errors}"
        # Each thread should get the max_tokens it requested, not the other's
        if "a" in results and "b" in results:
            assert results["a"] == 100, f"Thread 'a' got max_tokens={results['a']}, expected 100"
            assert results["b"] == 999, f"Thread 'b' got max_tokens={results['b']}, expected 999"


# ---------------------------------------------------------------------------
# Embedding singleton tests
# ---------------------------------------------------------------------------

# Expected exceptions during model loading in test environments
_EMBEDDING_LOAD_EXCEPTIONS = (ImportError, RuntimeError, OSError)


class TestEmbeddingManagerNeverReturnsNone:
    """get_embedding_manager() must never return None, even under concurrent reset."""

    def test_concurrent_get_and_reset(self):
        from lasagna.infrastructure.embedding.manager import (
            get_embedding_manager,
            reset_embedding_manager,
        )

        errors = []

        def getter():
            for _ in range(50):
                try:
                    em = get_embedding_manager()
                    assert em is not None, "get_embedding_manager() returned None"
                    _ = em.model_name  # access an attribute
                except AssertionError as e:
                    errors.append(("None-return", str(e)))
                except _EMBEDDING_LOAD_EXCEPTIONS:
                    pass  # Model loading may fail in test env — tolerated
                except Exception as e:
                    errors.append(("unexpected", f"{type(e).__name__}: {e}"))

        def resetter():
            for _ in range(25):
                reset_embedding_manager()
                time.sleep(0.001)

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=getter))
            threads.append(threading.Thread(target=resetter))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Embedding singleton race failures: {errors}"


# ---------------------------------------------------------------------------
# Search cache singleton tests
# ---------------------------------------------------------------------------


class TestSearchCacheSingletonNeverReturnsNone:
    """get_search_cache() must never return None, even under concurrent reset."""

    def test_concurrent_get_and_reset(self):
        from lasagna.infrastructure.cache.search_cache import (
            get_search_cache,
            reset_search_cache,
        )

        errors = []

        def getter():
            for _ in range(100):
                try:
                    sc = get_search_cache()
                    assert sc is not None, "get_search_cache() returned None"
                    _ = sc.stats()
                except AssertionError as e:
                    errors.append(("None-return", str(e)))
                except Exception as e:
                    # SearchCache has no heavy deps — any exception is unexpected
                    errors.append(("unexpected", f"{type(e).__name__}: {e}"))

        def resetter():
            for _ in range(50):
                reset_search_cache()
                time.sleep(0.001)

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=getter))
            threads.append(threading.Thread(target=resetter))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Search cache singleton race failures: {errors}"
