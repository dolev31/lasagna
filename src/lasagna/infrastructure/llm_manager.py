"""LLM Manager for Lasagna.

Provides a singleton manager for creating and caching LangChain LLM instances
across multiple platforms (OpenAI, Azure, Groq, WatsonX, Google GenAI, OpenRouter).

This is a self-contained copy for standalone deployment — no dependencies on any host application.
"""

import copy
import threading
from collections import OrderedDict
from datetime import date
from typing import Dict, Any, Optional
import hashlib
import json
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from lasagna.providers.llm_provider import SimpleLLMProvider

# Guarded like every other optional provider in this file.
#
# This was a bare module-level import, and `langchain-ibm` is declared in no
# dependency group -- pyproject's own comment lists it among the packages
# "deliberately absent" after the extraction from the host project. So importing
# this module on a clean install raised ImportError, and the 12 tests that
# exercise LLMManager passed only on machines that happened to have it. It is a
# vendor SDK pulling `ibm-watsonx-ai`; a user on OpenAI should not need it.
try:
    from langchain_ibm import ChatWatsonx
except ImportError:
    ChatWatsonx = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


#: Ids this manager never sent a temperature, kept for the ones the capability
#: table does not know (gpt-5-mini): it may not take one, and a temperature set
#: on a cached model later is sent as is.
_LEGACY_NO_TEMPERATURE = ("o1", "o3", "gpt-5")

# Configurable max cached LLM instances (prevents unbounded memory growth)
_DEFAULT_MAX_MODELS = 50


class LLMManager:
    """Singleton class to manage LLM instances based on agent name and settings.

    Thread-safe: all cache operations are protected by ``_cache_lock``.
    Model instances are cached in an LRU ``OrderedDict`` with a configurable
    maximum size (default 50, override via ``RAG_LLM_CACHE_MAX_SIZE`` env var).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._max_models = self._parse_max_models()
            self._models: OrderedDict[str, BaseChatModel] = OrderedDict()
            self._cache_lock = threading.Lock()
            self._pre_instantiated_model: Optional[BaseChatModel] = None
            self._initialized = True

    @staticmethod
    def _parse_max_models() -> int:
        """Parse ``RAG_LLM_CACHE_MAX_SIZE`` with validation and safe fallback."""
        raw = os.environ.get("RAG_LLM_CACHE_MAX_SIZE")
        if raw is None:
            return _DEFAULT_MAX_MODELS
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError(f"must be positive, got {value}")
            return value
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"Invalid RAG_LLM_CACHE_MAX_SIZE={raw!r} ({exc}); "
                f"falling back to default {_DEFAULT_MAX_MODELS}"
            )
            return _DEFAULT_MAX_MODELS

    def convert_dates_to_strings(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_dates_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_dates_to_strings(item) for item in obj]
        elif isinstance(obj, date):
            return obj.isoformat()
        else:
            return obj

    def set_llm(self, model: BaseChatModel) -> None:
        """Set a pre-instantiated model to use for all tasks.

        Thread-safe: acquires ``_cache_lock`` so that concurrent
        ``get_model()`` calls see a consistent model source.

        Args:
            model: Pre-instantiated ChatOpenAI or BaseChatModel instance
        """
        if not isinstance(model, BaseChatModel):
            raise ValueError("Model must be an instance of BaseChatModel or its subclass")

        with self._cache_lock:
            self._pre_instantiated_model = model
        logger.info(f"Pre-instantiated model set: {type(model).__name__}")

    def _update_model_parameters(
        self,
        model: BaseChatModel,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        max_completion_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """Update model parameters (temperature, max_tokens, and max_completion_tokens) for the task.

        NOTE: This mutates the model in-place.  When called on a cached model,
        the caller should hold ``_cache_lock`` or operate on a copy to prevent
        concurrent mutation from other threads.
        """
        model_kwargs = {}
        if hasattr(model, "model_kwargs") and model.model_kwargs is not None:
            model_kwargs = model.model_kwargs.copy()

        model_name = getattr(model, "model_name", "") or getattr(model, "model", "")
        # A reasoning model built with reasoning_effort="none" takes temperature.
        takes_temperature = (
            not self._is_reasoning_model(model_name)
            or getattr(model, "reasoning_effort", None) == "none"
        )

        if takes_temperature:
            if hasattr(model, "temperature"):
                model.temperature = temperature
            elif "temperature" in model_kwargs:
                model_kwargs["temperature"] = temperature

        completion_tokens = (
            max_completion_tokens if max_completion_tokens is not None else max_tokens
        )

        if hasattr(model, "max_tokens"):
            model.max_tokens = max_tokens
        elif "max_tokens" in model_kwargs:
            model_kwargs["max_tokens"] = max_tokens

        if hasattr(model, "max_completion_tokens"):
            model.max_completion_tokens = completion_tokens
        elif "max_completion_tokens" in model_kwargs:
            model_kwargs["max_completion_tokens"] = completion_tokens

        if hasattr(model, "model_kwargs") and model.model_kwargs is not None:
            model.model_kwargs = model_kwargs

        return model

    @staticmethod
    def _clone_model_instance(model: BaseChatModel) -> BaseChatModel:
        """Create a per-call clone so parameter mutation does not leak across threads.

        Tries Pydantic v2 ``model_copy(deep=True)`` first (LangChain >=0.2),
        then falls back to ``copy.deepcopy``.
        """
        if hasattr(model, "model_copy"):
            return model.model_copy(deep=True)
        return copy.deepcopy(model)

    def clear_pre_instantiated_model(self) -> None:
        """Clear the pre-instantiated model and return to normal model creation.

        Thread-safe: acquires ``_cache_lock`` so that concurrent
        ``get_model()`` calls see a consistent model source.
        """
        with self._cache_lock:
            self._pre_instantiated_model = None

    def _create_cache_key(self, model_settings: Dict[str, Any]) -> str:
        """Create a unique cache key from model settings including resolved values"""
        if hasattr(model_settings, "to_dict"):
            settings_dict = model_settings.to_dict()
        else:
            settings_dict = dict(model_settings)
        d = self.convert_dates_to_strings(settings_dict)
        keys_to_delete = [key for key in d if "prompt" in key]

        for key in keys_to_delete:
            del d[key]

        platform = model_settings.get("platform")
        if platform:
            d["resolved_model_name"] = self._get_model_name(model_settings, platform)
            d["resolved_api_version"] = self._get_api_version(model_settings, platform)
            d["resolved_base_url"] = self._get_base_url(model_settings, platform)

        settings_str = json.dumps(d, sort_keys=True)
        return hashlib.md5(settings_str.encode()).hexdigest()

    def _get_model_name(self, model_settings: Dict[str, Any], platform: str) -> str:
        """Get model name with environment variable override support"""
        toml_model_name = model_settings.get("model_name")

        if platform == "openai":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                # Same default as SimpleLLMProvider, with the source cited there.
                return SimpleLLMProvider.SUPPORTED_PLATFORMS["openai"]
        elif platform == "groq":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return "openai/gpt-oss-20b"
        elif platform == "watsonx":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        elif platform == "azure":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return SimpleLLMProvider.SUPPORTED_PLATFORMS["azure"]
        elif platform == "google-genai":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                # Gemini 1.5 Pro was shut down 2025-09-29
                # (https://ai.google.dev/gemini-api/docs/changelog);
                # gemini-2.5-pro is served, with no shutdown date announced.
                # https://ai.google.dev/gemini-api/docs/deprecations
                return "gemini-2.5-pro"
        elif platform == "openrouter":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return SimpleLLMProvider.SUPPORTED_PLATFORMS["openrouter"]
        elif platform == "ollama":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return "llama3.2"
        elif platform == "huggingface":
            env_model_name = os.environ.get("HF_MODEL_ID") or os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return "microsoft/Phi-3-mini-4k-instruct"
        elif platform == "llamacpp":
            # llamacpp uses model_path, not model_name
            return toml_model_name or ""
        elif platform == "vllm":
            env_model_name = os.environ.get("MODEL_NAME")
            if env_model_name:
                return env_model_name
            elif toml_model_name:
                return toml_model_name
            else:
                return "default"
        else:
            if toml_model_name:
                return toml_model_name
            else:
                raise ValueError(f"model_name must be specified for platform: {platform}")

    def _get_api_version(self, model_settings: Dict[str, Any], platform: str) -> Optional[str]:
        """Get API version with environment variable override support"""
        if platform == "openai":
            env_api_version = os.environ.get("OPENAI_API_VERSION")
            if env_api_version:
                return env_api_version
            toml_api_version = model_settings.get("api_version")
            if toml_api_version:
                if isinstance(toml_api_version, date):
                    toml_api_version = toml_api_version.isoformat()
                return toml_api_version
            # OpenAI (non-Azure) does not require an api_version
            return None
        else:
            api_version = model_settings.get("api_version", "2024-08-06")
            if isinstance(api_version, date):
                api_version = api_version.isoformat()
            return api_version

    def _get_base_url(self, model_settings: Dict[str, Any], platform: str) -> str:
        """Get base URL with environment variable override support"""
        if platform == "openai":
            env_base_url = os.environ.get("OPENAI_BASE_URL")
            if env_base_url:
                return env_base_url
            toml_url = model_settings.get("url")
            if toml_url:
                return toml_url
            return None
        elif platform == "openrouter":
            env_base_url = os.environ.get("OPENROUTER_BASE_URL")
            if env_base_url:
                return env_base_url
            toml_url = model_settings.get("url")
            if toml_url:
                return toml_url
            return "https://openrouter.ai/api/v1"
        elif platform == "ollama":
            return model_settings.get("url") or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        elif platform == "vllm":
            return model_settings.get("url") or os.environ.get(
                "VLLM_BASE_URL", "http://localhost:8000/v1"
            )
        elif platform in ("huggingface", "llamacpp"):
            return None  # Local in-process — no URL needed
        else:
            return model_settings.get("url")

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Whether the model takes temperature only with reasoning off, or never.

        Read from :mod:`lasagna.providers.model_capabilities`, the one table of
        what each model accepts. The prefix list this replaced matched every
        ``gpt-5*`` and nothing newer, so gpt-6-luna was sent a temperature at
        its default reasoning effort, which the API rejects.
        """
        from lasagna.providers.model_capabilities import ACCEPTED, model_capabilities

        if not model_name:
            return False
        return model_capabilities(model_name).temperature != ACCEPTED or model_name.startswith(
            _LEGACY_NO_TEMPERATURE
        )

    @staticmethod
    def _sampling(model_name: str, temperature: float, platform: str) -> Dict[str, Any]:
        """``temperature`` and ``reasoning_effort`` as the model takes them."""
        from lasagna.providers.model_capabilities import sampling_kwargs

        kwargs, _ = sampling_kwargs(model_name, temperature, platform)
        if (model_name or "").startswith(
            _LEGACY_NO_TEMPERATURE
        ) and "reasoning_effort" not in kwargs:
            kwargs.pop("temperature", None)
        return kwargs

    def _create_llm_instance(self, model_settings: Dict[str, Any]):
        """Create LLM instance based on platform and settings"""
        platform = model_settings.get("platform")
        temperature = model_settings.get("temperature", 0.7)
        max_tokens = model_settings.get("max_tokens")
        if max_tokens is None:
            raise ValueError("max_tokens must be specified in model_settings")
        model_name = self._get_model_name(model_settings, platform)
        api_version = self._get_api_version(model_settings, platform)
        base_url = self._get_base_url(model_settings, platform)
        if platform == "azure":
            api_version = str(model_settings.get("api_version"))
            is_reasoning = self._is_reasoning_model(model_name)

            if is_reasoning:
                llm = AzureChatOpenAI(
                    model_version=api_version,
                    timeout=61,
                    api_version="2025-04-01-preview",
                    azure_deployment=model_name + "-" + api_version,
                    max_completion_tokens=max_tokens,
                    **self._sampling(model_name, temperature, platform),
                )
            else:
                llm = AzureChatOpenAI(
                    timeout=61,
                    azure_deployment=model_name + "-" + api_version,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        elif platform == "openai":
            openai_params = {
                "model_name": model_name,
                "max_tokens": max_tokens,
                "timeout": 61,
                **self._sampling(model_name, temperature, platform),
            }
            apikey_name = model_settings.get("apikey_name")
            if apikey_name:
                openai_params["openai_api_key"] = os.environ.get(apikey_name)
            if base_url:
                openai_params["openai_api_base"] = base_url
            llm = ChatOpenAI(**openai_params)
        elif platform == "groq":
            llm = ChatGroq(
                max_tokens=max_tokens,
                model=model_name,
                temperature=temperature,
            )
        elif platform == "watsonx":
            if ChatWatsonx is None:
                # Named, with the fix. Deferring the guard to the call site
                # rather than the import is only an improvement if the failure
                # still says what to do -- otherwise it becomes
                # "'NoneType' object is not callable" several frames deep.
                raise ImportError(
                    "The watsonx platform requires langchain-ibm, which is not "
                    "installed. Install it with: pip install langchain-ibm"
                )
            project_id = os.environ.get("WATSONX_PROJECT_ID")
            if not project_id:
                raise ValueError(
                    "WATSONX_PROJECT_ID environment variable is required for the WatsonX platform"
                )
            llm = ChatWatsonx(
                model_id=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                project_id=project_id,
            )
        elif platform == "google-genai":
            llm = ChatGoogleGenerativeAI(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif platform == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            openrouter_params = {
                "model_name": model_name,
                "max_tokens": max_tokens,
                "timeout": 61,
                "openai_api_key": api_key,
                "openai_api_base": base_url,
                **self._sampling(model_name, temperature, platform),
            }
            default_headers = {}
            site_url = model_settings.get("site_url") or os.environ.get("OPENROUTER_SITE_URL")
            app_name = model_settings.get("app_name") or os.environ.get("OPENROUTER_APP_NAME")
            if site_url:
                default_headers["HTTP-Referer"] = site_url
            if app_name:
                default_headers["X-Title"] = app_name
            if default_headers:
                openrouter_params["default_headers"] = default_headers
            llm = ChatOpenAI(**openrouter_params)
        elif platform == "ollama":
            ollama_base_url = model_settings.get("base_url") or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            try:
                from langchain_ollama import ChatOllama

                # ChatOllama uses num_predict (not max_tokens)
                llm = ChatOllama(
                    model=model_name,
                    base_url=ollama_base_url,
                    temperature=temperature,
                    num_predict=max_tokens,
                )
            except ImportError:
                # Fall back to OpenAI-compatible
                if not ollama_base_url.endswith("/v1"):
                    ollama_base_url = ollama_base_url.rstrip("/") + "/v1"
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key="ollama",
                    openai_api_base=ollama_base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        elif platform == "huggingface":
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            except ImportError:
                raise ValueError(
                    "HuggingFace support requires langchain-huggingface. "
                    "Install with: pip install langchain-huggingface transformers torch accelerate"
                )
            model_id = model_settings.get("model_id", model_name)
            device_map = model_settings.get("device_map", "auto")
            user_pipeline_kwargs = model_settings.get("pipeline_kwargs", {})
            gen_kwargs = {"max_new_tokens": max_tokens, **user_pipeline_kwargs}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
            hf_llm = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task=model_settings.get("task", "text-generation"),
                device_map=device_map,
                pipeline_kwargs=gen_kwargs,
            )
            llm = ChatHuggingFace(llm=hf_llm)
        elif platform == "llamacpp":
            try:
                from langchain_community.chat_models import ChatLlamaCpp
            except ImportError:
                raise ValueError(
                    "llama.cpp support requires langchain-community and llama-cpp-python. "
                    "Install with: pip install langchain-community llama-cpp-python"
                )
            model_path = model_settings.get("model_path") or os.environ.get("LLAMACPP_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "llamacpp platform requires 'model_path' in model_settings "
                    "or LLAMACPP_MODEL_PATH environment variable"
                )
            llm = ChatLlamaCpp(
                model_path=model_path,
                n_ctx=model_settings.get("n_ctx", 4096),
                n_gpu_layers=model_settings.get("n_gpu_layers", -1),
                n_batch=model_settings.get("n_batch", 512),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif platform == "vllm":
            vllm_base_url = model_settings.get("base_url") or os.environ.get(
                "VLLM_BASE_URL", "http://localhost:8000/v1"
            )
            if not vllm_base_url.endswith("/v1"):
                vllm_base_url = vllm_base_url.rstrip("/") + "/v1"
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                openai_api_base=vllm_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        return llm

    def get_model(self, model_settings: Dict[str, Any]):
        """Get or create LLM instance for the given model settings.

        Thread-safe: the pre-instantiated model check, cache lookup, creation,
        insertion, and LRU eviction are all protected by ``_cache_lock``.
        """
        max_tokens = model_settings.get("max_tokens")
        if max_tokens is None:
            raise ValueError("max_tokens must be specified in model_settings")

        cache_key = self._create_cache_key(model_settings)

        with self._cache_lock:
            if self._pre_instantiated_model is not None:
                # Clone per call so parameter mutation does not leak across threads.
                cloned = self._clone_model_instance(self._pre_instantiated_model)
                self._update_model_parameters(cloned, temperature=0.1, max_tokens=max_tokens)
                return cloned
            if cache_key in self._models:
                # LRU: move to end on hit
                self._models.move_to_end(cache_key)
                cached_model = self._models[cache_key]
                self._update_model_parameters(
                    cached_model,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_tokens,
                )
                return cached_model

            # Cache miss — create outside the hot path of other cache hits,
            # but still under lock to prevent duplicate creation for the same key.
            platform = model_settings.get("platform", "unknown")
            model_name = self._get_model_name(model_settings, platform)
            api_version = self._get_api_version(model_settings, platform)
            base_url = self._get_base_url(model_settings, platform)

            logger.debug(
                f"Creating new model: {platform}/{model_name} (api_version={api_version}, base_url={base_url})"
            )
            model = self._create_llm_instance(model_settings)

            # LRU eviction: remove oldest entries when at capacity
            while len(self._models) >= self._max_models:
                evicted_key, _ = self._models.popitem(last=False)
                logger.debug(f"LLM cache eviction (LRU): {evicted_key[:8]}...")

            self._models[cache_key] = model
            self._update_model_parameters(model, temperature=0.1, max_tokens=max_tokens)
            return model
