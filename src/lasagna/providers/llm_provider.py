"""LLM provider interface for RAG system.

This interface abstracts LLM model creation, allowing the RAG module
to work with any LLM framework.

Supports 18+ platforms out of the box:

  Tier 1 (native LangChain integrations):
    openai, anthropic, azure, google, groq, watsonx, mistral, cohere, nvidia

  Tier 2 (OpenAI-compatible — zero extra deps):
    openrouter, together, deepseek, fireworks, perplexity

  Tier 3 (local inference — no API key needed):
    ollama, huggingface, llamacpp, vllm

Example usage::

    from lasagna.providers import SimpleLLMProvider

    llm = SimpleLLMProvider(default_provider="openai")
    model = llm.get_chat_model(
        {"platform": "openai", "model_name": "gpt-6-luna"},
        temperature=0.1,
        max_tokens=1000,
    )
"""

import functools
import inspect
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, NamedTuple, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def _instrument_chat_model(
    model: Any, provider: "LLMProvider", model_settings: Dict[str, Any], temperature: Any
) -> Any:
    """Give a chat model from any provider the token meter and the response cache.

    `SimpleLLMProvider` attaches both itself, with what it knows about each
    model's sampling; this covers a provider that builds its models some other
    way. Such a model is cached only when temperature 0 was asked for and the
    model kept it (one without a ``temperature`` attribute is taken at its
    word). A model that is not a LangChain chat model cannot carry the cache,
    and the run records that instead of an empty cache.
    """
    from lasagna.utils.llm_cache import attach_llm_cache, carries_llm_cache, note_not_applied
    from lasagna.utils.token_usage import attach_usage_meter

    model = attach_usage_meter(model)
    if carries_llm_cache(model):
        return model
    name = f"{type(provider).__module__}.{type(provider).__qualname__}"
    try:
        from langchain_core.language_models import BaseLanguageModel
    except ImportError:  # pragma: no cover - langchain-core is a core dependency
        BaseLanguageModel = None  # type: ignore[assignment,misc]
    if BaseLanguageModel is None or not isinstance(model, BaseLanguageModel):
        note_not_applied(
            f"the LLM provider {name} returned a {type(model).__name__}, not a LangChain "
            f"chat model, so its calls are not cached"
        )
        return model
    kept = getattr(model, "temperature", temperature)
    deterministic = temperature == 0 and kept == 0
    platform = f"provider:{name}:{model_settings.get('model_name', '')}"
    return attach_llm_cache(model, platform, deterministic)


def _instrumented(get_chat_model: Any) -> Any:
    """Wrap a subclass's ``get_chat_model`` with `_instrument_chat_model`."""
    signature = inspect.signature(get_chat_model)

    @functools.wraps(get_chat_model)
    def wrapper(self: "LLMProvider", *args: Any, **kwargs: Any) -> Any:
        model = get_chat_model(self, *args, **kwargs)
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            settings = bound.arguments.get("model_settings") or {}
            temperature = bound.arguments.get("temperature", 0.0)
        except TypeError:
            settings, temperature = {}, None
        return _instrument_chat_model(
            model, self, settings if isinstance(settings, dict) else {}, temperature
        )

    wrapper._lasagna_instrumented = True  # type: ignore[attr-defined]
    return wrapper


class LLMProvider(ABC):
    """Abstract interface for LLM model access.

    Implementations provide chat models for:
    - Query refinement in agentic RAG
    - Document summarization
    - HyDE (Hypothetical Document Embeddings) generation
    - Query enhancement and decomposition

    A subclass's ``get_chat_model`` is wrapped so the model it returns carries
    the token meter and the experiment's response cache, as the models of the
    built-in provider do: a provider set with ``set_llm_provider`` used to
    bypass both.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("get_chat_model")
        if (
            callable(own)
            and not getattr(own, "__isabstractmethod__", False)
            and not getattr(own, "_lasagna_instrumented", False)
        ):
            cls.get_chat_model = _instrumented(own)  # type: ignore[method-assign]

    @abstractmethod
    def get_chat_model(
        self,
        model_settings: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "BaseChatModel":
        """Get a chat model instance.

        Args:
            model_settings: Dict with model configuration. Expected keys:
                - 'platform': Provider name ('openai', 'anthropic', etc.)
                - 'model_name': Model identifier ('gpt-6-luna', etc.)
                - Additional provider-specific settings
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum output tokens (None = model default)

        Returns:
            LangChain-compatible chat model instance

        Raises:
            ValueError: If platform not supported or model unavailable
        """
        ...

    def get_llm(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> "BaseChatModel":
        """Get a chat model using defaults, with optional model name override.

        Convenience wrapper around get_chat_model() for callers that don't
        need to specify a full model_settings dict.

        Args:
            model_name: Optional model identifier override
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum output tokens

        Returns:
            LangChain-compatible chat model instance
        """
        settings: Dict[str, Any] = {}
        if model_name:
            settings["model_name"] = model_name
        return self.get_chat_model(settings, temperature=temperature, max_tokens=max_tokens)

    @abstractmethod
    def validate_model_settings(self, model_settings: Dict[str, Any]) -> bool:
        """Check if model settings are valid and model is available.

        Args:
            model_settings: Settings dict to validate

        Returns:
            True if model can be created, False otherwise
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible platforms (Tier 2)
# These use ChatOpenAI with a custom base_url — zero extra dependencies.
# ---------------------------------------------------------------------------


class OpenAICompatibleConfig(NamedTuple):
    base_url: str
    default_model: str
    api_key_env_var: str


_OPENAI_COMPATIBLE: Dict[str, OpenAICompatibleConfig] = {
    "openrouter": OpenAICompatibleConfig(
        "https://openrouter.ai/api/v1",
        # The Claude 3.5 Sonnet route is gone from the catalog (404). OpenRouter
        # spells Claude versions with a dot. https://openrouter.ai/api/v1/models
        "anthropic/claude-sonnet-5.5",
        "OPENROUTER_API_KEY",
    ),
    "together": OpenAICompatibleConfig(
        "https://api.together.xyz/v1",
        # Llama 3.3 70B Instruct Turbo leaves serverless inference on
        # 2026-10-22; this is the replacement Together names.
        # https://docs.together.ai/docs/deprecations
        "meta-models/Muse-Glimmer-30B",
        "TOGETHER_API_KEY",
    ),
    "deepseek": OpenAICompatibleConfig(
        "https://api.deepseek.com/v1",
        # The old chat model was retired after 2026-07-24 ("Use deepseek-flash
        # as the model name"). https://api-docs.deepseek.com/
        "deepseek-flash",
        "DEEPSEEK_API_KEY",
    ),
    "fireworks": OpenAICompatibleConfig(
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "FIREWORKS_API_KEY",
    ),
    "perplexity": OpenAICompatibleConfig(
        "https://api.perplexity.ai",
        "sonar",
        "PERPLEXITY_API_KEY",
    ),
    # Tier 3 — Local servers (OpenAI-compatible)
    "ollama": OpenAICompatibleConfig(
        "http://localhost:11434/v1",
        "llama3.2",
        "OLLAMA_BASE_URL",  # Not a key — presence signals Ollama
    ),
    "vllm": OpenAICompatibleConfig(
        "http://localhost:8000/v1",
        "default",
        "VLLM_API_KEY",  # Usually not needed; presence signals vLLM
    ),
}

# Platforms that don't require an API key (local inference)
_LOCAL_PLATFORMS = {"ollama", "huggingface", "llamacpp", "vllm"}

# Tier 1 platform API key environment variables (for lightweight validation)
_TIER1_API_KEY_ENV: Dict[str, Any] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    # A tuple where more than one spelling is in the wild. IBM's own docs and
    # langchain_ibm use WATSONX_APIKEY; checking only WATSONX_API_KEY made the
    # package report watsonx unavailable to a user holding valid credentials.
    "watsonx": ("WATSONX_API_KEY", "WATSONX_APIKEY"),
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}


class SimpleLLMProvider(LLMProvider):
    """LLM provider with comprehensive platform support.

    Tier 1 — Native LangChain integrations (dedicated packages):
      - openai:     OPENAI_API_KEY       (gpt-6-luna, gpt-6-sol, etc.)
      - anthropic:  ANTHROPIC_API_KEY    (claude-sonnet-5-5, claude-haiku-5-5, etc.)
      - azure:      AZURE_OPENAI_API_KEY (Azure OpenAI deployments)
      - google:     GOOGLE_API_KEY       (gemini-3.6-flash, gemini-2.5-pro)
      - google-genai: (alias for google)
      - groq:       GROQ_API_KEY         (llama-3.3-70b, mixtral, etc.)
      - watsonx:    WATSONX_APIKEY or WATSONX_API_KEY (granite, llama,
                    openai/gpt-oss-120b; also needs WATSONX_PROJECT_ID
                    and WATSONX_URL)
      - mistral:    MISTRAL_API_KEY      (mistral-large, codestral, etc.)
      - cohere:     COHERE_API_KEY       (command-a-03-2025, etc.)
      - nvidia:     NVIDIA_API_KEY       (NIM endpoints)

    Tier 2 — OpenAI-compatible (zero extra dependencies):
      - openrouter: OPENROUTER_API_KEY   (any model via OpenRouter)
      - together:   TOGETHER_API_KEY     (Llama, Mixtral, etc.)
      - deepseek:   DEEPSEEK_API_KEY     (deepseek-flash, deepseek-v4-pro)
      - fireworks:  FIREWORKS_API_KEY    (fast inference)
      - perplexity: PERPLEXITY_API_KEY   (sonar, etc.)

    Tier 3 — Local inference (no API key needed):
      - ollama:      OLLAMA_BASE_URL      (local Ollama server)
      - huggingface: HF_MODEL_ID          (in-process HuggingFace Transformers)
      - llamacpp:    LLAMACPP_MODEL_PATH  (in-process GGUF via llama.cpp)
      - vllm:        VLLM_BASE_URL        (local vLLM / OpenAI-compatible server)

    Example::

        llm = SimpleLLMProvider(default_provider="openai")
        model = llm.get_chat_model(
            {"platform": "openai", "model_name": "gpt-6-luna"},
            temperature=0.0,
            max_tokens=1000,
        )
        response = model.invoke("Hello!")

    Local model example::

        llm = SimpleLLMProvider(default_provider="ollama")
        model = llm.get_chat_model(
            {"platform": "ollama", "model_name": "llama3.2"},
            temperature=0.0,
        )
        response = model.invoke("Hello!")
    """

    # Supported platforms and their default models. Each replaced default was
    # retired or closed to new accounts; the comment cites the provider's page
    # (read 2026-10-15). The rest are still served and are unchanged.
    SUPPORTED_PLATFORMS = {
        # Tier 1 — native integrations
        # GPT-4o mini shuts down 2027-01-14 and OpenAI names gpt-6-luna as its
        # replacement. https://developers.openai.com/api/docs/deprecations
        "openai": "gpt-6-luna",
        # Claude Sonnet 4 was retired 2026-06-15; this is its named
        # replacement. https://platform.claude.com/docs/en/about-claude/model-deprecations
        "anthropic": "claude-sonnet-5-5",
        # An Azure deployment name. GPT-4o mini is deprecated on Azure (closed
        # to new customers) and gpt-6-luna is not offered there; gpt-5.6-luna
        # is GA. https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirement-schedule
        "azure": "gpt-5.6-luna",
        # Gemini 2.0 Flash was shut down 2026-06-01; this is its named replacement.
        # https://ai.google.dev/gemini-api/docs/deprecations
        "google": "gemini-3.6-flash",
        "google-genai": "gemini-3.6-flash",
        # Llama 3.3 70B Versatile was shut down 2026-08-16 for free and
        # developer tiers; this is its named replacement. https://console.groq.com/docs/deprecations
        "groq": "openai/gpt-oss-120b",
        "watsonx": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "mistral": "mistral-large-latest",
        # Command R+ is deprecated, "no longer available to new customers";
        # this is the replacement Cohere recommends. https://docs.cohere.com/docs/deprecations
        "cohere": "command-a-03-2025",
        "nvidia": "meta/llama-3.1-70b-instruct",
        # Tier 2 — OpenAI-compatible (sources beside _OPENAI_COMPATIBLE)
        "openrouter": "anthropic/claude-sonnet-5.5",
        "together": "meta-models/Muse-Glimmer-30B",
        "deepseek": "deepseek-flash",
        "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "perplexity": "sonar",
        # Tier 3 — Local inference (no API key needed)
        "ollama": "llama3.2",
        "huggingface": "microsoft/Phi-3-mini-4k-instruct",
        "llamacpp": "",  # requires model_path in settings
        "vllm": "default",  # model name depends on what's loaded in server
    }

    def __init__(self, default_provider: str = "openai"):
        """Initialize LLM provider.

        Args:
            default_provider: Default platform to use when not specified
                in model_settings.
        """
        if default_provider not in self.SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported default provider: {default_provider}. "
                f"Supported: {list(self.SUPPORTED_PLATFORMS.keys())}"
            )
        self._default_provider = default_provider

    @property
    def default_provider(self) -> str:
        """The default platform used when model_settings omits 'platform'."""
        return self._default_provider

    def get_chat_model(
        self,
        model_settings: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "BaseChatModel":
        """Get a chat model instance, counting whatever it goes on to spend.

        Args:
            model_settings: Model configuration dict with 'platform' and
                'model_name' keys. Additional keys are passed to the
                provider (e.g. 'deployment_name' for Azure).
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            LangChain chat model instance

        Every chat model in the package is built here, which makes this the one
        place where token accounting can be complete. Instrumenting call sites
        instead leaves each new call uncounted until somebody remembers, and an
        undercount flatters whichever technique spends the most -- exactly the
        comparison this package exists to get right. The meter is inert unless
        a :class:`~lasagna.utils.token_usage.UsageMeter` is active.

        The experiment's response cache is attached here for the same reason.
        It serves a call only inside an experiment, and only when the model
        decodes greedily: temperature 0 was asked for, the model accepts it
        (see :mod:`lasagna.providers.model_capabilities`), and the client kept
        it. See :mod:`lasagna.utils.llm_cache`.
        """
        from lasagna.utils.llm_cache import attach_llm_cache
        from lasagna.utils.token_usage import attach_usage_meter

        from .model_capabilities import sampling_kwargs

        platform, model_name = self._resolve_model(model_settings)
        model = attach_usage_meter(
            self._build_chat_model(model_settings, temperature=temperature, max_tokens=max_tokens)
        )
        _, pinned = sampling_kwargs(
            model_name, temperature, platform, model_settings.get("reasoning_effort")
        )
        # langchain-openai drops a temperature it believes the model rejects
        # (gpt-5-mini) without a word, so read back what the client kept. A
        # model with no such attribute sends what it was built with.
        kept = getattr(model, "temperature", temperature)
        deterministic = pinned and temperature == 0 and kept == 0
        return attach_llm_cache(model, f"{platform}:{model_name}", deterministic)

    def _resolve_model(self, model_settings: Dict[str, Any]) -> "tuple[str, str]":
        """The platform and model name ``model_settings`` resolves to."""
        platform = model_settings.get("platform", self._default_provider)
        # Normalize platform aliases
        if platform == "google-genai":
            platform = "google"
        # Fall back to MODEL_NAME env var before using hardcoded platform defaults
        from lasagna.utils.model_utils import get_default_model_name

        default_model = get_default_model_name(self.SUPPORTED_PLATFORMS.get(platform, "gpt-6-luna"))
        return platform, model_settings.get("model_name", default_model)

    def _build_chat_model(
        self,
        model_settings: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "BaseChatModel":
        """Dispatch to the per-platform constructor. See ``get_chat_model``."""
        from .model_capabilities import sampling_kwargs

        platform, model_name = self._resolve_model(model_settings)

        # Sampling as the model takes it: temperature as asked, with
        # reasoning_effort="none" where only that admits it, or not at all
        # (Claude 4.7 and later answer 400 to any temperature but the default).
        # `reasoning_effort` in model_settings is the user's and wins.
        kwargs, _ = sampling_kwargs(
            model_name, temperature, platform, model_settings.get("reasoning_effort")
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # ---- Tier 3: Local inference ----
        if platform == "ollama":
            return self._create_ollama(model_name, model_settings, kwargs)

        if platform == "huggingface":
            return self._create_huggingface(model_name, model_settings, kwargs)

        if platform == "llamacpp":
            return self._create_llamacpp(model_name, model_settings, kwargs)

        # ---- Tier 2: OpenAI-compatible platforms (includes vllm) ----
        if platform in _OPENAI_COMPATIBLE:
            return self._create_openai_compatible(platform, model_name, model_settings, kwargs)

        # ---- Tier 1: Native integrations ----
        if platform == "openai":
            return self._create_openai(model_name, model_settings, kwargs)

        if platform == "anthropic":
            return self._create_anthropic(model_name, kwargs)

        if platform == "azure":
            return self._create_azure(model_name, model_settings, kwargs)

        if platform == "google":
            return self._create_google(model_name, kwargs)

        if platform == "groq":
            return self._create_groq(model_name, kwargs)

        if platform == "watsonx":
            return self._create_watsonx(model_name, model_settings, kwargs)

        if platform == "mistral":
            return self._create_mistral(model_name, kwargs)

        if platform == "cohere":
            return self._create_cohere(model_name, kwargs)

        if platform == "nvidia":
            return self._create_nvidia(model_name, kwargs)

        raise ValueError(
            f"Unsupported platform: {platform}. Supported: {list(self.SUPPORTED_PLATFORMS.keys())}"
        )

    # -- Tier 1: Native integrations -----------------------------------------

    @staticmethod
    def _create_openai(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI

        params = {"model": model_name, **kwargs}
        # Support custom base_url (e.g. for proxies or vLLM)
        base_url = model_settings.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            params["openai_api_base"] = base_url
        apikey_name = model_settings.get("apikey_name")
        if apikey_name:
            params["openai_api_key"] = os.environ.get(apikey_name)
        # Add request timeout to prevent hanging on proxy issues
        if "request_timeout" not in params and "timeout" not in params:
            params["request_timeout"] = 60
        return ChatOpenAI(**params)

    @staticmethod
    def _create_anthropic(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, **kwargs)

    @staticmethod
    def _create_azure(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        from langchain_openai import AzureChatOpenAI

        deployment = model_settings.get("deployment_name", model_name)
        azure_endpoint = model_settings.get("azure_endpoint")
        api_version = model_settings.get("api_version", "2024-08-06")
        if hasattr(api_version, "isoformat"):
            api_version = api_version.isoformat()
        return AzureChatOpenAI(
            deployment_name=deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )

    @staticmethod
    def _create_google(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ValueError(
                "Google GenAI support requires langchain-google-genai. "
                "Install with: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(model=model_name, **kwargs)

    @staticmethod
    def _create_groq(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ValueError(
                "Groq support requires langchain-groq. Install with: pip install langchain-groq"
            )
        return ChatGroq(model=model_name, **kwargs)

    @staticmethod
    def _create_watsonx(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        try:
            from langchain_ibm import ChatWatsonx
        except ImportError:
            raise ValueError(
                "WatsonX support requires langchain-ibm. Install with: pip install langchain-ibm"
            )
        project_id = model_settings.get("project_id") or os.environ.get("WATSONX_PROJECT_ID")
        if not project_id:
            raise ValueError("WatsonX requires WATSONX_PROJECT_ID environment variable")
        return ChatWatsonx(model_id=model_name, project_id=project_id, **kwargs)

    @staticmethod
    def _create_mistral(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise ValueError(
                "Mistral support requires langchain-mistralai. "
                "Install with: pip install langchain-mistralai"
            )
        return ChatMistralAI(model=model_name, **kwargs)

    @staticmethod
    def _create_cohere(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        try:
            from langchain_cohere import ChatCohere
        except ImportError:
            raise ValueError(
                "Cohere support requires langchain-cohere. "
                "Install with: pip install langchain-cohere"
            )
        return ChatCohere(model=model_name, **kwargs)

    @staticmethod
    def _create_nvidia(model_name: str, kwargs: Dict[str, Any]) -> "BaseChatModel":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise ValueError(
                "NVIDIA support requires langchain-nvidia-ai-endpoints. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            )
        return ChatNVIDIA(model=model_name, **kwargs)

    # -- Tier 3: Local inference -----------------------------------------------

    @staticmethod
    def _create_ollama(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        """Create an Ollama model, preferring native ChatOllama over OpenAI-compat."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fall back to OpenAI-compatible path (zero extra deps)
            return SimpleLLMProvider._create_openai_compatible(
                "ollama", model_name, model_settings, kwargs
            )

        base_url = model_settings.get("base_url") or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        # ChatOllama uses num_predict (not max_tokens) for output length
        ollama_kwargs = dict(kwargs)
        max_tokens = ollama_kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            ollama_kwargs["num_predict"] = max_tokens
        return ChatOllama(model=model_name, base_url=base_url, **ollama_kwargs)

    @staticmethod
    def _create_huggingface(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        """Create a HuggingFace Transformers model (in-process inference)."""
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

        # Extract generation params from kwargs (don't pass to ChatHuggingFace)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)

        # Build pipeline generation kwargs
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            **user_pipeline_kwargs,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task=model_settings.get("task", "text-generation"),
            device_map=device_map,
            pipeline_kwargs=gen_kwargs,
        )
        return ChatHuggingFace(llm=llm)

    @staticmethod
    def _create_llamacpp(
        model_name: str, model_settings: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> "BaseChatModel":
        """Create a llama.cpp model from a local GGUF file."""
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
                "or LLAMACPP_MODEL_PATH environment variable pointing to a .gguf file"
            )

        return ChatLlamaCpp(
            model_path=model_path,
            n_ctx=model_settings.get("n_ctx", 4096),
            n_gpu_layers=model_settings.get("n_gpu_layers", -1),
            n_batch=model_settings.get("n_batch", 512),
            **kwargs,
        )

    # -- Tier 2: OpenAI-compatible -------------------------------------------

    @staticmethod
    def _create_openai_compatible(
        platform: str,
        model_name: str,
        model_settings: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> "BaseChatModel":
        """Create a model using an OpenAI-compatible API endpoint."""
        from langchain_openai import ChatOpenAI

        base_url, _default_model, api_key_env = _OPENAI_COMPATIBLE[platform]

        # Allow base_url override from settings or env
        actual_base_url = model_settings.get("base_url") or base_url
        if platform == "ollama":
            actual_base_url = os.environ.get("OLLAMA_BASE_URL", actual_base_url)
            if not actual_base_url.endswith("/v1"):
                actual_base_url = actual_base_url.rstrip("/") + "/v1"
        elif platform == "vllm":
            actual_base_url = os.environ.get("VLLM_BASE_URL", actual_base_url)
            if not actual_base_url.endswith("/v1"):
                actual_base_url = actual_base_url.rstrip("/") + "/v1"

        # Resolve API key — local platforms don't require a real key
        api_key = model_settings.get("api_key") or os.environ.get(api_key_env, "")
        if platform in _LOCAL_PLATFORMS:
            api_key = api_key or "EMPTY"

        params: Dict[str, Any] = {
            "model": model_name,
            "openai_api_key": api_key,
            "openai_api_base": actual_base_url,
            **kwargs,
        }

        # OpenRouter custom headers
        if platform == "openrouter":
            headers: Dict[str, str] = {}
            site_url = model_settings.get("site_url") or os.environ.get("OPENROUTER_SITE_URL")
            app_name = model_settings.get("app_name") or os.environ.get("OPENROUTER_APP_NAME")
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name
            if headers:
                params["default_headers"] = headers

        return ChatOpenAI(**params)

    # -- Convenience ---------------------------------------------------------

    def get_llm(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> "BaseChatModel":
        """Get a chat model using the default provider and optional model name.

        Convenience wrapper around get_chat_model() for callers that don't
        need to specify a full model_settings dict.
        """
        settings: Dict[str, Any] = {"platform": self._default_provider}
        if model_name:
            settings["model_name"] = model_name
        else:
            # Respect MODEL_NAME env var (or host application config) before falling
            # back to the hardcoded platform default (e.g. "gpt-6-luna").
            from lasagna.utils.model_utils import get_default_model_name

            settings["model_name"] = get_default_model_name(
                self.SUPPORTED_PLATFORMS[self._default_provider]
            )
        return self.get_chat_model(settings, temperature=temperature, max_tokens=max_tokens)

    # -- Validation ----------------------------------------------------------

    def validate_model_settings(self, model_settings: Dict[str, Any]) -> bool:
        """Validate that model settings are usable.

        Checks platform support and API key presence. Does NOT instantiate
        the LLM client (avoids network calls and heavy imports).

        Args:
            model_settings: Settings to validate

        Returns:
            True if platform is supported and API key is available
        """
        try:
            platform = model_settings.get("platform", self._default_provider)
            if platform not in self.SUPPORTED_PLATFORMS:
                return False

            # Normalize alias before validation
            if platform == "google-genai":
                platform = "google"

            # Tier 3: Local platforms — no API key required
            if platform in _LOCAL_PLATFORMS:
                if platform == "llamacpp":
                    # llamacpp needs a model_path
                    return bool(
                        model_settings.get("model_path") or os.environ.get("LLAMACPP_MODEL_PATH")
                    )
                return True

            # Tier 2: OpenAI-compatible platforms
            if platform in _OPENAI_COMPATIBLE:
                cfg = _OPENAI_COMPATIBLE[platform]
                return bool(os.environ.get(cfg.api_key_env_var))

            # Tier 1: Native integrations
            if platform in _TIER1_API_KEY_ENV:
                names = _TIER1_API_KEY_ENV[platform]
                if isinstance(names, str):
                    names = (names,)
                return any(os.environ.get(n) for n in names)

            # Unknown platform that passed SUPPORTED_PLATFORMS check
            return True
        except Exception as e:
            logger.debug(f"Model validation failed for {model_settings}: {e}")
            return False
