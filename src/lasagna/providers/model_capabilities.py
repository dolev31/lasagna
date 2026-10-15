"""What a model accepts, so each request is built for the model it goes to.

Two provider changes broke the request this package sent on every LLM call,
``temperature=0``:

* **Claude 4.7 and later** answer HTTP 400 when ``temperature``, ``top_p`` or
  ``top_k`` is set to a non-default value, and the Anthropic Python SDK 1.0
  removes the parameters, so passing one raises ``TypeError``.
  https://platform.claude.com/docs/en/about-claude/model-deprecations
* **OpenAI reasoning models** accept ``temperature`` only when
  ``reasoning_effort`` is ``"none"`` ("When reasoning effort is not none,
  remove temperature, top_p, and top_logprobs"), and the default effort of the
  current ones is ``"medium"``.
  https://developers.openai.com/api/docs/guides/latest-model

So a model takes temperature as asked (``ACCEPTED``), takes it once reasoning
is off (``NEEDS_EFFORT_NONE``), or never (``REJECTED``). A model of the last
kind answers with the provider's own sampling. Its run-to-run spread is then a
property of the model rather than of the pipeline, and the harness records
``sampling: "provider-default"`` instead of claiming temperature 0.

An id this table does not know keeps the old behavior: temperature is sent. A
guess that dropped it would unpin a model that honors it, which is the failure
this module exists to prevent, moved somewhere quieter.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: The model takes ``temperature`` (and ``top_p``) as asked.
ACCEPTED = "accepted"
#: An OpenAI reasoning model: temperature only with ``reasoning_effort="none"``.
NEEDS_EFFORT_NONE = "needs_reasoning_effort_none"
#: Temperature cannot be set at all; the provider samples as it likes.
REJECTED = "rejected"

#: What the run metadata says about sampling.
PINNED = "pinned"
PROVIDER_DEFAULT = "provider-default"


@dataclass(frozen=True)
class ModelCapabilities:
    """What one model id accepts."""

    #: ``ACCEPTED``, ``NEEDS_EFFORT_NONE`` or ``REJECTED``.
    temperature: str = ACCEPTED
    #: An OpenAI reasoning model, which takes ``reasoning_effort``.
    reasoning: bool = False
    #: The id names fixed weights without a date in it (see ``bundle.py``).
    pinned_snapshot: bool = False


_UNKNOWN = ModelCapabilities()

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

#: ``claude-<family>-<major>[-<minor>][-<yyyymmdd>]``, the id scheme from
#: Claude 4 on. Claude 3 ids put the version before the family; those models
#: are retired and match nothing here.
_CLAUDE = re.compile(
    r"^claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:-(\d{8}))?$"
)

#: "Returns a 400 error when set to a non-default value on Claude 4.7 and
#: later models and Claude Mythos Preview."
#: https://platform.claude.com/docs/en/about-claude/model-deprecations
_CLAUDE_REJECTS_SAMPLING_FROM = (4, 7)

#: "Every Claude model ID is a pinned snapshot, including the dateless IDs used
#: from the 4.6 generation on." Before 4.6 a dateless id is an alias.
#: https://platform.claude.com/docs/en/about-claude/models/overview
_CLAUDE_DATELESS_PINNED_FROM = (4, 6)


def _claude(model: str) -> Optional[ModelCapabilities]:
    if model == "claude-mythos-preview":
        return ModelCapabilities(temperature=REJECTED)
    match = _CLAUDE.match(model)
    if not match:
        return None
    version = (int(match.group(1)), int(match.group(2) or 0))
    dated = match.group(3) is not None
    return ModelCapabilities(
        temperature=REJECTED if version >= _CLAUDE_REJECTS_SAMPLING_FROM else ACCEPTED,
        pinned_snapshot=dated or version >= _CLAUDE_DATELESS_PINNED_FROM,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

#: Reasoning models whose model page lists ``none`` among the reasoning
#: efforts, read on https://developers.openai.com/api/docs/models/<id>
#: (2026-10-15). With effort ``none`` they take temperature.
_OPENAI_EFFORT_NONE = frozenset(
    {
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.5",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.1",
    }
)

#: Reasoning models whose page lists no ``none`` effort, so temperature can
#: never be sent: gpt-6-astra and gpt-6.1-sol ("The none and minimal reasoning
#: efforts are not supported"), gpt-5 (minimal, low, medium, high).
_OPENAI_NO_EFFORT_NONE = frozenset({"gpt-6-astra", "gpt-6.1-sol", "gpt-5"})

#: o1 and o3 take no temperature; this package skipped it for them before the
#: table existed, and still does.
_OPENAI_LEGACY_REASONING_PREFIXES = ("o1", "o3")

#: Models whose page lists the dateless id itself as the only snapshot ("Use
#: gpt-6-luna in your API requests. - gpt-6-luna"). Elsewhere a dateless OpenAI
#: id is an alias for a dated snapshot.
_OPENAI_DATELESS_SNAPSHOTS = frozenset(
    {"gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol"}
    | {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}
)

_OPENAI_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _openai(model: str) -> Optional[ModelCapabilities]:
    base = _OPENAI_SNAPSHOT_DATE.sub("", model)
    pinned = base != model or base in _OPENAI_DATELESS_SNAPSHOTS
    if base in _OPENAI_EFFORT_NONE:
        return ModelCapabilities(NEEDS_EFFORT_NONE, reasoning=True, pinned_snapshot=pinned)
    if base in _OPENAI_NO_EFFORT_NONE or base.startswith(_OPENAI_LEGACY_REASONING_PREFIXES):
        return ModelCapabilities(REJECTED, reasoning=True, pinned_snapshot=pinned)
    return None


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

#: Platforms that speak OpenAI's request format, so ``reasoning_effort`` can
#: be sent.
_EFFORT_PLATFORMS = frozenset({"openai", "azure", "openrouter"})


def model_capabilities(model: Optional[str], platform: Optional[str] = None) -> ModelCapabilities:
    """What ``model`` accepts on ``platform``.

    The model id decides, not the platform: a Claude id rejects temperature
    whether it is called directly, through OpenRouter or through an
    OpenAI-compatible proxy. A routed name (``anthropic/claude-sonnet-5.5``) is
    read as the model it routes to.

    OpenRouter is the exception for OpenAI's reasoning models: its catalog
    lists no ``temperature`` for any of them, ``reasoning_effort="none"``
    included (``supported_parameters`` at https://openrouter.ai/api/v1/models),
    so there they cannot be pinned.
    """
    if not model:
        return _UNKNOWN
    name = model.strip().rsplit("/", 1)[-1]
    # OpenRouter spells Claude versions with a dot (claude-sonnet-5.5).
    caps = _claude(name.replace(".", "-")) if name.startswith("claude-") else _openai(name)
    if caps is None:
        return _UNKNOWN
    if platform == "openrouter" and caps.temperature == NEEDS_EFFORT_NONE:
        return ModelCapabilities(REJECTED, caps.reasoning, caps.pinned_snapshot)
    return caps


def sampling_kwargs(
    model: Optional[str],
    temperature: Optional[float],
    platform: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """The sampling arguments to send, and whether ``temperature`` is among them.

    * ``ACCEPTED``: ``temperature`` as asked.
    * ``NEEDS_EFFORT_NONE``: ``reasoning_effort="none"`` with the temperature,
      which makes the model answer the way the chat models it replaces did.
      An effort the user set wins: anything but ``"none"`` means no
      temperature can go with it.
    * ``REJECTED``: no temperature at all.

    An effort the user set is passed on to a reasoning model, or to any model
    on a platform that speaks OpenAI's format. The flag is False when no
    temperature is sent, which is what "sampling could not be pinned" means.
    """
    caps = model_capabilities(model, platform)
    effort = reasoning_effort
    if caps.temperature == NEEDS_EFFORT_NONE and effort is None:
        effort = "none"
    out: Dict[str, Any] = {}
    if effort is not None and (caps.reasoning or platform in _EFFORT_PLATFORMS):
        out["reasoning_effort"] = effort
    sends = temperature is not None and (
        caps.temperature == ACCEPTED or (caps.temperature == NEEDS_EFFORT_NONE and effort == "none")
    )
    if sends:
        out["temperature"] = temperature
    return out, sends


def sampling_label(
    model: Optional[str], platform: Optional[str] = None, reasoning_effort: Optional[str] = None
) -> str:
    """``"pinned"`` or ``"provider-default"``, for the run metadata."""
    _, sends = sampling_kwargs(model, 0.0, platform, reasoning_effort)
    return PINNED if sends else PROVIDER_DEFAULT


def unpinned_note(
    model: Optional[str],
    platform: Optional[str] = None,
    what: str = "answers",
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """The one line a user sees when ``model`` cannot be pinned, or None."""
    if not model or sampling_label(model, platform, reasoning_effort) == PINNED:
        return None
    return (
        f"{model} does not accept temperature; {what} can vary between runs; "
        f"repeat runs (n_runs >= 3) to measure that variance"
    )


# ---------------------------------------------------------------------------
# A raw OpenAI client (the RAGAS judge)
# ---------------------------------------------------------------------------

#: Marks a client whose requests are already rewritten.
_PINNED_MARKER = "_lasagna_sampling_pinned"

#: Sampling parameters a model that rejects temperature rejects with it.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "top_logprobs")


def pin_openai_client(client: Any, temperature: float) -> Any:
    """Rewrite each ``chat.completions.create`` call for the model it names.

    RAGAS builds its judge on a raw OpenAI client and decides the sampling
    itself: 0.01 for most models, but 1.0 for any ``gpt-5`` or later, because
    it takes all of those to accept nothing else. On gpt-6-luna that is a
    sampled judge at the default effort. Here the request is rebuilt from the
    table: a model that takes temperature only with effort ``none`` gets
    ``reasoning_effort="none"`` and ``temperature``; a model that rejects it
    loses every sampling parameter; anything else goes out as RAGAS built it.
    An effort the caller set is kept.

    Patched in place, like :func:`~lasagna.utils.llm_cache.cache_openai_client`
    and for the same reason (instructor dispatches on ``isinstance``). Apply
    it last, so the cache keys on, and the meter counts, the request that is
    actually sent. A client pointed at OpenRouter is read as that platform.
    """
    try:
        completions = client.chat.completions
        original = completions.create
    except Exception:  # noqa: BLE001 - not a client this can rewrite
        return client
    if _PINNED_MARKER in getattr(completions, "__dict__", {}):
        return client
    platform = "openrouter" if "openrouter.ai" in str(getattr(client, "base_url", "")) else "openai"

    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        caps = model_capabilities(kwargs.get("model"), platform)
        if caps.temperature == NEEDS_EFFORT_NONE and kwargs.get("reasoning_effort") in (
            None,
            "none",
        ):
            kwargs["reasoning_effort"] = "none"
            kwargs["temperature"] = temperature
        elif caps.temperature != ACCEPTED:
            for key in _SAMPLING_KEYS:
                kwargs.pop(key, None)
        return original(*args, **kwargs)

    try:
        setattr(completions, "create", create)
        setattr(completions, _PINNED_MARKER, True)
    except Exception:  # noqa: BLE001 - never break the judge
        pass
    return client
