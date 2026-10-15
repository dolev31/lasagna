"""Count the tokens a run actually spent, and refuse to price them.

Every cost number this package reported was zero. `input_tokens` and
`output_tokens` were read off ``chunk.metadata`` in the experiment harness, and
nothing anywhere ever wrote those keys, so ``estimated_cost_usd``,
``system_cost_usd`` and ``eval_cost_usd`` summed zeros, every report printed
``$0.00000``, and ``quality_per_dollar`` divided by zero and came out ``inf``.
Meanwhile the ``AIMessage`` carrying the real usage was reduced to ``.content``
on the line after it arrived.

**Counted at the provider, not at the call sites.** Every chat model in the
package comes from ``SimpleLLMProvider.get_chat_model``, so attaching a callback
there counts the agentic graph, HyDE, query enhancement, generation and the
LangChain judge alike. Instrumenting call sites instead means the number is
wrong the moment somebody adds a call, and wrong in the direction that flatters
the result — a technique that quietly spends more looks free.

**Tokens, never dollars.** A price table in a published bundle is a fabricated
number: prices change, they differ by tier, region and commitment, and a reader
in a year cannot tell a stale rate from a real measurement. Tokens are what was
actually observed; anyone who wants a cost multiplies by the rate they actually
pay. This is why ``_get_cost_rates`` and ``_DEFAULT_COST_TABLE`` are gone rather
than fixed.

**Zero and unmeasured are different.** ``llm_calls == 0`` means no call was seen,
which for a retrieval-only arm is the truth and for an uninstrumented client is
a silent undercount. Consumers should read :meth:`TokenUsage.measured` rather
than treating a zero as a measurement.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from loguru import logger

#: The two things worth separating. A researcher comparing techniques cares
#: what the *system* spent; what the judge spent is the price of measuring, and
#: adding them together makes an expensive judge look like an expensive
#: pipeline.
SYSTEM = "system"
EVAL = "eval"


@dataclass
class TokenUsage:
    """Tokens and calls observed in one scope."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def measured(self) -> bool:
        """Whether anything was actually observed.

        An all-zero record is ambiguous — no calls were made, or the client
        making them was not instrumented — and the two must not be reported the
        same way.
        """
        return self.llm_calls > 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.llm_calls += other.llm_calls

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}input_tokens": float(self.input_tokens),
            f"{prefix}output_tokens": float(self.output_tokens),
            f"{prefix}llm_calls": float(self.llm_calls),
        }


def _first_int(mapping: Any, *keys: str) -> Optional[int]:
    """First present, integer-coercible value among ``keys``."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_usage(response: Any) -> TokenUsage:
    """Pull token counts off whatever an LLM call returned.

    Providers disagree on where usage lives and on what to call it, and a
    wrong guess here shows up as a confident zero rather than an error. The
    orderings below are widest-first: ``usage_metadata`` is the normalised
    LangChain field, then the provider-native blobs, then the aggregate
    ``LLMResult`` shape.

    A response with no recognisable usage returns a record with
    ``llm_calls == 0``, so it reads as unmeasured rather than as free.
    """
    # LangChain >=0.2 normalises onto the message itself.
    meta = getattr(response, "usage_metadata", None)
    tokens_in = _first_int(meta, "input_tokens", "prompt_tokens")
    tokens_out = _first_int(meta, "output_tokens", "completion_tokens")

    if tokens_in is None and tokens_out is None:
        for holder in ("response_metadata", "additional_kwargs", "llm_output"):
            blob = getattr(response, holder, None)
            if not isinstance(blob, dict):
                continue
            usage = blob.get("token_usage") or blob.get("usage") or {}
            tokens_in = _first_int(usage, "prompt_tokens", "input_tokens")
            tokens_out = _first_int(usage, "completion_tokens", "output_tokens")
            if tokens_in is not None or tokens_out is not None:
                break

    # OpenAI SDK responses carry a `usage` object, not a dict.
    if tokens_in is None and tokens_out is None:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None and not isinstance(usage_obj, dict):
            tokens_in = getattr(usage_obj, "prompt_tokens", None) or getattr(
                usage_obj, "input_tokens", None
            )
            tokens_out = getattr(usage_obj, "completion_tokens", None) or getattr(
                usage_obj, "output_tokens", None
            )
        elif isinstance(usage_obj, dict):
            tokens_in = _first_int(usage_obj, "prompt_tokens", "input_tokens")
            tokens_out = _first_int(usage_obj, "completion_tokens", "output_tokens")

    if tokens_in is None and tokens_out is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=int(tokens_in or 0),
        output_tokens=int(tokens_out or 0),
        llm_calls=1,
    )


class UsageMeter:
    """Accumulates token usage across everything running inside a ``with``.

    Held in a :mod:`contextvars` variable rather than passed down, because the
    calls being counted happen inside a graph several layers below the code
    that wants the total, and threading a counter through every signature would
    be a change to every signature.

    Nothing is counted when no meter is active, which is the normal case for
    library users. The cost of the instrumentation is then one attribute read
    per LLM call.
    """

    def __init__(self) -> None:
        self._by_scope: Dict[str, TokenUsage] = {}
        self._token: Any = None
        self._lock = threading.Lock()

    # -- accumulation -------------------------------------------------------

    def record(self, usage: TokenUsage, scope: Optional[str] = None) -> None:
        """Add one observation. Safe to call from several threads.

        The harness runs RAGAS in worker threads via ``asyncio.to_thread`` and
        can have more than one in flight, all reporting into the same meter.
        ``+=`` on an attribute is a read and a write, so without the lock two
        threads finishing together can drop one of the counts -- an undercount
        that looks like a plausible number.
        """
        if not usage.measured:
            return
        name = scope or _active_scope.get()
        with self._lock:
            self._by_scope.setdefault(name, TokenUsage()).add(usage)

    def usage(self, scope: str) -> TokenUsage:
        return self._by_scope.get(scope, TokenUsage())

    def total(self) -> TokenUsage:
        combined = TokenUsage()
        for usage in self._by_scope.values():
            combined.add(usage)
        return combined

    def snapshot(self) -> Dict[str, TokenUsage]:
        with self._lock:
            return {
                name: TokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    llm_calls=usage.llm_calls,
                )
                for name, usage in self._by_scope.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._by_scope.clear()

    # -- activation ---------------------------------------------------------

    def __enter__(self) -> "UsageMeter":
        self._token = _active_meter.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _active_meter.reset(self._token)
            self._token = None

    @contextmanager
    def scope(self, name: str) -> Iterator["UsageMeter"]:
        """Attribute everything inside to ``name`` (``SYSTEM`` or ``EVAL``)."""
        token = _active_scope.set(name)
        try:
            yield self
        finally:
            _active_scope.reset(token)


_active_meter: contextvars.ContextVar[Optional[UsageMeter]] = contextvars.ContextVar(
    "lasagna_usage_meter", default=None
)
_active_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lasagna_usage_scope", default=SYSTEM
)


def active_meter() -> Optional[UsageMeter]:
    return _active_meter.get()


def record_usage(response: Any, scope: Optional[str] = None) -> TokenUsage:
    """Count one response against the active meter, if there is one."""
    usage = extract_usage(response)
    meter = _active_meter.get()
    if meter is not None:
        meter.record(usage, scope)
    return usage


@contextmanager
def usage_scope(name: str) -> Iterator[None]:
    """Attribute usage to ``name`` whether or not a meter is active."""
    token = _active_scope.set(name)
    try:
        yield
    finally:
        _active_scope.reset(token)


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def usage_callback_handler() -> Any:
    """A LangChain callback that reports every completion to the active meter.

    Built lazily so importing this module does not import langchain_core, which
    the offline paths do without ever making a call.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageCallbackHandler(BaseCallbackHandler):
        #: Callbacks must never break the call they observe. A provider that
        #: reports usage in a shape not handled here should cost the run a
        #: missing number, not a failed benchmark.
        raise_error = False

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            _count_llm_call("ok")
            meter = _active_meter.get()
            if meter is None:
                return
            usage = extract_usage(response)
            if not usage.measured:
                # LLMResult keeps usage on the generated message when the
                # provider does not fill llm_output.
                for batch in getattr(response, "generations", None) or []:
                    for generation in batch or []:
                        message = getattr(generation, "message", None)
                        if message is not None:
                            usage = extract_usage(message)
                            if usage.measured:
                                break
                    if usage.measured:
                        break
            meter.record(usage)

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            _count_llm_call("error")

    return _UsageCallbackHandler()


def _count_llm_call(outcome: str) -> None:
    """``lasagna_llm_calls_total{outcome}`` on a server's /metrics; a no-op otherwise."""
    from lasagna.utils.monitoring import METRICS

    METRICS.counter("lasagna_llm_calls", 1, tags={"outcome": outcome})


def attach_usage_meter(model: Any) -> Any:
    """Add usage counting to a chat model, in place, and return it.

    Called from the one provider that constructs every chat model in the
    package. Failure is non-fatal: a model that cannot carry callbacks is worth
    less accounting, not a crash at construction time.
    """
    try:
        existing = list(getattr(model, "callbacks", None) or [])
        if any(type(h).__name__ == "_UsageCallbackHandler" for h in existing):
            return model
        model.callbacks = existing + [usage_callback_handler()]
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.debug(f"Could not attach the usage meter to {type(model).__name__}: {exc}")
    return model


# ---------------------------------------------------------------------------
# OpenAI SDK (RAGAS builds its judge on a raw client, not on LangChain)
# ---------------------------------------------------------------------------


#: Marks a client whose completions are already counted, so instrumenting the
#: same client twice does not double every judge call.
_COUNTED_MARKER = "_lasagna_usage_counted"


def instrument_openai_client(client: Any, scope: str = EVAL) -> Any:
    """Count an OpenAI client's completions, and return *that same client*.

    RAGAS builds its judge with ``llm_factory(model, client=OpenAI(...))``, so
    the judge never passes through the LangChain provider and its tokens — the
    entire cost of *measuring* — went uncounted.

    **The client is patched in place rather than wrapped**, and that is the
    whole design. The first version of this returned a transparent proxy object
    with ``__getattr__`` delegation, which reads as the safer choice and was
    not: RAGAS hands the client to ``instructor.from_openai``, which dispatches
    on ``isinstance(client, OpenAI)``. A proxy is not an ``OpenAI``, so the
    dispatch fell through and returned ``None``, ``InstructorLLM.client`` became
    ``None``, and every judge call died with ``AttributeError: 'NoneType' object
    has no attribute 'chat'`` before reaching the network — inside an
    ``except Exception`` that logged and moved on, so every RAGAS metric
    silently vanished from the results. Instrumentation that disables the thing
    it measures is worse than no instrumentation.

    Patching the bound method keeps the object a real ``OpenAI``, so every
    ``isinstance`` check downstream still passes.

    Returns the client untouched if it does not look like one, or if patching
    fails: an uncounted judge is a missing number, a broken judge is a missing
    evaluation.
    """
    try:
        completions = client.chat.completions
    except Exception as exc:  # noqa: BLE001 - see docstring: never break the judge
        # Deliberately broad. Narrowing this to AttributeError let any other
        # error from a client's `chat` property escape and take the judge down,
        # which contradicts the promise above -- a missing number must never
        # become a missing evaluation.
        logger.debug(f"Could not reach chat.completions to instrument it: {exc}")
        return client

    # `getattr(..., default)` is truthy on any object with a permissive
    # `__getattr__` (a MagicMock, most hand-rolled fakes), which would skip
    # instrumentation silently and report zero tokens -- indistinguishable from
    # "no calls were made". Check the instance dictionary instead.
    if _COUNTED_MARKER in getattr(completions, "__dict__", {}):
        return client

    def _wrap(method_name: str) -> None:
        original = getattr(completions, method_name, None)
        if original is None:
            return

        @functools.wraps(original)
        def counted(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            record_usage(response, scope)
            return response

        setattr(completions, method_name, counted)

    try:
        # `parse` is the structured-output path; instructor may use either.
        for name in ("create", "parse"):
            _wrap(name)
        setattr(completions, _COUNTED_MARKER, True)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.debug(f"Could not instrument the judge client: {exc}")

    return client
