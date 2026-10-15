"""What a sweep will cost before it runs, and a dollar estimate after.

A run records tokens, which are observed (:mod:`lasagna.utils.token_usage`).
Dollars are not: prices change and differ by tier, region and commitment, so a
rate written into a results file is indistinguishable from a measurement to a
reader a year later. Dollars are therefore opt-in (``prices:`` in the sweep),
always named ``cost_usd_estimate``, and the run metadata names the table they
came from and the rates it used.

:func:`dry_run_lines` is ``lasagna benchmark experiment --dry-run``: the arms
and tasks, the models the run needs and whether each is downloaded already, a
static token estimate, and (unless ``--no-probe``) a three-sample probe per
benchmark extrapolated to the whole run. Every number it prints is labelled an
estimate, and it writes nothing under the results folder.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .experiment import Experiment
    from .runner import BenchmarkTask, TaskResult

#: The built-in table's name, recorded with every estimate made from it.
BUILTIN_TABLE = "builtin-2026-10"

#: USD per million tokens (input, output), standard tier, as each provider
#: listed them on 2026-10-15. Prices change: check them, and put the rates you
#: pay under ``prices:`` in the sweep, which wins over this table. Models
#: priced by prompt length are left out, since one rate would be wrong.
_BUILTIN_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


@dataclass(frozen=True)
class Price:
    input_per_1m: float
    output_per_1m: float

    def of(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1e6


@dataclass(frozen=True)
class PriceTable:
    """The built-in table with the sweep's ``prices:`` on top of it."""

    name: str = BUILTIN_TABLE
    overrides: Mapping[str, Price] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any) -> Optional["PriceTable"]:
        """Read ``prices:``: absent (no dollar estimate), ``builtin``, or a
        mapping ``{model: {input_per_1m: ..., output_per_1m: ...}}``."""
        if value is None or isinstance(value, cls):
            return value
        if value == "builtin":
            return cls()
        if not isinstance(value, dict):
            raise ValueError(
                f"`prices:` must be `builtin` or a mapping of model to "
                f"{{input_per_1m: ..., output_per_1m: ...}}, got {value!r}"
            )
        overrides: Dict[str, Price] = {}
        for model, rates in value.items():
            keys = set(rates) if isinstance(rates, dict) else set()
            if keys != {"input_per_1m", "output_per_1m"}:
                raise ValueError(
                    f"`prices.{model}` needs exactly input_per_1m and output_per_1m "
                    f"(USD per million tokens), got {rates!r}"
                )
            for key in ("input_per_1m", "output_per_1m"):
                rate = rates[key]
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
                    raise ValueError(
                        f"`prices.{model}.{key}` must be a number of dollars, got {rate!r}"
                    )
            overrides[str(model)] = Price(
                float(rates["input_per_1m"]), float(rates["output_per_1m"])
            )
        return cls(overrides=overrides)

    def price(self, model: Optional[str]) -> Optional[Price]:
        """The rate for ``model``: the sweep's first, then the built-in table.
        A routed name (``openai/gpt-4o-mini``) is also tried without its prefix."""
        if not model:
            return None
        for name in dict.fromkeys((model, model.rsplit("/", 1)[-1])):
            if name in self.overrides:
                return self.overrides[name]
            if name in _BUILTIN_PRICES:
                return Price(*_BUILTIN_PRICES[name])
        return None

    def record(self, models: List[Optional[str]]) -> Dict[str, Any]:
        """What the run metadata says: the table, and the rates it used."""
        used = {m: self.price(m) for m in dict.fromkeys(m for m in models if m)}
        return {
            "price_table": self.name,
            "overrides": sorted(self.overrides),
            "is_estimate": True,
            "prices": {
                m: {"input_per_1m": p.input_per_1m, "output_per_1m": p.output_per_1m}
                for m, p in used.items()
                if p is not None
            },
        }


def estimate_cost(
    result: "TaskResult",
    table: PriceTable,
    generator_model: Optional[str],
    judge_model: Optional[str],
) -> Optional[float]:
    """Dollars for one task, from its recorded tokens; None when any spend is
    unpriced.

    System tokens are priced at the generator model's rate and the judge's at
    the judge's. Agentic steps or an LLM reranker that pin another model are
    priced at the generator's rate too, so name that model's rate under
    ``prices:`` if it differs. A partial sum would read as the whole bill, so
    a task with any spend the table cannot price gets no estimate.
    """
    if not result.tokens_measured:
        return None
    total = 0.0
    for tokens_in, tokens_out, model in (
        (result.system_input_tokens, result.system_output_tokens, generator_model),
        (result.eval_input_tokens, result.eval_output_tokens, judge_model),
    ):
        if not (tokens_in or tokens_out):
            continue
        price = table.price(model)
        if price is None:
            return None
        total += price.of(tokens_in, tokens_out)
    return total


def generator_model_name() -> Optional[str]:
    """The model the harness generates answers with, without building it."""
    from .experiment import _resolve_generator_model

    named = _resolve_generator_model()
    if named:
        return named
    try:
        from lasagna.providers import get_llm_provider
        from lasagna.providers.llm_provider import SimpleLLMProvider
        from lasagna.utils.model_utils import get_default_model_name

        platform = getattr(get_llm_provider(), "default_provider", None)
        default = SimpleLLMProvider.SUPPORTED_PLATFORMS.get(platform or "", "")
        return get_default_model_name(default) or None
    except Exception:  # noqa: BLE001 - no provider configured
        return None


def _platform() -> Optional[str]:
    try:
        from lasagna.providers import get_llm_provider

        return getattr(get_llm_provider(), "default_provider", None)
    except Exception:  # noqa: BLE001 - no provider configured
        return None


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

#: Samples per benchmark the dry-run probe runs, on the first arm.
PROBE_SAMPLES = 3


def _hf_cached(name: str) -> Optional[bool]:
    """Whether a Hugging Face model is downloaded; None when that cannot be told."""
    if os.path.isdir(os.path.expanduser(name)):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    for repo in [name] if "/" in name else [name, f"sentence-transformers/{name}"]:
        try:
            if isinstance(try_to_load_from_cache(repo, "config.json"), str):
                return True
        except Exception:  # noqa: BLE001 - an invalid repo id is "not cached"
            continue
    return False


def _arm_models(task: "BenchmarkTask") -> List[Tuple[str, str, Optional[bool]]]:
    """``(role, model, cached)`` for the local models one arm loads."""
    from lasagna.core.config import rag_config
    from lasagna.core.session_config._profiles import resolve_ingestion_defaults

    overrides = dict(task.config_overrides or {})
    settings = {**resolve_ingestion_defaults(task.profile or "standard"), **overrides}

    def get(key: str) -> Any:
        value = settings.get(key)
        return value if value is not None else getattr(rag_config, key, None)

    out: List[Tuple[str, str, Optional[bool]]] = []

    def model(role: str, name: Any) -> None:
        if name:
            out.append((role, str(name), _hf_cached(str(name))))

    def component(role: str, name: str) -> None:
        out.append((role, f"component {name}", None))

    if overrides.get("embedder"):
        component("embedder", overrides["embedder"])
    else:
        model("embedder", get("embedding_model"))
    if overrides.get("retriever") == "splade":
        model("retriever", get("splade_model"))
    reranker = overrides.get("reranker")
    if reranker or get("enable_reranking"):
        if reranker in (None, "cross_encoder", "cross_encoder_adapted"):
            model("reranker", get("reranker_model"))
        elif reranker == "monot5":
            model("reranker", get("monot5_model"))
        else:
            component("reranker", reranker)
    return out


def _token_counter() -> Tuple[Any, str]:
    """The token estimator, and its name for the printout."""
    try:
        from lasagna.utils import tokenization

        name = "tiktoken cl100k_base" if tokenization._get_tokenizer() else "characters / 4"
        return tokenization.estimate_tokens, name
    except Exception:  # noqa: BLE001 - fall back to the heuristic
        return (lambda text: len(text or "") // 4), "characters / 4"


def _probe(exp: "Experiment", first: "BenchmarkTask") -> Tuple["TaskResult", float]:
    """Run one arm on a few samples of one benchmark, in a folder of its own."""
    from .experiment import BenchmarkSpec, Experiment, ExperimentConfig, PipelineSweepSpec

    cfg = exp.config
    folder = tempfile.mkdtemp(prefix="lasagna-dry-run-")
    n = min(PROBE_SAMPLES, first.n_samples or PROBE_SAMPLES)
    probe = ExperimentConfig(
        name=f"{cfg.name} (dry-run probe)",
        benchmarks=[
            BenchmarkSpec(
                name=first.benchmark_name, split=first.split, n_samples=n, filters=first.filters
            )
        ],
        pipeline=PipelineSweepSpec(
            base_profile=first.profile,
            config_overrides=[{"name": first.config_name, **first.config_overrides}],
        ),
        mode=cfg.mode,
        n_runs=1,
        variance_source=cfg.variance_source,
        seed=cfg.seed,
        max_concurrency=1,
        checkpoint_dir=folder,
        output_dir=folder,
        output_formats=["json"],
        judge_model=cfg.judge_model,
        ingestion_per_config=cfg.ingestion_per_config,
        max_corpus_docs=cfg.max_corpus_docs,
        tracks=list(cfg.tracks),
        llm_cache=cfg.llm_cache,
        prices=cfg.prices,
        base_dir=cfg.base_dir,
    )
    start = time.monotonic()
    try:
        results = asyncio.run(Experiment(probe, eval_fn=exp._eval_fn).run())
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return results._results[0], time.monotonic() - start


def _count(n: int, noun: str, plural: str = "") -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {plural or noun + 's'}"


def dry_run_lines(exp: "Experiment", probe: bool = True) -> List[str]:
    """The printout of ``lasagna benchmark experiment --dry-run``.

    Raises what ``generate_tasks`` raises for a sweep that cannot run, so the
    dry run refuses exactly what the run would.
    """
    from .benchmarks import benchmark_registry
    from .experiment import _parse_tracks, resolve_mode

    cfg = exp.config
    tasks = exp.generate_tasks()  # names local-file benchmarks, so read them after
    arms = list(dict.fromkeys(t.config_name for t in tasks))
    benches = list(dict.fromkeys(t.benchmark_name for t in tasks))
    generation = "generation" in _parse_tracks(cfg.tracks)
    lines = [
        f'Dry run of "{cfg.name}": an estimate; nothing is written under the results folder.',
        f"Arms ({len(arms)}): {', '.join(arms)}",
        f"Tasks ({len(tasks)}): {_count(len(benches), 'benchmark')} x "
        f"{_count(len(arms), 'arm')} x {_count(cfg.n_runs, 'run')}",
    ]

    # Each benchmark's samples, loaded the way the runner loads them.
    samples_by_bench: Dict[str, List[Any]] = {}
    for bench in benches:
        first = next(t for t in tasks if t.benchmark_name == bench)
        try:
            samples = benchmark_registry.get(bench).load(
                split=first.split,
                n_samples=first.n_samples,
                seed=first.seed,
                filters=first.filters or None,
            )
            samples_by_bench[bench] = list(samples)
            lines.append(f"  {bench}  split={first.split}  {_count(len(samples), 'sample')}")
        except Exception as exc:  # noqa: BLE001 - say why, keep going
            lines.append(f"  {bench}  split={first.split}  samples not loaded: {exc}")
    queries = {
        b: len(samples_by_bench.get(b, [])) * sum(1 for t in tasks if t.benchmark_name == b)
        for b in benches
    }

    # Models.
    lines.append("Models:")
    seen: set = set()
    entries: List[Tuple[str, str, Optional[bool]]] = []
    for task in tasks:
        # generation_only scores gold context and never builds the pipeline.
        if resolve_mode(task.mode, task.benchmark_name)[0] == "live":
            entries += _arm_models(task)
    if generation:
        from .answer_metrics import similarity_model_name

        sim = similarity_model_name()
        if str(sim).lower() not in ("none", "off", "false", "0"):
            entries.append(("answer similarity", sim, _hf_cached(sim)))
    for role, name, cached in entries:
        if (role, name) in seen:
            continue
        seen.add((role, name))
        state = {True: "downloaded", False: "not downloaded yet"}.get(cached, "")
        lines.append(f"  {role:18s} {name}  {state}".rstrip())
    gen_model = generator_model_name() if generation else None
    judge = os.environ.get("RAG_EVAL_JUDGE_MODEL") or cfg.judge_model or None
    if generation:
        platform = _platform()
        lines.append(
            f"  {'generator':18s} {gen_model} ({platform})"
            if gen_model and platform
            else f"  {'generator':18s} none configured: generation will fail. Set an API key "
            f"(lasagna doctor lists them) or RAG_EVAL_GENERATOR_MODEL."
        )
        lines.append(
            f"  {'judge':18s} {judge}"
            if judge
            else f"  {'judge':18s} RAGAS default (set judge_model: or RAG_EVAL_JUDGE_MODEL)"
        )

    # Static estimate.
    count, counter = _token_counter()
    if generation and samples_by_bench:
        from lasagna.generation.strategies import _concise_prompt

        overhead = count(_concise_prompt("", [], ""))
        static = 0
        for bench, samples in samples_by_bench.items():
            per_task = sum(
                overhead + count(s.question or "") + count(getattr(s, "context", "") or "")
                for s in samples
            )
            static += per_task * sum(1 for t in tasks if t.benchmark_name == bench)
        lines.append(
            f"Tokens (static estimate, {counter}): about {static:,} input tokens to the "
            f"generator, counting each question with its gold passage in the default prompt. "
            f"Retrieved passages, answers and the judge are not counted before a run."
        )
    elif not generation:
        lines.append(
            "Tokens: no generation track, so no answer is generated or judged; an arm "
            "still calls an LLM if its pipeline does (HyDE, an agentic pipeline, an LLM "
            "reranker)."
        )

    if not probe:
        lines.append("Probe skipped (--no-probe).")
        return lines

    # Probe, extrapolated.
    table = cfg.prices or PriceTable()
    total_cost = 0.0
    priced = True
    for bench in benches:
        first = next(t for t in tasks if t.benchmark_name == bench and t.run_index == 0)
        result, wall = _probe(exp, first)
        if result.error:
            lines.append(f"Probe of {bench} on arm {first.config_name!r} failed: {result.error}")
            priced = False
            continue
        n = max(result.n_samples, 1)
        hits = int(result.metrics.get("llm_cache_hits", 0))
        lines.append(
            f"Probe of {bench} (estimate): arm {first.config_name!r} on "
            f"{_count(result.n_samples, 'sample')} took {wall:.1f} s including setup "
            f"(loading models, ingesting), {result.avg_latency_ms / 1000:.2f} s per query, "
            f"{_count(result.total_llm_calls, 'LLM call')}"
            + (f" ({hits} replayed from the cache, so the time is a lower bound)" if hits else "")
        )
        total_q = queries.get(bench, 0)
        sys_in = result.system_input_tokens / n * total_q
        sys_out = result.system_output_tokens / n * total_q
        ev_in = result.eval_input_tokens / n * total_q
        ev_out = result.eval_output_tokens / n * total_q
        seconds = result.avg_latency_ms / 1000 * total_q
        lines.append(
            f"  Extrapolated to {_count(total_q, 'query', 'queries')} (estimate, every arm "
            f"assumed to spend what this one did): about {seconds:,.0f} s of query time plus "
            f"setup, {sys_in + sys_out:,.0f} system tokens and {ev_in + ev_out:,.0f} judge tokens"
        )
        for tokens_in, tokens_out, model in ((sys_in, sys_out, gen_model), (ev_in, ev_out, judge)):
            if not (tokens_in or tokens_out):
                continue
            price = table.price(model)
            if price is None:
                priced = False
                lines.append(f"  No price for {model or 'the RAGAS default judge'} in {table.name}")
            else:
                total_cost += price.of(tokens_in, tokens_out)
    if priced and total_cost:
        dollars = f"{total_cost:,.2f}" if total_cost >= 0.01 else f"{total_cost:.4f}"
        lines.append(
            f"Cost (estimate, price table {table.name}"
            + (", with the sweep's prices" if table.overrides else "")
            + f"): about ${dollars}. Prices change; put your rates under prices:."
        )
    return lines


def with_cost_estimate(eval_fn: Any, table: PriceTable) -> Any:
    """Add ``cost_usd_estimate`` to each task whose spend the table prices."""

    async def run(task: "BenchmarkTask", samples: List[Any]) -> "TaskResult":
        result = await eval_fn(task, samples)
        cost = estimate_cost(result, table, generator_model_name(), judge_model_name())
        if cost is not None:
            result.metrics["cost_usd_estimate"] = cost
        return result

    return run


def judge_model_name() -> Optional[str]:
    """The judge model the run resolved (``judge_model:`` or ``RAG_EVAL_JUDGE_MODEL``)."""
    from . import experiment

    return experiment._active_judge_model or os.environ.get("RAG_EVAL_JUDGE_MODEL") or None
