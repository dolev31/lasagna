"""A chunking ablation must not silently produce identical arms.

`chunking_strategy` is not the top of the dispatch. `enable_hierarchical` wins,
then `preserve_headers`/`preserve_tables`, and only then the strategy — and all
three default to True. So out of the box, sweeping `fixed` against `semantic`
produced byte-identical chunks, a difference of exactly zero, and the confident
conclusion that chunking does not matter.

That is the failure mode this framework exists to prevent, sitting inside the
framework. It cannot be caught by looking at the output, because the output is
a perfectly plausible number.

Two smaller defects in the same area: `create_chunker` computed a strategy,
logged "Created chunker with strategy: X", and returned a chunker that ignored
it; and `chunking_strategy="recursive"` was accepted, documented as "splits by
separators", and silently ran fixed-size chunking. It was refused until it had
a real implementation; it now runs recursive separator splitting.
"""

from __future__ import annotations

import pytest

from lasagna.processing.chunker.semantic import (
    ChunkingStrategy,
    create_chunker,
    resolve_effective_strategy,
)


class _Config:
    """Only the fields the resolver reads."""

    def __init__(self, strategy="semantic", hierarchical=False, headers=False, tables=False):
        self.chunking_strategy = strategy
        self.enable_hierarchical = hierarchical
        self.preserve_headers = headers
        self.preserve_tables = tables


class TestTheEffectiveStrategyIsReported:
    def test_with_structural_flags_off_the_request_is_honoured(self):
        assert resolve_effective_strategy(_Config(strategy="fixed")) == "fixed"

    def test_hierarchical_overrides_the_request(self):
        assert (
            resolve_effective_strategy(_Config(strategy="fixed", hierarchical=True))
            == ChunkingStrategy.HIERARCHICAL.value
        )

    def test_structure_flags_override_the_request(self):
        assert (
            resolve_effective_strategy(_Config(strategy="fixed", headers=True)) == "structure_aware"
        )
        assert (
            resolve_effective_strategy(_Config(strategy="fixed", tables=True)) == "structure_aware"
        )

    def test_hierarchical_wins_over_structure(self):
        """Documents the precedence rather than leaving it to be rediscovered."""
        cfg = _Config(strategy="fixed", hierarchical=True, headers=True)
        assert resolve_effective_strategy(cfg) == ChunkingStrategy.HIERARCHICAL.value


class TestTheShadowingIsAnnounced:
    """The whole point: an ablation must be able to notice.

    Captured through a loguru sink rather than pytest's `caplog`: this package
    logs with loguru, which does not route through stdlib logging, so `caplog`
    silently sees nothing and a warning-assertion written with it would pass
    whether or not the warning existed.
    """

    @pytest.fixture(autouse=True)
    def _fresh_warning_state(self):
        # The warning is issued once per (requested, effective) pair per process.
        from lasagna.processing.chunker import semantic

        semantic._WARNED_OVERRIDES.clear()
        yield

    @pytest.fixture
    def warnings(self):
        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(captured.append, level="WARNING")
        yield captured
        logger.remove(sink_id)

    def test_a_shadowed_strategy_warns(self, warnings):
        resolve_effective_strategy(_Config(strategy="fixed", hierarchical=True))
        joined = "".join(warnings)
        assert "overridden" in joined, f"no warning when the strategy was discarded: {joined!r}"
        assert "identical chunks" in joined, "the warning should say why it matters"

    def test_the_warning_names_both_strategies(self, warnings):
        resolve_effective_strategy(_Config(strategy="fixed", hierarchical=True))
        joined = "".join(warnings)
        assert "fixed" in joined and "hierarchical" in joined

    def test_an_honoured_strategy_does_not_warn(self, warnings):
        resolve_effective_strategy(_Config(strategy="fixed"))
        assert "overridden" not in "".join(warnings)

    def test_two_strategies_resolve_identically_under_defaults(self):
        """The ablation-is-a-no-op condition, stated as a fact.

        Both arms of a `fixed` vs `semantic` sweep collapse to the same
        effective strategy when the structural flags are on — which is what
        made the comparison meaningless.
        """
        a = resolve_effective_strategy(_Config(strategy="fixed", hierarchical=True))
        b = resolve_effective_strategy(_Config(strategy="semantic", hierarchical=True))
        assert a == b


class TestRecursiveRunsTheRecursiveAlgorithm:
    def test_requesting_recursive_resolves_to_recursive(self):
        assert resolve_effective_strategy(_Config(strategy="recursive")) == "recursive"

    def test_it_is_still_in_the_enum(self):
        assert ChunkingStrategy.RECURSIVE.value == "recursive"

    def test_recursive_and_fixed_produce_different_chunks(self):
        """It used to fall through to fixed-size chunking under its own name."""
        from lasagna.core.config import rag_config
        from lasagna.processing.chunker.semantic import chunk_document

        text = ("A short paragraph about chunking.\n\n" + "word " * 60 + "\n\n") * 4
        flat = {"enable_hierarchical": False, "preserve_headers": False, "preserve_tables": False}

        def run(strategy):
            config = rag_config.with_query_overrides(
                {**flat, "chunking_strategy": strategy, "chunk_size": 120}
            )
            return [c for c, _ in chunk_document(text, "d", "s", "f", config=config)]

        recursive = run("recursive")
        assert recursive != run("fixed")
        assert "A short paragraph about chunking." in recursive


class TestCreateChunkerNoLongerLies:
    def test_the_chunker_carries_its_effective_strategy(self):
        chunker = create_chunker(strategy="fixed", config=_Config(strategy="fixed"))
        assert chunker.effective_strategy == "fixed"

    def test_a_shadowed_request_is_reflected_on_the_instance(self):
        """It used to log the requested strategy and return one ignoring it."""
        cfg = _Config(strategy="fixed", hierarchical=True)
        chunker = create_chunker(strategy="fixed", config=cfg)
        assert chunker.effective_strategy == ChunkingStrategy.HIERARCHICAL.value

    def test_create_chunker_reports_recursive(self):
        chunker = create_chunker(strategy="recursive", config=_Config(strategy="recursive"))
        assert chunker.effective_strategy == "recursive"
