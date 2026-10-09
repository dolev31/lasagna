"""Stage isolation, asserted by behaviour rather than by shape.

`test_abc_conformance.py` checks that components subclass the right base and
implement the right methods. That is necessary and not sufficient: it says
nothing about whether a component *stays inside its stage* at runtime.

The property that matters is the one the framework's value rests on — toggling
one component must not change what another sees. If a reranker annotates the
chunks it was handed, that annotation reaches the citation manager, the answer
generator and the evaluation harness, and any measured difference gets
attributed to whichever stage reads it next. An interaction effect measured
under those conditions is partly an artefact.

Nothing enforces this today: the inter-stage dataclasses (`Chunk`,
`ScoredDocument`, `ProcessedDocument`) are mutable, and no layer copies between
stages. So the contract is "components must not mutate their inputs", and this
module is what makes that contract real for anything registered.
"""

import copy
import dataclasses

import pytest

from lasagna.core.base_components import Chunk, ScoredDocument
from lasagna.core.registry import registry


#: Long enough that every chunker really splits it. The previous fixture was
#: "One. Two. Three. Four. Five." -- one chunk for all of them, so the
#: two-chunk guard skipped and `TestChunksAreIndependent` asserted nothing.
_ALIASING_FIXTURE_TEXT = (
    "Retrieval-augmented generation joins a retriever to a generator. "
    "The retriever selects passages; the generator conditions on them.\n\n"
    "Chunking decides what a passage is. Split too finely and a claim loses "
    "the sentence that supports it; split too coarsely and the embedding "
    "averages several topics into one vector.\n\n"
    "Reranking reorders candidates with a model that sees the query and the "
    "passage together, which a bi-encoder never does.\n\n"
    "Evaluation ties these together. A change to any one stage moves the "
    "final metric, so attributing the movement requires holding the rest "
    "fixed and testing whether the difference exceeds sampling noise.\n\n"
) * 3


def _snapshot(objs):
    """Deep, comparable copy of a stage's input."""
    return copy.deepcopy(objs)


def _describe_mutation(before, after) -> str:
    """Human-readable account of what a component changed in place."""
    diffs = []
    for i, (b, a) in enumerate(zip(before, after)):
        for f in dataclasses.fields(b):
            bv, av = getattr(b, f.name), getattr(a, f.name)
            if bv != av:
                diffs.append(f"item[{i}].{f.name}: {bv!r} -> {av!r}")
    return "; ".join(diffs)


@pytest.fixture
def documents():
    return [
        ScoredDocument(
            content=f"document number {i}",
            score=0.5,
            document_id=f"d{i}",
            chunk_id=f"c{i}",
            metadata={"origin": "fixture"},
        )
        for i in range(4)
    ]


class TestTheContractIsEnforceable:
    """Demonstrate the leak the contract exists to prevent.

    These do not test product code — they pin the reason the rest of this
    module exists, so that a future reader does not mistake the contract for
    pedantry.
    """

    def test_in_place_mutation_reaches_the_caller(self, documents):
        from lasagna.core.base_components import BaseReranker

        class Annotating(BaseReranker):
            """The ordinary accident: annotate in place, return the same list."""

            def rerank(self, query, docs, top_k=None):
                for d in docs:
                    d.metadata["seen_by"] = "reranker"
                return docs

        Annotating().rerank("q", documents)
        assert documents[0].metadata.get("seen_by") == "reranker", (
            "expected the demonstration to leak; if this fails the data "
            "contracts became immutable and this module can be simplified"
        )

    def test_inter_stage_contracts_are_still_mutable(self):
        """If these ever become frozen, the contract is enforced by the type.

        Kept as a tripwire: the day someone adds frozen=True, this fails and
        whoever did it can delete the runtime checks below.
        """
        chunk = Chunk(content="x")
        chunk.metadata["mutated"] = True  # would raise on a frozen dataclass
        assert chunk.metadata["mutated"] is True


#: Rerankers that claim `BaseReranker`, and therefore must accept the contract's
#: argument shape. The five duck-typed built-ins report
#: `implements_contract: False` and take a different shape by design -- calling
#: them through the ABC raises TypeError, which is not a defect in them.
#: `test_every_stage_has_conformance.py` is what holds that flag honest.
_CONFORMING_RERANKERS = sorted(
    e["name"] for e in registry.list_rerankers() if e.get("implements_contract")
)


class TestRegisteredRerankersDoNotMutateTheirInput:
    """Every contract-implementing reranker must leave the caller's documents alone."""

    @pytest.mark.parametrize("name", _CONFORMING_RERANKERS)
    def test_reranker_leaves_input_unmutated(self, name, documents):
        cls = registry.get_reranker(name)
        if cls is None:
            pytest.skip(f"reranker '{name}' did not resolve")

        try:
            instance = cls()
        except Exception as exc:  # noqa: BLE001 - construction may need a model
            pytest.skip(f"reranker '{name}' needs configuration to construct: {exc}")

        before = _snapshot(documents)
        try:
            instance.rerank("a query", documents, top_k=2)
        except (TypeError, AttributeError, NotImplementedError) as exc:
            # NOT a skip. These say the component does not implement the
            # interface it registered under -- a real defect, and exactly what
            # this suite exists to catch. Reporting them as "environment not
            # available" is how a broken component passes conformance.
            pytest.fail(f"reranker '{name}' does not honour BaseReranker.rerank: {exc!r}")
        except Exception as exc:  # noqa: BLE001 - genuinely needs a model/network
            pytest.skip(f"reranker '{name}' could not run offline: {exc}")

        mutated = _describe_mutation(before, documents)
        assert not mutated, (
            f"reranker '{name}' mutated the documents it was given: {mutated}. "
            f"Downstream stages and the caller share these objects, so the "
            f"change is attributed to whichever stage reads it next."
        )


class TestRegisteredChunkersDoNotMutateTheirInput:
    @pytest.mark.parametrize("name", sorted(registry.list("chunker").get("chunker", [])))
    def test_chunker_does_not_mutate_source_text(self, name):
        cls = registry.get_chunker(name)
        if cls is None:
            pytest.skip(f"chunker '{name}' did not resolve")
        try:
            instance = cls()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chunker '{name}' needs configuration: {exc}")

        text = "First sentence. Second sentence. Third sentence."
        metadata = {"source": "fixture"}
        metadata_before = dict(metadata)

        try:
            instance.chunk(text, metadata=metadata)
        except TypeError:
            try:
                instance.chunk(text)
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"chunker '{name}' could not run: {exc}")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chunker '{name}' could not run: {exc}")

        assert metadata == metadata_before, (
            f"chunker '{name}' mutated the metadata dict it was passed: "
            f"{metadata_before} -> {metadata}"
        )


class TestChunksAreIndependent:
    """A chunker must not hand out aliases of one shared object."""

    @pytest.mark.parametrize("name", sorted(registry.list("chunker").get("chunker", [])))
    def test_chunks_do_not_share_metadata_dicts(self, name):
        cls = registry.get_chunker(name)
        if cls is None:
            pytest.skip(f"chunker '{name}' did not resolve")
        try:
            # Long enough to actually split. "One. Two. Three. Four. Five."
            # produced a single chunk for every chunker, so the guard below
            # skipped and this test asserted nothing for any of them.
            chunks = cls().chunk(_ALIASING_FIXTURE_TEXT)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chunker '{name}' could not run: {exc}")

        assert chunks, f"chunker '{name}' produced no chunks at all"
        if len(chunks) < 2:
            # A chunker that will not split ~2 KB of multi-paragraph prose is
            # not exercising the property under test, and silently passing said
            # so to nobody.
            pytest.fail(
                f"chunker '{name}' produced {len(chunks)} chunk(s) from "
                f"{len(_ALIASING_FIXTURE_TEXT)} characters; aliasing cannot be checked"
            )

        metas = [getattr(c, "metadata", None) for c in chunks]
        metas = [m for m in metas if isinstance(m, dict)]
        if len(metas) < 2:
            pytest.skip("chunks carry no metadata dicts")

        ids = {id(m) for m in metas}
        assert len(ids) == len(metas), (
            f"chunker '{name}' gave several chunks the *same* metadata dict. "
            f"Annotating one chunk downstream would silently annotate them all."
        )


# ---------------------------------------------------------------------------
# Registry-wide contract conformance
# ---------------------------------------------------------------------------

# Components registered under a stage whose ABC they do not implement.
#
# The framework's premise is that a component "drops into one stage and is
# measured against the same surrounding stack". That holds only if everything
# registered under a stage speaks that stage's contract. These do not: they are
# the older dict-based implementations, kept registered alongside the *_adapted
# wrappers that do honour the ABC.
#
# The practical consequence is that swapping `cross_encoder` for
# `cross_encoder_adapted` is not a like-for-like comparison -- one takes
# ScoredDocument and returns ScoredDocument, the other takes dicts and returns
# RerankedResult. A researcher writing to the documented BaseReranker interface
# is interchangeable with the adapters but not with these.
#
# Recorded rather than fixed: re-typing them is a behavioural change that needs
# its own pass. The point of the allowlist is that it cannot grow silently --
# anything newly registered must conform.
KNOWN_NON_CONFORMING = {
    ("reranker", "cross_encoder"),
    ("reranker", "combined"),
    ("reranker", "ensemble"),
    ("reranker", "colbert"),
    ("reranker", "llm_listwise"),
    ("query_enhancer", "hyde"),
    ("query_enhancer", "query_expansion"),
    ("query_enhancer", "query_decomposition"),
    ("query_enhancer", "combined"),
}


def _stage_abcs():
    from lasagna.core import base_components as bc

    return {
        "reranker": (bc.BaseReranker, registry.get_reranker),
        "chunker": (bc.BaseChunker, registry.get_chunker),
        "embedder": (bc.BaseEmbedder, registry.get_embedder),
        "query_enhancer": (bc.BaseQueryEnhancer, registry.get_enhancer),
    }


class TestRegistryHonoursItsOwnContracts:
    def test_no_new_non_conforming_registrations(self):
        """Anything registered under a stage must implement that stage's ABC.

        The pre-existing exceptions are allowlisted above. This test exists so
        the list can only shrink: a new component that skips the contract fails
        here rather than being discovered by whoever tries to swap it in.
        """
        offenders = set()
        for ctype, (base, getter) in _stage_abcs().items():
            for name in registry.list(ctype).get(ctype, []):
                cls = getter(name)
                if cls is not None and not issubclass(cls, base):
                    offenders.add((ctype, name))

        new = offenders - KNOWN_NON_CONFORMING
        assert not new, (
            f"these are registered under a stage whose contract they do not "
            f"implement: {sorted(new)}. Either implement the stage's ABC, or "
            f"register an adapter that does."
        )

    def test_allowlist_does_not_outlive_the_problem(self):
        """Fail once an allowlisted component is fixed, so the list gets pruned.

        A stale allowlist is how known debt turns into invisible debt.
        """
        still_bad = set()
        for ctype, (base, getter) in _stage_abcs().items():
            for name in registry.list(ctype).get(ctype, []):
                cls = getter(name)
                if cls is not None and not issubclass(cls, base):
                    still_bad.add((ctype, name))

        fixed = KNOWN_NON_CONFORMING - still_bad
        assert not fixed, (
            f"these now conform and should be removed from KNOWN_NON_CONFORMING: {sorted(fixed)}"
        )


class TestInterchangeabilityIsVisible:
    """A researcher must be able to tell which components are comparable.

    Comparing components is the reason this package exists, and a comparison
    only means something between components speaking the same contract. Since
    nine built-ins do not (see KNOWN_NON_CONFORMING), the difference has to be
    discoverable — otherwise it is found from a TypeError three stages later,
    or worse, not found at all and reported as a result.
    """

    def test_listing_reports_contract_status(self):
        entries = registry.list_rerankers()
        assert entries, "expected built-in rerankers to be registered"
        assert all("implements_contract" in e for e in entries), (
            "every reranker listing must say whether it implements the contract"
        )
        assert any(e["implements_contract"] for e in entries), "expected some to conform"
        assert any(not e["implements_contract"] for e in entries), (
            "expected the known non-conforming built-ins to be reported as such"
        )

    def test_selecting_a_non_interchangeable_component_warns(self):
        from loguru import logger
        from lasagna.core.pipeline import KnowledgePipeline

        seen = []
        sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            KnowledgePipeline().replace("reranker", "cross_encoder")
        finally:
            logger.remove(sink)

        joined = "".join(seen)
        assert "not interchangeable" in joined, (
            f"choosing a non-conforming component for a comparison must say so; got: {joined!r}"
        )
        assert "cross_encoder_adapted" in joined, (
            "the warning must name a comparable alternative, not just complain"
        )

    def test_selecting_a_conforming_component_is_quiet(self):
        from loguru import logger
        from lasagna.core.pipeline import KnowledgePipeline

        seen = []
        sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            KnowledgePipeline().replace("reranker", "cross_encoder_adapted")
        finally:
            logger.remove(sink)

        assert not any("not interchangeable" in s for s in seen), (
            f"a contract-compliant component must not be warned about: {seen}"
        )

    def test_a_user_component_is_reported_as_interchangeable(self):
        """The whole point: your component should register as comparable."""
        from lasagna.core.base_components import BaseReranker

        @registry.register("reranker", "_audit_probe_reranker")
        class ProbeReranker(BaseReranker):
            def rerank(self, query, documents, top_k=None):
                return list(documents)[: top_k or len(documents)]

        try:
            entry = next(
                e for e in registry.list_rerankers() if e["name"] == "_audit_probe_reranker"
            )
            assert entry["implements_contract"] is True
        finally:
            registry._components["reranker"].pop("_audit_probe_reranker", None)
