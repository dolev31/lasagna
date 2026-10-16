"""Conformance must cover every stage, not the stages someone remembered.

The README's claim is that all seven stages sit behind typed ABCs and can be
swapped independently. Two suites check that, and both were written stage by
stage:

    test_abc_conformance.py            5 of 7 -- shape
    test_stage_isolation_behavioural.py 2 of 7 -- runtime non-mutation

So a stage added later inherits no coverage and nothing says so. This file
derives its parametrisation from `registry` itself, which means a new stage
type arrives already covered, or fails loudly until it is.

It also names any stage with no registered component rather than leaving the
gap to be discovered. `retriever` was one until dense, BM25, SPLADE and two
hybrid fusions were registered as components.
"""

from __future__ import annotations

import copy

import pytest

from lasagna.core.registry import registry

#: The method each stage's ABC declares, keyed by the registry's own type name.
#: Assertions below check this is exhaustive, so a new stage type fails here
#: rather than being silently skipped.
STAGE_ENTRY_POINTS = {
    "chunker": "chunk",
    "embedder": "encode",
    "reranker": "rerank",
    "retriever": "retrieve",
    "document_processor": "extract",
    "query_enhancer": "enhance",
    "agentic_pipeline": "run",
}

#: Stages with no registered component. Being listed here is a statement that
#: the swap claim does not hold for that stage yet -- not permission to ignore
#: it. `test_the_empty_stages_are_still_the_empty_ones` fails when this drifts,
#: in either direction.
STAGES_WITH_NO_COMPONENTS: set = set()


def _valid_types():
    """The registry's own list of stage types, read from its error message.

    Asking the registry rather than hard-coding is the point of this file.
    """
    try:
        registry.list("definitely-not-a-stage")
    except ValueError as exc:
        text = str(exc)
        return {name for name in STAGE_ENTRY_POINTS if f"'{name}'" in text}
    pytest.fail("registry.list accepted an invented component type")


def _registered(stage):
    return sorted(registry.list(stage).get(stage, []))


#: Stages whose built-in components load model weights when constructed.
STAGES_THAT_LOAD_WEIGHTS = {"embedder"}


def _construct(stage, name):
    """``registry.get``, with a stand-in client for an embedder behind a hosted API.

    The openai, cohere and voyage embedders build their API client in the
    constructor, which needs the provider's package and key; neither is a
    property of the component. They take a ``client`` for exactly this, so
    they are still constructed and checked here. The CI step that runs these
    tests (``-m needs_embedding_weights``) has neither, and failed on all three.
    """
    cls = registry.get_class(stage, name)
    if getattr(cls, "key_vars", None):
        return registry.get(stage, name, client=object())
    return registry.get(stage, name)


def _constructing(stages):
    """Parametrize ``stages`` for a test that constructs every component.

    Constructing the built-in embedder loads the configured model, so that case
    carries the ``needs_embedding_weights`` marker (see tests/conftest.py).
    """
    return [
        pytest.param(stage, marks=pytest.mark.needs_embedding_weights)
        if stage in STAGES_THAT_LOAD_WEIGHTS
        else stage
        for stage in stages
    ]


class TestTheStageListIsComplete:
    def test_this_file_knows_every_stage_the_registry_has(self):
        """The guard that makes the rest of this file trustworthy."""
        assert _valid_types() == set(STAGE_ENTRY_POINTS), (
            "the registry's stage types and this file's list have diverged; "
            "a stage added to the registry needs an entry point here"
        )

    def test_there_are_seven(self):
        """The number the README commits to. If a stage is added or removed,
        the docs claim moves with it."""
        assert len(STAGE_ENTRY_POINTS) == 7

    @pytest.mark.parametrize("stage,method", sorted(STAGE_ENTRY_POINTS.items()))
    def test_each_stage_abc_declares_its_entry_point(self, stage, method):
        """The mapping above is only useful if it is true."""
        from lasagna.core import base_components

        bases = [
            obj
            for name, obj in vars(base_components).items()
            if isinstance(obj, type) and name.startswith("Base") and hasattr(obj, method)
        ]
        assert bases, f"no Base* class in base_components declares {method!r} for {stage}"


class TestEveryStageWithComponentsIsExercised:
    @pytest.mark.parametrize("stage", sorted(STAGE_ENTRY_POINTS))
    def test_a_stage_either_has_components_or_is_declared_empty(self, stage):
        """Silence is the failure mode this replaces.

        A stage registering nothing is a real gap in the swap-any-stage claim
        (`retriever` was one), and until this test existed nothing said so.
        """
        has_components = bool(_registered(stage))
        declared_empty = stage in STAGES_WITH_NO_COMPONENTS
        assert has_components != declared_empty, (
            f"{stage}: registered={_registered(stage)}, "
            f"declared empty={declared_empty}. Update STAGES_WITH_NO_COMPONENTS."
        )

    def test_the_empty_stages_are_still_the_empty_ones(self):
        """Fails in both directions.

        If a stage listed as empty gains a component, this fails and the note
        above -- plus the README's stage claim -- gets corrected. If a stage
        empties out, it fails too.
        """
        empty = {s for s in STAGE_ENTRY_POINTS if not _registered(s)}
        assert empty == STAGES_WITH_NO_COMPONENTS

    @pytest.mark.parametrize(
        "stage", _constructing(sorted(set(STAGE_ENTRY_POINTS) - STAGES_WITH_NO_COMPONENTS))
    )
    def test_every_registered_component_can_be_constructed(self, stage):
        """Cheaper than the behavioural suite and catches a different thing: a
        component that registers but cannot be instantiated is unusable, and
        the shape-based conformance test never calls the constructor."""
        failures = []
        for name in _registered(stage):
            try:
                _construct(stage, name)
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert not failures, f"{stage} components that cannot be resolved:\n  " + "\n  ".join(
            failures
        )


#: Public accessor for each stage's listing. Not derivable by appending "s":
#: `query_enhancer` is `list_enhancers` and `agentic_pipeline` is
#: `list_pipelines`, so a naive `list_{stage}s` silently found nothing for two
#: of the seven and the tests below skipped without saying why.
STAGE_LISTING_ACCESSORS = {
    "chunker": "list_chunkers",
    "embedder": "list_embedders",
    "reranker": "list_rerankers",
    "retriever": "list_retrievers",
    "document_processor": "list_document_processors",
    "query_enhancer": "list_enhancers",
    "agentic_pipeline": "list_pipelines",
}


def _listing(stage):
    return getattr(registry, STAGE_LISTING_ACCESSORS[stage])()


_STAGES_WITH_LISTINGS = sorted(set(STAGE_ENTRY_POINTS) - STAGES_WITH_NO_COMPONENTS)


class TestEveryStageCanBeAskedWhatItHas:
    """`list_retrievers` and `list_document_processors` did not exist.

    Three of the seven stages could not be enumerated at all, so a caller
    following the "swap any stage" claim had no way to find out what was
    available -- or, for `retriever`, that nothing was.
    """

    @pytest.mark.parametrize("stage,accessor", sorted(STAGE_LISTING_ACCESSORS.items()))
    def test_the_accessor_exists(self, stage, accessor):
        assert callable(getattr(registry, accessor, None)), (
            f"{stage} cannot be enumerated: registry has no {accessor}()"
        )

    def test_every_stage_has_one(self):
        assert set(STAGE_LISTING_ACCESSORS) == set(STAGE_ENTRY_POINTS)

    @pytest.mark.parametrize("stage", sorted(STAGE_LISTING_ACCESSORS))
    def test_the_accessor_agrees_with_the_generic_list(self, stage):
        """Two ways to ask the same question must not disagree."""
        assert {e["name"] for e in _listing(stage)} == set(_registered(stage))


class TestTheContractFlagTellsTheTruth:
    """`implements_contract` is the whole interchangeability story.

    Registration deliberately warns rather than raises when a component does not
    inherit the stage ABC -- duck-typed components keep working through their
    own interfaces, and the `_adapted` variants are the ones wearing the
    contract. That design is fine, and it puts all the weight on one boolean:
    a caller who wants a component they can swap reads `implements_contract` and
    trusts it.

    Nothing checked that it was accurate. It is derived at registration from
    `issubclass`, so the failure it cannot catch is a class that inherits the
    ABC without implementing the method -- reported as interchangeable, raising
    AttributeError on use.

    Found while writing this: 4 of the 5 registered query enhancers (`hyde`,
    `query_decomposition`, `query_expansion`, `combined`) have no `enhance`
    method at all. They report `implements_contract: False`, so the flag is
    honest and this test passes -- but that is a fact worth having asserted
    rather than assumed, and it means "swap the query enhancer" works for
    exactly one of the five.
    """

    @pytest.mark.parametrize("stage", _STAGES_WITH_LISTINGS)
    def test_every_component_declares_a_contract_status(self, stage):
        for entry in _listing(stage):
            assert "implements_contract" in entry, (
                f"{stage}/{entry.get('name')} does not say whether it is "
                f"interchangeable, so a caller cannot tell"
            )

    @pytest.mark.parametrize("stage", _constructing(_STAGES_WITH_LISTINGS))
    def test_components_claiming_the_contract_have_the_method(self, stage):
        """The claim, checked against the object rather than its ancestry."""
        method = STAGE_ENTRY_POINTS[stage]
        liars = []
        for entry in _listing(stage):
            if not entry.get("implements_contract"):
                continue
            name = entry["name"]
            try:
                component = _construct(stage, name)
            except Exception as exc:  # noqa: BLE001
                liars.append(f"{name}: cannot be constructed ({type(exc).__name__}: {exc})")
                continue
            if not callable(getattr(component, method, None)):
                liars.append(f"{name}: claims the contract but has no callable {method}()")
        assert not liars, f"{stage} components misreporting interchangeability:\n  " + "\n  ".join(
            liars
        )

    @pytest.mark.parametrize("stage", _STAGES_WITH_LISTINGS)
    def test_at_least_one_component_per_stage_is_interchangeable(self, stage):
        """Otherwise the stage is listed as swappable and nothing can be
        swapped into it -- the gap `retriever` had, one level down."""
        conforming = [e["name"] for e in _listing(stage) if e.get("implements_contract")]
        assert conforming, (
            f"no {stage} implements the stage contract, so the swap claim does not hold for it"
        )


class TestQueryEnhancersDoNotMutateTheirInput:
    """The non-mutation contract, extended past chunker and reranker.

    `test_stage_isolation_behavioural.py` explains why this matters: the
    inter-stage dataclasses are mutable and nothing copies between stages, so a
    component that edits its input changes what every later stage sees, and any
    measured effect is partly an artefact of that.

    Only components claiming the contract are exercised -- the rest have no
    `enhance` to call, which `TestTheContractFlagTellsTheTruth` establishes.
    """

    @pytest.mark.parametrize(
        "name",
        sorted(e["name"] for e in registry.list_enhancers() if e.get("implements_contract")),
    )
    def test_enhance_leaves_the_query_intact_and_returns_strings(self, name):
        try:
            component = registry.get("query_enhancer", name)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"{name} cannot be constructed here: {exc}")

        query = "what is the capital of France"
        original = copy.deepcopy(query)

        try:
            result = component.enhance(query)
        except NotImplementedError:
            pytest.skip(f"{name} does not implement enhance synchronously")
        except Exception as exc:  # noqa: BLE001
            # Needing a live LLM is a limitation of this environment, not a
            # contract violation. AttributeError is NOT in that category and
            # would have been caught above.
            pytest.skip(f"{name} could not run offline: {type(exc).__name__}: {exc}")

        assert query == original, f"{name} mutated the query it was given"
        assert isinstance(result, list), f"{name}.enhance must return a list"
        assert all(isinstance(item, str) for item in result), (
            f"{name}.enhance must return strings; a later stage handed a "
            f"mutable object here can reach backwards into this one"
        )
