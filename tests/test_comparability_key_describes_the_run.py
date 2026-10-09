"""The comparability key described the machine, not the run.

`_system_under_test` sourced five of the fourteen comparability-key fields from
the process-global ``rag_config``. No run uses that object: every override path
in the harness is session-scoped (``sp.set_config_overrides``) and
``from_profile`` stores the profile on the pipeline instance. The global is
never mutated.

So a ``speed``-profile run — ``all-MiniLM-L6-v2`` at 384 dimensions — published
``held_fixed.embedder_model = BAAI/bge-m3, embedder_dim = 1024``, whatever
happened to be in the operator's settings file. Two runs on *different embedding
models* hashed to the same key, passed ``lasagna results validate``, and were
ranked in one leaderboard table.

The comment six lines above the call site states the stakes exactly: *"two runs
on different embedding models hash to the same key and get ranked in the same
leaderboard table. A key that cannot tell those apart is worse than no key,
because it looks like it can."*

Verified before the fix::

    >>> sp.set_config_overrides("probe", {"embedding_model": "all-MiniLM-L6-v2",
    ...                                   "embedding_dim": 384})
    >>> _system_under_test()["embedder_model"]
    'BAAI/bge-m3'
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import HELD_FIXED_FIELDS
from lasagna.knowledge_evaluation_sdk.experiment import _system_under_test


def _profiles():
    from lasagna.core.session_config import get_knowledge_config_manager

    return {p.name: p.config for p in get_knowledge_config_manager()._default_profiles}


class TestThePremise:
    """These tests are only meaningful if the built-in profiles really differ.
    If they ever stop differing, this file proves nothing and should fail
    loudly rather than pass vacuously."""

    def test_speed_and_standard_pin_different_embedders(self):
        profiles = _profiles()
        speed, standard = profiles["speed"], profiles["standard"]
        assert speed["embedding_model"] != standard["embedding_model"]
        assert speed["embedding_dim"] != standard["embedding_dim"]

    def test_the_embedder_is_in_the_key(self):
        assert "embedder_model" in HELD_FIXED_FIELDS
        assert "embedder_dim" in HELD_FIXED_FIELDS


class TestTheBlockDescribesTheProfileTheRunNamed:
    def test_speed_reports_its_own_embedder(self):
        sut = _system_under_test("speed")
        assert sut["embedder_model"] == "all-MiniLM-L6-v2"
        assert sut["embedder_dim"] == 384

    def test_standard_reports_its_own_embedder(self):
        sut = _system_under_test("standard")
        assert sut["embedder_model"] == "BAAI/bge-m3"
        assert sut["embedder_dim"] == 1024

    def test_two_profiles_do_not_look_identical(self):
        """The defect, stated as the property that matters. Both reported
        bge-m3/1024 before the fix, so the key could not separate them."""
        a = _system_under_test("speed")
        b = _system_under_test("standard")
        key_a = tuple(a.get(f) for f in HELD_FIXED_FIELDS if f in a)
        key_b = tuple(b.get(f) for f in HELD_FIXED_FIELDS if f in b)
        assert key_a != key_b

    def test_a_profile_config_is_a_dict_not_an_object(self):
        """The trap that made the first fix look right while changing nothing.

        `RAGProfile.config` is a plain dict, so `getattr(config, "embedding_model")`
        returns None for every profile and the code falls through to the global
        it was supposed to stop reading.
        """
        assert isinstance(_profiles()["speed"], dict)


class TestTheFallbackIsAVisibleGapNotAConfidentGuess:
    def test_an_unresolvable_profile_reports_nothing(self):
        """A gap a reader can see is honest. The machine's defaults presented as
        the run's configuration are not — that is the original bug."""
        sut = _system_under_test("no-such-profile-exists")
        assert sut["embedder_model"] is None
        assert sut["embedder_dim"] is None

    def test_no_profile_named_still_reports_the_global(self):
        """Nothing was claimed, so the process config is the honest answer."""
        from lasagna.core.config import rag_config

        assert _system_under_test(None)["embedder_model"] == rag_config.embedding_model

    def test_the_bundle_still_builds_with_gaps(self):
        """Provenance must never be the reason a run fails to produce output."""
        sut = _system_under_test("no-such-profile-exists")
        assert isinstance(sut, dict)
        assert sut  # populated, just with Nones in the unresolvable fields


class TestItIsWiredToTheRunsProfile:
    def test_the_call_site_passes_the_configured_profile(self):
        """A correct helper nothing calls with the right argument is the shape
        of fix this project keeps mistaking for a fix."""
        import inspect

        from lasagna.knowledge_evaluation_sdk import experiment

        source = inspect.getsource(experiment)
        assert "_system_under_test(cfg.pipeline.base_profile)" in source

    def test_it_uses_the_same_lookup_as_from_profile(self):
        """`list_profiles()` returns {name, description, builtin} metadata with
        no config attached, so reading `.config` off one yields None for every
        profile — silently reproducing the original bug.

        Parsed, not grepped. The first version of this asserted
        ``"list_profiles()" not in source`` and failed against correct code,
        because the docstring explains why that method is wrong. Substring
        matching over source has now produced a false failure three times in
        this project; the call graph is the claim, so read the call graph.
        """
        import ast
        import inspect
        import textwrap

        from lasagna.knowledge_evaluation_sdk import experiment

        tree = ast.parse(textwrap.dedent(inspect.getsource(experiment._resolve_profile_config)))
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_default_profiles" in attrs
        assert "list_profiles" not in called

    @pytest.mark.parametrize("field", ["embedder_model", "embedder_dim", "chunker"])
    def test_the_key_fields_it_feeds_are_populated(self, field):
        assert _system_under_test("standard").get(field) is not None
