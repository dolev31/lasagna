"""Agentic prompts must be overridable, and an override must fail early.

The six prompts driving the agentic graph were module constants with no way to
change them, which closes off a whole category of research: "does a better
hallucination-check prompt reduce false positives" is a legitimate question and
the answer was "fork the package".

It also made agentic comparisons quietly incomplete. Two runs could differ in
the prompt deciding whether an answer was grounded, and nothing recorded it, so
the bundles compared equal while measuring different systems.

Validation happens when the override is registered rather than when the prompt
is used. A prompt is applied with `.format(**kwargs)` deep inside a graph node,
so a missing placeholder surfaces as a `KeyError` minutes into a benchmark from
a stack that never mentions prompts.
"""

from __future__ import annotations

import pytest

from lasagna.agentic.prompts import (
    NODE_PLACEHOLDERS,
    PromptError,
    active_prompt_fingerprint,
    describe_nodes,
    resolve_node_model,
    resolve_prompt,
    validate_prompt,
)


class _Config:
    def __init__(self, prompts=None, models=None):
        self.agentic_prompts = prompts or {}
        self.agentic_models = models or {}


class TestDefaultsAreUnchanged:
    """Adding a knob must not move anything for someone who does not turn it."""

    def test_no_config_returns_the_default(self):
        assert resolve_prompt("generate", "DEFAULT") == "DEFAULT"

    def test_empty_overrides_return_the_default(self):
        assert resolve_prompt("generate", "DEFAULT", _Config()) == "DEFAULT"

    def test_an_unrelated_override_returns_the_default(self):
        cfg = _Config(
            {"rewrite_query": "Use {original_question} {current_query} {results_summary}"}
        )
        assert resolve_prompt("generate", "DEFAULT", cfg) == "DEFAULT"

    def test_a_missing_key_falls_back(self):
        assert resolve_prompt("generate", "DEFAULT", _Config({})) == "DEFAULT"

    def test_a_whitespace_override_raises_rather_than_falling_back(self):
        """Deliberately not a silent fallback.

        `generate: "   "` in a config file is a mistake. Quietly using the
        shipped default would run a different prompt than its author believes
        and record no sign of it — the exact failure this package exists to
        prevent. An explicit key means an explicit intent, so a broken one is
        an error rather than a shrug.
        """
        with pytest.raises(PromptError, match="empty"):
            resolve_prompt("generate", "DEFAULT", _Config({"generate": "   "}))


class TestOverridesApply:
    def test_a_valid_override_is_returned(self):
        cfg = _Config({"generate": "Q: {question}\nC: {context}"})
        assert resolve_prompt("generate", "DEFAULT", cfg) == "Q: {question}\nC: {context}"

    @pytest.mark.parametrize("node", sorted(NODE_PLACEHOLDERS))
    def test_every_node_is_overridable(self, node):
        """A registry that covers five of six nodes is a trap for the sixth."""
        template = " ".join("{" + f + "}" for f in sorted(NODE_PLACEHOLDERS[node]))
        assert resolve_prompt(node, "DEFAULT", _Config({node: template})) == template


class TestValidationFailsEarlyAndClearly:
    def test_an_unknown_placeholder_is_rejected(self):
        with pytest.raises(PromptError, match="documents"):
            validate_prompt("generate", "Use {question} and {documents}")

    def test_the_error_lists_what_is_available(self):
        with pytest.raises(PromptError) as exc:
            validate_prompt("generate", "{nonsense}")
        assert "context" in str(exc.value) and "question" in str(exc.value)

    def test_an_unknown_node_is_rejected(self):
        with pytest.raises(PromptError, match="Unknown agentic node"):
            validate_prompt("no_such_node", "{question}")

    def test_an_empty_prompt_is_rejected(self):
        with pytest.raises(PromptError, match="empty"):
            validate_prompt("generate", "   ")

    def test_a_subset_of_placeholders_is_allowed(self):
        """Ignoring an available field is a legitimate design choice."""
        validate_prompt("generate", "Only the question: {question}")

    def test_resolve_validates_too(self):
        """Config can be assembled from a file that never called validate."""
        cfg = _Config({"generate": "{bogus}"})
        with pytest.raises(PromptError):
            resolve_prompt("generate", "DEFAULT", cfg)

    def test_a_prompt_using_no_placeholders_warns_but_is_allowed(self):
        from loguru import logger

        captured: list = []
        sink = logger.add(captured.append, level="WARNING")
        try:
            validate_prompt("generate", "Answer in one word.")
        finally:
            logger.remove(sink)
        assert any("placeholder" in m for m in captured)


class TestPlaceholdersMatchTheNodes:
    """The registry is a claim about the code; check it against the code."""

    #: Every node, not a sample. Covering three of six is how the registry came
    #: to omit `missing_info` from rewrite_query: the shipped prompt used a
    #: placeholder the registry did not list, so a faithful copy of the default
    #: would have been rejected as an override.
    ALL_NODES = [
        ("generate", "generate", "_GENERATION_PROMPT"),
        ("plan_retrieval", "plan_retrieval", "_DECOMPOSITION_PROMPT"),
        ("check_hallucination", "quality_gates", "_HALLUCINATION_PROMPT"),
        ("check_answer_quality", "quality_gates", "_ANSWER_QUALITY_PROMPT"),
        ("rewrite_query", "rewrite_query", "_QUERY_REFINEMENT_PROMPT"),
        ("select_chunks", "select_chunks", "_AGENT_DECIDES_PROMPT"),
    ]

    @pytest.mark.parametrize("node,module,constant", ALL_NODES)
    def test_the_shipped_default_validates_against_its_own_node(self, node, module, constant):
        """If the default would be rejected as an override, the registry is wrong."""
        import importlib

        mod = importlib.import_module(f"lasagna.agentic.nodes.{module}")
        validate_prompt(node, getattr(mod, constant))

    def test_every_node_in_the_registry_is_covered_here(self):
        """Guards the guard: a new node with no case would slip through."""
        assert {n for n, _, _ in self.ALL_NODES} == set(NODE_PLACEHOLDERS)

    @pytest.mark.parametrize("node,module,constant", ALL_NODES)
    def test_the_node_actually_consults_the_registry(self, node, module, constant):
        """A declared-but-unwired node is worse than an undeclared one.

        It appears in `describe_nodes()`, accepts an override, validates it —
        and then runs the hardcoded constant anyway.
        """
        import importlib
        import inspect

        mod = importlib.import_module(f"lasagna.agentic.nodes.{module}")
        src = inspect.getsource(mod)
        assert f'resolve_prompt(\n            "{node}"' in src or f'"{node}", {constant}' in src, (
            f"{node} declares a prompt override but never calls resolve_prompt"
        )

    def test_every_registered_node_declares_placeholders(self):
        for node, fields in NODE_PLACEHOLDERS.items():
            assert isinstance(fields, frozenset), node


class TestPerNodeModels:
    def test_no_override_means_pipeline_default(self):
        assert resolve_node_model("generate", _Config()) is None

    def test_an_override_is_returned(self):
        cfg = _Config(models={"check_hallucination": "gpt-4o-mini-2024-07-18"})
        assert resolve_node_model("check_hallucination", cfg) == "gpt-4o-mini-2024-07-18"

    def test_nodes_are_independent(self):
        cfg = _Config(models={"generate": "big-model"})
        assert resolve_node_model("check_hallucination", cfg) is None


class TestProvenance:
    """Two agentic runs with different prompts must not compare equal."""

    def test_no_overrides_records_nothing(self):
        assert active_prompt_fingerprint(_Config()) == {}

    def test_an_override_is_fingerprinted(self):
        fp = active_prompt_fingerprint(_Config({"generate": "{question} {context}"}))
        assert set(fp) == {"generate"}
        assert len(fp["generate"]) == 12

    def test_different_prompts_fingerprint_differently(self):
        a = active_prompt_fingerprint(_Config({"generate": "{question}"}))
        b = active_prompt_fingerprint(_Config({"generate": "{question} "}))
        assert a["generate"] != b["generate"]

    def test_the_same_prompt_fingerprints_identically(self):
        a = active_prompt_fingerprint(_Config({"generate": "{question}"}))
        b = active_prompt_fingerprint(_Config({"generate": "{question}"}))
        assert a == b

    def test_the_text_itself_is_not_recorded(self):
        """Paragraphs of prose do not belong in every bundle."""
        prompt = "A very long instruction " * 50
        fp = active_prompt_fingerprint(_Config({"generate": prompt + "{question}"}))
        assert prompt not in str(fp)


class TestDiscovery:
    def test_describe_nodes_lists_every_node_and_its_fields(self):
        described = {d["node"] for d in describe_nodes()}
        assert described == set(NODE_PLACEHOLDERS)
        for entry in describe_nodes():
            assert entry["placeholders"] == sorted(NODE_PLACEHOLDERS[entry["node"]])
