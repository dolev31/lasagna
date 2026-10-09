"""Every agentic prompt, overridable and validated.

The six prompts driving the agentic graph were module-level constants with no
way to change them. That closes off an entire category of research: "does a
better hallucination-check prompt reduce false positives?" is a legitimate
question, and the answer was "fork the package".

It also made agentic comparisons incomplete in a subtler way. Two runs could
differ in the prompt that decided whether an answer was grounded, and nothing
recorded it — so the bundles compared equal while measuring different systems.

**Placeholders are validated on override, not on use.** A prompt is applied with
``.format(**kwargs)`` deep inside a graph node; a custom prompt missing
``{question}`` raises ``KeyError`` several minutes into a benchmark run, from a
stack that does not mention prompts. Checking at the boundary turns that into an
error naming the node and the missing field.

Extra placeholders are rejected too. ``{documents}`` in a prompt for a node that
only supplies ``{question}`` is a silent no-op today and a confusing failure
tomorrow, and either way the prompt does not do what its author believed.
"""

from __future__ import annotations

import string
from typing import Any, Dict, Iterable, Optional

from loguru import logger

#: Node name -> the placeholders that node supplies to ``.format()``.
#:
#: Derived from the call sites rather than the prompt text: what a prompt *may*
#: reference is decided by what the node passes, not by what the shipped default
#: happens to use. A node could pass a field its own default ignores, and an
#: override is entitled to use it.
NODE_PLACEHOLDERS: Dict[str, frozenset] = {
    "generate": frozenset({"question", "context"}),
    "plan_retrieval": frozenset({"question"}),
    "check_hallucination": frozenset({"context", "generation"}),
    "check_answer_quality": frozenset({"question", "generation"}),
    "rewrite_query": frozenset(
        {"original_question", "current_query", "results_summary", "missing_info"}
    ),
    "select_chunks": frozenset({"generation", "chunks_text"}),
}


def _placeholders_in(template: str) -> set:
    """Field names referenced by a format string, ignoring positional ones."""
    return {
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field and not field.isdigit()
    }


class PromptError(ValueError):
    """A prompt override that cannot work, reported before anything runs."""


def validate_prompt(node: str, template: str) -> None:
    """Raise if ``template`` cannot be formatted by ``node``.

    Called when an override is registered, so the failure names the node and
    the field rather than surfacing as a ``KeyError`` mid-benchmark.
    """
    if node not in NODE_PLACEHOLDERS:
        raise PromptError(
            f"Unknown agentic node {node!r}. Known nodes: {sorted(NODE_PLACEHOLDERS)}"
        )
    if not isinstance(template, str) or not template.strip():
        raise PromptError(f"Prompt for {node!r} is empty.")

    available = NODE_PLACEHOLDERS[node]
    used = _placeholders_in(template)

    unknown = used - available
    if unknown:
        raise PromptError(
            f"Prompt for {node!r} references {sorted(unknown)}, which this node "
            f"does not provide. Available: {sorted(available)}."
        )

    # A prompt that uses none of them is legal but almost always a mistake --
    # it ignores the question and the retrieved context entirely.
    if available and not used:
        logger.warning(
            f"Prompt override for {node!r} uses none of its available "
            f"placeholders {sorted(available)}; the node's inputs will be "
            f"ignored. This is allowed, but it is rarely intended."
        )


def resolve_prompt(node: str, default: str, config: Optional[Any] = None) -> str:
    """The prompt this node should use: an override if configured, else default.

    Overrides live under ``agentic_prompts`` on the resolved config, keyed by
    node name, so they travel through the same session-config path as every
    other setting and land in the run metadata with it.
    """
    if config is None:
        return default

    overrides = getattr(config, "agentic_prompts", None) or {}
    if not isinstance(overrides, dict):
        return default

    if node not in overrides:
        return default
    template = overrides[node]

    # An explicit key means an explicit intent, so a broken value is an error
    # rather than a silent fallback to the shipped prompt -- which would run
    # something other than what its author configured and record no sign of it.
    validate_prompt(node, template)

    return template


def resolve_node_model(node: str, config: Optional[Any] = None) -> Optional[str]:
    """Per-node model override, or None to use the pipeline default.

    Nodes do different jobs -- a grounding check is a classification and a
    generation is not -- so pinning one small model for the gates and a larger
    one for generation is a normal thing to want to measure. Keyed the same way
    as prompts, under ``agentic_models``.
    """
    if config is None:
        return None
    models = getattr(config, "agentic_models", None) or {}
    if not isinstance(models, dict):
        return None
    return models.get(node) or None


def active_prompt_fingerprint(config: Optional[Any] = None) -> Dict[str, str]:
    """Which nodes are running a non-default prompt, and a digest of each.

    Goes into run metadata. Recording the full text would put paragraphs of
    prose into every bundle and into the comparability key; a digest is enough
    to answer "were these two runs using the same prompts?", which is the
    question that matters when comparing agentic results.
    """
    import hashlib

    if config is None:
        return {}
    overrides = getattr(config, "agentic_prompts", None) or {}
    if not isinstance(overrides, dict):
        return {}
    return {
        node: hashlib.sha256(text.encode()).hexdigest()[:12]
        for node, text in sorted(overrides.items())
        if isinstance(text, str) and text.strip()
    }


def describe_nodes() -> Iterable[Dict[str, Any]]:
    """Every overridable node and what it may reference, for discovery."""
    return [
        {"node": node, "placeholders": sorted(fields)}
        for node, fields in sorted(NODE_PLACEHOLDERS.items())
    ]
