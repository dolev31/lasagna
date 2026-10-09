"""Configuration validation — cross-field rules and constraint checking.

Used by ``POST /configurator/api/validate`` to give immediate feedback
without saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import PARAMETERS, get_reindex_keys


@dataclass
class ValidationResult:
    """Result of validating a RAG configuration."""

    valid: bool = True
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    requires_reindex: bool = False
    reindex_keys: list[str] = field(default_factory=list)

    def add_error(self, key: str, message: str) -> None:
        self.errors.append({"key": key, "message": message})
        self.valid = False

    def add_warning(self, key: str, message: str) -> None:
        self.warnings.append({"key": key, "message": message})

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "requires_reindex": self.requires_reindex,
            "reindex_keys": self.reindex_keys,
        }


def validate_config(
    config: dict[str, Any], baseline: dict[str, Any] | None = None
) -> ValidationResult:
    """Validate a RAG configuration dict.

    Args:
        config: The configuration to validate.
        baseline: Optional baseline config to detect reindex-requiring changes.

    Returns:
        ValidationResult with errors, warnings, and reindex info.
    """
    result = ValidationResult()

    # --- Range checks for known parameters ---
    _param_map = {p.key: p for p in PARAMETERS}

    for p in PARAMETERS:
        # Resolve value (handle nested params)
        if p.parent_key:
            parent_obj = config.get(p.parent_key, {})
            if not isinstance(parent_obj, dict):
                continue
            child_key = p.key.split(".")[-1]
            value = parent_obj.get(child_key)
        else:
            value = config.get(p.key)

        if value is None:
            continue

        # Type checks
        if p.type == "boolean" and not isinstance(value, bool):
            result.add_error(p.key, f"Expected boolean, got {type(value).__name__}")
        elif p.type in ("integer", "float") and not isinstance(value, (int, float)):
            result.add_error(p.key, f"Expected number, got {type(value).__name__}")
        elif p.type in ("integer", "float") and isinstance(value, (int, float)):
            if p.min_value is not None and value < p.min_value:
                result.add_error(p.key, f"Value {value} is below minimum {p.min_value}")
            if p.max_value is not None and value > p.max_value:
                result.add_error(p.key, f"Value {value} exceeds maximum {p.max_value}")
        elif p.type == "enum" and p.enum_options:
            valid_values = {o.value for o in p.enum_options}
            if value not in valid_values:
                result.add_error(p.key, f"Invalid value '{value}'. Valid: {sorted(valid_values)}")
        elif p.validate_as == "string_list":
            if not isinstance(value, list):
                result.add_error(p.key, f"Expected list, got {type(value).__name__}")
            elif not all(isinstance(item, str) for item in value):
                result.add_error(p.key, "All items must be strings")

    # --- Cross-field validation ---
    chunk_size = config.get("chunk_size", 800)
    chunk_overlap = config.get("chunk_overlap", 0)
    if isinstance(chunk_size, (int, float)) and isinstance(chunk_overlap, (int, float)):
        if chunk_overlap >= chunk_size:
            result.add_warning(
                "chunk_overlap",
                f"Overlap ({chunk_overlap}) is >= chunk_size ({chunk_size}). "
                f"Will auto-correct to {int(chunk_size * 0.15)}.",
            )

    min_chunk = config.get("min_chunk_size", 200)
    max_chunk = config.get("max_chunk_size", 1500)
    if isinstance(min_chunk, (int, float)) and isinstance(max_chunk, (int, float)):
        if min_chunk >= max_chunk:
            result.add_error(
                "min_chunk_size",
                f"min_chunk_size ({min_chunk}) must be less than max_chunk_size ({max_chunk})",
            )

    parent_size = config.get("parent_chunk_size", 2000)
    child_size = config.get("child_chunk_size", 500)
    if isinstance(parent_size, (int, float)) and isinstance(child_size, (int, float)):
        if child_size >= parent_size:
            result.add_error(
                "child_chunk_size",
                f"child_chunk_size ({child_size}) must be less than parent_chunk_size ({parent_size})",
            )

    # Embedding dim / model consistency warning
    model = config.get("embedding_model", "")
    dim = config.get("embedding_dim")
    model_dims = {
        "all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-m3": 1024,
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat/multilingual-e5-large": 1024,
    }
    if model in model_dims and dim is not None and dim != model_dims[model]:
        result.add_warning(
            "embedding_dim",
            f"embedding_dim ({dim}) doesn't match expected dimension for {model} ({model_dims[model]}). "
            f"This will be auto-corrected.",
        )

    # Chunk retrieval: min_chunks <= max_chunks
    ccr = config.get("chunk_cite_retrieval", {})
    if isinstance(ccr, dict):
        min_c = ccr.get("min_chunks", 2)
        max_c = ccr.get("max_chunks", 20)
        if isinstance(min_c, (int, float)) and isinstance(max_c, (int, float)):
            if min_c > max_c:
                result.add_error(
                    "chunk_cite_retrieval.min_chunks",
                    f"min_chunks ({min_c}) cannot exceed max_chunks ({max_c})",
                )

    # Routing: comprehensive tier requires agentic RAG
    routing_tier = config.get("routing_default_tier", "standard")
    use_agentic = config.get("use_agentic_rag", True)
    if routing_tier == "comprehensive" and not use_agentic:
        result.add_warning(
            "routing_default_tier",
            "routing_default_tier is 'comprehensive' but use_agentic_rag is disabled. "
            "Comprehensive queries will be downgraded to 'standard'.",
        )

    # Quality gates require structured graph or compatible pipeline
    use_structured_graph = config.get("use_structured_graph", False)
    agentic_pipeline = config.get("agentic_pipeline", None)
    enable_hallucination = config.get("enable_hallucination_check", False)
    enable_quality = config.get("enable_answer_quality_check", False)

    # Pipeline supports quality gates if it's "default" (structured graph) or
    # if use_structured_graph is True and no explicit pipeline overrides it.
    pipeline_supports_gates = agentic_pipeline == "default" or (
        agentic_pipeline is None and use_structured_graph
    )
    if (enable_hallucination or enable_quality) and not pipeline_supports_gates:
        result.add_warning(
            "agentic_pipeline",
            "enable_hallucination_check or enable_answer_quality_check is enabled "
            "but the active pipeline does not support quality gates. "
            "Quality gates require agentic_pipeline='default' or use_structured_graph=True.",
        )

    # Structured graph requires agentic RAG
    if use_structured_graph and not use_agentic:
        result.add_warning(
            "use_structured_graph",
            "use_structured_graph is enabled but use_agentic_rag is disabled. "
            "Structured graph mode is a feature of the agentic RAG system — "
            "it will be ignored.",
        )

    # Agentic pipeline requires agentic RAG enabled
    if agentic_pipeline and agentic_pipeline != "legacy_react" and not use_agentic:
        result.add_warning(
            "agentic_pipeline",
            f"agentic_pipeline='{agentic_pipeline}' but use_agentic_rag is disabled. "
            "Pipeline selection is ignored at runtime.",
        )

    # Unknown pipeline name — warn for any non-builtin value.
    # Attempt registry lookup if available; otherwise fall back to
    # builtin-name check so the warning is always emitted.
    _BUILTIN_PIPELINES = {"default", "legacy_react"}
    if agentic_pipeline and agentic_pipeline not in (None, *_BUILTIN_PIPELINES):
        is_registered = False
        try:
            from lasagna.core.registry import registry

            is_registered = registry.has("agentic_pipeline", agentic_pipeline)
        except Exception:
            pass  # Registry unavailable — rely on builtin check only
        if not is_registered:
            result.add_warning(
                "agentic_pipeline",
                f"agentic_pipeline '{agentic_pipeline}' is not a built-in pipeline "
                f"and was not found in the component registry. "
                f"Runtime will raise KeyError if it is not registered before use. "
                f"Built-in values: {sorted(_BUILTIN_PIPELINES)}.",
            )

    # Conflicting use_structured_graph + agentic_pipeline
    if use_structured_graph and agentic_pipeline and agentic_pipeline != "default":
        result.add_warning(
            "use_structured_graph",
            f"use_structured_graph=True but agentic_pipeline='{agentic_pipeline}'. "
            "agentic_pipeline takes precedence. use_structured_graph is a legacy alias.",
        )

    # --- Reindex detection ---
    if baseline:
        reindex_keys = get_reindex_keys()
        for key in reindex_keys:
            old_val = baseline.get(key)
            new_val = config.get(key)
            if old_val != new_val and new_val is not None:
                result.requires_reindex = True
                result.reindex_keys.append(key)

    return result
