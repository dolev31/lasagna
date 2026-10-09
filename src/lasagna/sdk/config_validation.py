"""Configuration validation rules for the Knowledge SDK.

Provides static validation (at config creation) and runtime validation
(during search pipeline) to catch parameter conflicts early and log
when adaptive systems override user settings.

Adding a new rule:
    @static_rule
    def my_check(config: "KnowledgeConfig") -> List[ValidationIssue]:
        if bad_condition(config):
            return [ValidationIssue(
                rule_id="my_check",
                severity=Severity.WARNING,
                message="What went wrong.",
            )]
        return []

    No other files need changes — the runner picks it up automatically.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TYPE_CHECKING

from loguru import logger

from .types import _StrValueEnum

if TYPE_CHECKING:
    from .config import KnowledgeConfig

# =============================================================================
# Types
# =============================================================================


class Severity(_StrValueEnum):
    """Validation issue severity level."""

    ERROR = "error"  # Raises ConfigurationError
    WARNING = "warning"  # Logged at WARNING level
    INFO = "info"  # Logged at INFO level (typically auto-fix applied)


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    rule_id: str
    severity: Severity
    message: str
    fix_applied: bool = False


# =============================================================================
# Rule Registration
# =============================================================================

StaticRuleFn = Callable[["KnowledgeConfig"], List[ValidationIssue]]
RuntimeRuleFn = Callable[[Dict[str, Any]], List[ValidationIssue]]

STATIC_RULES: List[StaticRuleFn] = []
RUNTIME_RULES: List[RuntimeRuleFn] = []


def static_rule(fn: StaticRuleFn) -> StaticRuleFn:
    """Register a static validation rule (runs at config creation)."""
    STATIC_RULES.append(fn)
    return fn


def runtime_rule(fn: RuntimeRuleFn) -> RuntimeRuleFn:
    """Register a runtime validation rule (runs during search)."""
    RUNTIME_RULES.append(fn)
    return fn


# =============================================================================
# Static Rules
# =============================================================================


@static_rule
def mmr_requires_reranking(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """MMR only runs inside the reranking pipeline — silently skipped otherwise."""
    if config.enable_mmr and not config.enable_reranking:
        return [
            ValidationIssue(
                rule_id="mmr_requires_reranking",
                severity=Severity.WARNING,
                message=(
                    "enable_mmr=True has no effect when enable_reranking=False. "
                    "MMR diversity runs inside the reranking pipeline."
                ),
            )
        ]
    return []


@static_rule
def top_k_consistency(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """rerank_top_k must be <= default_top_k to avoid starving the reranker."""
    if config.enable_reranking and config.rerank_top_k > config.default_top_k:
        old_rerank = config.rerank_top_k
        config.rerank_top_k = config.default_top_k
        return [
            ValidationIssue(
                rule_id="top_k_consistency",
                severity=Severity.INFO,
                message=(
                    f"rerank_top_k ({old_rerank}) > default_top_k ({config.default_top_k}). "
                    f"Clamped rerank_top_k to {config.default_top_k}."
                ),
                fix_applied=True,
            )
        ]
    return []


@static_rule
def hyde_strategy_noop(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """Warn if HyDE is disabled but strategy was explicitly set to non-default."""
    from .types import HyDEStrategy

    if not config.enable_hyde and config.hyde_strategy not in (
        HyDEStrategy.ADAPTIVE,
        HyDEStrategy.OFF,
    ):
        return [
            ValidationIssue(
                rule_id="hyde_strategy_noop",
                severity=Severity.WARNING,
                message=(
                    f"enable_hyde=False but hyde_strategy={config.hyde_strategy.value!r}. "
                    f"The strategy has no effect when HyDE is disabled."
                ),
            )
        ]
    return []


@static_rule
def range_checks(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """Clamp numeric parameters to valid ranges."""
    issues: List[ValidationIssue] = []

    for field_name, lo, hi in [
        ("hybrid_alpha", 0.0, 1.0),
        ("mmr_lambda", 0.0, 1.0),
        ("similarity_threshold", 0.0, 1.0),
    ]:
        value = getattr(config, field_name)
        if not (lo <= value <= hi):
            clamped = max(lo, min(hi, value))
            setattr(config, field_name, clamped)
            issues.append(
                ValidationIssue(
                    rule_id=f"range_{field_name}",
                    severity=Severity.INFO,
                    message=f"{field_name}={value} out of [{lo}, {hi}]. Clamped to {clamped}.",
                    fix_applied=True,
                )
            )

    return issues


@static_rule
def quality_gates_require_structured_graph(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """Quality gates (hallucination/answer checks) require a compatible pipeline.

    Quality gates are supported when:
    - agentic_pipeline is explicitly set to "default", OR
    - agentic_pipeline is None and use_structured_graph is True
    """
    issues: List[ValidationIssue] = []

    # Determine if the active pipeline supports quality gates.
    # When agentic_pipeline is explicitly set, it takes full precedence —
    # only "default" supports gates.  The use_structured_graph fallback
    # only applies when no explicit pipeline is configured (None).
    pipeline = getattr(config, "agentic_pipeline", None)
    pipeline_supports_gates = pipeline == "default" or (
        pipeline is None and config.use_structured_graph
    )

    if config.enable_hallucination_check and not pipeline_supports_gates:
        issues.append(
            ValidationIssue(
                rule_id="quality_gates_require_structured_graph",
                severity=Severity.WARNING,
                message=(
                    "enable_hallucination_check=True has no effect without a "
                    "compatible pipeline. Set agentic_pipeline='default' or "
                    "use_structured_graph=True. Quality gates run inside "
                    "the structured StateGraph pipeline."
                ),
            )
        )
    if config.enable_answer_quality_check and not pipeline_supports_gates:
        issues.append(
            ValidationIssue(
                rule_id="quality_gates_require_structured_graph",
                severity=Severity.WARNING,
                message=(
                    "enable_answer_quality_check=True has no effect without a "
                    "compatible pipeline. Set agentic_pipeline='default' or "
                    "use_structured_graph=True. Quality gates run inside "
                    "the structured StateGraph pipeline."
                ),
            )
        )
    return issues


@static_rule
def chunk_overlap_vs_size(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """chunk_overlap must be < chunk_size."""
    if config.chunk_overlap >= config.chunk_size:
        old_overlap = config.chunk_overlap
        new_overlap = config.chunk_size // 5
        config.chunk_overlap = new_overlap
        return [
            ValidationIssue(
                rule_id="chunk_overlap_vs_size",
                severity=Severity.INFO,
                message=(
                    f"chunk_overlap ({old_overlap}) >= chunk_size ({config.chunk_size}). "
                    f"Auto-set overlap to {new_overlap}."
                ),
                fix_applied=True,
            )
        ]
    return []


@static_rule
def hierarchical_chunk_sizes(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """parent_chunk_size must be > child_chunk_size for hierarchical chunking."""
    if not config.enable_hierarchical:
        return []

    issues: List[ValidationIssue] = []

    if config.parent_chunk_size <= config.child_chunk_size:
        old_parent = config.parent_chunk_size
        config.parent_chunk_size = config.child_chunk_size * 4
        issues.append(
            ValidationIssue(
                rule_id="hierarchical_parent_too_small",
                severity=Severity.INFO,
                message=(
                    f"parent_chunk_size ({old_parent}) must be > "
                    f"child_chunk_size ({config.child_chunk_size}). "
                    f"Auto-set parent_chunk_size to {config.parent_chunk_size}."
                ),
                fix_applied=True,
            )
        )

    if not (0.0 <= config.auto_merge_threshold <= 1.0):
        old = config.auto_merge_threshold
        config.auto_merge_threshold = max(0.0, min(1.0, old))
        issues.append(
            ValidationIssue(
                rule_id="auto_merge_threshold_range",
                severity=Severity.INFO,
                message=(
                    f"auto_merge_threshold={old} out of [0.0, 1.0]. "
                    f"Clamped to {config.auto_merge_threshold}."
                ),
                fix_applied=True,
            )
        )

    return issues


# =============================================================================
# Runtime Rules
# =============================================================================


@runtime_rule
def classification_alpha_override(params: Dict[str, Any]) -> List[ValidationIssue]:
    """Log when query classification overrides hybrid_alpha."""
    classification = params.get("query_classification")
    if classification is None:
        return []

    config_alpha = params.get("config_hybrid_alpha")
    effective = params.get("effective_alpha")
    if config_alpha is not None and effective is not None and config_alpha != effective:
        return [
            ValidationIssue(
                rule_id="classification_alpha_override",
                severity=Severity.INFO,
                message=(
                    f"Query classification ({classification.query_type.value}) "
                    f"changed hybrid_alpha: {config_alpha} -> {effective}"
                ),
            )
        ]
    return []


@runtime_rule
def classification_hyde_override(params: Dict[str, Any]) -> List[ValidationIssue]:
    """Log when query classification overrides HyDE setting."""
    classification = params.get("query_classification")
    if classification is None:
        return []

    config_hyde = params.get("config_enable_hyde")
    effective = params.get("effective_use_hyde")
    if config_hyde is not None and effective is not None and config_hyde != effective:
        return [
            ValidationIssue(
                rule_id="classification_hyde_override",
                severity=Severity.INFO,
                message=(
                    f"Query classification ({classification.query_type.value}) "
                    f"changed use_hyde: {config_hyde} -> {effective}"
                ),
            )
        ]
    return []


# =============================================================================
# Runners
# =============================================================================


def validate_static(config: "KnowledgeConfig") -> List[ValidationIssue]:
    """Run all static rules against a KnowledgeConfig.

    Called from KnowledgeConfig.__post_init__().
    Logs warnings/info and raises ConfigurationError for ERROR severity.
    """
    all_issues: List[ValidationIssue] = []

    for rule_fn in STATIC_RULES:
        try:
            issues = rule_fn(config)
            all_issues.extend(issues)
        except Exception as e:
            logger.debug(f"Validation rule {rule_fn.__name__} failed: {e}")

    for issue in all_issues:
        if issue.severity == Severity.ERROR:
            from .exceptions import ConfigurationError

            raise ConfigurationError(
                f"[{issue.rule_id}] {issue.message}",
                config_key=issue.rule_id,
            )
        elif issue.severity == Severity.WARNING:
            logger.warning(f"Config [{issue.rule_id}]: {issue.message}")
        elif issue.severity == Severity.INFO and issue.fix_applied:
            logger.info(f"Config auto-fix [{issue.rule_id}]: {issue.message}")

    return all_issues


def validate_runtime(effective_params: Dict[str, Any]) -> List[ValidationIssue]:
    """Run all runtime rules against effective search parameters.

    Called from RAGBackend.search() after query classification.
    """
    all_issues: List[ValidationIssue] = []

    for rule_fn in RUNTIME_RULES:
        try:
            issues = rule_fn(effective_params)
            all_issues.extend(issues)
        except Exception as e:
            logger.debug(f"Runtime validation rule {rule_fn.__name__} failed: {e}")

    for issue in all_issues:
        if issue.severity == Severity.WARNING:
            logger.warning(f"Search [{issue.rule_id}]: {issue.message}")
        elif issue.severity == Severity.INFO:
            logger.info(f"Search [{issue.rule_id}]: {issue.message}")

    return all_issues
