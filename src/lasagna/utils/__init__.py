"""Utility layer for Lasagna.

This module provides common utilities including:
- Monitoring and metrics
- Model utilities for LLM operations
- Logging configuration
- Score normalization utilities
- Token estimation utilities
"""

from .monitoring import (
    METRICS,
    MetricsClient,
    configure_metrics,
    trunc_text,
    hash_text,
    summarize_document_ids,
    log_json,
    log_json_debug,
    StageTimer,
)
from .tracing import (
    TRACER,
    Tracer,
    configure_tracing,
)
from .model_utils import resolve_model_settings
from .logging_utils import configure_knowledge_logging
from .normalization import (
    normalize_score_sigmoid,
    normalize_scores_minmax,
    normalize_scores_for_mmr,
    normalize_score_for_display,
    normalize_score_relative,
    adaptive_k_selection,
)
from .tokenization import (
    estimate_tokens,
    estimate_tokens_batch,
    count_tokens_for_messages,
    fits_in_context,
    truncate_to_token_limit,
    TIKTOKEN_AVAILABLE,
)

__all__ = [
    # Monitoring
    "METRICS",
    "MetricsClient",
    "configure_metrics",
    "trunc_text",
    "hash_text",
    "summarize_document_ids",
    "log_json",
    "log_json_debug",
    "StageTimer",
    # Tracing
    "TRACER",
    "Tracer",
    "configure_tracing",
    # Model utilities
    "resolve_model_settings",
    # Logging
    "configure_knowledge_logging",
    # Normalization
    "normalize_score_sigmoid",
    "normalize_scores_minmax",
    "normalize_scores_for_mmr",
    "normalize_score_for_display",
    "normalize_score_relative",
    "adaptive_k_selection",
    # Tokenization
    "estimate_tokens",
    "estimate_tokens_batch",
    "count_tokens_for_messages",
    "fits_in_context",
    "truncate_to_token_limit",
    "TIKTOKEN_AVAILABLE",
]
