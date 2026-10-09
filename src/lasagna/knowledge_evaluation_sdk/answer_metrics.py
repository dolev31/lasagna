"""Exact Match and Token-F1 metrics for RAG answer evaluation.

Standard QA metrics (SQuAD-style normalization) that complement
LLM-judged RAGAS metrics. These are free to compute (no API calls).

Example::

    from lasagna.knowledge_evaluation_sdk.answer_metrics import (
        exact_match,
        token_f1,
    )

    em = exact_match("the Eiffel Tower", "Eiffel Tower", aliases=["Tour Eiffel"])
    f1 = token_f1("the Eiffel Tower is in Paris", "Eiffel Tower")
"""

from __future__ import annotations

import re
import string
from typing import List, Optional, Sequence


_ABSTENTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i don'?t know",
        r"cannot (be )?answer",
        r"not enough (context|information)",
        r"the (provided |given )?(context|passages?) (does|do) not",
        r"no relevant (information|context|passage)",
        r"unable to (answer|determine|find)",
        r"there is no (information|mention|evidence)",
    ]
]


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization.

    Lowercase, strip articles, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def is_abstention(text: str) -> bool:
    """Detect whether an answer is an abstention (refusal to answer)."""
    if not text or not text.strip():
        return True
    for pattern in _ABSTENTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def exact_match(
    prediction: str,
    ground_truth: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """Exact match with SQuAD-style normalization.

    Args:
        prediction: Generated answer.
        ground_truth: Expected answer (canonical).
        aliases: Additional acceptable answer strings.

    Returns:
        1.0 if normalized prediction matches any reference, 0.0 otherwise.

    No-answer policy:
        - Both empty/abstention → 1.0 (correct abstention)
        - Prediction empty, GT non-empty → 0.0
        - Prediction non-empty, GT empty → 0.0
    """
    pred_empty = is_abstention(prediction)
    gt_empty = not ground_truth or not ground_truth.strip()

    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0

    norm_pred = normalize_answer(prediction)

    references: List[str] = [ground_truth]
    if aliases:
        references.extend(aliases)

    valid_refs = [ref for ref in references if ref and ref.strip()]
    if not valid_refs:
        return 0.0
    return max(1.0 if normalize_answer(ref) == norm_pred else 0.0 for ref in valid_refs)


def token_f1(
    prediction: str,
    ground_truth: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """Token-level F1 score with SQuAD-style normalization.

    Args:
        prediction: Generated answer.
        ground_truth: Expected answer (canonical).
        aliases: Additional acceptable answer strings.

    Returns:
        Maximum F1 score across all references. Range [0.0, 1.0].

    No-answer policy:
        - Both empty → 1.0
        - Either empty (not both) → 0.0
    """
    pred_empty = is_abstention(prediction)
    gt_empty = not ground_truth or not ground_truth.strip()

    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    references: List[str] = [ground_truth]
    if aliases:
        references.extend(aliases)

    best_f1 = 0.0
    for ref in references:
        if not ref or not ref.strip():
            continue
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue

        common = set(pred_tokens) & set(ref_tokens)
        n_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

        if n_common == 0:
            continue

        precision = n_common / len(pred_tokens)
        recall = n_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1
