"""Exact Match and Token-F1 metrics for RAG answer evaluation.

Standard QA metrics (the official SQuAD v2.0 normalization) that complement
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


#: What an abstention opens with. Matched at the start of the answer's first
#: sentence (after a lead-in like "Sorry," or "Based on the context,"), never
#: anywhere in it: searching the whole answer counted "Paris. There is no
#: mention of the year." as a refusal and scored a correct answer 0.
_ABSTENTION_OPENINGS = re.compile(
    "|".join(
        [
            r"(?:i|we) (?:do not|don't|dont|cannot|can ?not|can't|am unable to|are unable to"
            r"|was unable to|were unable to|could not|couldn't) (?:know|answer|determine|find"
            r"|say|tell|provide|locate|identify)",
            r"i(?:'m| am) (?:not sure|unable to|not able to)",
            r"(?:the |this )?(?:question|answer|information) (?:cannot|can ?not|can't|could not"
            r"|couldn't) be (?:answered|determined|found|provided|identified)",
            r"(?:it is |it's )?(?:not possible|impossible) to (?:answer|determine|say|tell)",
            r"cannot (?:be )?(?:answer|determin)",
            r"unable to (?:answer|determine|find)",
            r"not enough (?:context|information)",
            r"(?:insufficient|not sufficient) (?:context|information)",
            r"no (?:relevant )?(?:information|context|passages?|mention|evidence|answer)",
            r"there (?:is|are|was|were|'s) no (?:relevant |specific |such )?(?:information"
            r"|mention|evidence|context|answer|data|details?)",
            r"(?:the )?(?:provided |given |retrieved |available )?(?:context|passages?"
            r"|documents?|text|sources?|information) (?:does|do|did) not",
            r"(?:the )?(?:provided |given |retrieved |available )?(?:context|passages?"
            r"|documents?|text|sources?) (?:doesn't|don't|lacks?|contains? no|provides? no"
            r"|has no|have no|mentions? no|makes? no mention)",
        ]
    ),
    re.IGNORECASE,
)

#: Courtesy and framing an abstention is often wrapped in, stripped before the
#: opening is matched: "I'm sorry, but", "Based on the provided context,".
_LEAD_IN = re.compile(
    r"^(?:(?:i'm |i am |we're )?(?:sorry|afraid)|unfortunately|regrettably|apologies"
    r"|i apologi[sz]e|however|(?:based on|according to|from|given|in|after reviewing"
    r"|looking at) (?:the |this |these )?(?:provided |given |available |retrieved "
    r"|supplied )?(?:context|passages?|documents?|information|text|sources?)"
    r"(?: provided| given| above)?)\s*[,.:;!]?\s*(?:but\s+)?",
    re.IGNORECASE,
)

#: A contrast after the opening means an answer follows the hedge: "The
#: context does not give the year, but it says Paris."
_CONTRAST = re.compile(r"\b(?:but|however|although|though|yet|nevertheless)\b", re.IGNORECASE)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n")


def normalize_answer(text: str) -> str:
    """The official SQuAD v2.0 answer normalization.

    Lowercase, remove punctuation, remove articles, collapse whitespace, in
    that order. Articles used to be removed before punctuation, so
    ``the-end`` normalized to ``end`` where the official script gives
    ``theend``.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_abstention(text: str) -> bool:
    """Whether an answer is primarily an abstention (a refusal to answer).

    True for an empty answer, or when the first sentence opens with an
    abstention ("I don't know", "The context does not ...", "There is no
    information ..."), after any lead-in such as "Sorry," or "Based on the
    provided context,", and no contrast ("but", "however") follows it. A
    correct answer that also hedges ("Paris. There is no mention of the
    year.") is an answer.
    """
    if not text or not text.strip():
        return True
    # Curly apostrophes are what most LLMs emit for "don't".
    flat = text.replace("’", "'").replace("‘", "'").strip()
    first = next((part for part in _SENTENCE_END.split(flat) if part.strip()), "")
    lead = first.lstrip(" \t*_\"'`>-#")
    while True:
        stripped = _LEAD_IN.sub("", lead, count=1)
        if stripped == lead:
            break
        lead = stripped
    match = _ABSTENTION_OPENINGS.match(lead)
    return bool(match) and not _CONTRAST.search(lead[match.end() :])


def abstention_rate(answers: Sequence[str]) -> Optional[float]:
    """Share of ``answers`` that :func:`is_abstention` counts as abstentions.

    Worth reporting next to exact match and token F1: an abstention scores 0
    on an answerable question and 1 on an unanswerable one, so two generators
    can tie on F1 while abstaining at very different rates. None for no
    answers.
    """
    if not answers:
        return None
    return sum(1 for a in answers if is_abstention(a)) / len(answers)


def exact_match(
    prediction: str,
    ground_truth: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """Exact match with the official SQuAD v2.0 normalization.

    Args:
        prediction: Generated answer.
        ground_truth: Expected answer (canonical).
        aliases: Additional acceptable answer strings.

    Returns:
        1.0 if normalized prediction matches any reference, 0.0 otherwise.

    No-answer policy (an abstention is judged by :func:`is_abstention`):
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
    """Token-level F1 score with the official SQuAD v2.0 normalization.

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
