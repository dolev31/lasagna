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

import functools
import math
import os
import re
import string
from typing import Callable, Dict, List, Optional, Sequence, Tuple


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


def _references(ground_truth: str, aliases: Optional[Sequence[str]]) -> List[str]:
    """The canonical answer and its aliases, without the empty ones."""
    return [r for r in [ground_truth, *(aliases or [])] if r and r.strip()]


def _no_answer_policy(prediction: str, ground_truth: str) -> Optional[float]:
    """The SQuAD v2 rule :func:`token_f1` applies, for the metrics beside it.

    1.0 when both sides are empty or abstain, 0.0 when exactly one is, None
    when both carry an answer and the metric itself decides.
    """
    pred_empty = is_abstention(prediction)
    gt_empty = not ground_truth or not ground_truth.strip()
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    return None


def _rouge_tokens(text: str) -> List[str]:
    """``rouge_score``'s default tokenizer: lowercase, alphanumeric runs, no stemming."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, one row at a time."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(
    prediction: str,
    ground_truth: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """ROUGE-L F1 (Lin, 2004): longest-common-subsequence overlap.

    ``P = LCS / |prediction|``, ``R = LCS / |reference|``, ``F = 2PR / (P + R)``
    over ``rouge_score``'s default tokens (lowercase alphanumeric runs, no
    stemming), so a value matches ``RougeScorer(["rougeL"]).score(ref, pred)``'s
    ``fmeasure``. Token F1 counts shared words in any order; ROUGE-L rewards
    them in the reference's order, which is what separates a long answer that
    says the right thing from one that lists the right words. The maximum over
    the reference and its aliases. No-answer policy as :func:`token_f1`.

    Returns:
        ROUGE-L F1 in [0.0, 1.0].
    """
    policy = _no_answer_policy(prediction, ground_truth)
    if policy is not None:
        return policy
    pred_tokens = _rouge_tokens(prediction)
    best = 0.0
    for ref in _references(ground_truth, aliases):
        ref_tokens = _rouge_tokens(ref)
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs:
            p, r = lcs / len(pred_tokens), lcs / len(ref_tokens)
            best = max(best, 2 * p * r / (p + r))
    return best


def contains_answer(
    prediction: str,
    ground_truth: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """1.0 if a reference answer appears in the prediction, else 0.0.

    The "accuracy" of PopQA (Mallen et al., 2023) and Self-RAG, FlashRAG's
    Sub-EM and ALCE's STR-EM for one required answer: a generator asked for a
    sentence answers "The Eiffel Tower is in Paris.", which exact match scores
    0 against "Paris". Both sides are normalized as in :func:`exact_match` and
    the reference must appear as a whole run of tokens, which is stricter than
    the raw substring test those papers use: "no" is not found in "I do not
    know", and "1" is not found in "1990". No-answer policy as
    :func:`exact_match`.

    Returns:
        1.0 or 0.0.
    """
    policy = _no_answer_policy(prediction, ground_truth)
    if policy is not None:
        return policy
    padded = f" {normalize_answer(prediction)} "
    for ref in _references(ground_truth, aliases):
        norm = normalize_answer(ref)
        if norm and f" {norm} " in padded:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Semantic answer similarity (no LLM; a small local embedder)
# ---------------------------------------------------------------------------

#: The embedder :func:`answer_similarity` uses unless ``RAG_EVAL_SIMILARITY_MODEL``
#: names another. Small enough for CPU, and a different model gives different
#: numbers, so change it for a whole comparison or not at all.
DEFAULT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Encoder = Callable[[List[str]], Sequence[Sequence[float]]]


def similarity_model_name() -> str:
    """The embedder answer similarity is computed with."""
    return os.environ.get("RAG_EVAL_SIMILARITY_MODEL") or DEFAULT_SIMILARITY_MODEL


@functools.lru_cache(maxsize=2)
def load_similarity_encoder(model_name: Optional[str] = None) -> Encoder:
    """A ``texts -> vectors`` function backed by a sentence-transformers model.

    On CPU, separate from the pipeline's embedder: the instrument must not be
    the system under test, and must not change with its device.
    """
    from sentence_transformers import SentenceTransformer

    from lasagna.core import model_downloads

    name = model_name or similarity_model_name()
    model_downloads.check(name, kind="answer similarity model")
    with model_downloads.one_weights_copy():
        model = SentenceTransformer(name, device="cpu")

    def encode(texts: List[str]) -> Sequence[Sequence[float]]:
        return model.encode(texts, convert_to_numpy=True).tolist()

    return encode


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    norm = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    return dot / norm if norm else 0.0


def answer_similarity(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    encode: Encoder,
) -> List[Optional[float]]:
    """Cosine similarity of each answer to its closest reference answer.

    RAGAS's ``answer_similarity`` and the bi-encoder form of SAS (Risch et al.,
    2021): both texts are embedded and compared, so "two Nobel Prizes" and "she
    won the Nobel twice" score high where token F1 scores near zero. Batched,
    one ``encode`` call for the whole run.

    Args:
        predictions: Generated answers.
        references: For each answer, its acceptable references (canonical
            answer and aliases).
        encode: ``texts -> vectors``, e.g. :func:`load_similarity_encoder`.

    Returns:
        One value per answer: the maximum cosine over its references, in
        [-1, 1] (in practice [0, 1]); 0.0 for an empty answer, which shares
        nothing with any reference; None when there is no reference to
        compare with (an unanswerable question), never 0.0.
    """
    refs = [[r for r in rs if r and r.strip()] for rs in references]
    texts: List[str] = []
    for pred, rs in zip(predictions, refs):
        if rs and pred and pred.strip():
            texts.append(pred)
            texts.extend(rs)
    vectors = iter(encode(texts) if texts else [])
    out: List[Optional[float]] = []
    for pred, rs in zip(predictions, refs):
        if not rs:
            out.append(None)
        elif not (pred and pred.strip()):
            out.append(0.0)
        else:
            p = next(vectors)
            out.append(max(_cosine(p, next(vectors)) for _ in rs))
    return out


# ---------------------------------------------------------------------------
# Citation quality (ALCE)
# ---------------------------------------------------------------------------

#: ``[1]``, ``[2][3]`` or ``[2, 3]``: 1-based passage numbers, as ALCE prompts.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_LEADING_CITATIONS = re.compile(r"^(?:\s*\[\d+(?:\s*,\s*\d+)*\])+")

#: ``premise, hypothesis -> entailed``; None when the judge could not decide.
Entails = Callable[[str, str], Optional[bool]]


def has_citations(text: str) -> bool:
    """Whether an answer cites passages with ``[n]`` markers."""
    return bool(text) and bool(_CITATION.search(text))


def strip_citations(text: str) -> str:
    """The answer without its ``[n]`` markers: "Paris [1]." -> "Paris.".

    What answer correctness is scored on, as ALCE does: the markers are not
    part of the answer, and left in, "Paris [1]" fails exact match against
    "Paris" and every citing generator loses token F1 to its own "1" tokens.
    Text without markers is returned unchanged.
    """
    if not has_citations(text):
        return text
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_CITATION.sub(" ", text).split()))


def _cited_statements(text: str) -> List[Tuple[str, List[int]]]:
    """Split an answer into sentences, each with the passage numbers it cites.

    Markers that open a sentence ("Paris. [1] It is ...") belong to the
    sentence before them, which is where the claim they support was made.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]
    statements: List[Tuple[str, List[int]]] = []
    for sentence in sentences:
        lead = _LEADING_CITATIONS.match(sentence)
        if lead and statements:
            prior, ids = statements[-1]
            statements[-1] = (prior, ids + _citation_ids(lead.group(0)))
            sentence = sentence[lead.end() :].strip()
        if not sentence:
            continue
        claim = strip_citations(sentence)
        if claim:
            statements.append((claim, _citation_ids(sentence)))
    return [(claim, list(dict.fromkeys(ids))) for claim, ids in statements]


def _citation_ids(text: str) -> List[int]:
    return [int(n) for group in _CITATION.findall(text) for n in group.split(",")]


def citation_precision_recall(
    answer: str,
    passages: Sequence[str],
    entails: Entails,
) -> Optional[Dict[str, float]]:
    """ALCE citation recall and precision (Gao et al., 2023).

    The answer is split into sentences; each cites passages ``[n]`` (1-based
    into ``passages``, the contexts the generator was shown).

    * **Citation recall**: the share of sentences whose cited passages,
      concatenated, entail the sentence. A sentence with no citation, or citing
      a passage number that was never shown, scores 0.
    * **Citation precision**: the share of citations that are not irrelevant.
      A citation on a single-citation sentence counts if it entails the
      sentence. On a multi-citation sentence that its citations jointly
      entail, a citation is irrelevant when it does not entail the sentence
      alone and the others still do without it; on one they do not jointly
      entail, none counts.

    ``entails(premise, hypothesis)`` is the NLI judge (ALCE uses TRUE; the
    harness uses the judge LLM).

    Returns:
        ``{"citation_recall": r, "citation_precision": p}``, or None when the
        answer cites nothing (the metric does not apply, which is not 0.0) or
        the judge could not decide one of the entailments.
    """
    if not has_citations(answer):
        return None
    statements = _cited_statements(answer)
    if not statements:
        return None

    def joined(ids: Sequence[int]) -> str:
        return "\n\n".join(passages[i - 1] for i in ids)

    supported = 0
    credited = 0
    n_citations = 0
    for claim, ids in statements:
        n_citations += len(ids)
        if not ids or any(i < 1 or i > len(passages) for i in ids):
            continue
        joint = entails(joined(ids), claim)
        if joint is None:
            return None
        if not joint:
            continue
        supported += 1
        if len(ids) == 1:
            credited += 1
            continue
        for i in ids:
            alone = entails(passages[i - 1], claim)
            if alone is None:
                return None
            if alone:
                credited += 1
                continue
            rest = entails(joined([j for j in ids if j != i]), claim)
            if rest is None:
                return None
            if not rest:
                credited += 1
    if not n_citations:
        return None
    return {
        "citation_recall": supported / len(statements),
        "citation_precision": credited / n_citations,
    }
