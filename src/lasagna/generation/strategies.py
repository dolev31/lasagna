"""Built-in answer generation strategies.

Each is a published prompting strategy behind ``BaseGenerator``, so a sweep can
hold retrieval fixed and vary only how the answer is written. The run's
generator model is passed in as ``llm``; with none, a strategy raises rather
than return an empty answer that a judge would then score.
"""

from __future__ import annotations

import re
from typing import Any, List

from ..core.base_components import BaseGenerator

#: The marker the reasoning strategies put before their final answer.
_ANSWER_LINE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+)$")


def _invoke(llm: Any, prompt: str, who: str) -> str:
    if llm is None:
        raise RuntimeError(f"{who} needs an LLM, and none is configured")
    raw = llm.invoke(prompt)
    return raw.content if hasattr(raw, "content") else str(raw)


def _final_answer(text: str) -> str:
    """The text after the last 'Answer:' line; the whole reply when none.

    Only the answer is scored: the reasoning before it would otherwise be
    judged for faithfulness and matched for exact match, and two strategies
    would differ in how much they wrote rather than in what they answered.
    """
    found = _ANSWER_LINE.findall(text or "")
    return found[-1].strip() if found else (text or "").strip()


def _numbered(contexts: List[str]) -> str:
    return "\n\n".join(f"Document [{i}]: {c}" for i, c in enumerate(contexts, 1))


class ConciseGenerator(BaseGenerator):
    """One concise sentence from the context only: the harness's default prompt.

    Kept word for word, so naming it reproduces a run that named no generator.
    """

    def generate(self, query, contexts, *, llm=None, instructions="", **kwargs) -> str:
        ctx_text = "\n\n".join(contexts)
        prompt = (
            "Answer the following question in ONE concise sentence "
            "based only on the provided context. "
            "Do not explain or elaborate. "
            f"{instructions}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        return _invoke(llm, prompt, "the concise generator")


class CitedGenerator(BaseGenerator):
    """Answers that cite their supporting passages as [n] (ALCE, Gao et al. 2023).

    The instruction follows ALCE's: use only the provided documents, cite every
    factual claim, one to three documents per sentence. The citations are what
    citation precision and recall measure.
    """

    def generate(self, query, contexts, *, llm=None, instructions="", **kwargs) -> str:
        prompt = (
            "Instruction: Write an accurate and concise answer for the given question "
            "using only the provided documents (some of which might be irrelevant) and "
            "cite them properly. Always cite for any factual claim. When citing several "
            "documents, use [1][2][3]. Cite at least one document and at most three "
            f"documents in each sentence. {instructions}\n\n"
            f"Question: {query}\n\n{_numbered(contexts)}\n\nAnswer:"
        )
        return _invoke(llm, prompt, "the cited generator").strip()


class ChainOfThoughtGenerator(BaseGenerator):
    """Reason step by step, then answer (Wei et al. 2022; Kojima et al. 2022).

    Only the final answer line is returned and scored.
    """

    def generate(self, query, contexts, *, llm=None, instructions="", **kwargs) -> str:
        prompt = (
            "Use the context to answer the question. Think step by step, then give "
            "the final answer on a last line that starts with 'Answer:'. "
            f"{instructions}\n\n"
            f"Context:\n{_numbered(contexts)}\n\n"
            f"Question: {query}\n\nLet's think step by step."
        )
        return _final_answer(_invoke(llm, prompt, "the chain-of-thought generator"))


class ChainOfNoteGenerator(BaseGenerator):
    """Reading notes per passage, then the answer (Yu et al. 2023, Chain-of-Note).

    Writing a note on each passage's relevance before answering makes the model
    robust to retrieved passages that are irrelevant, and lets it say the
    answer is unknown when no passage holds it.
    """

    def generate(self, query, contexts, *, llm=None, instructions="", **kwargs) -> str:
        prompt = (
            "Task: read the documents and write a brief reading note on each, saying "
            "whether and how it helps answer the question. Then answer the question "
            "using the relevant notes. If no document contains the answer, answer "
            "'unknown'. Give the final answer on a last line that starts with "
            f"'Answer:'. {instructions}\n\n"
            f"{_numbered(contexts)}\n\nQuestion: {query}\n\nNotes:"
        )
        return _final_answer(_invoke(llm, prompt, "the chain-of-note generator"))
