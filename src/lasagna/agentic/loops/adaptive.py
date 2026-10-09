"""Adaptive-RAG (Jeong et al. 2024): route each question by its complexity.

Jeong, Baek, Cho, Hwang, Park. "Adaptive-RAG: Learning to Adapt
Retrieval-Augmented Large Language Models through Question Complexity".
NAACL 2024. arXiv:2403.14403.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult
from lasagna.agentic.loops._base import LoopPipeline, LoopRun, interleave, message_text, setting
from lasagna.agentic.loops.ircot import ircot_read, ircot_retrieve

#: The classifier, prompted with the paper's own definitions of its three
#: classes (Section 3.2, verbatim). The paper trains a T5 classifier on these
#: labels; it publishes no prompted classifier.
CLASSIFIER_PROMPT = """Classify the complexity of a question q into one of three classes:
'A' indicates that q is straightforward and answerable by LLM(q) itself, 'B' indicates that q has the moderate complexity where at least a single-step approach LLM(q, d) is needed, and 'C' indicates that q is complex, requiring the most extensive solution LLM(q, d, c).
Here LLM(q) is a language model answering alone, LLM(q, d) answers from documents d retrieved once for q, and LLM(q, d, c) interleaves several retrieval and reasoning steps c.

q: {question}
Class (A, B or C):"""

ROUTES = {"A": "no_retrieval", "B": "single_step", "C": "multi_step"}


def complexity_label(text: str) -> str:
    """The class (A, B or C) a classifier reply names, or ValueError.

    A reply that starts with a class is that class ("B (moderate). A single
    retrieval..."); otherwise the last class named on its first line wins, so
    an echoed "Class (A, B or C): C" reads as C.
    """
    line = text.strip().splitlines()[0] if text.strip() else ""
    leading = re.match(r"\W*([ABC])\b", line)
    named = re.findall(r"\b([ABC])\b", line)
    if leading:
        return leading.group(1)
    if named:
        return named[-1]
    raise ValueError(f"adaptive's classifier named no class (A, B or C): {text.strip()[:80]!r}")


class AdaptiveRAGPipeline(LoopPipeline):
    """Adaptive-RAG: a complexity classifier picks no, single-step or multi-step retrieval.

    Jeong et al. 2024, "Adaptive-RAG: Learning to Adapt Retrieval-Augmented
    Large Language Models through Question Complexity" (arXiv:2403.14403).

    The classifier labels the question, and the label picks the strategy, each
    answered by the IRCoT reader the paper reuses (Section 7):

    * A: no retrieval; the reader answers from no documents.
    * B: single-step; one search for the question
      (``adaptive_rag_single_step_docs`` documents), then the reader.
    * C: multi-step; IRCoT (the ``ircot_*`` settings), then the reader.

    The paper's classifier is a trained T5-large; neither the paper nor its
    code has a prompted one. Here an LLM is prompted with the paper's own
    definitions of the three classes (``CLASSIFIER_PROMPT``), on
    ``adaptive_rag_classifier_model`` when set, else the loop's model. A
    reply naming no class is an error, never a default route. The label and
    route are recorded in ``loop_trace``. A class-A answer retrieved nothing
    and returns no passages: judged for faithfulness to its context, it is the
    ungrounded answer it is.

    The single-step depth is 5, the authors' base configuration (their run
    script sets 15). Settings: ``adaptive_rag_classifier_model``,
    ``adaptive_rag_single_step_docs``, ``ircot_*``.
    """

    method = "adaptive"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        question = run.input.question
        reply = await run.ask(
            CLASSIFIER_PROMPT.format(question=question), llm=self._classifier(run)
        )
        label = complexity_label(message_text(reply))
        route = ROUTES[label]
        run.trace.update(complexity=label, route=route)
        logger.info(f"[adaptive] complexity={label} route={route}")

        if route == "no_retrieval":
            answer = await ircot_read(run, [])
            return run.result(answer, [], stop_reason=route, steps=0, max_steps=0)

        if route == "single_step":
            k = int(setting(run.config, "adaptive_rag_single_step_docs"))
            documents = run.search(question, k)
            answer = await ircot_read(run, documents)
            return run.result(answer, documents, stop_reason=route, steps=1, max_steps=1)

        collected, cot, stop_reason, per_hop = await ircot_retrieve(run)
        answer = await ircot_read(run, collected)
        read = {c.chunk_id for c in collected}
        return run.result(
            answer,
            [c for c in interleave(per_hop) if c.chunk_id in read],
            stop_reason=f"{route}:{stop_reason}",
            steps=len(cot),
            max_steps=int(setting(run.config, "ircot_max_steps")),
        )

    def _classifier(self, run: LoopRun) -> Any:
        model = str(setting(run.config, "adaptive_rag_classifier_model") or "")
        if not model:
            return run.llm
        from lasagna.providers import get_llm_provider

        return get_llm_provider().get_llm(model_name=model)
