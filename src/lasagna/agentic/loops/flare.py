"""FLARE (Jiang et al. 2023): Forward-Looking Active REtrieval augmented generation.

Jiang, Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan, Neubig. "Active Retrieval
Augmented Generation". EMNLP 2023. arXiv:2305.06983. Defaults from the
authors' 2WikiMultihopQA configuration (github.com/jzbjyb/FLARE,
``configs/2wikihop_flare_config.json``).
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional, Sequence, Tuple

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult, RetrievedChunkData
from lasagna.agentic.loops._base import (
    LoopPipeline,
    LoopRun,
    first_sentence,
    interleave,
    message_text,
    setting,
)
from lasagna.agentic.loops.ircot import ANSWER, stated_answer

#: FLARE-direct's exemplar format for 2WikiMultihopQA (Prompt D.4): the first
#: exemplar verbatim from D.4, the second from the same exemplar pool (D.3).
FLARE_EXEMPLARS = """Question: When did the director of film Hypocrite (Film) die?
Answer: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.

Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no."""

#: Retrieved documents, formatted as in Prompt D.1, right before the test case.
SEARCH_RESULTS = "Search results:\n{documents}\n"

#: The explicit-confidence fallback for a model that returns no token
#: probabilities: the model names the spans of its draft it is unsure of.
CONFIDENCE_PROMPT = """Here is a question and a draft of the next sentence of its answer.

Question: {question}
Draft sentence: {sentence}

List the words or phrases in the draft sentence that you are not confident are correct, separated by " | ". If you are confident about every part of it, reply with exactly: NONE"""

LOGPROBS = "logprobs"
SELF_REPORTED = "self_reported"
SOURCES = ("auto", LOGPROBS, SELF_REPORTED)


def flare_prompt(question: str, answer_so_far: str, documents: Sequence[RetrievedChunkData]) -> str:
    """Exemplars, then the search results, then the question and the answer so far."""
    results = ""
    if documents:
        lines = "\n".join(f"[{i}] {d.content.strip()}" for i, d in enumerate(documents, start=1))
        results = SEARCH_RESULTS.format(documents=lines)
    return f"{FLARE_EXEMPLARS}\n\n{results}Question: {question}\nAnswer: {answer_so_far}".rstrip()


def token_probabilities(response: Any) -> Optional[List[Tuple[str, float]]]:
    """``(token, probability)`` pairs from an OpenAI-style reply, else None."""
    metadata = getattr(response, "response_metadata", None) or {}
    logprobs = metadata.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else logprobs
    if not content:
        return None
    pairs = []
    for entry in content:
        get = entry.get if isinstance(entry, dict) else lambda k, e=entry: getattr(e, k, None)
        token, logprob = get("token"), get("logprob")
        if token is None or logprob is None:
            return None
        pairs.append((token, math.exp(logprob)))
    return pairs


def tokens_within(
    text: str, part: str, tokens: Sequence[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """The tokens of ``text`` that fall inside its substring ``part``."""
    start = text.find(part)
    if start == -1:
        return list(tokens)
    end = start + len(part)
    out, pos = [], 0
    for token, p in tokens:
        token_start, pos = pos, pos + len(token)
        if pos <= start:
            continue
        if token_start >= end:
            break
        out.append((token, p))
    return out


def masked_query(tokens: Sequence[Tuple[str, float]], beta: float) -> str:
    """FLARE's implicit query: the sentence with its tokens below ``beta`` masked out."""
    kept = "".join(token if p >= beta else " " for token, p in tokens)
    return re.sub(r"\s+", " ", kept).strip()


def mask_spans(sentence: str, spans: Sequence[str]) -> str:
    """The sentence with each named span masked out (the self-reported masked query)."""
    out = sentence
    for span in spans:
        if span:
            out = re.sub(re.escape(span), " ", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,;")


def _continuation(text: str, answer_so_far: str) -> str:
    """What the model added, when a chat model restates the answer so far first."""
    text, so_far = text.strip(), answer_so_far.strip()
    if so_far and text.startswith(so_far):
        return text[len(so_far) :].strip()
    return text


class FLAREPipeline(LoopPipeline):
    """FLARE-direct: retrieve when the model is unsure of the sentence it is about to write.

    Jiang et al. 2023, "Active Retrieval Augmented Generation" (arXiv:2305.06983).

    The question is the first query. The answer is then written a sentence at
    a time: the model writes a tentative next sentence with the current
    documents in its prompt (as the authors' code does; the paper describes
    this look-ahead without them). If every token of it has probability at
    least ``flare_retrieval_threshold`` (theta), it is kept. Otherwise it
    becomes the query with its tokens below ``flare_mask_threshold`` (beta)
    masked out, so that an unsure guess does not steer the search (the
    implicit, masked query); the pipeline searches, its documents replace the
    previous ones, and the sentence is written again. Generation ends when a
    sentence states the answer ("So the answer is"), the model writes nothing
    more, or after ``flare_max_sentences`` sentences (the paper caps output at
    256 tokens). The answer is what the last sentence states, or the whole
    text when none does.

    Which confidence signal ran is recorded in ``loop_trace``
    (``confidence_source``):

    * ``logprobs``: token probabilities, as in the paper. Requested with
      ``logprobs=True`` and used when the reply carries them.
    * ``self_reported``: for a model that returns none, the model lists the
      spans of its draft it is unsure of (``CONFIDENCE_PROMPT``). A listed span
      counts as a token below theta, and the listed spans are what is masked.

    ``flare_confidence_source`` forces either; ``auto`` uses logprobs when the
    model returns them and falls back otherwise.

    Defaults are the authors' 2WikiMultihopQA configuration: theta 0.8,
    beta 0.4, 2 documents per retrieval. Settings:
    ``flare_retrieval_threshold``, ``flare_mask_threshold``,
    ``flare_max_sentences``, ``flare_docs_per_retrieval``,
    ``flare_confidence_source``.
    """

    method = "flare"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        config = run.config
        question = run.input.question
        theta = float(setting(config, "flare_retrieval_threshold"))
        beta = float(setting(config, "flare_mask_threshold"))
        max_sentences = int(setting(config, "flare_max_sentences"))
        k = int(setting(config, "flare_docs_per_retrieval"))
        source = str(setting(config, "flare_confidence_source"))
        if source not in SOURCES:
            raise ValueError(f"flare_confidence_source must be one of {SOURCES}, not {source!r}")

        documents = run.search(question, k)
        per_hop: List[List[RetrievedChunkData]] = [documents]
        sentences: List[str] = []
        retrievals = 0
        stop_reason = "max_sentences"
        for _ in range(max_sentences):
            so_far = " ".join(sentences)
            tentative, tokens, source = await self._look_ahead(run, so_far, documents, source)
            if not tentative:
                stop_reason = "answer_complete"
                break
            if tokens is not None:
                confident = all(p >= theta for _, p in tokens)
                query = masked_query(tokens, beta)
                self._record(run, LOGPROBS)
            else:
                spans = await self._unsure_spans(run, tentative)
                confident = not spans
                query = mask_spans(tentative, spans)
                self._record(run, SELF_REPORTED)

            sentence = tentative
            if not confident:
                documents = run.search(query or question, k)
                per_hop.append(documents)
                retrievals += 1
                rewritten = await run.complete(flare_prompt(question, so_far, documents))
                sentence = first_sentence(_continuation(rewritten, so_far))
                if not sentence:
                    stop_reason = "answer_complete"
                    break
            sentences.append(sentence)
            if ANSWER.match(sentence):
                stop_reason = "answer"
                break

        text = " ".join(sentences)
        run.trace.update(retrievals_triggered=retrievals, generation=text)
        logger.info(
            f"[flare] {len(sentences)} sentence(s), {retrievals} triggered retrieval(s), "
            f"confidence={run.trace.get('confidence_source')}, stop={stop_reason}"
        )
        answer = stated_answer(sentences[-1]) if stop_reason == "answer" else text
        return run.result(
            answer,
            interleave(per_hop),
            stop_reason=stop_reason,
            steps=len(sentences),
            max_steps=max_sentences,
        )

    async def _look_ahead(
        self,
        run: LoopRun,
        so_far: str,
        documents: List[RetrievedChunkData],
        source: str,
    ) -> Tuple[str, Optional[List[Tuple[str, float]]], str]:
        """The tentative next sentence, its token probabilities (or None), and the source.

        In ``auto`` a model that rejects the ``logprobs`` request is asked again
        without it, and the run continues on self-reported confidence.
        """
        prompt = flare_prompt(run.input.question, so_far, documents)
        response = None
        if source in ("auto", LOGPROBS) and hasattr(run.llm, "bind"):
            try:
                response = await run.ask(prompt, llm=run.llm.bind(logprobs=True))
            except Exception as exc:
                if source == LOGPROBS:
                    raise
                logger.warning(
                    f"[flare] model refused logprobs ({type(exc).__name__}: {exc}); "
                    "using self-reported confidence"
                )
                run.trace["logprobs_refused"] = f"{type(exc).__name__}"
                source = SELF_REPORTED
        if response is None:
            response = await run.ask(prompt)
        text = message_text(response)
        sentence = first_sentence(_continuation(text, so_far))
        tokens = token_probabilities(response) if source != SELF_REPORTED else None
        if tokens is not None:
            tokens = tokens_within(text, sentence, tokens)
        elif source == LOGPROBS:
            raise RuntimeError(
                "flare_confidence_source is logprobs, but the model returned no token logprobs"
            )
        elif source == "auto":
            source = SELF_REPORTED  # it accepted the request and returned none
        return sentence, tokens, source

    async def _unsure_spans(self, run: LoopRun, sentence: str) -> List[str]:
        reply = (
            await run.complete(
                CONFIDENCE_PROMPT.format(question=run.input.question, sentence=sentence)
            )
        ).strip()
        if not reply or reply.upper().startswith("NONE"):
            return []
        return [s.strip().strip('"') for s in reply.split("|") if s.strip()]

    @staticmethod
    def _record(run: LoopRun, used: str) -> None:
        seen = run.trace.setdefault("confidence_source", used)
        if seen != used:
            run.trace["confidence_source"] = "mixed"
