"""IRCoT (Trivedi et al. 2023): retrieval interleaved with chain-of-thought steps.

Trivedi, Balasubramanian, Khot, Sabharwal. "Interleaving Retrieval with
Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions".
ACL 2023. arXiv:2212.10509. Prompt layout from the authors' code
(github.com/StonyBrookNLP/ircot, ``commaqa/inference/ircot.py``).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult, RetrievedChunkData
from lasagna.agentic.loops._base import (
    LoopPipeline,
    LoopRun,
    first_sentence,
    interleave,
    passage_block,
    setting,
)

#: Chain-of-thought demonstrations, verbatim from the paper (Listing 1,
#: HotpotQA). The authors' prompt files also put each demonstration's gold and
#: distractor paragraphs before it; those are left out to keep the prompt short.
IRCOT_DEMOS = "\n\n\n".join(
    [
        "Q: Jeremy Theobald and Christopher Nolan share what profession?\n"
        "A: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, "
        "producer, and screenwriter. Therefore, they both share the profession of being a "
        "producer. So the answer is: producer.",
        "Q: What film directed by Brian Patrick Butler was inspired by a film directed by "
        "F.W. Murnau?\n"
        "A: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was "
        "inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these "
        "Nosferatu was directed by F.W. Murnau. So the answer is: The Phantom Hour.",
        'Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The '
        'Right Stuff"?\n'
        'A: The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. '
        "The movie Vertical Limit also starred Scott Glenn. So the answer is: Scott Glenn.",
        "Q: In what country was Lost Gravity manufactured?\n"
        "A: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a "
        "German company. So the answer is: Germany.",
    ]
)

#: The paper's stop pattern: a sentence that states the answer.
ANSWER = re.compile(r".* answer is:? (.*)\.?", re.IGNORECASE | re.DOTALL)


def ircot_prompt(paragraphs: Sequence[RetrievedChunkData], question: str, cot: str) -> str:
    """Demonstrations, then the paragraphs, then ``Q:`` and ``A:`` with the chain so far."""
    context = passage_block(paragraphs, header="Wikipedia Title")
    test = f"{context}\n\nQ: {question}\nA: {cot}" if context else f"Q: {question}\nA: {cot}"
    return f"{IRCOT_DEMOS}\n\n\n{test}"


def stated_answer(text: str) -> str:
    """The answer a chain of thought states ("... answer is: X."), else the whole text."""
    match = ANSWER.match(text.strip())
    if not match:
        return text.strip()
    return match.group(1).strip().splitlines()[0].strip().rstrip(".").strip() or text.strip()


def _add_capped(
    collected: List[RetrievedChunkData], new: Sequence[RetrievedChunkData], cap: int
) -> List[RetrievedChunkData]:
    """``collected`` plus the unseen paragraphs of ``new``, at most ``cap`` in all."""
    seen = {c.chunk_id for c in collected}
    out = list(collected)
    for c in new:
        if len(out) >= cap:
            break
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


async def ircot_retrieve(
    run: LoopRun,
) -> Tuple[List[RetrievedChunkData], List[str], str, List[List[RetrievedChunkData]]]:
    """IRCoT's retriever: alternate a reasoning sentence and a search.

    Returns the collected paragraphs (in retrieval order, as the paper keeps
    them), the chain of thought, why it stopped, and each search's results.
    """
    question = run.input.question
    k = int(setting(run.config, "ircot_paragraphs_per_step"))
    max_steps = int(setting(run.config, "ircot_max_steps"))
    cap = int(setting(run.config, "ircot_max_paragraphs"))

    first = run.search(question, k)
    per_hop = [first]
    collected = _add_capped([], first, cap)
    cot: List[str] = []
    stop_reason = "max_steps"
    for _ in range(max_steps):
        sentence = first_sentence(
            await run.complete(ircot_prompt(collected, question, " ".join(cot)))
        )
        if not sentence:
            stop_reason = "empty_step"
            break
        cot.append(sentence)
        if ANSWER.match(sentence):
            stop_reason = "answer"
            break
        new = run.search(sentence, k)
        per_hop.append(new)
        collected = _add_capped(collected, new, cap)
    run.trace["chain_of_thought"] = list(cot)
    return collected, cot, stop_reason, per_hop


async def ircot_read(run: LoopRun, paragraphs: Sequence[RetrievedChunkData]) -> str:
    """IRCoT's reader: one chain of thought over the paragraphs; its stated answer."""
    reasoning = await run.complete(ircot_prompt(paragraphs, run.input.question, ""))
    run.trace["reader_reasoning"] = reasoning.strip()
    return stated_answer(reasoning)


class IRCoTPipeline(LoopPipeline):
    """IRCoT: each chain-of-thought sentence is the next retrieval query.

    Trivedi et al. 2023, "Interleaving Retrieval with Chain-of-Thought
    Reasoning for Knowledge-Intensive Multi-Step Questions" (arXiv:2212.10509).

    Retrieve ``ircot_paragraphs_per_step`` (K) paragraphs for the question,
    then alternate two steps. Reason: given the demonstrations, every
    paragraph collected so far, the question and the chain so far, the model
    writes on; only its first sentence is kept. Retrieve: that sentence is the
    query, and its paragraphs join the collection, capped at
    ``ircot_max_paragraphs`` (the paper's 15). The loop stops when a sentence
    states the answer ("answer is") or after ``ircot_max_steps`` (the paper's
    8). As in the paper's IRCoT QA, a separate reader then answers from all
    collected paragraphs with a chain of thought, and its stated answer is the
    answer: the paper found that better than reading the answer off the
    retrieval chain.

    K is tuned per dataset in the paper ({2, 4, 6, 8}); the default 6 is the
    authors' base configuration. Settings: ``ircot_paragraphs_per_step``,
    ``ircot_max_steps``, ``ircot_max_paragraphs``.
    """

    method = "ircot"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        collected, cot, stop_reason, per_hop = await ircot_retrieve(run)
        answer = await ircot_read(run, collected)
        logger.info(f"[ircot] {len(cot)} step(s), {len(run.hops)} searches, stop={stop_reason}")
        return run.result(
            answer,
            # Every collected paragraph was read, so each hop's best leads.
            [c for c in interleave(per_hop) if c.chunk_id in {p.chunk_id for p in collected}],
            stop_reason=stop_reason,
            steps=len(cot),
            max_steps=int(setting(run.config, "ircot_max_steps")),
        )
