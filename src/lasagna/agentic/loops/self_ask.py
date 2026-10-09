"""Self-Ask (Press et al. 2022), with the pipeline's search answering follow-ups.

Press, Zhang, Min, Schmidt, Smith, Lewis. "Measuring and Narrowing the
Compositionality Gap in Language Models". Findings of EMNLP 2023.
arXiv:2210.03350.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult, RetrievedChunkData
from lasagna.agentic.loops._base import LoopPipeline, LoopRun, interleave, passage_block, setting

#: The four demonstrations of the paper's self-ask prompt, verbatim from the
#: authors' repository (github.com/ofirpress/self-ask), including the
#: "Intermediate Answer" capitalisation of the last one.
SELF_ASK_PROMPT = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No

Question: {question}
Are follow up questions needed here:"""

#: Reads the retrieved passages into the short answer a search engine's answer
#: box gave the paper. The paper's search returned an answer, not passages.
INTERMEDIATE_ANSWER_PROMPT = """Answer the question in one short sentence, using only the passages below. If they do not contain the answer, say what they do say that is closest.

{passages}

Question: {question}
Answer:"""

FOLLOW_UP = "Follow up:"
INTERMEDIATE = "\nIntermediate answer:"
FINAL = "So the final answer is:"


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _append(transcript: str, generated: str) -> str:
    """The transcript continued by the model, laid out as the demonstrations are.

    Chat models drop the leading space or newline a completion model would
    have written, which glued "here:" to "Yes." and an intermediate answer to
    the next "Follow up:".
    """
    generated = generated.rstrip()
    if transcript.endswith(":"):
        return f"{transcript} {generated.lstrip()}"
    return f"{transcript}\n{generated.lstrip()}"


class SelfAskPipeline(LoopPipeline):
    """Self-Ask: ask follow-up questions, answer each from retrieval, then conclude.

    Press et al. 2022, "Measuring and Narrowing the Compositionality Gap in
    Language Models" (arXiv:2210.03350), self-ask + search engine.

    The model writes in the paper's few-shot format. Generation stops at
    ``Intermediate answer:``; while the last line written is a ``Follow up:``
    question, that question is searched through the pipeline, the answer read
    off the retrieved passages is inserted as the intermediate answer, and
    generation resumes. The loop ends when the model writes
    ``So the final answer is:``; if it stops without one, the prompt is ended
    with that line and the model completes it, as in the authors' code.

    Two adaptations, both forced by the setting rather than chosen:

    * The paper's search engine returned an answer (Google's answer box or top
      snippet). A passage retriever returns passages, so one LLM call reads the
      retrieved passages into that short answer (``INTERMEDIATE_ANSWER_PROMPT``).
      When a search returns nothing, the model writes the intermediate answer
      itself, as the authors' code does when Google has no answer.
    * ``self_ask_max_follow_ups`` (default 5) bounds the loop. The paper has no
      bound; this only stops a model that never concludes.

    Settings: ``self_ask_max_follow_ups``; passages per follow-up come from the
    call's ``top_k``.
    """

    method = "self_ask"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        question = run.input.question
        max_follow_ups = int(setting(run.config, "self_ask_max_follow_ups"))
        prompt = SELF_ASK_PROMPT.format(question=question)
        # Both spellings: the last demonstration writes "Intermediate Answer",
        # and a model that copies it must still stop for the search.
        stops = [INTERMEDIATE, "\nIntermediate Answer:"]
        per_hop: List[List[RetrievedChunkData]] = []

        generated = await run.complete(prompt, stop=stops)
        follow_ups = 0
        stop_reason = "final_answer"
        while FOLLOW_UP in _last_line(generated):
            if follow_ups >= max_follow_ups:
                stop_reason = "max_follow_ups"
                # The pending question is never answered, so it is not left in
                # the transcript the model concludes from.
                lines = generated.rstrip().splitlines()[:-1]
                generated = "\n".join(lines)
                break
            follow_up = _last_line(generated).split(FOLLOW_UP, 1)[1].strip()
            prompt = _append(prompt, generated)
            passages = run.search(follow_up, run.input.top_k)
            per_hop.append(passages)
            follow_ups += 1
            if passages:
                answer = await run.complete(
                    INTERMEDIATE_ANSWER_PROMPT.format(
                        passages=passage_block(passages), question=follow_up
                    )
                )
                answer = answer.strip().splitlines()[0].strip() if answer.strip() else ""
            else:
                # The authors' code, when the search engine has no answer: the
                # model writes the intermediate answer itself.
                answer = await run.complete(
                    prompt + INTERMEDIATE, stop=["\n" + FOLLOW_UP, "\n" + FINAL]
                )
                answer = answer.strip()
            prompt += f"{INTERMEDIATE} {answer.rstrip('.')}."
            generated = await run.complete(prompt, stop=stops)

        if FINAL in generated:
            final = generated.split(FINAL, 1)[1]
        else:
            if stop_reason == "final_answer":
                stop_reason = "concluded_without_marker"
            if generated.strip():
                prompt = _append(prompt, generated)
            prompt += "\n" + FINAL
            final = await run.complete(prompt, stop=["\n"])
        final = final.strip().splitlines()[0].strip() if final.strip() else ""

        logger.info(f"[self_ask] {follow_ups} follow-up(s), stop={stop_reason}")
        run.trace["follow_ups"] = follow_ups
        return run.result(
            final,
            interleave(per_hop),
            stop_reason=stop_reason,
            steps=follow_ups,
            max_steps=max_follow_ups,
        )
