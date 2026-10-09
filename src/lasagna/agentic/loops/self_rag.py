"""Self-RAG (Asai et al. 2023), prompting variant: on-demand retrieval and self-critique.

Asai, Wu, Wang, Sil, Hajishirzi. "Self-RAG: Learning to Retrieve, Generate,
and Critique through Self-Reflection". ICLR 2024. arXiv:2310.11511.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult, RetrievedChunkData
from lasagna.agentic.loops._base import LoopPipeline, LoopRun, setting

# The reflection tokens, asked for with the authors' critic prompts, verbatim
# from their repository (data_creation/critic/gpt4_reward; paper Tables 8-12),
# demonstrations and typos included. These are the instructions that labelled
# the trained critic's data; here a general LLM answers them directly.

RETRIEVE_PROMPT = """Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation.

##
Instruction: Give three tips for staying healthy.
Need retrieval?: [Yes]
Explanation: There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query.

##
Instruction: Describe a time when you had to make a difficult decision.
Need retrieval?: [No]
Explanation: This instruction is asking about some personal experience and thus it does not require one to find some external documents.

##
Instruction: Write a short story in third person narration about a protagonist who has to make an important career decision.
Need retrieval?: [No]
Explanation: This instruction asks us to write a short story, which does not require external evidence to verify.

##
Instruction: What is the capital of France?
Need retrieval?: [Yes]
Explanation: While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.

##
 Instruction: Find the area of a circle given its radius. Radius = 4
Need retrieval?: [No]
Explanation: This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.

##
Instruction: Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped
Need retrieval?: [No]
Explanation: This task doesn't require any external evidence, as it is a simple grammatical question.

##
Instruction: Explain the process of cellular respiration in plants.Need retrieval?: [Yes]
Explanation: This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.

##
Instruction:{instruction}
Need retrieval?: """

ISREL_PROMPT = """You'll be provided with an instruction, along with evidence and possibly some preceding sentences. When there are preceding sentences, your focus should be on the sentence that comes after them. Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].

###
Instruction: Given four answer options, A, B, C, and D, choose the best answer.

Input: Earth rotating causes
A: the cycling of AM and PM
B: the creation of volcanic eruptions
C: the cycling of the tides
D: the creation of gravity

Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.

Rating: [Relevant]
Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.

###
Instruction: age to run for us house of representatives

Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.

Rating: [Irrelevant]
Explanation: The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.

###
Instruction: {instruction}

Evidence: {evidence}

Rating:"""

ISSUP_PROMPT = """You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement.
Use the following entailment scale to generate a score:
[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.
[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].
[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.

Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.

###
Instruction: Explain the use of word embeddings in Natural Language Processing.
Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.
Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.
Evidence: Word embedding
Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well as analogy detection.
Score: [Fully supported]
Explanation: The output sentence discusses the application of word embeddings, and the evidence mentions all of the applications syntactic parsing, sentiment analysis, next token predictions as well as analogy detection as the applications. Therefore, the score should be [Fully supported].

###
Instruction: {instruction}
Output: {target_output}
Evidence: {evidence}
Score: """

ISUSE_PROMPT = """Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.

The detailed criterion is as follows:
5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.
4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. 
3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.
2: The response still addresses the main request, but it is not complete or not relevant to the query.
1: The response is barely on-topic or completely irrelevant.
##
 Instruction: Who is the current prime minister of the UK as of 2023?
Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.
Perceived utility: 2
Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.

##
Instruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.
Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.
Perceived utility: 3
Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. 

##
Instruction: {instruction}
Output:{output}

"""

#: The generator's input, in Self-RAG's own format (an instruction, then the
#: retrieved paragraph after a Retrieval token); the model continues with the
#: response. Any reflection token it copies is removed from the answer, as the
#: authors' code removes the trained model's.
GENERATE_PROMPT = """### Instruction:
{instruction}

### Response:
{evidence}"""

GENERATE_EVIDENCE = "[Retrieval]<paragraph>{title}\n{text}</paragraph>"

_REFLECTION_TOKENS = re.compile(
    r"\[(?:Retrieval|No Retrieval|Continue to Use Evidence|Relevant|Irrelevant|"
    r"Fully supported|Partially supported|No support / Contradictory|Utility:\d)\]"
    r"|</?paragraph>"
)


def _response(text: str) -> str:
    return re.sub(r"\s+", " ", _REFLECTION_TOKENS.sub(" ", text)).strip()


#: The paper's critique scores, with the token probabilities of the trained
#: model replaced by the prompted critic's single label (probability 1).
_SUPPORT = {"fully": 1.0, "partially": 0.5, "no": 0.0}
_UTILITY = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}


def _first_line(text: str) -> str:
    text = text.strip()
    return text.splitlines()[0] if text else ""


def _retrieve(text: str) -> bool:
    head = _first_line(text).lower()
    if "[no]" in head or re.match(r"^\W*no\b", head):
        return False
    if "[yes]" in head or re.match(r"^\W*yes\b", head):
        return True
    raise ValueError(f"self_rag's retrieval judgement was neither [Yes] nor [No]: {text[:80]!r}")


def _relevant(text: str) -> float:
    head = _first_line(text).lower()
    if "irrelevant" in head:
        return 0.0
    if "relevant" in head:
        return 1.0
    raise ValueError(f"self_rag's relevance critique was neither label: {text[:80]!r}")


def _support(text: str) -> float:
    head = _first_line(text).lower()
    if "no support" in head or "contradictory" in head:
        return 0.0
    if "partially" in head:
        return _SUPPORT["partially"]
    if "fully" in head:
        return _SUPPORT["fully"]
    raise ValueError(f"self_rag's support critique named no level: {text[:80]!r}")


def _utility(text: str) -> float:
    match = re.search(r"[1-5]", _first_line(text))
    if not match:
        raise ValueError(f"self_rag's utility critique gave no 1-5 score: {text[:80]!r}")
    return _UTILITY[int(match.group(0))]


class SelfRAGPipeline(LoopPipeline):
    """Self-RAG, prompting variant: retrieve on demand, then critique each passage's answer.

    Asai et al. 2023, "Self-RAG: Learning to Retrieve, Generate, and Critique
    through Self-Reflection" (arXiv:2310.11511).

    Self-RAG trains one model to emit reflection tokens. Here a general LLM is
    prompted with the authors' critic instructions in place of the trained
    model, and the inference is the paper's short-form one: judge whether to
    retrieve (Retrieve); if so, retrieve ``self_rag_passages`` passages (the
    paper's 5) and, for each, judge its relevance (ISREL), answer from it,
    judge whether the answer is supported by it (ISSUP) and how useful it is
    (ISUSE); keep the answer with the highest critique score
    ``w_rel*ISREL + w_sup*ISSUP + w_use*ISUSE`` (the paper's weights 1.0, 1.0,
    0.5). Without retrieval the model answers alone. The answer comes back
    with the one passage it was written from, so its faithfulness is judged
    against that passage.

    This is not the default graph again. That graph grades a retrieved set,
    writes one answer from all of it and then checks that answer; Self-RAG
    writes one answer per passage and chooses among them by their critiques.

    Prompting has no reflection-token probabilities, so each critique counts at
    its label's value (Relevant 1; Fully 1, Partially 0.5, No support 0;
    utility 1..5 as -1, -0.5, 0, 0.5, 1, the paper's weights), the Retrieve
    judgement is a yes or no rather than a probability against the paper's
    threshold (0.2), and the answer's own likelihood term is left out. One
    segment per answer: the long-form, segment-level beam search is not
    implemented. ``self_rag_retrieval: always`` skips the Retrieve judgement,
    the paper's always-retrieve setting.

    Settings: ``self_rag_passages``, ``self_rag_weight_rel``,
    ``self_rag_weight_sup``, ``self_rag_weight_use``, ``self_rag_retrieval``
    (``adaptive`` or ``always``).
    """

    method = "self_rag"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        config = run.config
        question = run.input.question
        k = int(setting(config, "self_rag_passages"))
        weights = (
            float(setting(config, "self_rag_weight_rel")),
            float(setting(config, "self_rag_weight_sup")),
            float(setting(config, "self_rag_weight_use")),
        )
        mode = str(setting(config, "self_rag_retrieval"))
        if mode not in ("adaptive", "always"):
            raise ValueError(f"self_rag_retrieval must be adaptive or always, not {mode!r}")

        retrieve = True
        if mode == "adaptive":
            retrieve = _retrieve(await run.complete(RETRIEVE_PROMPT.format(instruction=question)))
        run.trace["retrieved"] = retrieve

        passages = run.search(question, k) if retrieve else []
        if not passages:
            answer = _response(
                await run.complete(GENERATE_PROMPT.format(instruction=question, evidence=""))
            )
            reason = "no_retrieval" if not retrieve else "nothing_retrieved"
            return run.result(answer, [], stop_reason=reason, steps=0, max_steps=1)

        scored: List[Tuple[float, str, RetrievedChunkData, dict]] = []
        for passage in passages:
            scored.append(await self._critiqued_answer(run, question, passage, weights))
        best = max(scored, key=lambda s: s[0])  # a tie goes to the higher-ranked passage
        run.trace["critiques"] = [s[3] for s in scored]
        logger.info(
            f"[self_rag] chose passage {scored.index(best) + 1}/{len(scored)} (score={best[0]:.2f})"
        )
        return run.result(best[1], [best[2]], stop_reason="critique_selected", steps=1, max_steps=1)

    async def _critiqued_answer(
        self,
        run: LoopRun,
        question: str,
        passage: RetrievedChunkData,
        weights: Tuple[float, float, float],
    ) -> Tuple[float, str, RetrievedChunkData, dict]:
        text = passage.content.strip()
        rel = _relevant(
            await run.complete(ISREL_PROMPT.format(instruction=question, evidence=text))
        )
        evidence = GENERATE_EVIDENCE.format(title=passage.document_name, text=text)
        answer = _response(
            await run.complete(GENERATE_PROMPT.format(instruction=question, evidence=evidence))
        )
        sup = _support(
            await run.complete(
                ISSUP_PROMPT.format(instruction=question, target_output=answer, evidence=text)
            )
        )
        use = _utility(await run.complete(ISUSE_PROMPT.format(instruction=question, output=answer)))
        w_rel, w_sup, w_use = weights
        score = w_rel * rel + w_sup * sup + w_use * use
        critique = {"chunk_id": passage.chunk_id, "isrel": rel, "issup": sup, "isuse": use}
        return score, answer, passage, critique
