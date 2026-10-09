"""Published query-augmentation methods behind the public ``BaseQueryEnhancer``.

* ``step_back`` -- Step-Back Prompting (Zheng et al. 2023)
* ``query2doc`` -- Query2doc (Wang, Yang and Wei 2023)
* ``rewrite`` -- Rewrite-Retrieve-Read, LLM rewriter (Ma et al. 2023)
* ``multi_query`` -- RAG-Fusion query generation (Rackauckas 2024)

Each returns the original query first, then what its method adds. The
pipeline searches every returned query and fuses the ranked lists with
weighted reciprocal rank fusion (``SearchOrchestrator.search_multi_query``:
the original at weight 2, each addition at weight 1, ``rrf_k``). That fusion
belongs to the pipeline and is held fixed across enhancers, so comparing two
enhancers measures what each one adds to the original query. It also means a
method that searches *only* its rewritten query in its paper (Query2doc,
Rewrite-Retrieve-Read) is measured fused with the original here.

The pipeline caps the additions at its variation count (``max_query_variations``,
or ``rag_fusion_n_queries`` when ``enable_rag_fusion`` is on) and calls each
enhancer with that count as ``n``. The LLM is the query-enhancement model,
``hyde_model``, shared with the other LLM enhancers so a sweep across
enhancers holds the model fixed. Sampling is greedy (temperature 0) for the
reason given at ``_DEFAULT_ENHANCER_TEMPERATURE``.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.core.base_components import BaseQueryEnhancer
from lasagna.core.config import rag_config
from lasagna.providers import get_llm_provider
from lasagna.retrieval.enhancement.enhancer import (
    _DEFAULT_ENHANCER_TEMPERATURE,
    _numbered_lines,
)
from lasagna.utils.model_utils import get_default_model_settings, resolve_model_settings


class _PromptedEnhancer(BaseQueryEnhancer):
    """An enhancer whose method is one LLM prompt, run through the stage's model."""

    #: Registry name, used in error messages.
    method = ""
    #: Output budget for the method's one LLM call.
    max_tokens = 300

    # The pipeline passes the session config (hyde_model) and the variation count.
    _forwards_session_config = True

    def __init__(self, model: Optional[str] = None, n: Optional[int] = None, **kwargs: Any):
        self.model = model
        self._n = n
        self._llm = None
        self._llm_settings: Optional[Dict[str, Any]] = None
        self._llm_lock = threading.Lock()

    def _llm_for(self, config: Any, max_tokens: Optional[int] = None) -> Any:
        """The stage's chat model (``hyde_model``), or RuntimeError when there is none."""
        model_ref = self.model or getattr(config, "hyde_model", None) or rag_config.hyde_model
        try:
            settings = resolve_model_settings(
                model_ref=model_ref,
                default_settings=get_default_model_settings(),
                max_tokens=max_tokens or self.max_tokens,
                context=f"{self.method}_model",
            )
            if settings.get("temperature") is None:
                settings["temperature"] = _DEFAULT_ENHANCER_TEMPERATURE
            with self._llm_lock:
                if self._llm is None or self._llm_settings != settings:
                    self._llm_settings = settings
                    self._llm = get_llm_provider().get_chat_model(settings)
                llm = self._llm
        except Exception as exc:
            logger.error(f"Failed to initialize LLM for {self.method}: {exc}")
            llm = None
        if llm is None:
            raise RuntimeError(f"{self.method} needs an LLM, and none is configured")
        return llm

    def _complete(self, prompt: str, config: Any, max_tokens: Optional[int] = None) -> str:
        return self._complete_messages([{"role": "user", "content": prompt}], config, max_tokens)

    def _complete_messages(
        self, messages: List[Dict[str, str]], config: Any, max_tokens: Optional[int] = None
    ) -> str:
        response = self._llm_for(config, max_tokens).invoke(messages)
        reply = (response.content if hasattr(response, "content") else str(response)).strip()
        if not reply:
            # Every one of these methods always writes something; an empty
            # reply is a failed call, not a query with nothing to add.
            raise RuntimeError(f"{self.method} got an empty reply from its LLM")
        return reply

    def _count(self, config: Any, n: Optional[int] = None) -> int:
        """How many queries to add: the pipeline's count, else the configured one."""
        return int(
            n
            or self._n
            or getattr(config, "max_query_variations", None)
            or rag_config.max_query_variations
            or 3
        )


# ---------------------------------------------------------------------------
# Step-back prompting
# ---------------------------------------------------------------------------

#: The knowledge-QA step-back prompt of Zheng et al. 2023 (Table 12), with five
#: of the paper's exemplars (Tables 13 and 15), one or two from each of its
#: knowledge datasets: TimeQA, SituatedQA, MuSiQue and StrategyQA.
STEP_BACK_PROMPT = """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:
Original Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956?
Stepback Question: Which positions have Knox Cunningham held in his career?
Original Question: Who was the spouse of Anna Karina from 1968 to 1974?
Stepback Question: Who were the spouses of Anna Karina?
Original Question: when was the last time a team from canada won the stanley cup as of 2002
Stepback Question: which years did a team from canada won the stanley cup as of 2002
Original Question: Jan Šindel's was born in what country?
Stepback Question: what is Jan Šindel's personal history?
Original Question: Could the members of The Police perform lawful arrests?
Stepback Question: what can the members of The Police do?
Original Question: {query}
Stepback Question:"""


class StepBackEnhancer(_PromptedEnhancer):
    """Step-back prompting: the question plus a more generic step-back question.

    Zheng, Mishra, Chen, Cheng, Chi, Le, Zhou. "Take a Step Back: Evoking
    Reasoning via Abstraction in Large Language Models". ICLR 2024.
    arXiv:2310.06117.

    The LLM abstracts the question into a step-back question with the paper's
    knowledge-QA prompt (Table 12) and five of its exemplars ("up to 5",
    Appendix D.2). As in the paper's retrieval-augmented setting, both the
    original and the step-back question are searched; the pipeline fuses the
    two lists. (The paper's final-answer prompt, which reads both retrievals,
    belongs to generation, not to this stage.) Greedy decoding, as in the
    paper. Settings: ``hyde_model`` (the LLM).
    """

    method = "step_back"
    max_tokens = 100

    def enhance(self, query: str, **kwargs: Any) -> List[str]:
        if not query.strip():
            return [query]
        reply = self._complete(STEP_BACK_PROMPT.format(query=query), kwargs.get("config"))
        step_back = reply.splitlines()[0].strip()
        step_back = re.sub(r"^step[- ]?back question:\s*", "", step_back, flags=re.I)
        if not step_back or step_back == query.strip():
            return [query]  # the question is already as general as it gets
        return [query, step_back]


# ---------------------------------------------------------------------------
# Query2doc
# ---------------------------------------------------------------------------

#: Query2doc's prompt, verbatim from the paper (Table 11, the full prompt of
#: Figure 1): the instruction and four MS MARCO query/passage demonstrations,
#: typos included.
QUERY2DOC_PROMPT = """Write a passage that answers the given query:
Query: what state is this zip code 85282
Passage: Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn't the place to look for housing bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the same latitude as Damascus, Syria!) and -112 degrees longitude.
Query: why is gibbs model of reflection good
Passage: In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able to improve my nursing practice continuously and learning from the experience for better practice in the future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some discussion on the important of doing reflection in nursing practice.
Query: what does a thousand pardons means
Passage: Oh, that's all right, that's all right, give us a rest; never mind about the direction, hang the direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize, it is an old habit, an old, bad habit, and hard to get rid of when one's digestion is all disordered with eating food that was raised forever and ever before he was born; good land! a man can't keep his functions regular on spring chickens thirteen hundred years old.
Query: what is a macro warning
Passage: Macro virus warning appears when no macros exist in the file in Word. When you open a Microsoft Word 2002 document or template, you may receive the following macro virus warning, even though the document or template does not contain macros: C:\\<path>\\<file name>contains macros. Macros may contain viruses.
Query: {query}
Passage:"""

#: The two expansions of the paper (Sec. 2): dense retrieval reads q [SEP] d';
#: sparse retrieval repeats q n times before d' to keep its terms' weight.
QUERY2DOC_FORMATS = ("dense", "sparse")


class Query2DocEnhancer(_PromptedEnhancer):
    """Query2doc: the query expanded with an LLM-written pseudo-document.

    Wang, Yang, Wei. "Query2doc: Query Expansion with Large Language Models".
    EMNLP 2023. arXiv:2303.07678.

    The LLM writes a passage that answers the query (the paper's few-shot
    prompt, Table 11), and the query is expanded with it. Unlike HyDE, which
    searches with the generated passage *in place of* the query, the expanded
    query keeps the query in front of the passage:

    * ``dense`` (default): ``q [SEP] d'``, the paper's dense-retrieval form;
    * ``sparse``: the query repeated ``query2doc_query_repeats`` times (the
      paper's n = 5), then ``d'``, which keeps the query's own terms weighted.

    The pipeline's hybrid search sends one string to both its dense and its
    sparse leg, so ``query2doc_format`` picks the form the arm is about.

    Two departures from the paper, both for reproducibility: the four
    demonstrations are the fixed ones the paper prints rather than four drawn
    at random per call, and decoding is greedy rather than sampled at
    temperature 1. Settings: ``hyde_model`` (the LLM), ``query2doc_format``,
    ``query2doc_query_repeats``, ``query2doc_max_tokens`` (128, the paper's).
    """

    method = "query2doc"

    def enhance(self, query: str, **kwargs: Any) -> List[str]:
        if not query.strip():
            return [query]
        config = kwargs.get("config")
        form = str(_setting(config, "query2doc_format"))
        if form not in QUERY2DOC_FORMATS:
            raise ValueError(f"query2doc_format must be dense or sparse, not {form!r}")
        max_tokens = int(_setting(config, "query2doc_max_tokens"))
        passage = self._complete(QUERY2DOC_PROMPT.format(query=query), config, max_tokens)
        passage = re.sub(r"^passage:\s*", "", passage, flags=re.I).strip()
        if form == "dense":
            return [query, f"{query.strip()} [SEP] {passage}"]
        repeats = max(1, int(_setting(config, "query2doc_query_repeats")))
        return [query, " ".join([query.strip()] * repeats + [passage])]


# ---------------------------------------------------------------------------
# Rewrite-Retrieve-Read
# ---------------------------------------------------------------------------

#: The frozen-LLM rewriter prompt for open-domain QA (Ma et al. 2023, Table 1),
#: with three demonstrations: the paper's own LLM rewrite (Figure 3) and two
#: from the authors' prompt file (generate/inprompts/myprompt.jsonl).
REWRITE_PROMPT = """Think step by step to answer this question, and provide search engine queries for knowledge that you need. Split the queries with ';' and end the queries with '**'.

Question: Rita Coolidge sang the title song for which Bond film?
Answer: Bond song Rita Coolidge; Bond film**

Question: What 2000 movie does the song "All Star" appear in?
Answer: movie "All Star" 2000**

Question: Ezzard Charles was a world champion in which sport?
Answer: Ezzard Charles champion**

Question: {query}
Answer:"""


class RewriteEnhancer(_PromptedEnhancer):
    """Rewrite-Retrieve-Read: an LLM rewrites the question into search queries.

    Ma, Gong, He, Zhao, Duan. "Query Rewriting for Retrieval-Augmented Large
    Language Models". EMNLP 2023. arXiv:2305.14283.

    The paper's frozen-LLM rewriter (its trained T5 rewriter is a model, not a
    prompt, and is not shipped here): the LLM writes one or more search
    queries split by ``;`` and ended by ``**`` (Table 1's open-domain prompt;
    the paper's demonstrations are unpublished, so they come from the
    authors' repository and Figure 3). Each query is searched. The paper
    searches only the rewrites; this pipeline fuses them with the original
    question (see the module notes). Greedy decoding, 300 tokens, as in the
    authors' code. Settings: ``hyde_model`` (the LLM); at most ``n`` rewrites.
    """

    method = "rewrite"

    def enhance(self, query: str, **kwargs: Any) -> List[str]:
        if not query.strip():
            return [query]
        config = kwargs.get("config")
        reply = self._complete(REWRITE_PROMPT.format(query=query), config)
        lines = [line for line in reply.split("**", 1)[0].splitlines() if line.strip()]
        # The queries close the reply; anything a chat model writes first is reasoning.
        last = re.sub(
            r"^(answer|queries|query):\s*", "", lines[-1].strip() if lines else "", flags=re.I
        )
        rewrites: List[str] = []
        for part in last.split(";"):
            part = part.strip()
            if part and part != query.strip() and part not in rewrites:
                rewrites.append(part)
        return [query] + rewrites[: self._count(config, kwargs.get("n"))]


# ---------------------------------------------------------------------------
# RAG-Fusion multi-query generation
# ---------------------------------------------------------------------------

#: RAG-Fusion's query generation, verbatim from the reference implementation
#: (Raudaschl/rag-fusion, 2023): a system message and two user messages.
MULTI_QUERY_SYSTEM = "You are a helpful assistant that generates multiple search queries based on a single input query."
MULTI_QUERY_USER = "Generate multiple search queries related to: {query}"
MULTI_QUERY_OUTPUT = "OUTPUT ({n} queries):"


class MultiQueryEnhancer(_PromptedEnhancer):
    """RAG-Fusion: LLM-generated search queries, fused by reciprocal rank.

    Rackauckas. "RAG-Fusion: a New Take on Retrieval-Augmented Generation".
    IJNLC 2024. arXiv:2402.03367. Query generation as in the reference
    implementation (github.com/Raudaschl/rag-fusion).

    The LLM generates search queries related to the input with RAG-Fusion's
    messages; the pipeline searches the original and each generated query
    and fuses the ranked lists by reciprocal rank (``rrf_k``, 60 by default,
    RAG-Fusion's k). Searching the original alongside the generated queries
    follows the method's current reference code; its 2023 code searched the
    generated ones only. The pipeline's fusion weights the original twice.

    RAG-Fusion generates 4 queries. Here the count is the pipeline's
    variation count ``n`` (``max_query_variations``, default 3, or
    ``rag_fusion_n_queries`` with ``enable_rag_fusion``), the count every
    enhancer gets, so a comparison between enhancers holds it fixed; set it to
    4 to reproduce the paper. Settings: ``hyde_model`` (the LLM).
    """

    method = "multi_query"

    def enhance(self, query: str, **kwargs: Any) -> List[str]:
        if not query.strip():
            return [query]
        config = kwargs.get("config")
        n = self._count(config, kwargs.get("n"))
        reply = self._complete_messages(
            [
                {"role": "system", "content": MULTI_QUERY_SYSTEM},
                {"role": "user", "content": MULTI_QUERY_USER.format(query=query)},
                {"role": "user", "content": MULTI_QUERY_OUTPUT.format(n=n)},
            ],
            config,
        )
        generated: List[str] = []
        for line in _numbered_lines(reply, exclude=query.strip()):
            line = line.lstrip("-*• ").strip().strip('"').strip()
            if line and line not in generated:
                generated.append(line)
        return [query] + generated[:n]


def _setting(config: Any, key: str) -> Any:
    """A setting from the arm's config, else the process config (its defaults)."""
    value = getattr(config, key, None) if config is not None else None
    return getattr(rag_config, key) if value is None else value
