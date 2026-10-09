"""Query Enhancement for RAG System - SOTA Implementation.

This module provides:
- HyDE (Hypothetical Document Embeddings) for improved retrieval
- Query expansion with paraphrasing
- Query decomposition for complex questions
"""

import asyncio
import atexit
import concurrent.futures
import contextvars
import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from loguru import logger

from lasagna.core.config import rag_config
from lasagna.utils.model_utils import (
    resolve_model_settings,
    get_default_model_settings,
    is_langfuse_tracing_enabled,
)
from lasagna.providers import get_llm_provider

# Shared thread pool for parallel HyDE generation (avoids per-call overhead)
_hyde_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_hyde_executor_lock = threading.Lock()


def _get_hyde_executor(max_workers: int = 8) -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared HyDE thread pool executor."""
    global _hyde_executor
    if _hyde_executor is None:
        with _hyde_executor_lock:
            if _hyde_executor is None:
                _hyde_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="hyde-gen",
                )
                atexit.register(_hyde_executor.shutdown, wait=False)
    return _hyde_executor


@dataclass
class EnhancedQuery:
    """Enhanced query with variations and hypothetical documents."""

    original_query: str
    hyde_document: Optional[str] = None
    query_variations: List[str] = None
    sub_queries: List[str] = None

    def __post_init__(self):
        if self.query_variations is None:
            self.query_variations = []
        if self.sub_queries is None:
            self.sub_queries = []

    def get_all_queries(self) -> List[str]:
        """Get all query variations for multi-query retrieval."""
        queries = [self.original_query]
        queries.extend(self.query_variations)
        queries.extend(self.sub_queries)
        return queries


class HyDEGenerator:
    """Hypothetical Document Embeddings (HyDE) generator.

    HyDE generates a hypothetical answer to the query, then uses
    that answer's embedding for retrieval. This bridges the gap
    between query and document embedding spaces.
    """

    HYDE_PROMPT = """Given the following question, write a short passage (2-3 sentences)
that would answer this question. Write as if you're citing from a document.
Do not include phrases like "According to" or "The document states".
Just write the factual content that would answer the question.

Question: {query}

Hypothetical answer:"""

    def __init__(self, model: Optional[str] = None):
        """Initialize HyDE generator.

        Args:
            model: LLM model to use for generation (default from config)
        """
        self.model = model
        self._llm = None
        self._llm_settings: Optional[Dict[str, Any]] = None
        # CRITICAL FIX: Thread safety for LLM initialization
        # Prevents race conditions in multi-threaded environments (FastAPI)
        self._llm_lock = threading.Lock()

    def _resolve_model_settings(self, model_ref: Optional[str], max_tokens: int) -> Dict[str, Any]:
        return resolve_model_settings(
            model_ref=model_ref,
            default_settings=get_default_model_settings(),
            max_tokens=max_tokens,
            context="hyde_model",
        )

    def _get_llm(self, model_ref: Optional[str], max_tokens: int) -> Optional[Any]:
        """Lazy load LLM client (thread-safe).

        CRITICAL FIX: Uses lock to prevent race conditions during LLM initialization.
        Without this, concurrent requests could trigger duplicate LLM initializations
        or read stale _llm/_llm_settings values.
        """
        try:
            model_settings = self._resolve_model_settings(model_ref, max_tokens)

            with self._llm_lock:
                if self._llm is None or self._llm_settings != model_settings:
                    self._llm_settings = model_settings
                    self._llm = get_llm_provider().get_chat_model(model_settings)
                return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM for HyDE: {e}")
            return None

    def _get_langfuse_handler(self, trace_id: Optional[str] = None) -> Optional[Any]:
        if not is_langfuse_tracing_enabled():
            return None
        try:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
        except ImportError:
            try:
                from langfuse.callback.langchain import (
                    LangchainCallbackHandler as LangfuseCallbackHandler,
                )
            except ImportError:
                LangfuseCallbackHandler = None
        if LangfuseCallbackHandler is None:
            return None
        try:
            trace_context = {"trace_id": trace_id} if trace_id else None
            return LangfuseCallbackHandler(trace_context=trace_context)
        except Exception as e:
            logger.debug(f"Failed to initialize Langfuse handler for HyDE: {e}")
            return None

    def _bind_callbacks(self, llm: Any, callbacks: Optional[List[Any]]) -> Any:
        """Attach callbacks to an LLM without passing them twice to invoke."""
        if not callbacks:
            return llm
        if hasattr(llm, "with_config"):
            try:
                return llm.with_config({"callbacks": callbacks})
            except Exception as e:
                logger.debug(f"Failed to bind callbacks for HyDE: {e}")
        return llm

    async def generate_hypothetical_document(
        self,
        query: str,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a hypothetical document for the query.

        Args:
            query: User query

        Returns:
            Hypothetical document text or None
        """
        model_ref = model_ref or self.model or rag_config.hyde_model
        base_llm = self._get_llm(model_ref, max_tokens)
        if not base_llm:
            return None
        langfuse_handler = self._get_langfuse_handler(trace_id)
        llm = self._bind_callbacks(base_llm, [langfuse_handler] if langfuse_handler else None)

        try:
            prompt = self.HYDE_PROMPT.format(query=query)
            messages = [
                {"role": "user", "content": prompt},
            ]
            logger.debug(f"HyDE generating hypothetical document with model={model_ref}")
            try:
                response = await llm.ainvoke(messages)
            except TypeError as e:
                if "callbacks" in str(e):
                    logger.warning("HyDE invoke failed with callbacks; retrying without callbacks")
                    response = await base_llm.ainvoke(messages)
                else:
                    raise
            hyde_doc = (
                response.content.strip() if hasattr(response, "content") else str(response).strip()
            )

            logger.debug(f"HyDE document generated: {hyde_doc[:100]}...")
            return hyde_doc

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return None

    def generate_hypothetical_document_sync(
        self,
        query: str,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Synchronous version of HyDE generation.

        Args:
            query: User query

        Returns:
            Hypothetical document text or None
        """
        try:
            model_ref = model_ref or self.model or rag_config.hyde_model
            base_llm = self._get_llm(model_ref, max_tokens)
            if not base_llm:
                return None
            langfuse_handler = self._get_langfuse_handler(trace_id)
            llm = self._bind_callbacks(base_llm, [langfuse_handler] if langfuse_handler else None)
            prompt = self.HYDE_PROMPT.format(query=query)
            messages = [
                {"role": "user", "content": prompt},
            ]
            logger.debug(f"HyDE generating hypothetical document with model={model_ref}")
            try:
                response = llm.invoke(messages)
            except TypeError as e:
                if "callbacks" in str(e):
                    logger.warning("HyDE invoke failed with callbacks; retrying without callbacks")
                    response = base_llm.invoke(messages)
                else:
                    raise
            return (
                response.content.strip() if hasattr(response, "content") else str(response).strip()
            )
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return None

    async def generate_hypothetical_documents_batch(
        self,
        query: str,
        n_docs: int = 3,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
    ) -> List[str]:
        """Generate multiple hypothetical documents for embedding averaging.

        Multi-document HyDE reduces variance from single LLM output and
        provides better coverage of the query intent space. The embeddings
        from multiple hypothetical documents are averaged to create a more
        robust query representation.

        Args:
            query: User query
            n_docs: Number of hypothetical documents to generate (recommended: 3-5)
            model_ref: LLM model reference
            max_tokens: Max tokens per document
            trace_id: Trace ID for observability

        Returns:
            List of hypothetical document strings (may be fewer than n_docs if some fail)
        """
        if n_docs <= 0:
            return []

        if n_docs == 1:
            # Optimization: use single-doc path for n=1
            doc = await self.generate_hypothetical_document(query, model_ref, max_tokens, trace_id)
            return [doc] if doc else []

        # Generate n_docs in parallel for efficiency
        tasks = [
            self.generate_hypothetical_document(query, model_ref, max_tokens, trace_id)
            for _ in range(n_docs)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures and empty results
        docs = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.debug(f"HyDE batch doc {i + 1}/{n_docs} failed: {r}")
            elif r:
                docs.append(r)

        if not docs:
            logger.warning(f"All {n_docs} HyDE generations failed, returning empty list")
            return []

        logger.debug(f"Generated {len(docs)}/{n_docs} hypothetical documents for multi-HyDE")
        return docs

    def generate_hypothetical_documents_batch_sync(
        self,
        query: str,
        n_docs: int = 3,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
    ) -> List[str]:
        """Synchronous version of batch HyDE generation.

        P1-2 FIX: Uses ThreadPoolExecutor for parallel generation even in async contexts.
        This avoids the 3-5x slowdown from sequential fallback in FastAPI/async servers.

        Args:
            query: User query
            n_docs: Number of hypothetical documents to generate
            model_ref: LLM model reference
            max_tokens: Max tokens per document
            trace_id: Trace ID for observability

        Returns:
            List of hypothetical document strings
        """
        if n_docs <= 0:
            return []

        if n_docs == 1:
            # Optimization: single doc doesn't need parallelization
            doc = self.generate_hypothetical_document_sync(query, model_ref, max_tokens, trace_id)
            return [doc] if doc else []

        # Always use the shared ThreadPoolExecutor with the sync per-doc path.
        # generate_hypothetical_document_sync uses llm.invoke() (no asyncio), so
        # it is safe whether we are inside an event loop, in a worker thread with
        # no loop (e.g. an evaluation harness calling through asyncio.to_thread),
        # or in a plain sync context.
        #
        # There used to be an `except RuntimeError: asyncio.run(...)` fallback
        # here. It re-entered the async path with a *fresh event loop on every
        # call*, so a cached LangChain client whose httpx.AsyncClient holds an
        # asyncio.Lock bound to the first loop would raise "is bound to a
        # different event loop" from then on. The error was swallowed upstream
        # and HyDE silently returned zero documents, degrading retrieval to
        # no-HyDE without any visible failure.
        executor = _get_hyde_executor()
        # Carry the caller's context into the workers.
        #
        # A bare `executor.submit` starts each worker with a *fresh* context, so
        # any contextvar set by the caller is invisible inside it. Multi-doc
        # HyDE is on by default, so its LLM calls -- the most expensive part of
        # an enhanced query -- were absent from every token count, and a
        # technique that quietly spends more looked free. `asyncio.to_thread`
        # copies the context for exactly this reason; a raw ThreadPoolExecutor
        # does not.
        contexts = [contextvars.copy_context() for _ in range(n_docs)]
        futures = [
            executor.submit(
                ctx.run,
                self.generate_hypothetical_document_sync,
                query,
                model_ref,
                max_tokens,
                trace_id,
            )
            for ctx in contexts
        ]

        docs = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                doc = future.result(timeout=30)  # 30s timeout per doc
                if doc:
                    docs.append(doc)
            except concurrent.futures.TimeoutError:
                logger.warning(f"HyDE doc {i + 1}/{n_docs} timed out")
            except Exception as e:
                logger.warning(f"HyDE doc {i + 1}/{n_docs} failed [{type(e).__name__}]: {e}")

        logger.debug(f"Multi-HyDE: Generated {len(docs)}/{n_docs} docs via ThreadPool")
        return docs


#: Default sampling temperature for query-enhancement LLM calls.
#:
#: Zero, because these components are swept dimensions in benchmark runs and
#: nothing in the harness seeds the sampler. A nonzero default made two runs of
#: an identical configuration disagree, and attributed that disagreement to the
#: component under test. Override per-model settings to trade reproducibility
#: for diversity deliberately.
_DEFAULT_ENHANCER_TEMPERATURE = 0.0


def _numbered_lines(content: str, exclude: Optional[str] = None) -> List[str]:
    """One query per line from an LLM reply, with list numbering removed."""
    out: List[str] = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            if line[0].isdigit() and line[1] in ".):":
                line = line[2:].strip()
            elif line[:2].isdigit() and line[2] in ".):":
                line = line[3:].strip()
            if line and line != exclude:
                out.append(line)
    return out


def _invoke_sync(llm: Any, prompt: str) -> str:
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content if hasattr(response, "content") else str(response)


class QueryExpander:
    """Query expansion through paraphrasing.

    Generates multiple versions of the query to improve recall.
    """

    EXPANSION_PROMPT = """Generate {n} alternative ways to ask this question.
Keep the same meaning but use different words and phrasing.
Return only the questions, one per line.

Original question: {query}

Alternative questions:"""

    def __init__(self, model: Optional[str] = None):
        """Initialize query expander.

        Args:
            model: LLM model for expansion (default from config)
        """
        self.model = model
        self._llm = None
        self._llm_settings: Optional[Dict[str, Any]] = None
        self._llm_lock = threading.Lock()

    def _get_llm(self, model_ref: Optional[str], max_tokens: int) -> Optional[Any]:
        """Lazy load LLM client (thread-safe)."""
        try:
            model_settings = resolve_model_settings(
                model_ref=model_ref,
                default_settings=get_default_model_settings(),
                max_tokens=max_tokens,
                context="query_expansion_model",
            )
            if model_settings.get("temperature") is None:
                # 0.0, not 0.8. Query expansion is a swept dimension in the
                # shipped experiment configs, and nothing seeds the sampler --
                # so a nonzero default made the A/B comparison this framework
                # exists to support non-reproducible, with the variation
                # attributed to the component rather than to the temperature.
                # Still configurable for anyone who wants diverse expansions.
                model_settings["temperature"] = _DEFAULT_ENHANCER_TEMPERATURE

            with self._llm_lock:
                if self._llm is None or self._llm_settings != model_settings:
                    self._llm_settings = model_settings
                    self._llm = get_llm_provider().get_chat_model(model_settings)
                return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM for query expansion: {e}")
            return None

    async def expand_query(
        self,
        query: str,
        n_variations: Optional[int] = None,
        model_ref: Optional[str] = None,
        max_tokens: int = 300,
    ) -> List[str]:
        """Generate query variations.

        Args:
            query: Original query
            n_variations: Number of variations (default from config)

        Returns:
            List of query variations
        """
        if not rag_config.enable_query_expansion:
            return []

        n_variations = n_variations or rag_config.max_query_variations

        model_ref = model_ref or self.model or rag_config.hyde_model
        llm = self._get_llm(model_ref, max_tokens)
        if not llm:
            return []

        try:
            prompt = self.EXPANSION_PROMPT.format(query=query, n=n_variations)
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, "content") else str(response)

            variations = []
            for line in content.strip().split("\n"):
                line = line.strip()
                # Remove numbering if present
                if line and len(line) > 5:
                    if line[0].isdigit() and line[1] in ".):":
                        line = line[2:].strip()
                    elif line[:2].isdigit() and line[2] in ".):":
                        line = line[3:].strip()
                    if line:
                        variations.append(line)

            logger.debug(f"Generated {len(variations)} query variations")
            return variations[:n_variations]

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []


def _expand_query_sync(
    expander: "QueryExpander",
    query: str,
    n: int,
    max_tokens: int = 300,
    model_ref: Optional[str] = None,
) -> List[str]:
    """``QueryExpander.expand_query`` without an event loop, ungated by config.

    Raises when there is no LLM or the call fails; the caller decides whether
    that falls back (serving) or fails the task (evaluation).
    """
    llm = expander._get_llm(model_ref or expander.model or rag_config.hyde_model, max_tokens)
    if not llm:
        raise RuntimeError("query_expansion needs an LLM, and none is configured")
    content = _invoke_sync(llm, expander.EXPANSION_PROMPT.format(query=query, n=n))
    return _numbered_lines(content)[:n]


class QueryDecomposer:
    """Least-to-most decomposition of a question into the sub-questions it needs first.

    Zhou et al. 2022, "Least-to-Most Prompting Enables Complex Reasoning in
    Large Language Models" (arXiv:2205.10625): the decomposition stage, with
    the paper's few-shot format ('To answer the question "Q", we need to know:
    "q1", "q2".') and two of its DROP exemplars (Appendix 9.2.4), verbatim.
    The paper then answers the sub-questions in order; for retrieval, each
    sub-question is searched alongside the question and the lists are fused.

    This replaced a zero-shot "break it into simpler sub-questions" prompt
    with no source, so results recorded before the change came from a
    different method under this name.
    """

    DECOMPOSITION_PROMPT = """Q: The median age in the city was 22.1 years. 10.1% of residents were under the age of 18; 56.2% were between the ages of 18 and 24; 16.1% were from 25 to 44; 10.5% were from 45 to 64; and 7% were 65 years of age or older. Which age group is larger: under the age of 18 or 18 and 24?
A: To answer the question "Which age group is larger: under the age of 18 or 18 and 24?", we need to know: "How many percent were under the age of 18?", "How many percent were between the ages of 18 and 24?".
Q: Non-nationals make up more than half of the population of Bahrain. According to government statistics dated between 2005-2009 roughly 290,000 Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians. How many Pakistanis and Indonesians are in Bahrain?
A: To answer the question "How many Pakistanis and Indonesians are in Bahrain?", we need to know: "How many Pakistanis are in Bahrain?", "How many Indonesians are in Bahrain?".
Q: {query}
A:"""

    def __init__(self, model: Optional[str] = None):
        """Initialize query decomposer.

        Args:
            model: LLM model for decomposition
        """
        self.model = model
        self._llm = None
        self._llm_settings: Optional[Dict[str, Any]] = None
        self._llm_lock = threading.Lock()

    def _get_llm(self, model_ref: Optional[str], max_tokens: int) -> Optional[Any]:
        """Lazy load LLM client (thread-safe)."""
        try:
            model_settings = resolve_model_settings(
                model_ref=model_ref,
                default_settings=get_default_model_settings(),
                max_tokens=max_tokens,
                context="query_decomposition_model",
            )
            if model_settings.get("temperature") is None:
                model_settings["temperature"] = _DEFAULT_ENHANCER_TEMPERATURE
            with self._llm_lock:
                if self._llm is None or self._llm_settings != model_settings:
                    self._llm_settings = model_settings
                    self._llm = get_llm_provider().get_chat_model(model_settings)
                return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM for query decomposition: {e}")
            return None

    async def decompose_query(
        self,
        query: str,
        model_ref: Optional[str] = None,
        max_tokens: int = 300,
    ) -> List[str]:
        """Decompose a complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        if not rag_config.enable_query_decomposition:
            return []

        model_ref = model_ref or self.model or rag_config.hyde_model
        llm = self._get_llm(model_ref, max_tokens)
        if not llm:
            return []

        try:
            prompt = self.DECOMPOSITION_PROMPT.format(query=query)
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, "content") else str(response)
            sub_queries = _sub_questions(content, query)

            logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return []


#: The sub-question list of a least-to-most decomposition, after its lead-in.
_NEED_TO_KNOW = re.compile(r"we need to (?:know|first solve)\s*:(.*)", re.IGNORECASE | re.DOTALL)
_QUOTED = re.compile(r'["\u201c]([^"\u201d]+)["\u201d]')


def _sub_questions(content: str, query: str) -> List[str]:
    """The sub-questions a least-to-most decomposition lists, in order, without the question."""
    match = _NEED_TO_KNOW.search(content)
    if match:
        found = _QUOTED.findall(match.group(1))
    else:
        # A model that ignored the format: one sub-question per line.
        found = _numbered_lines(content)
    out: List[str] = []
    for sub in found:
        sub = sub.strip()
        if sub and sub != query.strip() and sub not in out:
            out.append(sub)
    return out


def _decompose_query_sync(
    decomposer: "QueryDecomposer",
    query: str,
    max_tokens: int = 300,
    model_ref: Optional[str] = None,
) -> List[str]:
    """``QueryDecomposer.decompose_query`` without an event loop, ungated by config.

    Raises when there is no LLM or the call fails.
    """
    llm = decomposer._get_llm(model_ref or decomposer.model or rag_config.hyde_model, max_tokens)
    if not llm:
        raise RuntimeError("query_decomposition needs an LLM, and none is configured")
    content = _invoke_sync(llm, decomposer.DECOMPOSITION_PROMPT.format(query=query))
    return _sub_questions(content, query)


class QueryEnhancer:
    """Combined query enhancement pipeline.

    Applies HyDE, expansion, and decomposition as configured.
    """

    def __init__(self):
        """Initialize query enhancer."""
        self.hyde = HyDEGenerator()
        self.expander = QueryExpander()
        self.decomposer = QueryDecomposer()

    def get_hyde_embedding(
        self,
        query: str,
        embedding_manager,
        n_docs: int = 3,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
        strategy: str = "average",
    ) -> Optional[Any]:
        """Get embedding from multiple HyDE documents with averaging.

        This is the recommended method for multi-document HyDE. It generates
        multiple hypothetical documents, embeds each, and returns an averaged
        embedding. This reduces variance and provides better retrieval quality
        compared to single-document HyDE.

        Args:
            query: Original user query
            embedding_manager: EmbeddingManager instance for encoding
            n_docs: Number of hypothetical documents to generate (default: 3)
            model_ref: LLM model for HyDE generation
            max_tokens: Max tokens per hypothetical document
            trace_id: Trace ID for observability
            strategy: Embedding combination strategy:
                - 'average': Average all embeddings (recommended)
                - 'max_similarity': Use embedding with highest similarity to query

        Returns:
            Averaged embedding as numpy array, or None if all generations fail
        """
        import numpy as np

        # Generate multiple hypothetical documents
        hyde_docs = self.hyde.generate_hypothetical_documents_batch_sync(
            query, n_docs, model_ref, max_tokens, trace_id
        )

        if not hyde_docs:
            logger.warning("Multi-HyDE: No hypothetical documents generated")
            return None

        # Embed all documents
        try:
            embeddings = embedding_manager.encode_documents_batch(hyde_docs, show_progress=False)
        except Exception as e:
            logger.error(f"Multi-HyDE: Failed to embed documents: {e}")
            return None

        if not embeddings:
            return None

        # Combine embeddings based on strategy
        if strategy == "max_similarity":
            # Use the embedding most similar to the query embedding
            query_embedding = embedding_manager.encode_query(query)
            best_idx = 0
            best_sim = -1.0
            for i, emb in enumerate(embeddings):
                sim = embedding_manager.similarity(query_embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            logger.debug(
                f"Multi-HyDE: Selected doc {best_idx + 1}/{len(embeddings)} (sim={best_sim:.3f})"
            )
            return embeddings[best_idx]

        # Default: average strategy
        stacked = np.stack(embeddings)
        averaged = np.mean(stacked, axis=0)

        # Re-normalize for cosine similarity
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        logger.debug(f"Multi-HyDE: Averaged {len(embeddings)} embeddings")
        return averaged

    async def get_hyde_embedding_async(
        self,
        query: str,
        embedding_manager,
        n_docs: int = 3,
        model_ref: Optional[str] = None,
        max_tokens: int = 200,
        trace_id: Optional[str] = None,
        strategy: str = "average",
    ) -> Optional[Any]:
        """Async version of get_hyde_embedding."""
        import numpy as np

        # Generate multiple hypothetical documents
        hyde_docs = await self.hyde.generate_hypothetical_documents_batch(
            query, n_docs, model_ref, max_tokens, trace_id
        )

        if not hyde_docs:
            logger.warning("Multi-HyDE: No hypothetical documents generated")
            return None

        # Embed all documents
        try:
            embeddings = embedding_manager.encode_documents_batch(hyde_docs, show_progress=False)
        except Exception as e:
            logger.error(f"Multi-HyDE: Failed to embed documents: {e}")
            return None

        if not embeddings:
            return None

        # Combine embeddings based on strategy
        if strategy == "max_similarity":
            query_embedding = embedding_manager.encode_query(query)
            best_idx = 0
            best_sim = -1.0
            for i, emb in enumerate(embeddings):
                sim = embedding_manager.similarity(query_embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            return embeddings[best_idx]

        # Default: average strategy
        stacked = np.stack(embeddings)
        averaged = np.mean(stacked, axis=0)

        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged

    def generate_query_variations(self, query: str, n: int = 3) -> List[str]:
        """Generate diverse query reformulations for RAG Fusion.

        Creates n alternative phrasings of the original query to capture
        different aspects and terminology. Results are merged using RRF.

        Unlike expand_query(), this method:
        - Is synchronous (suitable for the sync search pipeline)
        - Does NOT gate on enable_query_expansion (RAG Fusion has its own toggle)
        - Uses a prompt tuned for diverse angles rather than paraphrasing

        Reference: arXiv:2402.03367 (+8-10% accuracy, +30-40% comprehensiveness)

        Args:
            query: Original search query
            n: Number of alternative queries to generate (default: 3)

        Returns:
            List of alternative query strings (may be fewer than n)

        Raises:
            RuntimeError: When no LLM is configured or the call fails. It used
                to return ``[]``, so a fusion arm with no LLM searched the plain
                query and scored as the control; the caller decides whether to
                fall back (serving) or fail the task (evaluation).
        """
        prompt = (
            f"Generate {n} alternative search queries for the following question. "
            f"Each query should approach the topic from a different angle or use "
            f"different terminology. Return ONLY the queries, one per line.\n\n"
            f"Original query: {query}\n\n"
            f"Alternative queries:"
        )

        model_ref = self.expander.model or rag_config.hyde_model
        llm = self.expander._get_llm(model_ref, max_tokens=300)
        if not llm:
            raise RuntimeError("query variations need an LLM, and none is configured")

        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content if hasattr(response, "content") else str(response)

        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        # Filter out numbering patterns
        variations = []
        for line in lines[:n]:
            # Remove common numbering patterns like "1.", "1)", "1 -", etc.
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned and cleaned != query:
                variations.append(cleaned)

        logger.debug(f"RAG Fusion: generated {len(variations)} query variations")
        return variations

    async def enhance_query(self, query: str) -> EnhancedQuery:
        """Enhance a query with all available techniques.

        Args:
            query: Original query

        Returns:
            EnhancedQuery with all enhancements
        """
        enhanced = EnhancedQuery(original_query=query)

        # Run enhancements in parallel
        async def _noop():
            return None

        async def _empty_list():
            return []

        tasks = []

        if rag_config.enable_hyde:
            tasks.append(self.hyde.generate_hypothetical_document(query))
        else:
            tasks.append(_noop())

        if rag_config.enable_query_expansion:
            tasks.append(self.expander.expand_query(query))
        else:
            tasks.append(_empty_list())

        if rag_config.enable_query_decomposition:
            tasks.append(self.decomposer.decompose_query(query))
        else:
            tasks.append(_empty_list())

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # HyDE result
            if not isinstance(results[0], Exception) and results[0]:
                enhanced.hyde_document = results[0]

            # Expansion result
            if not isinstance(results[1], Exception) and results[1]:
                enhanced.query_variations = results[1]

            # Decomposition result
            if not isinstance(results[2], Exception) and results[2]:
                enhanced.sub_queries = results[2]

        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")

        return enhanced

    # NOTE: a former `enhance_query_sync()` was removed here. It had no callers
    # and carried two traps: inside a running loop it silently returned an
    # un-enhanced EnhancedQuery (no warning, so callers could not tell), and
    # outside one it used asyncio.run(), which creates a fresh event loop per
    # call and breaks any cached async LLM client -- the same defect fixed in
    # generate_hypothetical_documents() above. Use the async `enhance_query()`,
    # or the sync single-purpose helpers (`generate_query_variations`,
    # `generate_hypothetical_document_sync`) which call llm.invoke() directly.


# Module-level singleton
_query_enhancer: Optional[QueryEnhancer] = None
_query_enhancer_lock = threading.Lock()


def get_query_enhancer() -> QueryEnhancer:
    """Get or create the query enhancer singleton (thread-safe)."""
    global _query_enhancer
    if _query_enhancer is None:
        with _query_enhancer_lock:
            if _query_enhancer is None:
                _query_enhancer = QueryEnhancer()
    return _query_enhancer


def reset_query_enhancer():
    """Reset the query enhancer singleton."""
    global _query_enhancer
    with _query_enhancer_lock:
        _query_enhancer = None


# ---------------------------------------------------------------------------
# ABC-compliant adapter
# ---------------------------------------------------------------------------

from lasagna.core.base_components import BaseQueryEnhancer  # noqa: E402


class QueryEnhancerAdapter(BaseQueryEnhancer):
    """Adapts ``QueryEnhancer`` to the public ``BaseQueryEnhancer`` ABC.

    Maps ``enhance()`` → ``generate_query_variations()`` (synchronous,
    suitable for the sync search pipeline and RAG Fusion).

    Per the ``BaseQueryEnhancer`` contract, the returned list always
    includes the original query as the first element, followed by
    any generated variations.
    """

    # The pipeline passes the session config and the variation count.
    _forwards_session_config = True

    def __init__(self, **kwargs):
        self._inner = QueryEnhancer()

    def enhance(self, query: str, **kwargs) -> List[str]:
        n = kwargs.get("n") or 3
        variations = self._inner.generate_query_variations(query, n=n)
        # ABC contract: "may include the original" — always include it
        # so callers can rely on at least one result
        return [query] + variations


class HyDEEnhancerAdapter(BaseQueryEnhancer):
    """HyDE behind the public ``BaseQueryEnhancer`` contract.

    Returns the query and one hypothetical passage written to answer it; the
    pipeline searches both and fuses the lists. Raises when no passage can be
    generated (no LLM): returning the query alone made a HyDE arm the control.
    """

    # The pipeline passes the session config (hyde_model) to built-in adapters.
    _forwards_session_config = True

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._inner = HyDEGenerator(model=model)

    def enhance(self, query: str, **kwargs) -> List[str]:
        if not query.strip():
            return [query]
        config = kwargs.get("config")
        doc = self._inner.generate_hypothetical_document_sync(
            query, model_ref=getattr(config, "hyde_model", None)
        )
        if not doc:
            raise RuntimeError("HyDE produced no hypothetical passage (is an LLM configured?)")
        return [query, doc]


class QueryExpansionAdapter(BaseQueryEnhancer):
    """LLM paraphrasing behind the public ``BaseQueryEnhancer`` contract.

    Naming the component is the request, so it runs whatever
    ``enable_query_expansion`` says. Raises without an LLM.
    """

    # The pipeline passes the session config and the variation count.
    _forwards_session_config = True

    def __init__(self, model: Optional[str] = None, n: Optional[int] = None, **kwargs):
        self._inner = QueryExpander(model=model)
        self._n = n

    def enhance(self, query: str, **kwargs) -> List[str]:
        config = kwargs.get("config")
        n = int(
            kwargs.get("n")
            or self._n
            or getattr(config, "max_query_variations", None)
            or rag_config.max_query_variations
            or 3
        )
        if not query.strip():
            return [query]
        return [query] + _expand_query_sync(
            self._inner, query, n, model_ref=getattr(config, "hyde_model", None)
        )


class QueryDecompositionAdapter(BaseQueryEnhancer):
    """Least-to-most decomposition behind the public ``BaseQueryEnhancer`` contract.

    Returns the question and the sub-questions it needs first (see
    ``QueryDecomposer``), at most ``n`` of them once the pipeline caps them.
    """

    # The pipeline passes the session config.
    _forwards_session_config = True

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._inner = QueryDecomposer(model=model)

    def enhance(self, query: str, **kwargs) -> List[str]:
        if not query.strip():
            return [query]
        config = kwargs.get("config")
        return [query] + _decompose_query_sync(
            self._inner, query, model_ref=getattr(config, "hyde_model", None)
        )
